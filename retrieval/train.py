import os
import json
import pickle
import logging
import argparse
import collections

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from model import KGReasoning

logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s',
    level=logging.INFO
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────

class TextQuery2BoxDataset(Dataset):
    def __init__(self, queries, nentity, negative_sample_size,
                 tokenizer, max_length=128):
        self.queries               = queries
        self.nentity               = nentity
        self.negative_sample_size  = negative_sample_size
        self.tokenizer             = tokenizer
        self.max_length            = max_length

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        q = self.queries[idx]
        neg = np.random.choice(self.nentity,
                               size=self.negative_sample_size,
                               replace=False)
        # tokenize query_text
        enc = self.tokenizer(
            q['query_text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'positive_sample':  q['positive_sample'],
            'negative_samples': neg,
            'query':            q['query'],
            'query_structure':  q['structure'],
            'input_ids':        enc['input_ids'].squeeze(0),        # (seq_len,)
            'attention_mask':   enc['attention_mask'].squeeze(0),   # (seq_len,)
        }

    def collate_fn(self, batch):
        bs = len(batch)
        pos = torch.LongTensor([b['positive_sample'] for b in batch])
        neg = torch.LongTensor(
            np.stack([b['negative_samples'] for b in batch]))
        sub = torch.ones(bs, dtype=torch.float32)

        input_ids      = torch.stack([b['input_ids']      for b in batch])
        attention_mask = torch.stack([b['attention_mask'] for b in batch])

        # group queries by structure
        qdict, idict = collections.defaultdict(list), collections.defaultdict(list)
        for i, b in enumerate(batch):
            s = b['query_structure']
            qdict[s].append(b['query'])
            idict[s].append(i)
        for s in qdict:
            qdict[s] = torch.LongTensor(np.array(qdict[s], dtype=np.int64))

        return pos, neg, sub, dict(qdict), dict(idict), input_ids, attention_mask


# ──────────────────────────────────────────────────────────────────────
# Evaluation helper
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, queries, nentity, tokenizer, args, desc='Eval'):
    model.eval()
    device = torch.device('cuda' if args.cuda else 'cpu')

    all_entity_emb = model.entity_embedding.detach()  # (nentity, dim)

    all_mrr, all_h1, all_h3, all_h10, all_h20 = [], [], [], [], []
    all_ndcg10, all_ndcg20 = [], []
    bs = args.test_batch_size

    for start in tqdm(range(0, len(queries), bs), desc=desc, leave=False):
        batch_qs = queries[start : start + bs]
        actual_bs = len(batch_qs)

        # ── tokenize ────────────────────────────────────────────────
        texts = [q['query_text'] for q in batch_qs]
        enc = tokenizer(
            texts,
            max_length=args.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids      = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)

        # ── build qdict / idict ──────────────────────────────────────
        qdict, idict = collections.defaultdict(list), collections.defaultdict(list)
        for i, q in enumerate(batch_qs):
            s = q['structure']
            qdict[s].append(q['query'])
            idict[s].append(i)
        for s in qdict:
            qdict[s] = torch.LongTensor(
                np.array(qdict[s], dtype=np.int64)).to(device)

        query_text_emb = model.encode_query_text(input_ids, attention_mask)
        centers = torch.zeros(actual_bs, model.hidden_dim, device=device)
        offsets = torch.zeros(actual_bs, model.hidden_dim, device=device)

        for s, idxs_list in idict.items():
            idx_t   = torch.tensor(idxs_list, dtype=torch.long)
            t_emb   = query_text_emb[idx_t]               # (n, dim)
            queries_t = qdict[s]                           # (n, 2)
            c, o, _ = model.embed_query_box_with_text(
                t_emb, queries_t, s, 0)
            centers[idx_t] = c
            offsets[idx_t] = o

        # ── score all entities ────────────────────────────────────────
        # cal_logit_box: (nentity, dim) vs (actual_bs, 1, dim)
        centers_exp = centers.unsqueeze(1)    # (bs, 1, dim)
        offsets_exp = offsets.unsqueeze(1)    # (bs, 1, dim)
        ent_emb_exp = all_entity_emb.unsqueeze(0)  # (1, nentity, dim)

        delta        = (ent_emb_exp - centers_exp).abs()          # (bs, nentity, dim)
        dist_out     = F.relu(delta - offsets_exp)
        dist_in      = torch.min(delta, offsets_exp)
        logits       = (model.gamma
                        - dist_out.norm(p=1, dim=-1)
                        - model.cen * dist_in.norm(p=1, dim=-1))  # (bs, nentity)


        for i, q in enumerate(batch_qs):
            gt_team = set(q['ground_truth_team'])
            scores_base = logits[i]   # (nentity,)

            for pos_entity in q['ground_truth_team']:
                scores = scores_base.clone()
                for gt in gt_team:
                    if gt != pos_entity:
                        scores[gt] = float('-inf')

                argsort = torch.argsort(scores, descending=True).cpu()
                pos_positions = (argsort == pos_entity).nonzero(as_tuple=True)[0]
                if len(pos_positions) == 0:
                    continue
                rank = pos_positions[0].item() + 1  

                all_mrr.append(1.0 / rank)
                all_h1.append(float(rank <= 1))
                all_h3.append(float(rank <= 3))
                all_h10.append(float(rank <= 10))
                all_h20.append(float(rank <= 20))
                all_ndcg10.append(1.0 / np.log2(rank + 1) if rank <= 10 else 0.0)
                all_ndcg20.append(1.0 / np.log2(rank + 1) if rank <= 20 else 0.0)

    def safe_mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    return {
        'MRR':     safe_mean(all_mrr),
        'HR@1':    safe_mean(all_h1),
        'HR@3':    safe_mean(all_h3),
        'HR@10':   safe_mean(all_h10),
        'HR@20':   safe_mean(all_h20),
        'NDCG@10': safe_mean(all_ndcg10),
        'NDCG@20': safe_mean(all_ndcg20),
    }


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────

def train(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────
    log.info("Loading data from %s", args.data_dir)
    with open(f'{args.data_dir}/metadata.json') as f:
        meta = json.load(f)
    nentity  = meta['nentity']
    nrelation = meta['nrelation']

    with open(f'{args.data_dir}/train_queries.pkl', 'rb') as f:
        train_queries = pickle.load(f)
    with open(f'{args.data_dir}/val_queries.pkl', 'rb') as f:
        val_queries = pickle.load(f)

    log.info("Entities: %d | Train: %d | Val: %d",
             nentity, len(train_queries), len(val_queries))

    # ── Tokenizer ────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.scibert_model)

    # ── Model ────────────────────────────────────────────────────────
    query_name_dict = {('e', ('r',)): '1p'}

    model = KGReasoning(
        nentity          = nentity,
        nrelation        = nrelation,
        hidden_dim       = args.hidden_dim,
        gamma            = args.gamma,
        geo              = 'box',
        test_batch_size  = args.test_batch_size,
        use_cuda         = args.cuda,
        query_name_dict  = query_name_dict,
        freeze_bert      = True,
        scibert_model    = args.scibert_model,
    )

    if args.cuda:
        model = model.cuda()

    log.info("Parameters: %s",
             f"{sum(p.numel() for p in model.parameters()):,}")
    log.info("Trainable: %s",
             f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── Optimizer & Scheduler ─────────────────────────────────────────
    phase1_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(phase1_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=args.lr * 0.01
    )

    # ── Resume from checkpoint ────────────────────────────────────────
    start_step = 0
    best_mrr   = 0.0
    if args.resume and os.path.exists(args.resume):
        log.info("Resuming from: %s", args.resume)
        ckpt = torch.load(args.resume,
                          map_location='cuda' if args.cuda else 'cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        start_step = ckpt.get('step', 0)
        best_mrr   = ckpt.get('mrr', 0.0)
        if 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                log.info("Optimizer state restored")
            except ValueError:
                log.warning("Optimizer state mismatch (Phase 1→2), using fresh optimizer")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.max_steps,
            eta_min=args.lr * 0.01,
            last_epoch=start_step - 1
        )
        log.info("Resumed from step %d, best MRR: %.4f", start_step, best_mrr)
    else:
        best_mrr = 0.0

    # ── DataLoader ───────────────────────────────────────────────────
    train_dataset = TextQuery2BoxDataset(
        train_queries, nentity,
        negative_sample_size = args.negative_sample_size,
        tokenizer            = tokenizer,
        max_length           = args.max_length,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size   = args.batch_size,
        shuffle      = True,
        num_workers  = args.num_workers,
        collate_fn   = train_dataset.collate_fn,
        pin_memory   = args.cuda,
    )
    train_iter = iter(train_loader)

    # ── Training loop ────────────────────────────────────────────────
    patience_count = 0
    step           = start_step
    model.train()

    # Phase 2 unfreeze point
    unfreeze_step = int(args.max_steps * args.unfreeze_ratio)
    phase2_started = False

    log.info("Starting training (Phase 1: SciBERT frozen)")
    log.info("SciBERT will unfreeze at step %d / %d",
             unfreeze_step, args.max_steps)

    while step < args.max_steps:
        if step == unfreeze_step and not phase2_started:
            log.info("★ Phase 2: unfreezing SciBERT")
            
            model.unfreeze_bert()
            optimizer = torch.optim.Adam([
                {'params': [p for n, p in model.named_parameters()
                            if 'bert' not in n and p.requires_grad],
                 'lr': args.lr},
                {'params': model.text_encoder.bert.parameters(),
                 'lr': args.lr * 0.1},
            ])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.max_steps - unfreeze_step,
                eta_min=args.lr * 0.001
            )
            phase2_started = True

        # ── Get batch ─────────────────────────────────────────────────
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        (pos, neg, sub, qdict, idict,
         input_ids, attention_mask) = batch

        if args.cuda:
            pos            = pos.cuda()
            neg            = neg.cuda()
            sub            = sub.cuda()
            input_ids      = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            for s in qdict:
                qdict[s] = qdict[s].cuda()

        # ── Forward ───────────────────────────────────────────────────
        pos_logit, neg_logit, sub, _ = model(
            pos, neg, sub, qdict, idict,
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        neg_score = F.logsigmoid(-neg_logit).mean(dim=1)
        pos_score = F.logsigmoid(pos_logit).squeeze(dim=1)

        pos_loss = -(sub * pos_score).sum() / sub.sum()
        neg_loss = -(sub * neg_score).sum() / sub.sum()
        loss     = (pos_loss + neg_loss) / 2

        if torch.isnan(loss) or torch.isinf(loss):
            log.warning("NaN/Inf at step %d — skipping", step)
            step += 1
            continue

        # ── Backward ──────────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1

        # ── Log ───────────────────────────────────────────────────────
        if step % args.log_steps == 0:
            log.info("Step %6d/%d | Loss %.4f | Pos %.4f | Neg %.4f | LR %.2e",
                     step, args.max_steps,
                     loss.item(), pos_loss.item(), neg_loss.item(),
                     scheduler.get_last_lr()[0])

        # ── Validation ────────────────────────────────────────────────
        if step % args.val_steps == 0:
            metrics = evaluate(
                model, val_queries, nentity, tokenizer, args,
                desc=f'Val step {step}'
            )
            log.info("VAL  | MRR %.4f | HR@1 %.4f | HR@10 %.4f | HR@20 %.4f | NDCG@10 %.4f | NDCG@20 %.4f",
                     metrics['MRR'], metrics['HR@1'],
                     metrics['HR@10'], metrics['HR@20'],
                     metrics['NDCG@10'], metrics['NDCG@20'])

            if metrics['MRR'] > best_mrr:
                best_mrr       = metrics['MRR']
                patience_count = 0
                torch.save({
                    'model_state_dict':     model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'step':   step,
                    'mrr':    best_mrr,
                    'metrics': metrics,
                }, f'{args.checkpoint_dir}/best_model.pt')
                log.info("New best MRR: %.4f → saved", best_mrr)
            else:
                patience_count += 1
                log.info("No improvement (%d/%d)", patience_count, args.patience)
                if patience_count >= args.patience:
                    log.info("Early stopping at step %d", step)
                    break

            model.train()

        # ── Periodic checkpoint ───────────────────────────────────────
        if step % args.checkpoint_steps == 0:
            torch.save({
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step,
            }, f'{args.checkpoint_dir}/step_{step}.pt')

    log.info("Training complete. Best val MRR: %.4f", best_mrr)


# ──────────────────────────────────────────────────────────────────────
# Test
# ──────────────────────────────────────────────────────────────────────

def test(args):
    with open(f'{args.data_dir}/metadata.json') as f:
        meta = json.load(f)
    nentity  = meta['nentity']
    nrelation = meta['nrelation']

    with open(f'{args.data_dir}/test_queries.pkl', 'rb') as f:
        test_queries = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.scibert_model)
    query_name_dict = {('e', ('r',)): '1p'}

    model = KGReasoning(
        nentity=nentity, nrelation=nrelation,
        hidden_dim=args.hidden_dim, gamma=args.gamma,
        geo='box', test_batch_size=args.test_batch_size,
        use_cuda=args.cuda, query_name_dict=query_name_dict,
        freeze_bert=False, scibert_model=args.scibert_model,
    )

    ckpt = torch.load(f'{args.checkpoint_dir}/best_model.pt',
                      map_location='cuda' if args.cuda else 'cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    if args.cuda:
        model = model.cuda()

    metrics = evaluate(model, test_queries, nentity, tokenizer, args,
                       desc='Test')

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:8s}: {v:.4f}")
    print("=" * 60)

    source_queries = collections.defaultdict(list)
    for q in test_queries:
        source_queries[q.get('source', 'unknown')].append(q)

    print("\nResults by source:")
    print(f"  {'Source':<12} {'n':>6}  {'MRR':>7}  {'HR@1':>7}  "
          f"{'HR@10':>7}  {'NDCG@10':>9}")
    print(f"  {'-'*55}")
    for src, qs in sorted(source_queries.items()):
        m = evaluate(model, qs, nentity, tokenizer, args,
                     desc=f'Test-{src}')
        print(f"  [{src:<10}] n={len(qs):5d} | "
              f"MRR {m['MRR']:.4f} | HR@1 {m['HR@1']:.4f} | "
              f"HR@10 {m['HR@10']:.4f} | NDCG@10 {m['NDCG@10']:.4f}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data_dir',        default='training_data_v3')
    parser.add_argument('--checkpoint_dir',  default='checkpoints')

    # Model
    parser.add_argument('--hidden_dim',      type=int,   default=400)
    parser.add_argument('--gamma',           type=float, default=12.0)
    parser.add_argument('--scibert_model',   default='allenai/scibert_scivocab_uncased')
    parser.add_argument('--max_length',      type=int,   default=128)

    # Training
    parser.add_argument('--batch_size',           type=int,   default=256)
    parser.add_argument('--negative_sample_size', type=int,   default=128)
    parser.add_argument('--lr',                   type=float, default=1e-4)
    parser.add_argument('--max_steps',            type=int,   default=50000)
    parser.add_argument('--unfreeze_ratio',       type=float, default=0.5)
    parser.add_argument('--num_workers',          type=int,   default=4)
    parser.add_argument('--log_steps',            type=int,   default=100)
    parser.add_argument('--val_steps',            type=int,   default=2000)
    parser.add_argument('--checkpoint_steps',     type=int,   default=5000)
    parser.add_argument('--patience',             type=int,   default=5)

    # Test
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--do_test',         action='store_true')

    # System
    parser.add_argument('--cuda',   action='store_true', default=False)
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()
    if args.cuda and not torch.cuda.is_available():
        args.cuda = False
        log.warning("CUDA not available, using CPU")

    if args.do_test:
        test(args)
    else:
        train(args)
