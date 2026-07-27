import os
import json
import pickle
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s',
    level=logging.INFO
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────

class CATSScorer(nn.Module):
    def __init__(self, entity_dim=400, hidden_dim=256, dropout=0.1):
        super().__init__()
        input_dim = entity_dim * 4   # [q; t; q-t; q*t]

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.diversity_proj = nn.Linear(entity_dim, 64)
        self.final_proj     = nn.Linear(1 + 64, 1)
        self.sigmoid        = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, query_emb, team_embs):
        team_mean = team_embs.mean(dim=1)   # (batch, dim)

        interaction = torch.cat([
            query_emb,
            team_mean,
            query_emb - team_mean,
            query_emb * team_mean,
        ], dim=-1)   # (batch, dim*4)

        base_score = self.mlp(interaction)   # (batch, 1)


        team_std     = team_embs.std(dim=1)                   # (batch, dim)
        diversity    = F.relu(self.diversity_proj(team_std))   # (batch, 64)


        combined = torch.cat([base_score, diversity], dim=-1)  # (batch, 65)
        score    = self.sigmoid(self.final_proj(combined))     # (batch, 1)

        return score.squeeze(-1)   # (batch,)

    def score_team(self, query_emb, member_ids, entity_embedding):
        device = query_emb.device
        if not isinstance(member_ids, torch.Tensor):
            member_ids = torch.tensor(member_ids, dtype=torch.long, device=device)
        team_embs = entity_embedding[member_ids].unsqueeze(0)  # (1, n, dim)
        q_emb     = query_emb.unsqueeze(0)                     # (1, dim)
        return self.forward(q_emb, team_embs).item()


# ──────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────

class CATSDataset(Dataset):
    def __init__(self, queries, entity_embedding, tokenizer, model,
                 neg_per_pos=3, min_team_size=2, max_team_size=5,
                 max_length=128, device='cpu'):
        self.device          = device
        self.entity_embedding = entity_embedding.to(device)
        self.tokenizer       = tokenizer
        self.model           = model
        self.max_length      = max_length
        self.neg_per_pos     = neg_per_pos
        self.nentity         = entity_embedding.shape[0]

        self.samples = []
        for q in queries:
            if q.get('source') not in ('grant', 'pi_to_copi', 'copi_to_pi'):
                continue
            gt_team = q.get('ground_truth_team', [])
            if not (min_team_size <= len(gt_team) <= max_team_size):
                continue
            self.samples.append(q)

        log.info(f"CATSDataset: {len(self.samples)} positive samples "
                 f"(from grant/pi_copi queries)")

        log.info("Pre-computing query embeddings...")
        self.query_embs = self._precompute_query_embs()

    def _precompute_query_embs(self):
        import collections
        self.model.eval()
        query_embs = {}
        bs = 32

        with torch.no_grad():
            for start in tqdm(range(0, len(self.samples), bs),
                              desc="Query embeddings"):
                batch = self.samples[start:start+bs]
                texts = [q['query_text'] for q in batch]
                enc   = self.tokenizer(
                    texts, max_length=self.max_length,
                    padding='max_length', truncation=True,
                    return_tensors='pt'
                )
                input_ids      = enc['input_ids'].to(self.device)
                attention_mask = enc['attention_mask'].to(self.device)

                text_emb = self.model.encode_query_text(
                    input_ids, attention_mask)   # (bs, dim)

                import collections as col
                qdict = col.defaultdict(list)
                idict = col.defaultdict(list)
                for i, q in enumerate(batch):
                    s = q['structure']
                    qdict[s].append(q['query'])
                    idict[s].append(i)
                for s in qdict:
                    qdict[s] = torch.LongTensor(
                        np.array(qdict[s], dtype=np.int64)).to(self.device)

                centers = torch.zeros(len(batch), self.model.hidden_dim,
                                      device=self.device)
                for s, idxs in idict.items():
                    idx_t = torch.tensor(idxs, dtype=torch.long)
                    t_emb = text_emb[idx_t]
                    c, _, _ = self.model.embed_query_box_with_text(
                        t_emb, qdict[s], s, 0)
                    centers[idx_t] = c

                for i, q in enumerate(batch):
                    query_embs[q['query_id']] = centers[i].cpu()

        return query_embs

    def __len__(self):
        return len(self.samples) * (1 + self.neg_per_pos)

    def __getitem__(self, idx):
        pos_idx  = idx // (1 + self.neg_per_pos)
        is_neg   = (idx % (1 + self.neg_per_pos)) != 0

        q        = self.samples[pos_idx]
        q_emb    = self.query_embs[q['query_id']]   # (dim,)
        gt_team  = q['ground_truth_team']
        team_size = len(gt_team)

        if not is_neg:
            member_ids = gt_team
            label      = 1.0
        else:
            member_ids = random.sample(range(self.nentity), team_size)
            label      = 0.0

        team_embs = self.entity_embedding[
            torch.tensor(member_ids, dtype=torch.long)
        ].cpu()   # (team_size, dim)

        return q_emb, team_embs, torch.tensor(label, dtype=torch.float32)

    @staticmethod
    def collate_fn(batch):
        query_embs, team_embs_list, labels = zip(*batch)
        query_embs = torch.stack(query_embs)   # (bs, dim)
        labels     = torch.stack(labels)        # (bs,)

        max_size = max(t.shape[0] for t in team_embs_list)
        dim      = team_embs_list[0].shape[1]
        padded   = torch.zeros(len(team_embs_list), max_size, dim)
        for i, t in enumerate(team_embs_list):
            padded[i, :t.shape[0], :] = t

        return query_embs, padded, labels


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────

def train_cats(args):
    device = torch.device('cuda' if args.cuda else 'cpu')
    os.makedirs(args.cats_checkpoint_dir, exist_ok=True)

    log.info("Loading Stage 1 model...")
    from model import KGReasoning

    with open(f'{args.data_dir}/metadata.json') as f:
        meta = json.load(f)
    nentity  = meta['nentity']
    nrelation = meta['nrelation']

    tokenizer    = AutoTokenizer.from_pretrained(args.scibert_model)
    query_name_dict = {('e', ('r',)): '1p'}

    stage1_model = KGReasoning(
        nentity=nentity, nrelation=nrelation,
        hidden_dim=args.entity_dim, gamma=12.0,
        geo='box', use_cuda=args.cuda,
        query_name_dict=query_name_dict,
        freeze_bert=False,
        scibert_model=args.scibert_model,
    )
    ckpt = torch.load(args.stage1_checkpoint,
                      map_location=device)
    stage1_model.load_state_dict(ckpt['model_state_dict'])
    stage1_model = stage1_model.to(device)
    stage1_model.eval()
    for p in stage1_model.parameters():
        p.requires_grad = False

    entity_embedding = stage1_model.entity_embedding.detach()

    with open(f'{args.data_dir}/train_queries.pkl', 'rb') as f:
        train_queries = pickle.load(f)
    with open(f'{args.data_dir}/val_queries.pkl', 'rb') as f:
        val_queries = pickle.load(f)


    log.info("Building CATS training dataset...")
    train_dataset = CATSDataset(
        train_queries, entity_embedding,
        tokenizer, stage1_model,
        neg_per_pos=args.neg_per_pos, device=device,
    )
    val_dataset = CATSDataset(
        val_queries, entity_embedding,
        tokenizer, stage1_model,
        neg_per_pos=1, device=device,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0,
        collate_fn=CATSDataset.collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0,
        collate_fn=CATSDataset.collate_fn
    )

    cats = CATSScorer(
        entity_dim=args.entity_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(cats.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=args.lr * 0.01
    )

    log.info("CATS parameters: %s",
             f"{sum(p.numel() for p in cats.parameters()):,}")

    best_val_auc = 0.0
    criterion    = nn.BCELoss()

    for epoch in range(1, args.max_epochs + 1):
        # Train
        cats.train()
        train_losses = []
        for q_emb, team_embs, labels in tqdm(
                train_loader, desc=f"Epoch {epoch}", leave=False):
            q_emb     = q_emb.to(device)
            team_embs = team_embs.to(device)
            labels    = labels.to(device)

            scores = cats(q_emb, team_embs)
            loss   = criterion(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cats.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Validation
        cats.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for q_emb, team_embs, labels in val_loader:
                q_emb     = q_emb.to(device)
                team_embs = team_embs.to(device)
                scores    = cats(q_emb, team_embs).cpu().numpy()
                val_preds.extend(scores.tolist())
                val_labels.extend(labels.numpy().tolist())

        # AUC
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(val_labels, val_preds)
        except Exception:
            auc = 0.5

        # Accuracy（threshold=0.5）
        preds_binary = [1 if p > 0.5 else 0 for p in val_preds]
        acc = sum(p == l for p, l in zip(preds_binary, val_labels)) / len(val_labels)

        log.info("Epoch %2d | Loss %.4f | Val AUC %.4f | Val Acc %.4f",
                 epoch, np.mean(train_losses), auc, acc)

        if auc > best_val_auc:
            best_val_auc = auc
            torch.save({
                'model_state_dict': cats.state_dict(),
                'entity_dim':       args.entity_dim,
                'hidden_dim':       args.hidden_dim,
                'epoch':            epoch,
                'val_auc':          auc,
            }, f'{args.cats_checkpoint_dir}/cats_best.pt')
            log.info("New best AUC: %.4f → saved", auc)

    log.info("Training complete. Best Val AUC: %.4f", best_val_auc)


def load_cats(checkpoint_path, device='cpu'):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cats = CATSScorer(
        entity_dim=ckpt['entity_dim'],
        hidden_dim=ckpt['hidden_dim'],
    ).to(device)
    cats.load_state_dict(ckpt['model_state_dict'])
    cats.eval()
    log.info(f"CATS loaded from {checkpoint_path} "
             f"(Val AUC: {ckpt.get('val_auc', '?'):.4f})")
    return cats


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',    action='store_true')
    parser.add_argument('--data_dir', default='')
    parser.add_argument('--stage1_checkpoint',
                        default='')
    parser.add_argument('--scibert_model',
                        default='allenai/scibert_scivocab_uncased')
    parser.add_argument('--cats_checkpoint_dir',
                        default='checkpoints_cats')
    parser.add_argument('--entity_dim',  type=int,   default=400)
    parser.add_argument('--hidden_dim',  type=int,   default=256)
    parser.add_argument('--batch_size',  type=int,   default=256)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--max_epochs',  type=int,   default=20)
    parser.add_argument('--neg_per_pos', type=int,   default=3)
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()

    if args.train:
        train_cats(args)
    else:
        parser.print_help()
