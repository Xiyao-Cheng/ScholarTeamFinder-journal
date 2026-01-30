import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models import KGReasoning
import pickle
import json
import os
import logging
import numpy as np

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


class Query2BoxTrainDataset:
    
    def __init__(self, queries, nentity, nrelation, negative_sample_size):
        self.queries = queries
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        query_data = self.queries[idx]
        negative_samples = np.random.choice(
            self.nentity,
            size=self.negative_sample_size,
            replace=False
        )
        
        return {
            'positive_sample': query_data['positive_sample'],
            'negative_samples': negative_samples,
            'query': query_data['query'],
            'query_structure': query_data['structure']
        }
    
    def collate_fn(self, batch):
        batch_size = len(batch)
        
        positive_samples = np.array([item['positive_sample'] for item in batch], dtype=np.int64)
        positive_samples = torch.from_numpy(positive_samples)
        
        negative_samples = np.stack([item['negative_samples'] for item in batch], axis=0)
        negative_samples = torch.from_numpy(negative_samples).long()
        
        subsampling_weight = torch.ones(batch_size, dtype=torch.float32)
        
        # Group by structure
        batch_queries_dict = {}
        batch_idxs_dict = {}
        
        for idx, item in enumerate(batch):
            structure = item['query_structure']
            if structure not in batch_queries_dict:
                batch_queries_dict[structure] = []
                batch_idxs_dict[structure] = []
            
            batch_queries_dict[structure].append(item['query'])
            batch_idxs_dict[structure].append(idx)
        
        for structure in batch_queries_dict:
            queries_np = np.array(batch_queries_dict[structure], dtype=np.int64)
            batch_queries_dict[structure] = torch.from_numpy(queries_np)
        
        return (
            positive_samples,
            negative_samples,
            subsampling_weight,
            batch_queries_dict,
            batch_idxs_dict
        )


def ensure_model_on_cuda(model):
    # move to GPU
    model = model.to('cuda')
    
    for name, param in model.named_parameters():
        if param.device.type != 'cuda':
            param.data = param.data.cuda()
    
    for name, buffer in model.named_buffers():
        if buffer.device.type != 'cuda':
            buffer.data = buffer.data.cuda()
    
    if hasattr(model, 'batch_entity_range'):
        if model.batch_entity_range.device.type != 'cuda':
            model.batch_entity_range = model.batch_entity_range.cuda()
    
    return model


def train_query2box(args):
    # keep the same loss as model part
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print("="*70)
    print("LOADING DATA")
    print("="*70)
    
    with open(f'{args.data_dir}/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    nentity = metadata['nentity']
    nrelation = metadata['nrelation']
    print(f"Entities: {nentity}, Relations: {nrelation}")
    
    with open(f'{args.data_dir}/train_queries.pkl', 'rb') as f:
        train_queries = pickle.load(f)
    print(f"Train queries: {len(train_queries)}")
    
    # Query name dict
    query_name_dict = {
        ('e', ('r',)): '1p',
        ('e', ('r', 'r')): '2p',
        ('e', ('r', 'r', 'r')): '3p',
        (('e', ('r',)), ('e', ('r',))): '2i',
        (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
        ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
        (('e', ('r', 'r')), ('e', ('r',))): 'pi',
    }
    
    # Model
    print("\n" + "="*70)
    print("INITIALIZING MODEL")
    print("="*70)
    
    model = KGReasoning(
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        geo='box',
        use_cuda=False,
        query_name_dict=query_name_dict
    )
    
    if args.cuda:
        print("Moving to GPU...")
        model = ensure_model_on_cuda(model)
        print("✅ Model on GPU")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    
    # Dataset
    print("\n" + "="*70)
    print("CREATING DATALOADER")
    print("="*70)
    
    train_dataset = Query2BoxTrainDataset(
        train_queries,
        nentity=nentity,
        nrelation=nrelation,
        negative_sample_size=args.negative_sample_size
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_dataset.collate_fn
    )
    
    print(f"Batch size: {args.batch_size}")
    
    # Training
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    model.train()
    step = 0
    train_iterator = iter(train_dataloader)
    best_loss = float('inf')
    
    while step < args.max_steps:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dataloader)
            batch = next(train_iterator)
        
        positive_sample, negative_sample, subsampling_weight, batch_queries_dict, batch_idxs_dict = batch
        
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()
            
            for query_structure in batch_queries_dict:
                batch_queries_dict[query_structure] = batch_queries_dict[query_structure].cuda()
        
        # Forward
        positive_logit, negative_logit, subsampling_weight, _ = model(
            positive_sample,
            negative_sample,
            subsampling_weight,
            batch_queries_dict,
            batch_idxs_dict
        )
        
        negative_score = F.logsigmoid(-negative_logit).mean(dim=1)
        positive_score = F.logsigmoid(positive_logit).squeeze(dim=1)
        
        positive_sample_loss = - (subsampling_weight * positive_score).sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()
        
        positive_sample_loss /= subsampling_weight.sum()
        negative_sample_loss /= subsampling_weight.sum()
        
        loss = (positive_sample_loss + negative_sample_loss) / 2
        
        # Safety check
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"❌ NaN/Inf at step {step}")
            break
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        step += 1
        
        # Log
        if step % args.log_steps == 0:
            logging.info(
                f"Step {step:6d}/{args.max_steps} | "
                f"Loss: {loss.item():.4f} | "
                f"Pos: {positive_sample_loss.item():.4f} | "
                f"Neg: {negative_sample_loss.item():.4f}"
            )
        
        # Checkpoint
        if step % args.checkpoint_steps == 0:
            checkpoint_path = f'{args.checkpoint_dir}/query2box_step_{step}.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step,
                'loss': loss.item()
            }, checkpoint_path)
            logging.info(f"✅ Checkpoint: {checkpoint_path}")
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'step': step,
                    'loss': loss.item()
                }, f'{args.checkpoint_dir}/query2box_best.pt')
                logging.info(f"⭐ Best: {best_loss:.4f}")
    
    print("\n✅ TRAINING COMPLETE!")
    print(f"Best loss: {best_loss:.4f}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_dim', type=int, default=400)
    parser.add_argument('--gamma', type=float, default=12.0)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--negative_sample_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--data_dir', type=str, default='training_data_yelp/training_datav2')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_steps', type=int, default=100)
    parser.add_argument('--checkpoint_steps', type=int, default=5000)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_yelp')
    parser.add_argument('--cuda', action='store_true', default=False)
    
    args = parser.parse_args()
    
    if args.cuda and not torch.cuda.is_available():
        args.cuda = False
    
    train_query2box(args)