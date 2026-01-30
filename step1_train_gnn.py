import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
import numpy as np

class GraphSAGEEncoder(nn.Module):
    def __init__(self, 
                 in_features=10,        # scholar features
                 hidden_dim=256,
                 embedding_dim=128,     # final embedding
                 num_layers=3):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr='mean'))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr='mean'))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Last layer (output embedding)
        self.convs.append(SAGEConv(hidden_dim, embedding_dim, aggr='mean'))
        
        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        self.temperature = 0.07
    
    def forward(self, x, edge_index):
        """
        Forward pass
        
        Args:
            x: [num_nodes, in_features]
            edge_index: [2, num_edges]
        
        Returns:
            embeddings: [num_nodes, embedding_dim]
        """
        # Input projection
        h = self.input_proj(x)
        
        # GraphSAGE layers
        for i, (conv, bn) in enumerate(zip(self.convs[:-1], self.batch_norms)):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=0.3, training=self.training)
        
        # Last layer (no activation)
        h = self.convs[-1](h, edge_index)
        
        return h
    
    def get_projection(self, embeddings):
        return self.projection(embeddings)
    
    def contrastive_loss(self, embeddings, positive_pairs, negative_pairs):
        """
        InfoNCE loss (NT-Xent)
        
        Args:
            embeddings: [num_nodes, embedding_dim]
            positive_pairs: List[(anchor_idx, positive_idx)]
            negative_pairs: List[(anchor_idx, negative_idx)]
        """
        # Project embeddings
        z = self.get_projection(embeddings)
        z = F.normalize(z, p=2, dim=-1)
        
        total_loss = 0
        count = 0
        
        for (anchor_idx, pos_idx) in positive_pairs:
            # Positive similarity
            pos_sim = torch.dot(z[anchor_idx], z[pos_idx]) / self.temperature
            
            # Negative similarities
            neg_sims = []
            for (anc_idx, neg_idx) in negative_pairs:
                if anc_idx == anchor_idx:
                    neg_sim = torch.dot(z[anchor_idx], z[neg_idx]) / self.temperature
                    neg_sims.append(neg_sim)
            
            if neg_sims:
                # InfoNCE loss
                neg_sims = torch.stack(neg_sims)
                logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])
                labels = torch.zeros(1, dtype=torch.long, device=logits.device)
                
                loss = F.cross_entropy(logits.unsqueeze(0), labels)
                total_loss += loss
                count += 1
        
        return total_loss / max(count, 1)


def prepare_training_pairs(edge_index, num_nodes, num_negatives=5):
    import time
    start_time = time.time()
    
    print("Preparing training pairs (optimized version)...")
    
    positive_pairs = []
    negative_pairs = []

    print("  Building positive edge set...")
    edge_index_np = edge_index.cpu().numpy()
    positive_set = set()
    
    for i in range(edge_index_np.shape[1]):
        src, dst = edge_index_np[0, i], edge_index_np[1, i]
        positive_pairs.append((src, dst))
        positive_set.add((src, dst))
        positive_set.add((dst, src))  
    
    print(f"  Positive pairs: {len(positive_pairs)}")
    
    print("  Sampling negative pairs...")
    
    num_total_negatives = len(positive_pairs) * num_negatives
    
    batch_size = 100000
    sampled = 0
    
    while sampled < num_total_negatives:
        src_batch = np.random.randint(0, num_nodes, size=batch_size)
        dst_batch = np.random.randint(0, num_nodes, size=batch_size)
        

        for src, dst in zip(src_batch, dst_batch):
            if sampled >= num_total_negatives:
                break
            
            
            if src != dst and (src, dst) not in positive_set:
                negative_pairs.append((int(src), int(dst)))
                sampled += 1
        
        if sampled % 100000 == 0:
            print(f"    Sampled {sampled}/{num_total_negatives} negative pairs...")
    
    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f} seconds")
    print(f"  Positive pairs: {len(positive_pairs)}")
    print(f"  Negative pairs: {len(negative_pairs)}")
    
    return positive_pairs, negative_pairs


def train_gnn(scholar_features, edge_index, num_epochs=100, device='cuda'):
    
    num_scholars = scholar_features.size(0)
    
    print("="*70)
    print("TRAINING GRAPHSAGE")
    print("="*70)
    print(f"Number of scholars: {num_scholars}")
    print(f"Number of edges: {edge_index.size(1)}")
    print(f"Feature dimension: {scholar_features.size(1)}")
    
    # Model
    model = GraphSAGEEncoder(
        in_features=scholar_features.size(1),
        hidden_dim=256,
        embedding_dim=128,
        num_layers=3
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Move data to device
    scholar_features = scholar_features.to(device)
    edge_index = edge_index.to(device)
    
    # Prepare training pairs
    print("\nPreparing training pairs...")
    positive_pairs, negative_pairs = prepare_training_pairs(
        edge_index, 
        num_scholars, 
        num_negatives=2
    )
    print(f"Positive pairs: {len(positive_pairs)}")
    print(f"Negative pairs: {len(negative_pairs)}")
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        
        # Forward
        embeddings = model(scholar_features, edge_index)
        
        # Contrastive loss
        loss = model.contrastive_loss(embeddings, positive_pairs, negative_pairs)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item()
            }, 'checkpoints/graphsage_best.pt')
    
    # Get final embeddings
    print("\n" + "="*70)
    print("EXTRACTING FINAL EMBEDDINGS")
    print("="*70)
    
    model.eval()
    with torch.no_grad():
        final_embeddings = model(scholar_features, edge_index)
        final_embeddings = final_embeddings.cpu()
    
    print(f"Embeddings shape: {final_embeddings.shape}")
    
    # Save embeddings
    torch.save({
        'embeddings': final_embeddings,
        'num_scholars': num_scholars
    }, 'gnn_scholar_embeddings.pt')
    
    print("Saved embeddings to gnn_scholar_embeddings.pt")
    
    return model, final_embeddings


if __name__ == "__main__":
    # Load data
    print("Loading data...")
    
    # Scholar features [num_scholars, 10]
    scholar_features = load_scholar_features()  
    
    # Graph structure [2, num_edges]
    edge_index = load_edge_index()  
    
    # Train
    model, embeddings = train_gnn(
        scholar_features,
        edge_index,
        num_epochs=100,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )