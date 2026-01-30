import torch
import numpy as np
import pickle
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import os

class OfflineFeatureExtractor:
    def __init__(self,
                 gnn_embeddings_path='gnn_scholar_embeddings.pt',
                 scholar_metadata_path='scholar_metadata_offline.pkl',
                 scholar_id_mapping_path='scholar_features_from_neo4j.pt'):
        
        print("="*70)
        print("INITIALIZING OFFLINE FEATURE EXTRACTOR")
        print("="*70)
        
        # ===== 1. Load GNN embeddings =====
        print("\n1. Loading GNN embeddings...")
        if not os.path.exists(gnn_embeddings_path):
            raise FileNotFoundError(f"GNN embeddings not found: {gnn_embeddings_path}")
        
        gnn_data = torch.load(gnn_embeddings_path, map_location='cpu')
        self.gnn_embeddings = gnn_data['embeddings'].numpy()
        print(f"   GNN embeddings: {self.gnn_embeddings.shape}")
        
        # ===== 2. Load scholar ID mapping =====
        print("\n2. Loading scholar ID mapping...")
        if not os.path.exists(scholar_id_mapping_path):
            raise FileNotFoundError(f"Scholar ID mapping not found: {scholar_id_mapping_path}")
        
        id_mapping_data = torch.load(scholar_id_mapping_path, map_location='cpu')
        self.scholar_id_to_idx = id_mapping_data['scholar_id_to_idx']
        self.idx_to_scholar_id = id_mapping_data['idx_to_scholar_id']
        print(f"   Scholar ID mapping: {len(self.scholar_id_to_idx)} scholars")
        
        # ===== 3. Load scholar metadata =====
        print("\n3. Loading scholar metadata (offline)...")
        if not os.path.exists(scholar_metadata_path):
            raise FileNotFoundError(f"Scholar metadata not found: {scholar_metadata_path}")
        
        with open(scholar_metadata_path, 'rb') as f:
            self.scholar_metadata = pickle.load(f)
        
        print(f"   Scholar metadata: {len(self.scholar_metadata)} scholars")
        
        # ===== 4. Load SciBERT for query encoding =====
        print("\n4. Loading SciBERT...")
        self.query_encoder = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.query_encoder.eval()
        print("   SciBERT loaded")
        
        print("\n" + "="*70)
        print("OFFLINE FEATURE EXTRACTOR READY")
        print("="*70)
        print("\nAll data loaded from local files (no Neo4j connection needed)")
    
    def encode_query_text(self, nl_query):
        with torch.no_grad():
            inputs = self.tokenizer(
                nl_query,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            outputs = self.query_encoder(**inputs)
            query_emb = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        
        return query_emb
    
    def extract_features_for_candidate(self,
                                       nl_query,
                                       query_text_emb,
                                       candidate_scholar_id,
                                       q2b_retrieval_score,
                                       q2b_rank):
        
        features = {}
        
        # Check if candidate exists
        if candidate_scholar_id not in self.scholar_id_to_idx:
            return self._get_default_features()
        
        candidate_idx = self.scholar_id_to_idx[candidate_scholar_id]
        
        # 1. Query2Box Features
        features['q2b_retrieval_score'] = q2b_retrieval_score
        features['q2b_rank'] = q2b_rank / 100.0
        features['q2b_rank_reciprocal'] = 1.0 / (q2b_rank + 1)
        features['q2b_in_top10'] = 1.0 if q2b_rank < 10 else 0.0
        features['q2b_in_top20'] = 1.0 if q2b_rank < 20 else 0.0
        features['q2b_score_squared'] = q2b_retrieval_score ** 2
        
        # 2. GNN Graph Features 
        candidate_gnn_emb = self.gnn_embeddings[candidate_idx]
        
        features['gnn_emb_mean'] = candidate_gnn_emb.mean()
        features['gnn_emb_std'] = candidate_gnn_emb.std()
        features['gnn_emb_max'] = candidate_gnn_emb.max()
        features['gnn_emb_min'] = candidate_gnn_emb.min()
        features['gnn_emb_norm'] = np.linalg.norm(candidate_gnn_emb)
        
        # 3. Query Text Features
        features['query_text_norm'] = np.linalg.norm(query_text_emb)
        features['query_length'] = len(nl_query.split())
        
        # 4. Scholar Metadata Features
        metadata = self.scholar_metadata.get(candidate_scholar_id, {})
        
        features['h_index'] = metadata.get('h_index', 0) / 100.0
        features['citation_count'] = metadata.get('citation_count', 0) / 10000.0
        features['paper_count'] = metadata.get('paper_count', 0) / 100.0
        features['award_count'] = metadata.get('award_count', 0) / 10.0
        features['leadership_score'] = metadata.get('leadership_score', 0.0)
        features['experience_years'] = metadata.get('experience_years', 0) / 50.0
        features['num_collaborators'] = metadata.get('num_collaborators', 0) / 100.0
        features['recent_papers'] = metadata.get('recent_papers', 0) / 30.0
        features['recent_citations'] = metadata.get('recent_citations', 0) / 5000.0
        features['num_keywords'] = metadata.get('num_keywords', 0) / 50.0
        features['num_affiliations'] = metadata.get('num_affiliations', 0) / 10.0
        features['num_awards_participated'] = metadata.get('num_awards_participated', 0) / 20.0
        
        # 5. Derived Features
        paper_count_raw = metadata.get('paper_count', 0)
        citation_count_raw = metadata.get('citation_count', 0)
        experience_raw = metadata.get('experience_years', 0)
        
        if paper_count_raw > 0:
            features['avg_citations_per_paper'] = citation_count_raw / (paper_count_raw * 100.0)
        else:
            features['avg_citations_per_paper'] = 0.0
        
        if experience_raw > 0:
            features['papers_per_year'] = paper_count_raw / (experience_raw * 10.0)
        else:
            features['papers_per_year'] = 0.0
        
        features['recent_productivity'] = metadata.get('recent_papers', 0) / 30.0
        
        if citation_count_raw > 0:
            features['citation_growth_rate'] = metadata.get('recent_citations', 0) / citation_count_raw
        else:
            features['citation_growth_rate'] = 0.0
        
        if paper_count_raw > 0:
            features['collab_intensity'] = metadata.get('num_collaborators', 0) / (paper_count_raw * 10.0)
        else:
            features['collab_intensity'] = 0.0
        
        # H-index efficiency
        if paper_count_raw > 0:
            features['h_index_efficiency'] = metadata.get('h_index', 0) / paper_count_raw
        else:
            features['h_index_efficiency'] = 0.0
        
        # 6. Cross Features
        features['q2b_x_h_index'] = features['q2b_retrieval_score'] * features['h_index']
        features['q2b_x_citations'] = features['q2b_retrieval_score'] * features['citation_count']
        features['q2b_x_recent_papers'] = features['q2b_retrieval_score'] * features['recent_papers']
        features['q2b_x_leadership'] = features['q2b_retrieval_score'] * features['leadership_score']
        
        features['gnn_norm_x_h_index'] = features['gnn_emb_norm'] * features['h_index']
        features['gnn_norm_x_citations'] = features['gnn_emb_norm'] * features['citation_count']
        
        features['rank_x_h_index'] = features['q2b_rank'] * features['h_index']
        features['rank_x_citations'] = features['q2b_rank'] * features['citation_count']
        
        return features
    
    def _get_default_features(self):
        return {key: 0.0 for key in self._get_feature_names()}
    
    def _get_feature_names(self):
        dummy_features = {
            # Query2Box features (6)
            'q2b_retrieval_score': 0,
            'q2b_rank': 0,
            'q2b_rank_reciprocal': 0,
            'q2b_in_top10': 0,
            'q2b_in_top20': 0,
            'q2b_score_squared': 0,
            
            # GNN features (5)
            'gnn_emb_mean': 0,
            'gnn_emb_std': 0,
            'gnn_emb_max': 0,
            'gnn_emb_min': 0,
            'gnn_emb_norm': 0,
            
            # Query features (2)
            'query_text_norm': 0,
            'query_length': 0,
            
            # Scholar metadata (12)
            'h_index': 0,
            'citation_count': 0,
            'paper_count': 0,
            'award_count': 0,
            'leadership_score': 0,
            'experience_years': 0,
            'num_collaborators': 0,
            'recent_papers': 0,
            'recent_citations': 0,
            'num_keywords': 0,
            'num_affiliations': 0,
            'num_awards_participated': 0,
            
            # Derived features (6)
            'avg_citations_per_paper': 0,
            'papers_per_year': 0,
            'recent_productivity': 0,
            'citation_growth_rate': 0,
            'collab_intensity': 0,
            'h_index_efficiency': 0,
            
            # Cross features (8)
            'q2b_x_h_index': 0,
            'q2b_x_citations': 0,
            'q2b_x_recent_papers': 0,
            'q2b_x_leadership': 0,
            'gnn_norm_x_h_index': 0,
            'gnn_norm_x_citations': 0,
            'rank_x_h_index': 0,
            'rank_x_citations': 0
        }
        
        return sorted(dummy_features.keys())
    
    def extract_features_for_dataset(self, ranking_data_path, output_path):
        print(f"\n{'='*70}")
        print(f"EXTRACTING FEATURES FROM {ranking_data_path}")
        print(f"{'='*70}")
        
        # Load ranking data
        if not os.path.exists(ranking_data_path):
            raise FileNotFoundError(f"Ranking data not found: {ranking_data_path}")
        
        with open(ranking_data_path, 'rb') as f:
            ranking_data = pickle.load(f)
        
        print(f"Total queries: {len(ranking_data)}")
        
        all_features = []
        all_labels = []
        all_groups = []
        
        for sample in tqdm(ranking_data, desc="Extracting features"):
            nl_query = sample['nl_query']
            candidate_ids = sample['candidate_ids']
            retrieval_scores = sample['retrieval_scores']
            labels = sample['labels']
            
            # Encode query once
            query_text_emb = self.encode_query_text(nl_query)
            
            # Extract features for each candidate
            query_features = []
            
            for i, (candidate_id, retrieval_score) in enumerate(zip(candidate_ids, retrieval_scores)):
                features = self.extract_features_for_candidate(
                    nl_query=nl_query,
                    query_text_emb=query_text_emb,
                    candidate_scholar_id=candidate_id,
                    q2b_retrieval_score=retrieval_score,
                    q2b_rank=i
                )
                
                # Convert to array
                feature_array = [features[key] for key in sorted(features.keys())]
                query_features.append(feature_array)
            
            all_features.extend(query_features)
            all_labels.extend(labels)
            all_groups.append(len(candidate_ids))
        
        # Convert to numpy
        all_features = np.array(all_features, dtype=np.float32)
        all_labels = np.array(all_labels, dtype=np.float32)
        all_groups = np.array(all_groups, dtype=np.int32)
        
        print(f"\n{'='*70}")
        print("FEATURE EXTRACTION COMPLETE")
        print(f"{'='*70}")
        print(f"Features shape: {all_features.shape}")
        print(f"Labels shape: {all_labels.shape}")
        print(f"Queries: {len(all_groups)}")
        
        # Get feature names
        feature_names = self._get_feature_names()
        
        # Check for NaN or Inf
        if np.isnan(all_features).any():
            print("WARNING: Features contain NaN, replacing with 0")
            all_features = np.nan_to_num(all_features, nan=0.0)
        
        if np.isinf(all_features).any():
            print("WARNING: Features contain Inf, clipping")
            all_features = np.nan_to_num(all_features, posinf=1.0, neginf=-1.0)
        
        # Save
        output_data = {
            'features': all_features,
            'labels': all_labels,
            'groups': all_groups,
            'feature_names': feature_names
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(output_data, f)
        
        print(f"\nSaved to {output_path}")
        print(f"\nFeature summary ({len(feature_names)} features):")
        print(f"  Query2Box features: 6")
        print(f"  GNN features: 5")
        print(f"  Query features: 2")
        print(f"  Scholar metadata: 12")
        print(f"  Derived features: 6")
        print(f"  Cross features: 8")
        print(f"  Total: 39 features")
        
        return all_features, all_labels, all_groups, feature_names


if __name__ == "__main__":
    # Initialize offline extractor
    extractor = OfflineFeatureExtractor(
        gnn_embeddings_path='gnn_scholar_embeddings.pt',
        scholar_metadata_path='scholar_metadata_offline.pkl',
        scholar_id_mapping_path='scholar_features_from_neo4j.pt'
    )
    
    # Extract features for all datasets
    for split in ['train', 'val', 'test']:
        print(f"\n{'='*70}")
        print(f"PROCESSING {split.upper()} SET")
        print(f"{'='*70}")
        
        extractor.extract_features_for_dataset(
            ranking_data_path=f'ranking_data_final/{split}.pkl',
            output_path=f'xgboost_features/{split}_features.pkl'
        )
    
    print("\n" + "="*70)
    print("ALL FEATURE EXTRACTION COMPLETE!")
    print("="*70)