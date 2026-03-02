import numpy as np
import pickle
import json
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import torch

class AdvancedTeamFormation:
    def __init__(self,
                 ranker_model_path='checkpoints/xgboost_ranker.json',
                 gnn_embeddings_path='gnn_scholar_embeddings.pt',
                 metadata_path='scholar_metadata_offline.pkl'):
        
        print("="*70)
        print("ADVANCED TEAM FORMATION")
        print("="*70)
        
        # 1. load ranker
        import xgboost as xgb
        self.ranker = xgb.Booster()
        self.ranker.load_model(ranker_model_path)
        print(f"✅ Loaded ranker")
        
        # 2. get GNN embeddings 
        gnn_data = torch.load(gnn_embeddings_path, map_location='cpu')
        self.gnn_embeddings = gnn_data['embeddings'].numpy()
        self.scholar_id_to_idx = gnn_data.get('scholar_id_to_idx', {})
        print(f"✅ Loaded GNN embeddings: {self.gnn_embeddings.shape}")
        
        # 3. load metadata 
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        print(f"✅ Loaded metadata for {len(self.metadata)} scholars")
    
    def compute_quality_score(self, candidate_features):
        import xgboost as xgb
        
        dmatrix = xgb.DMatrix(candidate_features)
        scores = self.ranker.predict(dmatrix)
        
        return scores
    
    def compute_diversity_score(self, team_member_ids):
        embeddings = []
        for member_id in team_member_ids:
            idx = self.scholar_id_to_idx.get(member_id)
            if idx is not None:
                embeddings.append(self.gnn_embeddings[idx])
        
        if len(embeddings) < 2:
            return 0.0
        
        embeddings = np.array(embeddings)
        
        similarity_matrix = cosine_similarity(embeddings)
        
        n = len(embeddings)
        avg_similarity = (similarity_matrix.sum() - n) / (n * (n - 1))
        
        diversity = 1 - avg_similarity
        
        return diversity
    
    def compute_complementarity_score(self, team_member_ids):
        h_indices = []
        experience_years = []
        num_keywords_list = []
        
        for member_id in team_member_ids:
            metadata = self.metadata.get(member_id, {})
            h_indices.append(metadata.get('h_index', 0))
            experience_years.append(metadata.get('experience_years', 0))
            num_keywords_list.append(metadata.get('num_keywords', 0))
        

        h_std = np.std(h_indices) if h_indices else 0
        exp_std = np.std(experience_years) if experience_years else 0
        kw_std = np.std(num_keywords_list) if num_keywords_list else 0
        

        complementarity = (h_std / 50.0 + exp_std / 20.0 + kw_std / 20.0) / 3.0
        
        return complementarity
    
    def evaluate_team(self, 
                      team_member_ids,
                      candidate_ids,
                      candidate_features,
                      alpha=0.6,
                      beta=0.3,
                      gamma=0.1):

        # 1. Quality score
        quality_scores = self.compute_quality_score(candidate_features)
        
        team_indices = [candidate_ids.index(mid) for mid in team_member_ids]
        team_quality = np.mean([quality_scores[i] for i in team_indices])
        
        team_quality_norm = team_quality / quality_scores.max() if quality_scores.max() > 0 else 0
        
        # 2. Diversity score
        diversity = self.compute_diversity_score(team_member_ids)
        
        # 3. Complementarity score
        complementarity = self.compute_complementarity_score(team_member_ids)
        
        total_score = (
            alpha * team_quality_norm +
            beta * diversity +
            gamma * complementarity
        )
        
        return total_score, {
            'quality': team_quality_norm,
            'diversity': diversity,
            'complementarity': complementarity
        }
    
    def form_team_greedy(self,
                        candidate_ids,
                        candidate_features,
                        team_size=5,
                        alpha=0.6,
                        beta=0.3,
                        gamma=0.1):
        
        quality_scores = self.compute_quality_score(candidate_features)
        
        best_idx = np.argmax(quality_scores)
        team = [candidate_ids[best_idx]]
        remaining = [cid for cid in candidate_ids if cid != candidate_ids[best_idx]]
        
        print(f"\n{'='*70}")
        print("GREEDY TEAM FORMATION")
        print(f"{'='*70}")
        print(f"Initial member: {team[0]} (quality: {quality_scores[best_idx]:.4f})")
        
        for step in range(1, team_size):
            best_score = -1
            best_candidate = None
            best_breakdown = None
            
            
            for candidate in remaining:
                candidate_team = team + [candidate]
                
                score, breakdown = self.evaluate_team(
                    candidate_team,
                    candidate_ids,
                    candidate_features,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma
                )
                
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
                    best_breakdown = breakdown
            
            
            team.append(best_candidate)
            remaining.remove(best_candidate)
            
            print(f"\nStep {step+1}: Added {best_candidate}")
            print(f"  Total score: {best_score:.4f}")
            print(f"  Quality: {best_breakdown['quality']:.4f}")
            print(f"  Diversity: {best_breakdown['diversity']:.4f}")
            print(f"  Complementarity: {best_breakdown['complementarity']:.4f}")
        
       
        final_score, final_breakdown = self.evaluate_team(
            team,
            candidate_ids,
            candidate_features,
            alpha=alpha,
            beta=beta,
            gamma=gamma
        )
        
        return team, final_score, final_breakdown
    
    def form_team_beam_search(self,
                              candidate_ids,
                              candidate_features,
                              team_size=5,
                              beam_width=3,
                              alpha=0.6,
                              beta=0.3,
                              gamma=0.1):
        
        quality_scores = self.compute_quality_score(candidate_features)
        
        
        top_k_indices = np.argsort(-quality_scores)[:beam_width]
        
        beams = [([candidate_ids[i]], 0.0) for i in top_k_indices]
        
        print(f"\n{'='*70}")
        print("BEAM SEARCH TEAM FORMATION")
        print(f"{'='*70}")
        print(f"Beam width: {beam_width}")
        
        # Beam search
        for step in range(1, team_size):
            print(f"\nStep {step+1}:")
            
            candidates_for_next_beam = []
            
            
            for team, _ in beams:
                remaining = [cid for cid in candidate_ids if cid not in team]
                
                
                for candidate in remaining[:20]:  
                    new_team = team + [candidate]
                    
                    score, _ = self.evaluate_team(
                        new_team,
                        candidate_ids,
                        candidate_features,
                        alpha=alpha,
                        beta=beta,
                        gamma=gamma
                    )
                    
                    candidates_for_next_beam.append((new_team, score))
            
            
            candidates_for_next_beam.sort(key=lambda x: x[1], reverse=True)
            beams = candidates_for_next_beam[:beam_width]
            
            print(f"  Best score in beam: {beams[0][1]:.4f}")
        
        
        best_team, best_score = beams[0]
        
        _, breakdown = self.evaluate_team(
            best_team,
            candidate_ids,
            candidate_features,
            alpha=alpha,
            beta=beta,
            gamma=gamma
        )
        
        return best_team, best_score, breakdown


def main():
    team_former = AdvancedTeamFormation(
        ranker_model_path='checkpoints/xgboost_ranker.json',
        gnn_embeddings_path='gnn_scholar_embeddings.pt',
        metadata_path='scholar_metadata_offline.pkl'
    )
    
    candidate_ids = [f'scholar_{i}' for i in range(20)]
    candidate_features = np.random.rand(20, 39)
    
    
    print("\n" + "="*70)
    print("METHOD 1: GREEDY")
    print("="*70)
    
    team_greedy, score_greedy, breakdown_greedy = team_former.form_team_greedy(
        candidate_ids=candidate_ids,
        candidate_features=candidate_features,
        team_size=5,
        alpha=0.6,
        beta=0.3,
        gamma=0.1
    )
    
    print(f"\n{'='*70}")
    print("FINAL TEAM (GREEDY)")
    print(f"{'='*70}")
    for i, member in enumerate(team_greedy, 1):
        print(f"  {i}. {member}")
    print(f"\nTotal score: {score_greedy:.4f}")
    print(f"  Quality: {breakdown_greedy['quality']:.4f}")
    print(f"  Diversity: {breakdown_greedy['diversity']:.4f}")
    print(f"  Complementarity: {breakdown_greedy['complementarity']:.4f}")
    
    print("\n" + "="*70)
    print("METHOD 2: BEAM SEARCH")
    print("="*70)
    
    team_beam, score_beam, breakdown_beam = team_former.form_team_beam_search(
        candidate_ids=candidate_ids,
        candidate_features=candidate_features,
        team_size=5,
        beam_width=3,
        alpha=0.6,
        beta=0.3,
        gamma=0.1
    )
    
    print(f"\n{'='*70}")
    print("FINAL TEAM (BEAM SEARCH)")
    print(f"{'='*70}")
    for i, member in enumerate(team_beam, 1):
        print(f"  {i}. {member}")
    print(f"\nTotal score: {score_beam:.4f}")
    print(f"  Quality: {breakdown_beam['quality']:.4f}")
    print(f"  Diversity: {breakdown_beam['diversity']:.4f}")
    print(f"  Complementarity: {breakdown_beam['complementarity']:.4f}")


if __name__ == "__main__":
    main()
