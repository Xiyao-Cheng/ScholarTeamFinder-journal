# end_to_end_team_recommendation.py
"""
Completed End-to-End Team Recommendation Pipeline

Input：natural language query
Output：Recommended team which inclues K scholars

Steps：
1. Retrieve top-100 candidates
2. Features extraction（GNN + Metadata + Query features）
3. XGBoost ranking
4. Team formation (greedy/beam search)
"""

import torch
import numpy as np
import pickle
import json
from transformers import AutoModel, AutoTokenizer

class EndToEndTeamRecommendation:
    def __init__(self,
                 query2box_checkpoint='../checkpoints/query2box_best.pt',
                 ranker_checkpoint='checkpoints/xgboost_ranker.json',
                 gnn_embeddings_path='gnn_scholar_embeddings.pt',
                 metadata_path='scholar_metadata_offline.pkl',
                 scholar_id_mapping_path='scholar_features_from_neo4j.pt'):
        
        print("="*70)
        print("END-TO-END TEAM RECOMMENDATION SYSTEM")
        print("="*70)
        
        with open(f'../training_data/metadata.json', 'r') as f:
            metadata = json.load(f)

        nentity = metadata['nentity']
        nrelation = metadata['nrelation']
        print(f"Entities: {nentity}, Relations: {nrelation}")

        with open(f'../training_data/train_queries.pkl', 'rb') as f:
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
        
        from models import KGReasoning
        self.query2box = KGReasoning(
            nentity=nentity,
            nrelation=nrelation,
            hidden_dim=400,
            gamma=12.0,
            geo='box',
            use_cuda=False,  
            query_name_dict=query_name_dict
        )
        
        # 2. Load feature extractor
        from step2_extract_features_offline import OfflineFeatureExtractor
        self.feature_extractor = OfflineFeatureExtractor(
            gnn_embeddings_path=gnn_embeddings_path,
            scholar_metadata_path=metadata_path,
            scholar_id_mapping_path=scholar_id_mapping_path
        )
        
        # 3. Load ranker
        import xgboost as xgb
        self.ranker = xgb.Booster()
        self.ranker.load_model(ranker_checkpoint)
        
        # 4. Load team former
        from advanced_team_formation import AdvancedTeamFormation
        self.team_former = AdvancedTeamFormation(
            ranker_model_path=ranker_checkpoint,
            gnn_embeddings_path=gnn_embeddings_path,
            metadata_path=metadata_path
        )
        
        print("✅ All components loaded")
    
    def recommend_team_with_details(self,
                                nl_query,
                                team_size=5,
                                method='greedy',
                                alpha=0.6,
                                beta=0.3,
                                gamma=0.1):
     
        print(f"\n{'='*70}")
        print(f"QUERY: {nl_query}")
        print(f"{'='*70}")


        print("\n1. Retrieving candidates with Query2Box...")
        candidate_ids, retrieval_scores = self.retrieve_with_query2box(nl_query)
        q2b_candidates = candidate_ids.copy()
        q2b_scores = retrieval_scores.copy()
        print(f"   ✅ Retrieved {len(candidate_ids)} candidates")


        print("\n2. Extracting features...")
        query_text_emb = self.feature_extractor.encode_query_text(nl_query)

        all_features = []
        for i, (candidate_id, retrieval_score) in enumerate(zip(candidate_ids, retrieval_scores)):
            features = self.feature_extractor.extract_features_for_candidate(
                nl_query=nl_query,
                query_text_emb=query_text_emb,
                candidate_scholar_id=candidate_id,
                q2b_retrieval_score=retrieval_score,
                q2b_rank=i
            )

            feature_array = [features[key] for key in sorted(features.keys())]
            all_features.append(feature_array)

        all_features = np.array(all_features, dtype=np.float32)
        print(f"   ✅ Extracted features: {all_features.shape}")


        print("\n2.5. Re-ranking with XGBoost...")
        import xgboost as xgb
        dmatrix = xgb.DMatrix(all_features)
        reranked_scores = self.ranker.predict(dmatrix)


        sorted_indices = np.argsort(reranked_scores)[::-1]
        reranked_candidates = [candidate_ids[i] for i in sorted_indices]
        reranked_scores_sorted = [reranked_scores[i] for i in sorted_indices]

        candidate_ids = reranked_candidates
        all_features = all_features[sorted_indices]

        print(f"   ✅ Re-ranked {len(reranked_candidates)} candidates")


        print(f"\n3. Forming team with {method} method...")

        if method == 'greedy':
            team, score, breakdown = self.team_former.form_team_greedy(
                candidate_ids=candidate_ids,
                candidate_features=all_features,
                team_size=team_size,
                alpha=alpha,
                beta=beta,
                gamma=gamma
            )
        elif method == 'beam_search':
            team, score, breakdown = self.team_former.form_team_beam_search(
                candidate_ids=candidate_ids,
                candidate_features=all_features,
                team_size=team_size,
                beam_width=3,
                alpha=alpha,
                beta=beta,
                gamma=gamma
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        print(f"   ✅ Team formed")


        print("\n4. Fetching team member details...")
        team_details = self.get_team_details(team)

        return {
            'team': team,
            'team_score': score,
            'team_breakdown': breakdown,
            'team_details': team_details,
            'q2b_candidates': q2b_candidates,
            'q2b_scores': q2b_scores.tolist() if isinstance(q2b_scores, np.ndarray) else q2b_scores,
            'reranked_candidates': reranked_candidates,
            'reranked_scores': reranked_scores_sorted,
            'query': nl_query
        }
    
    def get_team_details(self, team_member_ids):
        details = []
        
        for member_id in team_member_ids:
            metadata = self.feature_extractor.scholar_metadata.get(member_id, {})
            
            details.append({
                'scholar_id': member_id,
                'name': metadata.get('name', 'Unknown'),
                'h_index': metadata.get('h_index', 0),
                'citations': metadata.get('citation_count', 0),
                'papers': metadata.get('paper_count', 0),
                'experience': metadata.get('experience_years', 0),
                'affiliation': metadata.get('primary_affiliation', 'Unknown'),
                'department': metadata.get('department', 'Unknown')
            })
        
        return details
    
    def print_recommendation(self, team, score, breakdown, team_details):
        print(f"\n{'='*70}")
        print("RECOMMENDED TEAM")
        print(f"{'='*70}")
        
        print(f"\nOverall Score: {score:.4f}")
        print(f"  Quality:         {breakdown['quality']:.4f}")
        print(f"  Diversity:       {breakdown['diversity']:.4f}")
        print(f"  Complementarity: {breakdown['complementarity']:.4f}")
        
        print(f"\nTeam Members:")
        print("-" * 70)
        
        for i, (member_id, details) in enumerate(zip(team, team_details), 1):
            print(f"\n{i}. {details['name']} ({member_id})")
            print(f"   H-index: {details['h_index']}")
            print(f"   Citations: {details['citations']}")
            print(f"   Papers: {details['papers']}")
            print(f"   Experience: {details['experience']} years")
            print(f"   Affiliation: {details['affiliation']}")
            print(f"   Department: {details['department']}")


def main():
    system = EndToEndTeamRecommendation(
        query2box_checkpoint='checkpoints/query2box_best.pt',
        ranker_checkpoint='checkpoints/xgboost_ranker.json',
        gnn_embeddings_path='gnn_scholar_embeddings.pt',
        metadata_path='scholar_metadata_offline.pkl',
        scholar_id_mapping_path='scholar_features_from_neo4j.pt'
    )
    
    nl_query = "Find a team of machine learning experts for medical imaging research"

    team, score, breakdown, team_details = system.recommend_team(
        nl_query=nl_query,
        team_size=5,
        method='greedy',
        alpha=0.6,  
        beta=0.3,   
        gamma=0.1   
    )
    

    system.print_recommendation(team, score, breakdown, team_details)


if __name__ == "__main__":
    main()
