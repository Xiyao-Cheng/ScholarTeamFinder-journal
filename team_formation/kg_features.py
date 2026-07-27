import os
import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────
# Collaboration Graph
# ──────────────────────────────────────────────────────────────────────

class CollaborationGraph:

    def __init__(self, scholars, s2id_to_idx):
        self.scholars    = scholars
        self.s2id_to_idx = s2id_to_idx
        self.idx_to_s2id = {v: k for k, v in s2id_to_idx.items()}
        self.nentity     = len(s2id_to_idx)

        self.neighbors   = defaultdict(set)
        self.communities = {}
        self.degrees     = {}

        self._built = False

    def build(self, cache_path=None):
        if cache_path and os.path.exists(cache_path):
            print(f"Loading graph from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            self.neighbors   = data['neighbors']
            self.communities = data['communities']
            self.degrees     = data['degrees']
            self._built      = True
            print(f"Graph loaded: {len(self.neighbors)} nodes, "
                  f"{sum(len(v) for v in self.neighbors.values())//2} edges")
            return

        print("Building collaboration graph from co_authors...")
        scholar_set = set(self.s2id_to_idx.keys())

        for s2_id, data in tqdm(self.scholars.items(), desc="Building graph"):
            if s2_id not in scholar_set:
                continue
            src_idx = self.s2id_to_idx[s2_id]
            co_authors = data.get('s2_profile', {}).get('co_authors', [])

            for co in co_authors:
                co_s2id = co.get('s2_author_id', '')
                if co_s2id in scholar_set:
                    dst_idx = self.s2id_to_idx[co_s2id]
                    self.neighbors[src_idx].add(dst_idx)
                    self.neighbors[dst_idx].add(src_idx)

        self.degrees = {
            node: len(nbrs)
            for node, nbrs in self.neighbors.items()
        }

        print("Running label propagation for community detection...")
        self.communities = self._label_propagation(max_iter=10)

        self._built = True
        n_edges = sum(len(v) for v in self.neighbors.values()) // 2
        n_communities = len(set(self.communities.values()))
        print(f"Graph built: {len(self.neighbors)} nodes, "
              f"{n_edges} edges, {n_communities} communities")

        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'neighbors':   dict(self.neighbors),
                    'communities': self.communities,
                    'degrees':     self.degrees,
                }, f)
            print(f"Graph cached to: {cache_path}")

    def _label_propagation(self, max_iter=10):
        labels = {node: node for node in self.neighbors}

        for _ in range(max_iter):
            changed = False
            nodes = list(self.neighbors.keys())
            np.random.shuffle(nodes)

            for node in nodes:
                nbrs = self.neighbors[node]
                if not nbrs:
                    continue

                label_count = defaultdict(int)
                for nbr in nbrs:
                    label_count[labels.get(nbr, nbr)] += 1

                best_label = max(label_count, key=label_count.get)
                if best_label != labels.get(node):
                    labels[node] = best_label
                    changed = True

            if not changed:
                break

        return labels

    def structural_diversity(self, team_ids):
        pairs = [
            (a, b) for i, a in enumerate(team_ids)
            for b in team_ids[i+1:]
        ]
        if not pairs:
            return 0.0

        novel = sum(
            1 for a, b in pairs
            if b not in self.neighbors.get(a, set())
        )
        return novel / len(pairs)

    def bridge_score(self, candidate_id, current_team_ids):
        cand_neighbors  = self.neighbors.get(candidate_id, set())
        if not cand_neighbors:
            return 0.0

        team_covered = set(current_team_ids)
        for m in current_team_ids:
            team_covered |= self.neighbors.get(m, set())


        new_connections = cand_neighbors - team_covered
        return len(new_connections) / len(cand_neighbors)

    def community_coverage(self, team_ids):
        if not team_ids:
            return 0.0
        communities = {
            self.communities.get(sid)
            for sid in team_ids
            if self.communities.get(sid) is not None
        }
        return len(communities) / len(team_ids)

    def novelty_score(self, team_ids):
        div = self.structural_diversity(team_ids)
        cov = self.community_coverage(team_ids)
        return div * cov

    def score_candidate(self, candidate_id, current_team_ids,
                         alpha=0.5, beta=0.5):
        bridge  = self.bridge_score(candidate_id, current_team_ids)


        before = self.community_coverage(current_team_ids)
        after  = self.community_coverage(current_team_ids + [candidate_id])
        cov_delta = after - before

        return alpha * bridge + beta * cov_delta

    def get_team_kg_features(self, team_ids):
        return {
            'structural_diversity': self.structural_diversity(team_ids),
            'community_coverage':   self.community_coverage(team_ids),
            'novelty_score':        self.novelty_score(team_ids),
            'avg_degree':           np.mean([
                self.degrees.get(sid, 0) for sid in team_ids
            ]),
            'team_size':            len(team_ids),
        }

    def find_best_by_kg(self, pool_ids, current_team_ids, top_n=5):
        scored = [
            (cid, self.score_candidate(cid, current_team_ids))
            for cid in pool_ids[:top_n * 3] 
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in scored[:top_n]]



def combined_candidate_score(candidate_id, current_team_ids,
                              llm_rank, kg_graph,
                              cats_score=0.5,
                              alpha=0.4, beta=0.3, gamma=0.3):

    llm_score = 1.0 / (1.0 + llm_rank)

    # KG score
    kg_score = kg_graph.score_candidate(candidate_id, current_team_ids)

    return alpha * llm_score + beta * cats_score + gamma * kg_score

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   default='')
    parser.add_argument('--cache_path', default='')
    args = parser.parse_args()

    with open(f'{args.data_dir}/scholar_id_map.pkl', 'rb') as f:
        s2id_to_idx = pickle.load(f)
    with open(f'{args.data_dir}/scholars.pkl', 'rb') as f:
        scholars = pickle.load(f)

    graph = CollaborationGraph(scholars, s2id_to_idx)
    graph.build(cache_path=args.cache_path)

    sample_team = list(s2id_to_idx.values())[:5]
    features = graph.get_team_kg_features(sample_team)

    print("\nSample team KG features:")
    for k, v in features.items():
        print(f"  {k:25s}: {v:.4f}")

    pool = list(s2id_to_idx.values())[5:25]
    best = graph.find_best_by_kg(pool, sample_team, top_n=3)
    print(f"\nTop-3 KG candidates: {best}")
    for cid in best:
        score = graph.score_candidate(cid, sample_team)
        print(f"  Scholar {cid}: KG score = {score:.4f}")

    total_nodes = len(graph.neighbors)
    nodes_with_edges = sum(1 for n, nbrs in graph.neighbors.items() if nbrs)
    total_edges = sum(len(v) for v in graph.neighbors.values()) // 2

    print(f"Total nodes with edges: {nodes_with_edges} / {total_nodes}")
    print(f"Total edges: {total_edges}")
    print(f"Avg degree: {total_edges*2/total_nodes:.2f}")
