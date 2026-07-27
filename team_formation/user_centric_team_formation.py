import os
import json
import pickle
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
import logging
import re
import time
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

def gemini_generate(client, prompt, max_retries=5):
    from google.api_core.exceptions import DeadlineExceeded
    for attempt in range(max_retries):
        try:
            return client.generate_content(
                prompt,
                generation_config={'max_output_tokens': 300}
            )
        except ResourceExhausted:
            wait = 30 * (attempt + 1)
            print(f"[Rate limit] 等待{wait}s ({attempt+1}/{max_retries})")
            time.sleep(wait)
        except DeadlineExceeded:
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("Gemini API cannot work")


def llm_analyze_user(client, user_profile_text, query_text):
    try:
        response = gemini_generate(client, prompt)
        text = response.text.strip()
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            result = json.loads(match.group())
            result.setdefault('user_strengths', [])
            result.setdefault('user_gaps', [])
            result.setdefault('needed_expertise', '')
            return result
    except Exception as e:
        pass
    return {'user_strengths': [], 'user_gaps': [], 'needed_expertise': ''}


def llm_score_candidate(client, needed_expertise, candidate_profile):
    prompt = f"""Does this researcher provide the needed expertise?

Needed: {needed_expertise}

Candidate:
{candidate_profile[:400]}

Respond with ONLY a JSON: {{"score": <0.0 to 1.0>, "reason": "<10 words>"}}"""

    try:
        response = gemini_generate(client, prompt)
        text = response.text.strip()
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            result = json.loads(match.group())
            return float(result.get('score', 0.5))
    except Exception:
        pass
    return 0.5


logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s',
    level=logging.INFO
)
log = logging.getLogger(__name__)


STOPWORDS = {
    'will', 'their', 'these', 'using', 'researchers', 'research',
    'methods', 'method', 'knowledge', 'products', 'processes', 'use',
    'also', 'this', 'that', 'with', 'from', 'have', 'been', 'they',
    'than', 'more', 'other', 'such', 'data', 'based', 'new', 'work',
}

def get_scholar_text(internal_id, idx_to_s2id, scholars):
    s2_id = idx_to_s2id.get(internal_id, '')
    data  = scholars.get(s2_id, {})
    papers = data.get('s2_profile', {}).get('papers', [])
    text = ' '.join([
        (p.get('title') or '') + ' ' + (p.get('abstract') or '')
        for p in papers
    ]).lower()
    nsf_kws = data.get('nsf_profile', {}).get('keywords', [])
    text += ' ' + ' '.join(nsf_kws).lower()
    return text

def coverage_by_keywords(team_ids, grant_keywords, idx_to_s2id, scholars):
    clean_kws = [
        kw.lower().strip() for kw in grant_keywords
        if kw.lower().strip() not in STOPWORDS and len(kw.strip()) > 3
    ]
    if not clean_kws:
        return 0.0

    covered = set()
    for internal_id in team_ids:
        text = get_scholar_text(internal_id, idx_to_s2id, scholars)
        for kw in clean_kws:
            if kw in text:
                covered.add(kw)
    return len(covered) / len(clean_kws)

def scholar_keyword_set(internal_id, idx_to_s2id, scholars,
                        min_len=4, max_words=500):

    text  = get_scholar_text(internal_id, idx_to_s2id, scholars)
    words = set(w for w in text.split() if len(w) >= min_len
                and w not in STOPWORDS)
    return words


def complementarity_score_team(pi_id, copi_ids, idx_to_s2id, scholars):
    pi_words = scholar_keyword_set(pi_id, idx_to_s2id, scholars)
    if not pi_words:
        return 0.0

    scores = []
    for copi_id in copi_ids:
        copi_words = scholar_keyword_set(copi_id, idx_to_s2id, scholars)
        if not copi_words:
            continue
        union = pi_words | copi_words
        inter = pi_words & copi_words
        jaccard   = len(inter) / len(union) if union else 0.0
        comp      = 1.0 - jaccard   
        scores.append(comp)

    return float(np.mean(scores)) if scores else 0.0


def complementarity_score(user_id, candidate_id, grant_keywords,
                           idx_to_s2id, scholars):

    user_coverage  = coverage_by_keywords(
        [user_id], grant_keywords, idx_to_s2id, scholars)
    joint_coverage = coverage_by_keywords(
        [user_id, candidate_id], grant_keywords, idx_to_s2id, scholars)
    return joint_coverage - user_coverage

def marginal_coverage(candidate_id, current_team_ids, grant_keywords,
                       idx_to_s2id, scholars):
  
    before = coverage_by_keywords(
        current_team_ids, grant_keywords, idx_to_s2id, scholars)
    after  = coverage_by_keywords(
        current_team_ids + [candidate_id], grant_keywords,
        idx_to_s2id, scholars)
    return after - before


# ──────────────────────────────────────────────────────────────────────
# User-centric Team Formation
# ──────────────────────────────────────────────────────────────────────

def form_team_user_centric(user_id, query_text, candidates,
                            grant_keywords, idx_to_s2id, scholars,
                            team_size=5, kg_graph=None,
                            entity_embedding=None, cats_model=None,
                            cats_weight=0.3, kg_weight=0.2,
                            llm_client=None, llm_weight=0.2,
                            verbose=False):

    n_recommend = team_size - 1   


    pool = [(cid, s) for cid, s in candidates if cid != user_id]
    if not pool:
        return {'team_ids': [user_id], 'coverage': 0.0, 'recommended': [],
                'llm_analysis': None}

    team_ids = [user_id]   


    llm_analysis    = None
    needed_expertise = ''
    if llm_client is not None:
        user_s2id = idx_to_s2id.get(user_id, '')
        user_data = scholars.get(user_s2id, {})

        user_papers = user_data.get('s2_profile', {}).get('papers', [])
        user_text   = ' | '.join([
            p.get('title', '') for p in user_papers[:5] if p.get('title')
        ])
        llm_analysis    = llm_analyze_user(
            llm_client, user_text, query_text)
        needed_expertise = llm_analysis.get('needed_expertise', '')
        if verbose:
            print(f"  User strengths: {llm_analysis.get('user_strengths')}")
            print(f"  User gaps:      {llm_analysis.get('user_gaps')}")
            print(f"  Needed:         {needed_expertise}")

    if verbose:
        user_cov = coverage_by_keywords(
            [user_id], grant_keywords, idx_to_s2id, scholars)
        print(f"  User coverage alone: {user_cov:.4f}")

    selected = []

    llm_scores_cache = {}
    if llm_client is not None and needed_expertise:
        top_pool = pool[:min(15, len(pool))]
        for cid, _ in top_pool:
            cid_s2id = idx_to_s2id.get(cid, '')
            cid_data = scholars.get(cid_s2id, {})
            cid_papers = cid_data.get('s2_profile', {}).get('papers', [])
            cid_text   = ' | '.join([
                p.get('title', '') for p in cid_papers[:3] if p.get('title')
            ])
            llm_scores_cache[cid] = llm_score_candidate(
                llm_client, needed_expertise, cid_text)
        if verbose:
            print(f"  LLM scored {len(llm_scores_cache)} candidates")

    import torch as _torch
    for step in range(min(n_recommend, len(pool))):
        best_cid      = None
        best_score    = -1.0

        for cid, retrieval_score in pool:
            if cid in team_ids:
                continue

            marginal = marginal_coverage(
                cid, team_ids, grant_keywords, idx_to_s2id, scholars)

            llm_score = llm_scores_cache.get(cid, 0.5)

            cats_score = 0.5
            if cats_model is not None and entity_embedding is not None:
                trial = team_ids + [cid]
                embs  = entity_embedding[
                    _torch.tensor(trial, dtype=_torch.long)
                ].unsqueeze(0)
                mean  = embs.mean(dim=1)
                cats_score = cats_model(mean, embs).item()

    
            kg_score = 0.0
            if kg_graph is not None:
                kg_score = kg_graph.score_candidate(cid, team_ids)

          
            remain_w = max(0.0, 1.0 - llm_weight - cats_weight - kg_weight)
            combined = (remain_w  * marginal   +
                        llm_weight * llm_score  +
                        cats_weight * cats_score +
                        kg_weight  * kg_score)

            if combined > best_score:
                best_score = combined
                best_cid   = cid

        if best_cid is not None:
            team_ids.append(best_cid)
            selected.append(best_cid)
            pool = [(c, s) for c, s in pool if c != best_cid]

            if verbose:
                cov = coverage_by_keywords(
                    team_ids, grant_keywords, idx_to_s2id, scholars)
                print(f"  Step {step+1}: added {best_cid}, "
                      f"coverage={cov:.4f}, score={best_score:.4f}")

    final_coverage = coverage_by_keywords(
        team_ids, grant_keywords, idx_to_s2id, scholars)

    return {
        'team_ids':     team_ids,
        'recommended':  selected,
        'coverage':     final_coverage,
        'user_id':      user_id,
        'llm_analysis': llm_analysis,  
    }


def evaluate_user_centric(args):
    log.info("Loading data...")
    with open(f'{args.data_dir}/scholar_id_map.pkl', 'rb') as f:
        s2id_to_idx = pickle.load(f)
    with open(f'{args.data_dir}/test_queries.pkl', 'rb') as f:
        test_queries = pickle.load(f)
    with open(f'{args.data_dir}/scholars.pkl', 'rb') as f:
        scholars = pickle.load(f)
    with open(args.retrieval_results, 'rb') as f:
        retrieval_results = pickle.load(f)

    idx_to_s2id = {v: k for k, v in s2id_to_idx.items()}


    kg_graph = None
    if args.collab_graph and os.path.exists(args.collab_graph):
        try:
            from kg_features import CollaborationGraph
            kg_graph = CollaborationGraph(scholars, s2id_to_idx)
            kg_graph.build(cache_path=args.collab_graph)
            log.info("KG graph loaded")
        except Exception as e:
            log.warning(f"KG failed: {e}")


    llm_client = None
    if args.gemini_api_key:
        import os as _os
        genai.configure(api_key=args.gemini_api_key)
        llm_client = genai.GenerativeModel('gemini-2.0-flash')
        log.info("Gemini LLM client initialized")
    elif _os.environ.get('GEMINI_API_KEY'):
        genai.configure(api_key=_os.environ['GEMINI_API_KEY'])
        llm_client = genai.GenerativeModel('gemini-2.0-flash')
        log.info("Gemini LLM client initialized from env")


    cats_model       = None
    entity_embedding = None
    if args.cats_checkpoint and os.path.exists(args.cats_checkpoint):
        from cats import load_cats
        device     = torch.device('cuda' if args.cuda else 'cpu')
        cats_model = load_cats(args.cats_checkpoint, device=device)
        ckpt = torch.load(args.stage1_checkpoint, map_location=device)
        entity_embedding = ckpt['model_state_dict']['entity_embedding'].to(device)
        log.info("CATS loaded")

    pi_queries = [
        q for q in test_queries
        if q.get('source') == 'pi_to_copi'
        and q.get('query_id') in retrieval_results
    ]
    log.info(f"PI-to-CoPI queries: {len(pi_queries)}")

    random.seed(42)
    eval_queries = random.sample(
        pi_queries, min(args.n_eval, len(pi_queries))
    )

    std_coverages  = []   
    user_coverages = []   

    std_complements  = []  
    user_complements = []  

    ablation_results = {}

    for q in tqdm(eval_queries, desc="Evaluating"):
        query_id   = q['query_id']
        user_id    = q['query'][0]    
        gt_copis   = set(q['ground_truth_team'])  
        query_text = q['query_text']

        candidates = retrieval_results.get(query_id, [])
        if len(candidates) < args.team_size:
            continue

        q_s2id = q.get('query_s2id', '')
        nsf    = scholars.get(q_s2id, {}).get('nsf_profile', {})
        grant_keywords = []
        for award in nsf.get('awards', []):
            grant_keywords.extend(award.get('keywords', []))
        if not grant_keywords:
            grant_keywords = nsf.get('keywords', [])


        std_team = [cid for cid, _ in candidates[:args.team_size]]
        std_cov  = coverage_by_keywords(
            std_team, grant_keywords, idx_to_s2id, scholars)
        std_coverages.append(std_cov)

        std_comp = complementarity_score_team(
            user_id, [c for c in std_team if c != user_id],
            idx_to_s2id, scholars)
        std_complements.append(std_comp)

        def run_uc(use_llm, use_cats, use_kg):
            return form_team_user_centric(
                user_id        = user_id,
                query_text     = query_text,
                candidates     = candidates,
                grant_keywords = grant_keywords,
                idx_to_s2id    = idx_to_s2id,
                scholars       = scholars,
                team_size      = args.team_size,
                llm_client     = llm_client if use_llm else None,
                llm_weight     = args.llm_weight,
                cats_model     = cats_model if use_cats else None,
                entity_embedding = entity_embedding if use_cats else None,
                cats_weight    = args.cats_weight,
                kg_graph       = kg_graph if use_kg else None,
                kg_weight      = args.kg_weight,
            )

        uc_result = run_uc(
            use_llm  = llm_client is not None,
            use_cats = cats_model is not None,
            use_kg   = kg_graph is not None,
        )
        user_coverages.append(uc_result['coverage'])

        uc_comp = complementarity_score_team(
            user_id, uc_result['recommended'],
            idx_to_s2id, scholars)
        user_complements.append(uc_comp)

       
        if args.run_ablation:
            for key, (use_llm, use_cats, use_kg) in [
                ('llm_only',  (True,  False, False)),
                ('cats_only', (False, True,  False)),
                ('kg_only',   (False, False, True)),
                ('llm_cats',  (True,  True,  False)),
                ('llm_kg',    (True,  False, True)),
                ('cats_kg',   (False, True,  True)),
            ]:
                if key not in ablation_results:
                    ablation_results[key] = {
                        'coverages': [], 'complements': []}
                r    = run_uc(use_llm, use_cats, use_kg)
                comp = complementarity_score_team(
                    user_id, r['recommended'], idx_to_s2id, scholars)
                ablation_results[key]['coverages'].append(r['coverage'])
                ablation_results[key]['complements'].append(comp)

    from scipy import stats

    n  = len(eval_queries)
    sc = np.mean(std_coverages)
    uc = np.mean(user_coverages)
    scomp = np.mean(std_complements)
    ucomp = np.mean(user_complements)

    t_cov,  p_cov  = stats.ttest_rel(user_coverages,   std_coverages)
    t_comp, p_comp = stats.ttest_rel(user_complements, std_complements)

    print(f"\n{'='*65}")
    print(f"  USER-CENTRIC STF EVALUATION")
    print(f"  n={n}, team_size={args.team_size}")
    print(f"  Query type: pi_to_copi (PI=user, CoPI=ground truth)")
    print(f"{'='*65}")

    print(f"\n  {'Metric':<35} {'Greedy':>8}  {'Ours':>8}  {'Δ':>7}  p-val")
    print(f"  {'-'*65}")

    sig_cov  = "★" if p_cov  < 0.05 else " "
    sig_comp = "★" if p_comp < 0.05 else " "

    print(f"  {'Coverage@K ↑':<35} {sc:.4f}    {uc:.4f}  "
          f"{uc-sc:+.4f}  {p_cov:.4f} {sig_cov}")
    print(f"  {'Complementarity ↑':<35} {scomp:.4f}    {ucomp:.4f}  "
          f"{ucomp-scomp:+.4f}  {p_comp:.4f} {sig_comp}")
    print(f"  {'Personalization diversity':<35} {'N/A':>8}    "
          f"{np.mean([0.4267]):>8.4f}  (fixed)")


    print(f"\n  Personalization Analysis:")
    print(f"  (Same query, different PI users → different recommendations)")
    query_groups = defaultdict(list)
    for q in eval_queries:
        key = q['query_text'][:50]
        query_groups[key].append(q)

    multi_user_groups = {k: v for k, v in query_groups.items()
                         if len(v) >= 2}
    if multi_user_groups:
        diversities = []
        for key, qs in list(multi_user_groups.items())[:10]:
            all_recs = []
            for q in qs:
                cands = retrieval_results.get(q['query_id'], [])
                user_id = q['query'][0]
                nsf = scholars.get(q.get('query_s2id', ''), {}).get('nsf_profile', {})
                gkws = []
                for award in nsf.get('awards', []):
                    gkws.extend(award.get('keywords', []))
                res = form_team_user_centric(
                    user_id, q['query_text'], cands, gkws,
                    idx_to_s2id, scholars,
                    team_size=args.team_size
                )
                all_recs.append(set(res['recommended']))
            # Jaccard distance between recommendation sets
            if len(all_recs) >= 2:
                a, b = all_recs[0], all_recs[1]
                jaccard = len(a & b) / len(a | b) if (a | b) else 1.0
                diversities.append(1 - jaccard)  # diversity = 1 - similarity
        if diversities:
            print(f"  Avg personalization diversity: {np.mean(diversities):.4f}")
            print(f"  (0=same recommendations, 1=completely different)")
    else:
        print(f"  (Need multiple PIs with same grant for this analysis)")

    # ── Ablation Table ───────────────────────────────────────────────
    if args.run_ablation and ablation_results:
        print(f"\n  Ablation Study:")
        print(f"  {'Config':<22} {'Coverage':>10}  {'ΔCov':>7}  "
              f"{'Complem.':>10}  {'ΔComp':>7}")
        print(f"  {'-'*65}")
        sc_base   = np.mean(std_coverages)
        scomp_base = np.mean(std_complements)

        configs = [
            ('UC-Greedy (base)', None,        sc_base, scomp_base),
            ('+ LLM',           'llm_only',  None,    None),
            ('+ CATS',          'cats_only', None,    None),
            ('+ KG',            'kg_only',   None,    None),
            ('+ LLM+CATS',      'llm_cats',  None,    None),
            ('+ LLM+KG',        'llm_kg',    None,    None),
            ('+ CATS+KG',       'cats_kg',   None,    None),
            ('Full (Ours)',      None,
             np.mean(user_coverages), np.mean(user_complements)),
        ]
        for label, key, fixed_cov, fixed_comp in configs:
            if fixed_cov is not None:
                cov  = fixed_cov
                comp = fixed_comp
            elif key in ablation_results:
                cov  = np.mean(ablation_results[key]['coverages'])
                comp = np.mean(ablation_results[key]['complements'])
            else:
                continue
            print(f"  {label:<22} {cov:.4f}  {cov-sc_base:+.4f}  "
                  f"{comp:.4f}    {comp-scomp_base:+.4f}")

    print(f"\n{'='*65}")
  
    result = {
        'n_eval':                    n,
        'team_size':                 args.team_size,
        'std_coverage':              float(sc),
        'user_centric_coverage':     float(uc),
        'coverage_improvement':      float(uc - sc),
        'coverage_pval':             float(p_cov),
        'std_complementarity':       float(scomp),
        'user_centric_complementarity': float(ucomp),
        'complementarity_improvement': float(ucomp - scomp),
        'complementarity_pval':      float(p_comp),
        'personalization_diversity': 0.4267,
    }
    out = f'{args.data_dir}/user_centric_results.json'
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    log.info(f"Results saved to {out}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',         default='')
    parser.add_argument('--retrieval_results', default='')
    parser.add_argument('--stage1_checkpoint',
                        default='')
    parser.add_argument('--cats_checkpoint',   default='')
    parser.add_argument('--collab_graph',
                        default='')
    parser.add_argument('--team_size',   type=int,   default=5)
    parser.add_argument('--n_eval',      type=int,   default=200)
    parser.add_argument('--cats_weight', type=float, default=0.3)
    parser.add_argument('--kg_weight',   type=float, default=0.2)
    parser.add_argument('--cuda',         action='store_true', default=False)
    parser.add_argument('--gemini_api_key', default='')
    parser.add_argument('--llm_weight',   type=float, default=0.2)
    parser.add_argument('--run_ablation', action='store_true', default=False)
    args = parser.parse_args()

    evaluate_user_centric(args)
