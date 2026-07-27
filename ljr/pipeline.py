import os
import json
import pickle
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Optional
from transformers import AutoTokenizer

from model import KGReasoning
from user_centric_team_formation import (
    form_team_user_centric,
    coverage_by_keywords,
    marginal_coverage,
    STOPWORDS,
)
from llm_judge import (
    LLMProvider, TeamMember, ljr_refinement_loop, build_user_prompt,
)
from query_understanding import (
    parse_query, ParsedQuery, TeamConstraints,
    apply_constraints_to_candidates,
    resolve_team_size,
    constraints_to_judge_context,
)

logger = logging.getLogger(__name__)


_idf_cache: dict[str, float] = {}
_idf_N: int = 0

def _build_idf(
    grant_keywords: list[str],
    scholars: dict,
    force: bool = False,
) -> dict[str, float]:
    global _idf_cache, _idf_N
    cache_key = "|".join(sorted(grant_keywords))
    if not force and cache_key in _idf_cache:
        return _idf_cache[cache_key]

    N = len(scholars)
    doc_freq: dict[str, int] = {}
    for kw in grant_keywords:
        kw_l = kw.lower()
        df = 0
        for s2_id, data in scholars.items():
            papers  = data.get("s2_profile", {}).get("papers", [])
            text    = " ".join(
                (p.get("title") or "") for p in papers[:10]
            ).lower()
            nsf_kws = " ".join(
                data.get("nsf_profile", {}).get("keywords", [])
            ).lower()
            if kw_l in text or kw_l in nsf_kws:
                df += 1
        doc_freq[kw] = df

    import math
    raw_idf = {
        kw: math.log((N + 1) / (doc_freq.get(kw, 0) + 1))
        for kw in grant_keywords
    }
    max_idf = max(raw_idf.values()) if raw_idf else 1.0
    norm_idf = {kw: v / max_idf for kw, v in raw_idf.items()}
    _idf_cache[cache_key] = norm_idf
    return norm_idf


def compute_tqs(
    team_ids: list,
    grant_keywords: list[str],
    idx_to_s2id: dict,
    scholars: dict,
    gold_team_ids: Optional[list] = None,
    h_index_norm: float = 50.0,
    use_idf: bool = True,          
    idf_weights: Optional[dict] = None,  
) -> tuple[float, dict]:
    clean_kws = [
        kw.lower().strip() for kw in grant_keywords
        if kw.lower().strip() not in STOPWORDS and len(kw.strip()) > 3
    ]
    if not clean_kws:
        return 0.0, {"coverage": 0.0, "cl": 0.0, "el": 0.0, "gm": 0.0}


    if use_idf:
        if idf_weights:
            idf = idf_weights
        else:

            import math

            idf = {kw: min(1.0, len(kw) / 20.0) for kw in clean_kws}
    else:
        idf = {kw: 1.0 for kw in clean_kws}

    total_idf = sum(idf.get(kw, 1.0) for kw in clean_kws) or 1.0

    member_texts: list[str] = []
    member_hindex: list[int] = []
    for internal_id in team_ids:
        s2_id = idx_to_s2id.get(internal_id, "")
        data  = scholars.get(s2_id, {})
        papers = data.get("s2_profile", {}).get("papers", [])
        text  = " ".join(
            (p.get("title") or "") + " " + (p.get("abstract") or "")
            for p in papers
        ).lower()
        nsf_kws = data.get("nsf_profile", {}).get("keywords", [])
        text += " " + " ".join(nsf_kws).lower()
        member_texts.append(text)
        h = data.get("s2_profile", {}).get("h_index", 0) or 0
        member_hindex.append(h)

    coverage_weighted = sum(
        idf.get(kw, 1.0)
        for kw in clean_kws
        if any(kw in text for text in member_texts)
    ) / total_idf

    cl_weighted = sum(
        idf.get(kw, 1.0)
        for kw in clean_kws
        if sum(1 for text in member_texts if kw in text) >= 2
    ) / total_idf


    el_sum = 0.0
    for kw in clean_kws:
        covering_h = [
            member_hindex[i]
            for i, text in enumerate(member_texts) if kw in text
        ]
        if covering_h:
            el_sum += idf.get(kw, 1.0) * min(max(covering_h) / h_index_norm, 1.0)
    el = el_sum / total_idf


    gm = 0.0
    if gold_team_ids:
        def to_int_set(ids):
            result = set()
            for x in ids:
                try:
                    result.add(int(x))
                except (ValueError, TypeError):
                    result.add(x)
            return result

        pred_set = to_int_set(team_ids)
        gold_set = to_int_set(gold_team_ids)
        tp = len(pred_set & gold_set)
        if tp > 0:
            p = tp / len(pred_set)
            r = tp / len(gold_set)
            gm = 2 * p * r / (p + r)

    tqs = 0.15 * coverage_weighted + 0.35 * cl_weighted + 0.35 * el + 0.15 * gm
    return tqs, {
        "coverage" : round(coverage_weighted, 4),
        "cl"       : round(cl_weighted, 4),
        "el"       : round(el, 4),
        "gm"       : round(gm, 4),
        "idf_used" : use_idf,
        "cats"     : 0.5,
    }


def diagnose_team(
    breakdown: dict,
    threshold: float = 0.80,
) -> str:

    if breakdown.get("coverage", 1.0) < 0.5 or breakdown.get("cl", 1.0) < 0.3:
        return "re_retrieve"   
    if breakdown.get("el", 1.0) < 0.4:
        return "re_form"      
    return "pass"


class STFResources:
    def __init__(
        self,
        model_ckpt: str,
        data_dir: str,              # training_data_v3/
        kg_graph_path: Optional[str] = None,
        cats_checkpoint: Optional[str] = None,
        nrelation: int  = 5,
        hidden_dim: int = 400,
        gamma: float    = 12.0,
        box_mode: tuple = ("none", 0.02),
        scibert_model: str = "allenai/scibert_scivocab_uncased",
        device: str     = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.data_dir = data_dir

        with open(f"{data_dir}/scholar_id_map.pkl", "rb") as f:
            self.s2id_to_idx: dict = pickle.load(f)
        self.idx_to_s2id = {v: k for k, v in self.s2id_to_idx.items()}

        with open(f"{data_dir}/scholars.pkl", "rb") as f:
            self.scholars: dict = pickle.load(f)

        nentity = len(self.s2id_to_idx)
        logger.info(f"Loading KGReasoning model (nentity={nentity}) ...")
        self.model = KGReasoning(
            nentity=nentity,
            nrelation=nrelation,
            hidden_dim=hidden_dim,
            gamma=gamma,
            geo="box",
            box_mode=box_mode,
            use_cuda=(self.device.type == "cuda"),
            query_name_dict={("e", ("r",)): "1p"},
            freeze_bert=True,
            scibert_model=scibert_model,
        )
        ckpt = torch.load(model_ckpt, map_location=self.device)
        state = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state)
        self.model.eval().to(self.device)


        self.entity_embedding = self.model.entity_embedding  # Parameter (N, dim)

        # ── Tokenizer ────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(scibert_model)


        self.scholar_ids = [
            self.idx_to_s2id[i]
            for i in range(len(self.idx_to_s2id))
            if i in self.idx_to_s2id
        ]

        self.kg_graph = None
        if kg_graph_path and os.path.exists(kg_graph_path):
            try:
                from kg_features import CollaborationGraph
                self.kg_graph = CollaborationGraph(
                    self.scholars, self.s2id_to_idx
                )
                self.kg_graph.build(cache_path=kg_graph_path)
                logger.info("KG graph loaded")
            except Exception as e:
                logger.warning(f"KG graph failed: {e}")


        self.cats_model = None
        if cats_checkpoint and os.path.exists(cats_checkpoint):
            try:
                from cats import load_cats
                self.cats_model = load_cats(cats_checkpoint, device=self.device)
                logger.info("CATS model loaded")
            except Exception as e:
                logger.warning(f"CATS failed: {e}")

        logger.info(
            f"STFResources ready | scholars={len(self.scholars)} "
            f"| device={self.device}"
        )


@torch.no_grad()
def tcsr_retrieve(
    query_text: str,
    resources: STFResources,
    top_k: int = 100,
    max_length: int = 256,
    relation_id: int = 0,
) -> list[tuple[int, float]]:
    model  = resources.model
    device = resources.device

    enc = resources.tokenizer(
        query_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # text_encoder: (1, seq) → (1, hidden_dim)
    query_center = model.text_encoder(input_ids, attention_mask)

    # 1-hop box
    r_emb   = model.relation_embedding[relation_id].unsqueeze(0)   # (1, dim)
    r_off   = model.offset_embedding[relation_id].unsqueeze(0)     # (1, dim)
    box_ctr = query_center + r_emb
    box_off = model.func(r_off)

    # score vs all entities: (nentity,)
    ent     = model.entity_embedding.unsqueeze(0)                  # (1, N, dim)
    scores  = model.cal_logit_box(
        ent,
        box_ctr.unsqueeze(1),
        box_off.unsqueeze(1),
    ).squeeze(0)

    k = min(top_k, scores.shape[0])
    vals, idxs = scores.topk(k)
    results = [(int(idxs[i].item()), float(vals[i].item())) for i in range(k)]

    logger.debug("TCSR top-10 candidates:")
    for rank, (iid, sc) in enumerate(results[:10]):
        sid  = resources.idx_to_s2id.get(iid, "")
        data = resources.scholars.get(sid, {})
        name = data.get("s2_profile", {}).get("name") or data.get("nsf_profile", {}).get("name", sid)
        kws  = data.get("nsf_profile", {}).get("keywords", [])[:4]
        logger.debug(f"  [{rank+1}] score={sc:.3f} {name}: {kws}")

    return results


def apply_constraints_to_internal_candidates(
    candidates: list[tuple[int, float]],
    constraints: TeamConstraints,
    idx_to_s2id: dict,
    scholars: dict,
    pi_collab_ids: Optional[set] = None,
) -> list[tuple[int, float]]:

    if constraints.is_empty():
        return candidates

    filtered = []
    for internal_id, score in candidates:
        s2_id = idx_to_s2id.get(internal_id, "")
        data  = scholars.get(s2_id, {})
        hard_fail = False

        h = data.get("s2_profile", {}).get("h_index", 0) or 0

        if constraints.min_h_index and h < constraints.min_h_index:
            hard_fail = True

        early_career_bonus = 0.0
        if constraints.max_h_index and h <= constraints.max_h_index:
            early_career_bonus = 0.15 

        # ── affiliation type ─────────────────────────────────
        if constraints.affiliation_types and not hard_fail:
            affil_list = data.get("s2_profile", {}).get("affiliations", [])
            affil = (
                data.get("affiliation")
                or (affil_list[0] if affil_list else "")
                or ""
            ).lower()
            # skip hard filter if affil data is missing (empty = unknown, not "not university")
            if affil and not any(
                _affil_matches(affil, t) for t in constraints.affiliation_types
            ):
                hard_fail = True

        # ── required institutions ─────────────────────────────
        if constraints.required_institutions and not hard_fail:
            affil_list = data.get("s2_profile", {}).get("affiliations", [])
            affil = (
                data.get("affiliation")
                or (affil_list[0] if affil_list else "")
                or ""
            ).lower()
            # same: skip if affil unknown
            if affil and not any(
                inst.lower() in affil
                for inst in constraints.required_institutions
            ):
                hard_fail = True

        # ── prior collab ─────────────────────────────────────
        if pi_collab_ids and not hard_fail:
            if constraints.exclude_prior_collabs and s2_id in pi_collab_ids:
                hard_fail = True
            if constraints.require_prior_collab and s2_id not in pi_collab_ids:
                hard_fail = True

        if hard_fail:
            continue

        # ── soft: required_roles + early_career bonus ───────
        role_bonus = 0.0
        if constraints.required_roles:
            papers = data.get("s2_profile", {}).get("papers", [])
            exp_text = " ".join(
                p.get("title", "") for p in papers[:5]
            ).lower()
            nsf_kws = " ".join(
                data.get("nsf_profile", {}).get("keywords", [])
            ).lower()
            expertise = exp_text + " " + nsf_kws
            role_bonus = sum(
                0.1 for role in constraints.required_roles
                if role.lower() in expertise
            )
        score = score + role_bonus + early_career_bonus

        filtered.append((internal_id, score))

    filtered.sort(key=lambda x: x[1], reverse=True)
    n_in = len(candidates)
    n_out = len(filtered)
    if n_out < n_in * 0.3: 
        logger.warning(
            f"[Constraint] Heavy filtering: {n_in} → {n_out} candidates "
            f"(constraints: max_h={constraints.max_h_index}, "
            f"min_h={constraints.min_h_index}, "
            f"affil={constraints.affiliation_types})"
        )
    else:
        logger.info(f"[Constraint] {n_in} → {n_out} candidates after filtering")
    return filtered


def _affil_matches(affil: str, affil_type: str) -> bool:
    rules = {
        "university":   ["university", "univ", "college", "institute of technology"],
        "industry":     ["inc", "corp", "ltd", "llc", "google", "microsoft",
                         "amazon", "meta", "apple", "nvidia", "ibm", "labs"],
        "government":   ["national", "federal", "department of", "nasa", "doe", "nih"],
        "national_lab": ["national lab", "argonne", "oak ridge", "brookhaven",
                         "sandia", "lawrence", "pacific northwest", "nrel"],
    }
    kws = rules.get(affil_type, [affil_type])
    return any(kw in affil for kw in kws)

_DISPLAY_STOPWORDS = {
    'will', 'their', 'these', 'using', 'researchers', 'research',
    'methods', 'method', 'knowledge', 'products', 'processes', 'use',
    'also', 'this', 'that', 'with', 'from', 'have', 'been', 'they',
    'than', 'more', 'other', 'such', 'data', 'based', 'new', 'work',
    'each', 'center', 'future', 'areas', 'other', 'existing',
}

def _clean_expertise(raw_keywords: list[str]) -> str:
    cleaned = [
        kw for kw in raw_keywords
        if kw.lower().strip() not in _DISPLAY_STOPWORDS
        and len(kw.strip()) > 3
    ]
    return ", ".join(cleaned[:8]) if cleaned else ""


def ids_to_team_members(
    internal_ids: list[int],
    idx_to_s2id: dict,
    scholars: dict,
) -> list[TeamMember]:
    members = []
    for iid in internal_ids:
        s2_id = idx_to_s2id.get(iid, "")
        data  = scholars.get(s2_id, {})
        s2    = data.get("s2_profile", {})
        nsf   = data.get("nsf_profile", {})
        papers = s2.get("papers", [])

        # ── recent titles ─────────────────────────────────────
        recent_titles = [
            p.get("title", "") for p in papers[:3] if p.get("title")
        ]
        if not recent_titles:
            for award in nsf.get("awards", [])[:3]:
                t = award.get("title", "")
                if t:
                    recent_titles.append(t)

        paper_words = []
        for p in papers[:10]:
            title = p.get("title", "")
            words = [
                w.lower().strip(".,;:()[]")
                for w in title.split()
                if len(w) > 4 and w.lower() not in _DISPLAY_STOPWORDS
            ]
            paper_words.extend(words)
        from collections import Counter
        word_freq = Counter(paper_words)
        top_words = [w for w, _ in word_freq.most_common(8)]
        expertise = ", ".join(top_words) if top_words else ""

        if not expertise:
            nsf_kws = nsf.get("keywords", [])
            expertise = _clean_expertise(nsf_kws)
        # fallback 2: NSF award keywords
        if not expertise:
            award_kws = []
            for award in nsf.get("awards", [])[:3]:
                award_kws.extend(award.get("keywords", []))
            expertise = _clean_expertise(award_kws)

        h = s2.get("h_index", 0) or 0

        name = (
            data.get("name")
            or s2.get("name")
            or nsf.get("pi_name")
            or f"Scholar_{iid}"
        )

        members.append(TeamMember(
            scholar_id        = s2_id or str(iid),
            name              = name,
            expertise_summary = expertise or "(no expertise data)",
            recent_titles     = recent_titles,
            h_index           = int(h) if h else None,
        ))
    return members


def make_constrained_fns(
    resources: STFResources,
    pi_internal_id: int,
    query_text: str,
    grant_keywords: list[str],
    constraints: TeamConstraints,
    gold_team_ids: Optional[list] = None,
    pi_collab_ids: Optional[set]  = None,
    top_k: int    = 100,
    team_size: int = 4,
    relation_id: int = 0,
    cats_weight: float = 0.3,
    kg_weight: float   = 0.2,
    llm_client=None,
    llm_weight: float  = 0.0,   
):

    _raw_cache: dict[str, list] = {}   # keywords_key → raw candidates

    def _get_candidates(keywords: list[str]) -> list[tuple[int, float]]:
        key = "|".join(sorted(keywords))
        if key not in _raw_cache:
            _raw_cache[key] = tcsr_retrieve(
                " ".join(keywords), resources,
                top_k=top_k, relation_id=relation_id,
            )
        return apply_constraints_to_internal_candidates(
            _raw_cache[key], constraints,
            resources.idx_to_s2id, resources.scholars,
            pi_collab_ids=pi_collab_ids,
        )

    _idf_weights = _build_idf(grant_keywords, resources.scholars)
    logger.info(
        f"{sorted(_idf_weights.items(), key=lambda x:-x[1])[:3]}"
    )

    def _form_and_score(candidates, kw_list, c_weight, k_weight):
        expanded_kws = []
        for kw in kw_list:
            expanded_kws.append(kw.lower())
            parts = [p for p in kw.lower().replace("-", " ").split() if len(p) > 3]
            expanded_kws.extend(parts)
        expanded_kws = list(dict.fromkeys(expanded_kws))

        result = form_team_user_centric(
            user_id          = pi_internal_id,
            query_text       = query_text,
            candidates       = candidates,
            grant_keywords   = expanded_kws,
            idx_to_s2id      = resources.idx_to_s2id,
            scholars         = resources.scholars,
            team_size        = team_size + 1,
            kg_graph         = resources.kg_graph,
            entity_embedding = resources.entity_embedding.detach(),
            cats_model       = resources.cats_model,
            cats_weight      = c_weight,
            kg_weight        = k_weight,
            llm_client       = llm_client,
            llm_weight       = llm_weight,
        )
        team_ids = result["team_ids"]
      
        tqs, bd = compute_tqs(
            team_ids, grant_keywords,
            resources.idx_to_s2id, resources.scholars,
            gold_team_ids  = gold_team_ids,
            use_idf        = True,
            idf_weights    = _idf_weights,
        )

        cats_score = 0.5 
        if resources.cats_model is not None and resources.entity_embedding is not None:
            import torch as _torch
            valid_ids = [
                iid for iid in team_ids
                if iid < resources.entity_embedding.shape[0]
            ]
            if len(valid_ids) >= 2:
                try:
                    embs = resources.entity_embedding[
                        _torch.tensor(valid_ids, dtype=_torch.long)
                    ].unsqueeze(0)
                    mean_emb = embs.mean(dim=1)
                    cats_score = float(
                        resources.cats_model(mean_emb, embs).item()
                    )
                    cats_score = max(0.0, min(1.0, cats_score))
                except Exception:
                    cats_score = 0.5
        bd["cats"] = round(cats_score, 4)

        members = ids_to_team_members(
            team_ids, resources.idx_to_s2id, resources.scholars
        )
        return members, tqs, bd

    def retrieve_and_form(keywords: list[str]):
        return _form_and_score(
            _get_candidates(keywords), keywords,
            cats_weight, kg_weight,
        )

    _excluded_ids: set = set()
    _rerun_count: list = [0] 

    def rerun_team_formation(keywords: list[str], hint: str = ""):
        _rerun_count[0] += 1
        candidates = _get_candidates(keywords)

        if _excluded_ids:
            candidates = [
                (cid, sc) for cid, sc in candidates
                if cid not in _excluded_ids
            ]
            logger.info(f"[Glue] Excluded {len(_excluded_ids)} rejected members from pool")

        c_w, k_w = cats_weight, kg_weight
        if hint:
            hl = hint.lower()
            if any(w in hl for w in ["diversity", "balance", "interdisciplin"]):
                k_w = min(kg_weight + 0.10 * _rerun_count[0], 0.45)
                c_w = max(cats_weight - 0.05 * _rerun_count[0], 0.10)
            elif any(w in hl for w in ["expertise", "missing", "lack", "irrelevant", "unrelated"]):
                c_w = min(cats_weight + 0.10 * _rerun_count[0], 0.55)
                k_w = max(kg_weight - 0.05 * _rerun_count[0], 0.05)
            logger.info(f"[Glue] rerun #{_rerun_count[0]}: cats_w={c_w:.2f} kg_w={k_w:.2f}")


        import random
        noise_scale = 0.05 * _rerun_count[0]   
        candidates_noisy = [
            (cid, sc * (1 + random.uniform(-noise_scale, noise_scale)))
            for cid, sc in candidates
        ]
        candidates_noisy.sort(key=lambda x: x[1], reverse=True)

        return _form_and_score(candidates_noisy, keywords, c_w, k_w)


    rerun_team_formation._excluded_ids = _excluded_ids

    return retrieve_and_form, rerun_team_formation


# ════════════════════════════════════════════════════════════
# 约束注入 LJR judge prompt
# ════════════════════════════════════════════════════════════

def _patched_build_prompt(inp, constraints: TeamConstraints) -> str:
    base = build_user_prompt(inp)
    ctx  = constraints_to_judge_context(constraints)
    return base + ("\n\n" + ctx if ctx else "")


def run_full_pipeline(
    raw_query: str,
    pi_s2_id: str,                 
    resources: STFResources,
    grant_id: str              = "unknown",
    gold_team_s2_ids: Optional[list] = None,  # ground truth CoPI s2_ids
    provider: LLMProvider      = LLMProvider.GEMINI,
    model_name: Optional[str]  = None,
    tqs_threshold: float       = 0.80,
    max_iter_diagnose: int     = 3,
    max_iter_ljr: int          = 3,
    default_team_size: int     = 4,   
    top_k: int                 = 100,
    relation_id: int           = 0,
    cats_weight: float         = 0.3,
    kg_weight: float           = 0.2,
    llm_client                 = None,
    llm_weight: float          = 0.0,
    pi_collab_ids: Optional[set] = None,
) -> dict:
   
    #  Query Understanding 
    logger.info(f"[Step 0] Parsing query for grant {grant_id} ...")
    parsed: ParsedQuery = parse_query(
        raw_query, provider=provider, model=model_name
    )
    logger.info(f"  keywords:        {parsed.keywords}")
    logger.info(f"  has_constraints: {parsed.has_constraints}")

    team_size = resolve_team_size(parsed.constraints, default=default_team_size)

    # PI internal_id
    pi_internal_id = resources.s2id_to_idx.get(pi_s2_id)
    if pi_internal_id is None:
        raise ValueError(f"PI s2_id '{pi_s2_id}' not found in scholar_id_map")

    gold_internal_ids = None
    if gold_team_s2_ids:
        gold_internal_ids = [
            resources.s2id_to_idx[s] for s in gold_team_s2_ids
            if s in resources.s2id_to_idx
        ]

    retrieve_and_form, rerun_team_formation = make_constrained_fns(
        resources       = resources,
        pi_internal_id  = pi_internal_id,
        query_text      = parsed.clean_query,
        grant_keywords  = parsed.keywords,
        constraints     = parsed.constraints,
        gold_team_ids   = gold_internal_ids,
        pi_collab_ids   = pi_collab_ids,
        top_k           = top_k,
        team_size       = team_size,
        relation_id     = relation_id,
        cats_weight     = cats_weight,
        kg_weight       = kg_weight,
        llm_client      = llm_client,
        llm_weight      = llm_weight,
    )


    logger.info("[Step 1] Initial retrieval + team formation ...")
    keywords = list(parsed.keywords)
    team, tqs, breakdown = retrieve_and_form(keywords)
    logger.info(f"  Initial TQS={tqs:.4f} | {breakdown}")

    logger.info("[Step 2] diagnose_team loop ...")
    diagnose_log = []
    for i in range(max_iter_diagnose):
        if tqs >= tqs_threshold:
            logger.info(f"  TQS {tqs:.4f} ≥ {tqs_threshold}, early stop.")
            break
        action = diagnose_team(breakdown, threshold=tqs_threshold)
        diagnose_log.append({"iter": i, "action": action, "tqs": round(tqs, 4)})
        logger.info(f"  iter {i}: {action}, TQS={tqs:.4f}")

        if action == "re_retrieve":
            extra = [
                w for w in parsed.clean_query.lower().split()
                if w not in keywords and len(w) > 4
            ][:2]
            keywords = keywords + extra
            team, tqs, breakdown = retrieve_and_form(keywords)
        elif action == "re_form":
            team, tqs, breakdown = rerun_team_formation(
                keywords, hint="expertise level low"
            )
        else:
            break

    # LJR loop
    logger.info("[Step 3] LJR loop ...")

    _initial_grant_kw_set = {
        k.lower() for kw in keywords
        for k in kw.lower().replace("-", " ").split()
        if len(k) > 3
    }
    for _member in team:
        _s2_id = _member.scholar_id
        _iid   = resources.s2id_to_idx.get(_s2_id)
        if _iid is None:
            continue
        _data   = resources.scholars.get(_s2_id, {})
        _papers = _data.get("s2_profile", {}).get("papers", [])
        _paper_words = {
            w.lower().strip(".,;:()")
            for p in _papers[:10]
            for w in p.get("title", "").split()
            if len(w) > 3
        }
        if (_initial_grant_kw_set and _paper_words
                and not _initial_grant_kw_set & _paper_words):
            rerun_team_formation._excluded_ids.add(_iid)
            logger.info(f"[Pre-filter] {_member.name} excluded before LJR loop")


    _pre_excluded_in_team = {
        resources.s2id_to_idx.get(m.scholar_id)
        for m in team
        if resources.s2id_to_idx.get(m.scholar_id) in rerun_team_formation._excluded_ids
    }

    _pre_filter_team = team
    _pre_filter_tqs  = tqs
    _pre_filter_bd   = breakdown

    if _pre_excluded_in_team:
        logger.info(f"[Pre-filter] Reforming initial team ({len(_pre_excluded_in_team)} excluded)")
        team, tqs, breakdown = rerun_team_formation(keywords, hint="pre-filter")
        if _pre_filter_tqs > tqs:
            logger.info(
                f"[Pre-filter] Pre-filter TQS {tqs:.4f} < original {_pre_filter_tqs:.4f}，"
                f"keep the origial team as LJR start point，but still remove unrelated members."
            )
            team, tqs, breakdown = _pre_filter_team, _pre_filter_tqs, _pre_filter_bd


    _current_team_ref = [team]   

    MAX_EXCLUSIONS = team_size 

    def rerun_with_exclusion(keywords_arg: list, hint: str = ""):
        current = _current_team_ref[0]

        grant_kw_set = {
            k.lower() for kw in keywords_arg
            for k in kw.lower().replace("-", " ").split()
            if len(k) > 3
        }

        for member in current:
            if len(rerun_team_formation._excluded_ids) >= MAX_EXCLUSIONS:
                break
            s2_id = member.scholar_id
            iid   = resources.s2id_to_idx.get(s2_id)
            if iid is None:
                continue
            data   = resources.scholars.get(s2_id, {})
            papers = data.get("s2_profile", {}).get("papers", [])
            paper_words = {
                w.lower().strip(".,;:()")
                for p in papers[:10]
                for w in p.get("title", "").split()
                if len(w) > 3
            }

            if grant_kw_set and paper_words and not grant_kw_set & paper_words:
                rerun_team_formation._excluded_ids.add(iid)
                logger.info(f"[Exclusion] {member.name} excluded (no overlap with grant keywords)")

        new_team, new_tqs, new_bd = rerun_team_formation(keywords_arg, hint=hint)
        _current_team_ref[0] = new_team
        return new_team, new_tqs, new_bd

    import llm_judge as _lj


    _orig = _lj.build_user_prompt
    _lj.build_user_prompt = lambda inp: _patched_build_prompt(
        inp, parsed.constraints
    )


    def _retrieve_starting_from_clean_team(kw):
        if not _retrieve_starting_from_clean_team._called:
            _retrieve_starting_from_clean_team._called = True
            return team, tqs, breakdown
        return retrieve_and_form(kw)
    _retrieve_starting_from_clean_team._called = False

    try:
        ljr_out = _lj.ljr_refinement_loop(
            grant_title             = parsed.clean_query,
            grant_abstract          = raw_query,
            initial_keywords        = keywords,
            retrieve_and_form_fn    = _retrieve_starting_from_clean_team,
            rerun_team_formation_fn = rerun_with_exclusion,
            provider                = provider,
            model                   = model_name,
            max_iter                = max_iter_ljr,
        )
    finally:
        _lj.build_user_prompt = _orig

    final_team = ljr_out["team"]
    excluded_ids = rerun_team_formation._excluded_ids
    excluded_in_best = [
        m for m in final_team
        if resources.s2id_to_idx.get(m.scholar_id) in excluded_ids
    ]

    if excluded_in_best:
        logger.info(
            f"[Post-filter] best team has {len(excluded_in_best)} removed members，"
            f"try to replace: {[m.name for m in excluded_in_best]}"
        )
        final_team_members, final_tqs_new, final_bd = rerun_team_formation(
            ljr_out["keywords"], hint="post-filter cleanup"
        )

        if final_tqs_new >= ljr_out["tqs"] * 0.95:
            final_team = final_team_members
            ljr_out["tqs"] = final_tqs_new
            logger.info(f"replaced finished，TQS {ljr_out['tqs']:.4f} → {final_tqs_new:.4f}")
        else:
            logger.warning(
                f"[Post-filter] 替换后 TQS={final_tqs_new:.4f} < best {ljr_out['tqs']:.4f}×0.95，"
                f"remove the old members, get new members from candidates."
            )
            keep = [m for m in final_team
                    if resources.s2id_to_idx.get(m.scholar_id) not in excluded_ids]
            if len(keep) < team_size:
                new_members = [
                    m for m in final_team_members
                    if resources.s2id_to_idx.get(m.scholar_id) not in excluded_ids
                    and m.scholar_id not in {k.scholar_id for k in keep}
                ]
                keep = keep + new_members[:team_size - len(keep)]
            final_team = keep
            logger.info(f"Supplyment finished，get final {len(final_team)} members")

    logger.info(
        f"[Done] TQS {tqs:.4f}→{ljr_out['tqs']:.4f} | "
        f"verdict={ljr_out['final_verdict']} | "
        f"LJR iters={len(ljr_out['iterations'])}"
    )

    return {
        "grant_id"           : grant_id,
        "parsed_keywords"    : parsed.keywords,
        "clean_query"        : parsed.clean_query,
        "has_constraints"    : parsed.has_constraints,
        "constraints_text"   : parsed.constraints.raw_constraint_text,
        "team_size_used"     : team_size,
        "final_team"         : final_team,
        "final_keywords"     : ljr_out["keywords"],
        "final_tqs"          : ljr_out["tqs"],
        "diagnose_log"       : diagnose_log,
        "ljr_iterations"     : ljr_out["iterations"],
        "ljr_final_verdict"  : ljr_out["final_verdict"],
    }

def run_batch(
    queries: list[dict],     # [{id, raw_query, pi_s2_id, gold_team_s2_ids?}, ...]
    resources: STFResources,
    output_path: str = "results/pipeline_out.json",
    checkpoint_every: int = 10,
    **pipeline_kwargs,
) -> list[dict]:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results = []
    for i, q in enumerate(queries):
        logger.info(f"[{i+1}/{len(queries)}] {q['id']}")
        try:
            out = run_full_pipeline(
                raw_query        = q["raw_query"],
                pi_s2_id         = q["pi_s2_id"],
                grant_id         = q["id"],
                gold_team_s2_ids = q.get("gold_team_s2_ids"),
                resources        = resources,
                **pipeline_kwargs,
            )
            results.append(out)
        except Exception as e:
            logger.error(f"  Failed: {e}", exc_info=True)
            results.append({"grant_id": q["id"], "error": str(e)})

        if (i + 1) % checkpoint_every == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


if __name__ == "__main__":
    import os
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    os.environ.setdefault("GEMINI_API_KEY", "")

    resources = STFResources(
        model_ckpt  = "",
        data_dir    = "",
        kg_graph_path = "",
        cats_checkpoint = "",
        # nentity     = 11238,    
        nrelation   = 5,
        hidden_dim  = 400,
        gamma       = 12.0,
        box_mode    = ("none", 0.02),
        device      = "cuda",
    )

    result = run_full_pipeline(
        raw_query = (
            "Neural-AI-Aided Discovery in Neuroscience: "
            "computational methods for brain-computer interfaces and fMRI. "
            "I need a team of 4, from different universities, "
            "with at least one early career researcher."
        ),
        pi_s2_id         = "2474084",
        resources        = resources,
        grant_id         = "naiad_test",
        provider         = LLMProvider.GEMINI,
        tqs_threshold    = 0.80,
        default_team_size = 4,
        relation_id      = 0,
        cats_weight      = 0.3,
        kg_weight        = 0.2,
    )

    print("\n" + "=" * 60)
    print(f"Keywords:        {result['parsed_keywords']}")
    print(f"Has constraints: {result['has_constraints']}")
    print(f"Team size used:  {result['team_size_used']}")
    print(f"Final TQS:       {result['final_tqs']:.4f}")
    print(f"LJR verdict:     {result['ljr_final_verdict']}")
    print("Team:")
    for m in result["final_team"]:
        print(f"  [{m.h_index or '?':>3}] {m.name}: {m.expertise_summary[:55]}")
