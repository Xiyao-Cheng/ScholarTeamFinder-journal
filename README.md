# ScholarTeamFinder (STF)

**ScholarTeamFinder** is a system that recommends multi-disciplinary research collaborator teams based on natural language queries (e.g., grant proposal descriptions), designed to support cross-disciplinary NSF grant proposal formation. Given a research need expressed in natural language and a fixed principal investigator (PI), STF recommends a complementary team of Co-PIs whose combined expertise best addresses the proposal's scope.

## Overview

Manually assembling cross-disciplinary research teams is increasingly difficult as the number of scholars and the granularity of their expertise grow. Existing approaches — knowledge graph-based scholar recommendation, dense retrieval over scientific document encoders, and historical team identification — each address only part of this problem: they rely on explicit pre-recorded connections, retrieve individuals without reasoning about team-level complementarity, or fail to account for the querying PI's own expertise.

**ScholarTeamFinder** addresses these gaps through a three-stage pipeline:

1. **TCSR (Text-Conditioned Scholar Retrieval)** — a box-embedding retrieval model built on SciBERT that retrieves candidate scholars relevant to a natural-language query, trained via a two-phase strategy that stabilizes box geometry before fine-tuning the encoder.
2. **User-Centric Team Formation** — a greedy team formation framework that fixes the querying PI and selects complementary Co-PIs by jointly optimizing marginal keyword coverage, a learned Collaboration-Aware Team Scorer (CATS), and structural diversity from the collaboration knowledge graph.
3. **LJR (LLM-as-Judge Refinement)** — a semantic validation layer that classifies a candidate team as `pass`, `query_issue`, or `team_issue`, triggering re-retrieval or re-formation when the assembled team does not genuinely fit the proposal's research scope.


## Repository Structure

```
scholarteamfinder/
├── data/                    # NSF grant data, Semantic Scholar scholar profiles
├── retrieval/                    # Text-Conditioned Scholar Retrieval (box embedding model)
│   ├── model.py
│   ├── train.py
├── team_formation/           # User-centric team formation module
│   ├── user_centric_team_formation.py
│   ├── cats.py               # Collaboration-Aware Team Scorer
│   └── kg_features.py        # Knowledge graph structural diversity features
├── ljr/                       # LLM-as-Judge Refinement module
│   ├── llm_judge.py
│   ├── query_understanding.py
│   └── pipeline.py
├── evaluation/                # Evaluation scripts (retrieval + team quality metrics)
├── case_studies/               # Case study generation scripts
└── README.md
```

> **Note:** paths above reflect the intended organization of this repository; adjust to match actual file layout before publishing.

**Requirements:**
- Python 3.10+
- PyTorch
- `transformers` (for SciBERT: `allenai/scibert_scivocab_uncased`)
- Access to a Gemini API key (for LJR; passed via `--gemini_key`, not environment variables)


## Data

ScholarTeamFinder operates on NSF grant data paired with scholar profiles drawn from Semantic Scholar and NSF databases (~11,238 entities). Each scholar is uniquely identified by `s2_author_id`.

> Due to data usage agreements, raw NSF and Semantic Scholar data are not redistributed in this repository. 

## Usage

### 1. Train the retrieval model (TCSR)

```bash
python tcsr/train.py --data_dir data/training_data --phase both
```

### 2. Run team formation

```bash
python team_formation/formation.py \
  --data_dir data/training_data \
  --retrieval_results retrieval_top50.pkl \
  --team_size 5
```

### 3. Run the full pipeline with LJR

```bash
python ljr/pipeline.py \
  --data_dir data/training_data \
  --gemini_key <YOUR_GEMINI_API_KEY> \
  --max_iterations 3
```

## Evaluation

We evaluate retrieval quality using MRR, HR@K, and NDCG@K, and team quality using a composite **Team Quality Score (TQS)** combining:

- **Coverage** — fraction of grant keywords covered by the team
- **Match Set Count Level (CL)** — fraction of keywords covered by ≥2 members
- **Expertise Level (EL)** — H-index-weighted keyword coverage
- **Gold Match (GM$_{F1}$)** — F1 score against the ground-truth Co-PI set


## Acknowledgments

This work was conducted at AERI Lab.
