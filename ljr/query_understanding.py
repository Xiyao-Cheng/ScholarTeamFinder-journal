import json
import os
import re
import logging
from dataclasses import dataclass, field
from typing import Optional
from openai import OpenAI

from llm_judge import LLMProvider, PROVIDER_CONFIG, get_client

logger = logging.getLogger(__name__)



@dataclass
class TeamConstraints:
    team_size: Optional[int] = None             # "team of 4" → 4

    affiliation_types: list[str] = field(default_factory=list)

    required_institutions: list[str] = field(default_factory=list)
    # 例: ["MIT", "Stanford"]


    min_h_index: Optional[int] = None           # "senior researcher" → 20
    max_h_index: Optional[int] = None           # "early career" → 10
    seniority: Optional[str] = None             # "senior" | "junior" | None

    require_interdisciplinary: bool = False     
    require_different_institutions: bool = False  

    exclude_prior_collabs: bool = False         
    require_prior_collab: bool = False          

    required_roles: list[str] = field(default_factory=list)

    raw_constraint_text: str = ""

    def is_empty(self) -> bool:
        return (
            self.team_size is None
            and not self.affiliation_types
            and not self.required_institutions
            and self.min_h_index is None
            and self.max_h_index is None
            and self.seniority is None
            and not self.require_interdisciplinary
            and not self.require_different_institutions
            and not self.exclude_prior_collabs
            and not self.require_prior_collab
            and not self.required_roles
        )


@dataclass
class ParsedQuery:
    original_query: str
    clean_query: str        
    keywords: list[str]     
    constraints: TeamConstraints
    has_constraints: bool   


# ─────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────

PARSE_SYSTEM_PROMPT = """You are a research query parser for an academic team recommendation system.

Your task: given a user's natural language query for finding research collaborators,
separate it into two parts:
  1. Research intent — what scientific topics/skills are needed (→ used for scholar retrieval)
  2. Team constraints — explicit requirements about the team composition (→ used to filter/rank)

Team constraints to detect:
  - team_size: explicit number of co-PIs wanted (e.g. "team of 4", "3 collaborators")
  - affiliation_types: institution type preferences ["university", "industry", "government", "national_lab"]
  - required_institutions: specific named institutions (e.g. ["MIT", "Stanford"])
  - min_h_index: minimum h-index requirement (e.g. "senior researcher" → 20, "h-index > 15" → 15)
  - max_h_index: maximum h-index (e.g. "early career" → 10)
  - seniority: "senior" | "junior" | null
  - require_interdisciplinary: true if user explicitly wants cross-disciplinary team
  - require_different_institutions: true if user wants members from different institutions
  - exclude_prior_collabs: true if user wants people they haven't worked with before
  - require_prior_collab: true if user wants existing collaborators
  - required_roles: specific expertise roles mentioned (e.g. ["statistician", "bioinformatician"])

Rules:
- Be conservative: only extract constraints that are EXPLICITLY stated. Do not infer.
- "top universities" → affiliation_types: ["university"], NOT required_institutions
- "someone with industry experience" → affiliation_types: ["industry"]
- "early career researcher" → max_h_index: 10, seniority: "junior"
- "established professor" → min_h_index: 20, seniority: "senior"
- The clean_query should be the query with constraint phrases removed, keeping only research content.
- Keywords: 3-8 specific research terms, noun phrases only, no verbs.

Respond ONLY with valid JSON, no preamble, no markdown fences.
Schema:
{
  "clean_query": "<research-only description>",
  "keywords": ["kw1", "kw2", ...],
  "constraints": {
    "team_size": <int or null>,
    "affiliation_types": [],
    "required_institutions": [],
    "min_h_index": <int or null>,
    "max_h_index": <int or null>,
    "seniority": "<senior|junior|null>",
    "require_interdisciplinary": false,
    "require_different_institutions": false,
    "exclude_prior_collabs": false,
    "require_prior_collab": false,
    "required_roles": [],
    "raw_constraint_text": "<verbatim constraint phrases from original query>"
  },
  "has_constraints": <bool>
}"""


def parse_query(
    raw_query: str,
    provider: LLMProvider = LLMProvider.GEMINI,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_retries: int = 2,
) -> ParsedQuery:
    constraint_signals = [
        "team of", "collaborator", "from", "university", "industry",
        "senior", "junior", "early career", "h-index", "experience",
        "institution", "interdisciplin", "diversity", "different",
        "prior", "before", "worked with", "statistician", "engineer",
        "expert in", "specialist",
    ]
    has_any_signal = any(sig in raw_query.lower() for sig in constraint_signals)

    if not has_any_signal:
        keywords = _rule_based_keywords(raw_query)
        return ParsedQuery(
            original_query = raw_query,
            clean_query    = raw_query,
            keywords       = keywords,
            constraints    = TeamConstraints(),
            has_constraints= False,
        )

    client, default_model_name = get_client(provider)
    model = model or default_model_name
    raw = ""

    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model    = model,
                temperature = temperature,
                messages = [
                    {"role": "system", "content": PARSE_SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Query: {raw_query}"},
                ],
                response_format = {"type": "json_object"},
            )
            raw = resp.choices[0].message.content.strip()
            parsed = json.loads(raw)

            constraints_dict = parsed.get("constraints", {})
            if constraints_dict.get("seniority") in ("null", "", None):
                constraints_dict["seniority"] = None

            constraints = TeamConstraints(
                team_size                  = constraints_dict.get("team_size"),
                affiliation_types          = constraints_dict.get("affiliation_types", []),
                required_institutions      = constraints_dict.get("required_institutions", []),
                min_h_index                = constraints_dict.get("min_h_index"),
                max_h_index                = constraints_dict.get("max_h_index"),
                seniority                  = constraints_dict.get("seniority"),
                require_interdisciplinary  = constraints_dict.get("require_interdisciplinary", False),
                require_different_institutions = constraints_dict.get("require_different_institutions", False),
                exclude_prior_collabs      = constraints_dict.get("exclude_prior_collabs", False),
                require_prior_collab       = constraints_dict.get("require_prior_collab", False),
                required_roles             = constraints_dict.get("required_roles", []),
                raw_constraint_text        = constraints_dict.get("raw_constraint_text", ""),
            )

            result = ParsedQuery(
                original_query  = raw_query,
                clean_query     = parsed.get("clean_query", raw_query),
                keywords        = parsed.get("keywords", _rule_based_keywords(raw_query)),
                constraints     = constraints,
                has_constraints = parsed.get("has_constraints", not constraints.is_empty()),
            )

            logger.info(
                f"[QueryParse] has_constraints={result.has_constraints} | "
                f"keywords={result.keywords} | "
                f"constraints={constraints.raw_constraint_text!r}"
            )
            return result

        except (json.JSONDecodeError, KeyError) as e:
            if attempt == max_retries:
                return ParsedQuery(
                    original_query  = raw_query,
                    clean_query     = raw_query,
                    keywords        = _rule_based_keywords(raw_query),
                    constraints     = TeamConstraints(),
                    has_constraints = False,
                )


def _rule_based_keywords(text: str, max_kw: int = 6) -> list[str]:
    stopwords = {
        "the", "a", "an", "and", "or", "for", "in", "of", "to", "with",
        "on", "at", "by", "from", "is", "are", "was", "we", "our", "this",
        "that", "it", "be", "as", "have", "has", "find", "need", "want",
        "looking", "research", "project", "study", "work", "team",
    }
    words = re.findall(r"\b[a-zA-Z][a-zA-Z\-]+\b", text.lower())
    filtered = [w for w in words if w not in stopwords and len(w) > 3]

    seen, keywords = set(), []
    for w in filtered:
        if w not in seen:
            seen.add(w)
            keywords.append(w)
        if len(keywords) >= max_kw:
            break
    return keywords


def apply_constraints_to_candidates(
    candidates: list[dict],
    constraints: TeamConstraints,
    pi_collab_ids: Optional[set[str]] = None,
    soft_only: bool = False,    
) -> list[dict]:
    if constraints.is_empty():
        return candidates

    filtered = []
    for c in candidates:
        score     = c.get("retrieval_score", 0.0)
        hard_fail = False

        h = c.get("h_index") or 0
        if constraints.min_h_index and h < constraints.min_h_index:
            hard_fail = True

        early_career_bonus = 0.0
        if constraints.max_h_index and h <= constraints.max_h_index:
            early_career_bonus = 0.15

        if constraints.affiliation_types and not hard_fail:
            affil = c.get("affiliation", "").lower()
            if affil and not any(
                _affil_matches(affil, t) for t in constraints.affiliation_types
            ):
                hard_fail = True

        if constraints.required_institutions and not hard_fail:
            affil = c.get("affiliation", "").lower()
            if affil and not any(
                inst.lower() in affil for inst in constraints.required_institutions
            ):
                hard_fail = True


        if pi_collab_ids and not hard_fail:
            sid = c.get("scholar_id", "")
            if constraints.exclude_prior_collabs and sid in pi_collab_ids:
                hard_fail = True
            if constraints.require_prior_collab and sid not in pi_collab_ids:
                hard_fail = True

        if hard_fail:
            continue


        role_bonus = 0.0
        if constraints.required_roles:
            expertise = c.get("expertise_summary", "").lower()
            role_bonus = sum(
                0.1 for role in constraints.required_roles
                if role.lower() in expertise
            )

        filtered_c = dict(c)
        filtered_c["retrieval_score"] = score + role_bonus + early_career_bonus
        filtered.append(filtered_c)

    filtered.sort(key=lambda x: x["retrieval_score"], reverse=True)
    return filtered


def _affil_matches(affil: str, affil_type: str) -> bool:
    rules = {
        "university":   ["university", "univ", "college", "institute of technology", "school of"],
        "industry":     ["inc", "corp", "ltd", "llc", "google", "microsoft", "amazon",
                         "meta", "apple", "nvidia", "ibm", "labs", "research center"],
        "government":   ["national", "federal", "department of", "agency", "nasa", "doe", "nih"],
        "national_lab": ["national lab", "argonne", "oak ridge", "brookhaven", "sandia",
                         "lawrence", "pacific northwest", "nrel"],
    }
    keywords = rules.get(affil_type, [affil_type])
    return any(kw in affil for kw in keywords)


def resolve_team_size(constraints: TeamConstraints, default: int = 4) -> int:
    if constraints.team_size is not None:
        copi_size = max(1, constraints.team_size - 1)
        logger.info(f"[Constraints] team_size override: {constraints.team_size} → {copi_size} Co-PIs")
        return copi_size
    return default


def constraints_to_judge_context(constraints: TeamConstraints) -> str:
    if constraints.is_empty():
        return ""

    lines = ["=== Explicit User Team Requirements ==="]
if constraints.team_size:
    lines.append(f"- Desired team size (including PI): {constraints.team_size}")
if constraints.affiliation_types:
    lines.append(f"- Preferred institution types: {', '.join(constraints.affiliation_types)}")
if constraints.required_institutions:
    lines.append(f"- Required institutions: {', '.join(constraints.required_institutions)}")
if constraints.min_h_index:
    lines.append(f"- Minimum h-index: {constraints.min_h_index}")
if constraints.max_h_index:
    lines.append(f"- Maximum h-index (early-career scholars): {constraints.max_h_index}")
if constraints.seniority:
    lines.append(f"- Seniority preference: {constraints.seniority}")
if constraints.require_interdisciplinary:
    lines.append("- Explicitly requires an interdisciplinary team")
if constraints.require_different_institutions:
    lines.append("- Requires members from different institutions")
if constraints.exclude_prior_collabs:
    lines.append("- Excludes prior collaborators")
if constraints.required_roles:
    lines.append(f"- Required expertise roles: {', '.join(constraints.required_roles)}")
lines.append(
    "\nWhen evaluating the team, also check whether the above requirements are satisfied. "
    "If the team violates a hard constraint (mismatched institution type, mismatched seniority), "
    "classify it as team_issue even if semantic coverage is otherwise adequate."
)
    return "\n".join(lines)
