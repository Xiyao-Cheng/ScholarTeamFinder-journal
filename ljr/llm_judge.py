import json
import re
import os
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    OPENAI = "openai"
    GEMINI = "gemini"


PROVIDER_CONFIG = {
    LLMProvider.OPENAI: {
        "base_url": None,                          
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o",
    },
    LLMProvider.GEMINI: {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "default_model": "gemini-2.5-flash",       
    },
}


def get_client(provider: LLMProvider) -> tuple[OpenAI, str]:
    cfg = PROVIDER_CONFIG[provider]
    client = OpenAI(
        api_key=os.environ[cfg["api_key_env"]],
        base_url=cfg["base_url"],   
    )
    return client, cfg["default_model"]


@dataclass
class TeamMember:
    scholar_id: str
    name: str
    expertise_summary: str       
    recent_titles: list[str]     
    h_index: Optional[int] = None


@dataclass
class JudgeInput:
    grant_title: str
    grant_abstract: str
    keywords: list[str]          
    team: list[TeamMember]
    tqs_score: float             
    tqs_breakdown: dict          
    iteration: int = 0         


@dataclass
class JudgeVerdict:
    verdict: str                 
    reason: str                
    suggestion: str             
    new_keywords: list           
    confidence: float           
    raw_response: str = ""     


SYSTEM_PROMPT = """You are an expert scientific program officer evaluating research team compositions for grant proposals.
Your role is to assess whether the assembled team is semantically well-matched to the grant's goals.

You have access to:
- The grant title and abstract
- The keywords used to retrieve candidate scholars
- The assembled team with their expertise profiles
- Quantitative quality scores (for context only — do NOT base your verdict solely on numbers)

Your verdict must be one of three categories:
  "query_issue"  — The keywords fail to capture the grant's core research scope.
                   Even good scholars are being pulled from the wrong area due to poor keywords.
  "team_issue"   — The keywords are reasonable, but the assembled team has compositional problems:
                   missing expertise, redundant coverage, poor disciplinary balance, or
                   lack of methodological diversity relative to the grant's needs.
  "pass"         — The team is semantically appropriate for this grant. Minor imperfections
                   are acceptable; pass if the team could realistically execute this work.

Decision guidelines:
- Prefer "query_issue" when >50% of team members' expertise is tangentially related or off-topic.
- Prefer "team_issue" when members are in the right area but team is unbalanced or missing a key role.
- Prefer "pass" when the team covers the grant's main intellectual thrusts, even if not perfectly.
- Do NOT fail a team purely because TQS is below a threshold — you are the semantic check.

Respond ONLY with a valid JSON object. No preamble, no markdown fences.
Schema:
{
  "verdict": "<query_issue|team_issue|pass>",
  "reason": "<one concise sentence explaining your verdict>",
  "suggestion": "<if team_issue: describe missing expertise in one sentence. if pass: empty string>",
  "new_keywords": ["kw1", "kw2", "kw3"],
  "confidence": <float 0.0-1.0>
}

IMPORTANT for new_keywords:
- Only populate new_keywords when verdict is "query_issue"
- Provide 2-4 specific ACADEMIC DOMAIN TERMS only (e.g. "neural signal processing", "fMRI connectivity")
- Each keyword must be a noun phrase that describes a research area, NOT a fragment of the original query
- Do NOT split or repeat words already in the keywords list (e.g. if "brain-computer interfaces" exists, do NOT add "brain-computer" or "interfaces")
- Do NOT include meta-instructions, verbs, or procedural text
- Leave as empty list [] for team_issue or pass"""


def build_user_prompt(inp: JudgeInput) -> str:
    team_lines = []
    for i, m in enumerate(inp.team, 1):
        titles = "; ".join(f'"{t}"' for t in m.recent_titles[:3])
        team_lines.append(
            f"  [{i}] {m.name}\n"
            f"      research direction: {m.expertise_summary}\n"
            f"      recent publications: {titles}"
        )

    bd = inp.tqs_breakdown
    tqs_str = (
        f"TQS={inp.tqs_score:.3f} "
        f"(Coverage={bd.get('coverage',0):.2f}, "
        f"CL={bd.get('cl',0):.2f}, "
        f"EL={bd.get('el',0):.2f}, "
        f"GM={bd.get('gm',0):.2f})"
    )

    return f"""=== grant proposal ===
title: {inp.grant_title}

abstract:
{inp.grant_abstract}

=== current keywords ===
{", ".join(inp.keywords)}

=== have built（no. {inp.iteration} iter）===
{chr(10).join(team_lines)}

=== ref TQS score===
{tqs_str}
"""


def call_llm_judge(
    inp: JudgeInput,
    provider: LLMProvider = LLMProvider.GEMINI,
    model: Optional[str] = None,
    temperature: float = 0.0,  
    max_retries: int = 2,
) -> JudgeVerdict:

    client, default_model = get_client(provider)
    model = model or default_model
    user_prompt = build_user_prompt(inp)
    raw = ""

    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content.strip()
            parsed = json.loads(raw)

            verdict = parsed.get("verdict", "").lower()
            if verdict not in {"query_issue", "team_issue", "pass"}:
                raise ValueError(f"unexpected verdict: {verdict}")

            return JudgeVerdict(
                verdict      = verdict,
                reason       = parsed.get("reason", ""),
                suggestion   = parsed.get("suggestion", ""),
                new_keywords = parsed.get("new_keywords", []),
                confidence   = float(parsed.get("confidence", 0.5)),
                raw_response = raw,
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Judge failed on（attempt no. {attempt+1}）: {e}\n orignal output: {raw}")
            if attempt == max_retries:
                logger.error("LLM judge retry failed, return pass")
                return JudgeVerdict(
                    verdict      = "pass",
                    reason       = "Judge fail，return pass",
                    suggestion   = "",
                    new_keywords = [],
                    confidence   = 0.0,
                    raw_response = raw,
                )



def _is_fragment_of_existing(kw: str, existing_keywords: list[str]) -> bool:
    kw_l = kw.lower().strip()
    if len(kw_l.split()) > 2: 
        return False
    for orig in existing_keywords:
        orig_l = orig.lower()
        orig_parts = set(orig_l.replace("-", " ").split())
        kw_parts   = set(kw_l.replace("-", " ").split())
        if kw_parts and kw_parts.issubset(orig_parts):
            return True
        if kw_l in orig_l:
            return True
    return False


_META_PHRASES = {
    "refine", "keywords", "specific", "restrictive", "add", "adding",
    "such as", "suggest", "include", "more", "relevant", "filter",
    "improve", "expand", "use", "better", "try", "consider",
}


def _is_meta_phrase(text: str) -> bool:
    tl = text.lower().strip()
    if len(tl.split()) > 6:   
        return True
    if any(m in tl for m in _META_PHRASES):
        return True
    return False


def expand_keywords(
    current_keywords: list[str],
    suggestion: str,
    max_new: int = 3,
) -> list[str]:
    quoted = re.findall(r'["\']([^"\']{2,40})["\']', suggestion)
    valid_quoted = [q.strip() for q in quoted if not _is_meta_phrase(q)]
    if valid_quoted:
        new_kws = valid_quoted[:max_new]
    else:
        parts = re.split(r"[,;]+", suggestion)
        candidates = [p.strip() for p in parts if 2 < len(p.strip()) <= 50]
        new_kws = [c for c in candidates if not _is_meta_phrase(c)][:max_new]

    existing_lower = {kw.lower() for kw in current_keywords}
    added = [kw for kw in new_kws if kw.lower() not in existing_lower]

    if not added:
        logger.warning(f"keyword extension：cannot get available keywords from suggestions: {str(suggestion)[:80]!r}")
        return current_keywords

    expanded = current_keywords + added
    logger.info(f"keyword extension: {current_keywords} → {expanded}")
    return expanded



def ljr_refinement_loop(
    grant_title: str,
    grant_abstract: str,
    initial_keywords: list[str],
    retrieve_and_form_fn,        # (keywords) → (team, tqs, tqs_breakdown)
    rerun_team_formation_fn,     # (keywords, hint) → (team, tqs, tqs_breakdown)
    provider: LLMProvider = LLMProvider.GEMINI,
    model: Optional[str] = None,
    max_iter: int = 3,
    keyword_expansion_once: bool = True, 
) -> dict:
    keywords = list(initial_keywords)
    team, tqs, tqs_breakdown = retrieve_and_form_fn(keywords)
    keyword_expanded = False
    iteration_logs = []

    best_team, best_tqs, best_keywords = team, tqs, list(keywords)

    for i in range(max_iter):
        inp = JudgeInput(
            grant_title    = grant_title,
            grant_abstract = grant_abstract,
            keywords       = keywords,
            team           = team,
            tqs_score      = tqs,
            tqs_breakdown  = tqs_breakdown,
            iteration      = i,
        )

        verdict = call_llm_judge(inp, provider=provider, model=model)
        logger.info(
            f"[LJR number {i} iter ] verdict={verdict.verdict} "
            f"conf={verdict.confidence:.2f} | {verdict.reason}"
        )

        iteration_logs.append({
            "iteration"  : i,
            "verdict"    : verdict.verdict,
            "reason"     : verdict.reason,
            "suggestion" : verdict.suggestion,
            "confidence" : verdict.confidence,
            "tqs"        : tqs,
            "keywords"   : list(keywords),
        })

        # 每轮开始时更新 best
        if tqs > best_tqs:
            best_team, best_tqs, best_keywords = team, tqs, list(keywords)

        if verdict.verdict == "pass":
            return {
                "team"          : team,
                "keywords"      : keywords,
                "tqs"           : tqs,
                "iterations"    : iteration_logs,
                "final_verdict" : "pass",
            }

        elif verdict.verdict == "query_issue":
            if keyword_expansion_once and keyword_expanded:
                logger.info("Keywords already extension, so go to team_issue.")
                team, tqs, tqs_breakdown = rerun_team_formation_fn(
                    keywords, hint=verdict.suggestion
                )
            else:
                existing_lower = {k.lower() for k in keywords}
                new_kws = [
                    kw for kw in verdict.new_keywords
                    if kw
                    and kw.lower() not in existing_lower      
                    and len(kw.strip()) > 3                   
                    and len(kw.split()) <= 5                
                    and len(kw.split()) >= 2                  
                    and not kw.endswith(":")                 
                    and "-" not in kw                        
                    and not _is_fragment_of_existing(kw, keywords)  
                ]
                if new_kws:
                    keywords = keywords + new_kws
                    logger.info(f"keyword extension: {keywords}")
                else:
                    logger.warning("It's query_issue, but new_keywords is null, so go to team_issue. ")
                keyword_expanded = True
                team, tqs, tqs_breakdown = retrieve_and_form_fn(keywords)

        elif verdict.verdict == "team_issue":
            team, tqs, tqs_breakdown = rerun_team_formation_fn(
                keywords, hint=verdict.suggestion
            )

        if tqs > best_tqs:
            best_team, best_tqs, best_keywords = team, tqs, list(keywords)

    logger.warning(
        f"LJR 超过 max_iter={max_iter}，"
        f"return the best team: TQS={best_tqs:.4f}）。"
    )
    return {
        "team"          : best_team,
        "keywords"      : best_keywords,
        "tqs"           : best_tqs,
        "iterations"    : iteration_logs,
        "final_verdict" : "max_iter_reached",
    }
