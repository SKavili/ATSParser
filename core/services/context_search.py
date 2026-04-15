"""Context query search + end-to-end test helper for ats_context_pipeline."""
import asyncio
import json
import re
from typing import Any, Dict, List, Optional

import httpx
from httpx import Timeout
from sqlalchemy import select

from app.config import settings
from app.database.connection import async_session_maker
from app.database.models import ResumeMetadata
from app.services.llm_service import generate as llm_generate
from app.services.llm_service import use_openai
from app.utils.logging import get_logger
from core.services.context_builder import build_context
from core.services.context_embedding import generate_embedding
from core.services.context_indexer import CONTEXT_INDEX_NAME, ContextIndexer

logger = get_logger(__name__)

CONTEXT_QUERY_JSON_SYSTEM = """You extract structured hiring intent for a candidate search profile.
Return ONLY valid JSON (no markdown, no code fences) with exactly this shape:
{"role": "", "experience": "", "skills": []}

Rules:
- "role": job title or primary role if clearly stated; else "".
- "experience": years as a short string if the query mentions them (e.g. "3", "5+"); else "".
- "skills": array of distinct skills/technologies explicitly mentioned; else [].
- Use only information supported by the query; do not invent employers or locations.
"""


def _extract_json_object(raw: str) -> Optional[Dict[str, Any]]:
    if not raw or not str(raw).strip():
        return None
    cleaned = str(raw).strip()
    cleaned = re.sub(r"```json\s*", "", cleaned)
    cleaned = re.sub(r"```\s*", "", cleaned)
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def _profile_from_llm_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    role = (data.get("role") or "").strip() if isinstance(data.get("role"), (str, int, float)) else ""
    if not isinstance(role, str):
        role = str(role).strip()
    exp = data.get("experience")
    if exp is not None and not isinstance(exp, str):
        exp = str(exp).strip()
    elif isinstance(exp, str):
        exp = exp.strip()
    else:
        exp = ""
    skills_raw = data.get("skills")
    skills: List[str] = []
    if isinstance(skills_raw, list):
        skills = [str(s).strip() for s in skills_raw if s is not None and str(s).strip()][:30]
    profile: Dict[str, Any] = {}
    if role:
        profile["role"] = role[:300]
    if exp:
        profile["experience"] = exp[:100]
    if skills:
        profile["skills"] = skills
    if not any(profile.get(k) for k in ("role", "experience", "skills")):
        return {}
    return profile


async def parse_natural_language_context_query(query: str) -> Dict[str, Any]:
    """
    Parse free-text query via LLM (OpenAI when LLM_MODEL=OpenAI, else Ollama chat), else heuristics.

    Returns profile keys: role, experience, skills — suitable for build_context.
    """
    q = (query or "").strip()
    if not q:
        return {}

    max_len = getattr(settings, "ai_search_query_max_length", 2000)
    if len(q) > max_len:
        q = q[:max_len]

    user_msg = f"Query:\n{q}\n\nRespond with JSON only."

    if use_openai():
        if not getattr(settings, "openai_api_key", None) or not str(settings.openai_api_key).strip():
            logger.warning("LLM_MODEL=OpenAI but OPENAI_API_KEY missing; using heuristic query parse")
            return extract_profile_from_query(q)
        try:
            raw = await llm_generate(
                prompt=user_msg,
                system_prompt=CONTEXT_QUERY_JSON_SYSTEM,
                temperature=0.1,
                timeout_seconds=120.0,
            )
            parsed = _extract_json_object(raw or "")
            if parsed:
                prof = _profile_from_llm_dict(parsed)
                if prof:
                    return prof
        except Exception as e:
            logger.warning(f"OpenAI context query parse failed: {e}; using heuristics")
        return extract_profile_from_query(q)

    # LLM_MODEL=OLLAMA: use Ollama /api/chat for parsing (embeddings still use separate Ollama embeddings API)
    try:
        raw = await _ollama_chat_context_json(user_msg)
        parsed = _extract_json_object(raw or "")
        if parsed:
            prof = _profile_from_llm_dict(parsed)
            if prof:
                return prof
    except Exception as e:
        logger.warning(f"Ollama context query parse failed: {e}; using heuristics")

    return extract_profile_from_query(q)


async def _ollama_chat_context_json(user_msg: str) -> str:
    host = (settings.ollama_host or "").rstrip("/")
    if not host:
        raise RuntimeError("OLLAMA_HOST not set")
    model = (getattr(settings, "context_query_ollama_model", None) or "llama3.1").strip()
    async with httpx.AsyncClient(timeout=Timeout(120.0)) as client:
        r = await client.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": CONTEXT_QUERY_JSON_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                "stream": False,
                "options": {"temperature": 0.1},
            },
        )
        if r.status_code == 404:
            r = await client.post(
                f"{host}/api/generate",
                json={
                    "model": model,
                    "prompt": f"{CONTEXT_QUERY_JSON_SYSTEM}\n\n{user_msg}",
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
            )
        r.raise_for_status()
        data = r.json()
        if "message" in data and isinstance(data["message"], dict):
            return (data["message"].get("content") or "").strip()
        return (data.get("response") or "").strip()


def _structured_search_profile(
    role: str,
    skills: List[str],
    experience: Optional[str],
) -> Dict[str, Any]:
    """Build profile dict for build_context from API structured fields."""
    profile: Dict[str, Any] = {}
    r = (role or "").strip()
    if r:
        profile["role"] = r
    if experience is not None and str(experience).strip():
        profile["experience"] = experience
    cleaned = [s.strip() for s in (skills or []) if s and str(s).strip()]
    if cleaned:
        profile["skills"] = cleaned
    return profile


def _ensure_query_text(text: str) -> str:
    """Non-empty string for embedding."""
    t = (text or "").strip()
    return t if t else "Candidate professional profile."


def extract_profile_from_query(text: str) -> Dict[str, Any]:
    """
    Parse a free-text hiring query into a profile dict for build_context.

    Heuristic only; on failure returns {} so caller can use fallback embedding text.
    """
    if not text or not str(text).strip():
        return {}

    s = str(text).strip()
    profile: Dict[str, Any] = {}

    # Optional skills: "skills: a, b" or "skilled in a, b"
    m_sk = re.search(r"(?:skilled in|skills?\s*:)\s*(.+?)(?:\.|$)", s, re.IGNORECASE)
    if m_sk:
        skills_part = m_sk.group(1).strip()
        skills_list = [x.strip() for x in re.split(r"[,;/]", skills_part) if x.strip()]
        if skills_list:
            profile["skills"] = skills_list[:30]
        s = (s[: m_sk.start()] + s[m_sk.end() :]).strip()
        s = re.sub(r"\s+", " ", s)

    # Experience: 3 years, 3+ years, 5 yrs
    m_exp = re.search(r"(\d+)\s*\+?\s*(?:years?|yrs?\.?)", s, re.IGNORECASE)
    if m_exp:
        profile["experience"] = m_exp.group(1)
        before = s[: m_exp.start()].strip()
        before = re.sub(r"\s+with\s*$", "", before, flags=re.IGNORECASE).strip()
        after_raw = s[m_exp.end() :].strip()
        after = re.sub(r"^(?:of\s+)?(?:experience\s+)?", "", after_raw, flags=re.IGNORECASE).strip()
        after = re.sub(r"^(?:in\s+)?", "", after, flags=re.IGNORECASE).strip()

        role_guess = ""
        if before:
            role_guess = before
        elif after:
            m_as = re.match(r"^(?:as|as a|for)\s+(.+)", after, re.IGNORECASE)
            if m_as:
                role_guess = m_as.group(1).strip()
            else:
                role_guess = after.split(".")[0].strip()
        if role_guess:
            profile["role"] = role_guess.rstrip(" -,/").strip(". ")[:300]
    else:
        if s:
            profile["role"] = s[:300]

    if profile.get("role") == "":
        profile.pop("role", None)

    if not any(profile.get(k) for k in ("role", "experience", "skills")):
        return {}

    return profile


def _extract_matches(raw_result: Any) -> List[Dict[str, Any]]:
    """
    Normalize Pinecone query response into a plain list of match dicts.

    Pinecone SDK may return either:
    - dict-like payload with "matches"
    - QueryResponse object with `.matches`
    """
    if raw_result is None:
        return []

    if isinstance(raw_result, dict):
        return raw_result.get("matches", []) or []

    matches = getattr(raw_result, "matches", None)
    if not matches:
        return []

    normalized: List[Dict[str, Any]] = []
    for m in matches:
        if isinstance(m, dict):
            normalized.append(m)
            continue

        normalized.append(
            {
                "id": getattr(m, "id", None),
                "score": getattr(m, "score", 0),
                "metadata": getattr(m, "metadata", {}) or {},
            }
        )
    return normalized


def search_candidates(
    role: str,
    skills: List[str],
    experience: Optional[str] = "",
    top_k: int = 5,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run context-aware semantic search on `all-ats-context`.

    Uses same natural-language format as indexing (build_context).
    """
    profile = _structured_search_profile(role, skills, experience)
    query_text = _ensure_query_text(build_context(profile))
    query_embedding = generate_embedding(query_text)

    indexer = ContextIndexer()
    indexer.ensure_index()
    return indexer.query(
        query_embedding=query_embedding,
        top_k=top_k,
        metadata_filter=metadata_filter,
    )


def extract_years(exp: Any) -> int:
    """Extract the first integer value from experience string."""
    if not exp:
        return 0
    match = re.search(r"\d+", str(exp))
    return int(match.group()) if match else 0


def split_skills(skills: Any) -> List[str]:
    """Convert comma-separated skills string to normalized list."""
    if not skills:
        return []
    return [s.strip().lower() for s in str(skills).split(",") if s.strip()]


def get_fit_tier(score: float) -> str:
    """Map raw similarity score to ATS fit tier labels."""
    if score >= 0.8:
        return "High Match"
    if score >= 0.6:
        return "Medium Match"
    return "Low Match"


def _tokenize_for_match(text: str) -> List[str]:
    """Tokenize free text for lightweight overlap matching."""
    if not text:
        return []
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def _calc_role_match_score(query_role: str, candidate_role_text: str) -> float:
    """Return role text overlap score in [0, 1]."""
    q_tokens = set(_tokenize_for_match(query_role))
    c_tokens = set(_tokenize_for_match(candidate_role_text))
    if not q_tokens or not c_tokens:
        return 0.0
    overlap = len(q_tokens.intersection(c_tokens))
    return overlap / float(len(q_tokens))


def _calc_skill_match_score(query_skills: List[str], candidate_skills: List[str]) -> float:
    """Return skills overlap score in [0, 1]."""
    q = {s.strip().lower() for s in (query_skills or []) if str(s).strip()}
    c = {s.strip().lower() for s in (candidate_skills or []) if str(s).strip()}
    if not q or not c:
        return 0.0
    overlap = len(q.intersection(c))
    return overlap / float(len(q))


def _calc_experience_match_score(query_exp: Any, candidate_exp_years: int) -> float:
    """Return experience alignment score in [0, 1]."""
    target = extract_years(query_exp)
    if target <= 0:
        return 0.0
    # Full score for meeting/exceeding target; soft penalty when below target.
    if candidate_exp_years >= target:
        return 1.0
    gap = target - max(candidate_exp_years, 0)
    return max(0.0, 1.0 - (gap / float(max(target, 1))))


def _weighted_context_score(
    similarity: float,
    query_role: str,
    query_skills: List[str],
    query_experience: Any,
    candidate_role_text: str,
    candidate_skills: List[str],
    candidate_experience_years: int,
) -> Dict[str, float]:
    """
    Blend Pinecone similarity with context-aware signals.

    Returns component scores and weighted score in [0, 100].
    """
    similarity_clamped = max(0.0, min(1.0, float(similarity or 0.0)))
    role_score = _calc_role_match_score(query_role, candidate_role_text)
    skill_score = _calc_skill_match_score(query_skills, candidate_skills)
    exp_score = _calc_experience_match_score(query_experience, candidate_experience_years)

    weights: Dict[str, float] = {"similarity": 0.7}
    if query_role and query_role.strip():
        weights["role"] = 0.1
    if query_skills:
        weights["skills"] = 0.15
    if extract_years(query_experience) > 0:
        weights["experience"] = 0.05

    total_weight = sum(weights.values()) or 1.0
    weighted_0_1 = (
        similarity_clamped * weights.get("similarity", 0.0)
        + role_score * weights.get("role", 0.0)
        + skill_score * weights.get("skills", 0.0)
        + exp_score * weights.get("experience", 0.0)
    ) / total_weight

    return {
        "similarity_score": round(similarity_clamped * 100.0, 2),
        "role_match_score": round(role_score * 100.0, 2),
        "skills_match_score": round(skill_score * 100.0, 2),
        "experience_match_score": round(exp_score * 100.0, 2),
        "weighted_score": round(max(0.0, min(100.0, weighted_0_1 * 100.0)), 2),
    }


def format_context_results(
    matches: List[Dict[str, Any]],
    search_profile: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Convert context index matches into ATS-compatible response rows."""
    profile = search_profile or {}
    query_role = str(profile.get("role") or "").strip()
    query_skills = [str(s).strip() for s in (profile.get("skills") or []) if str(s).strip()]
    query_experience = profile.get("experience")

    formatted_results: List[Dict[str, Any]] = []
    for match in matches:
        meta = match.get("metadata", {}) or {}
        raw_candidate_id = meta.get("candidate_id")
        score = float(match.get("score", 0) or 0)
        candidate_skills = split_skills(meta.get("skills"))
        candidate_exp_years = extract_years(meta.get("experience"))
        candidate_role_text = " ".join(
            str(v).strip()
            for v in [meta.get("role"), meta.get("jobrole"), meta.get("designation")]
            if str(v).strip()
        )
        scoring = _weighted_context_score(
            similarity=score,
            query_role=query_role,
            query_skills=query_skills,
            query_experience=query_experience,
            candidate_role_text=candidate_role_text,
            candidate_skills=candidate_skills,
            candidate_experience_years=candidate_exp_years,
        )
        resume_id = None
        candidate_label = "C0"

        if raw_candidate_id is not None and str(raw_candidate_id).strip():
            raw_text = str(raw_candidate_id).strip()
            try:
                raw_num = int(float(raw_text))
                resume_id = raw_num
                candidate_label = f"C{raw_num}"
            except ValueError:
                resume_id = raw_text
                candidate_label = f"C{raw_text}"

        formatted_results.append(
            {
                "candidate_id": candidate_label,
                "resume_id": resume_id,
                "name": meta.get("name"),
                "category": meta.get("category"),
                "mastercategory": meta.get("mastercategory"),
                "designation": meta.get("role"),
                "jobrole": meta.get("jobrole"),
                "experience_years": candidate_exp_years,
                "skills": candidate_skills,
                "location": meta.get("location", ""),
                "email": meta.get("email"),
                "mobile": meta.get("mobile"),
                "filename": meta.get("filename"),
                "score": scoring["weighted_score"],
                "weighted_score": scoring["weighted_score"],
                "similarity_score": scoring["similarity_score"],
                "skills_match_score": scoring["skills_match_score"],
                "experience_match_score": scoring["experience_match_score"],
                "role_match_score": scoring["role_match_score"],
                "fit_tier": get_fit_tier(scoring["weighted_score"] / 100.0),
            }
        )

    formatted_results.sort(
        key=lambda row: (
            float(row.get("weighted_score", 0) or 0),
            float(row.get("similarity_score", 0) or 0),
        ),
        reverse=True,
    )
    return formatted_results


def build_final_response(
    query: str,
    matches: List[Dict[str, Any]],
    results_from: str = "primary",
    search_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build ATS-style final payload for context-search API."""
    formatted = format_context_results(matches, search_profile=search_profile)
    return {
        "query": query,
        "total_results": len(formatted),
        "results": formatted,
        "search_type": "semantic",
        "results_from": results_from,
    }


def search_context_ats_response(
    role: str,
    skills: List[str],
    experience: Optional[str] = "",
    top_k: int = 20,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    New context-search response builder for ATS-compatible payloads.

    Structured fields are converted with build_context (same format as indexing).
    """
    profile = _structured_search_profile(role, skills, experience)
    query_text = _ensure_query_text(build_context(profile))
    query_embedding = generate_embedding(query_text)

    indexer = ContextIndexer()
    indexer.ensure_index()
    raw_result = indexer.query(
        query_embedding=query_embedding,
        top_k=top_k,
        metadata_filter=metadata_filter,
    )
    matches = _extract_matches(raw_result)
    return build_final_response(
        query_text,
        matches,
        results_from="primary",
        search_profile=profile,
    )


def _embed_and_query_context_index(
    query_text: str,
    top_k: int,
    metadata_filter: Optional[Dict[str, Any]],
    search_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Sync embedding + Pinecone query (runs in thread pool from async caller)."""
    query_embedding = generate_embedding(query_text)
    indexer = ContextIndexer()
    indexer.ensure_index()
    raw_result = indexer.query(
        query_embedding=query_embedding,
        top_k=top_k,
        metadata_filter=metadata_filter,
    )
    matches = _extract_matches(raw_result)
    return build_final_response(
        query_text,
        matches,
        results_from="primary",
        search_profile=search_profile,
    )


async def search_context_ats_response_query(
    query: str,
    top_k: int = 20,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Natural-language query: LLM parse → profile → build_context → Ollama embedding (never raw query).

    When LLM_MODEL=OpenAI, parsing uses llm_service (OpenAI). Otherwise Ollama chat for parsing;
    vectors always use context_embedding (Ollama /api/embeddings).
    """
    profile = await parse_natural_language_context_query(query or "")
    if not profile:
        query_text = _ensure_query_text("")
    else:
        query_text = _ensure_query_text(build_context(profile))

    return await asyncio.to_thread(
        _embed_and_query_context_index,
        query_text,
        top_k,
        metadata_filter,
        profile,
    )


async def _fetch_sample_candidates(limit: int = 5) -> List[Dict[str, Any]]:
    """Fetch sample candidates for test pipeline run."""
    async with async_session_maker() as session:
        stmt = (
            select(ResumeMetadata)
            .where(ResumeMetadata.resume_text.is_not(None))
            .limit(limit)
        )
        rows = (await session.execute(stmt)).scalars().all()

    candidates: List[Dict[str, Any]] = []
    for r in rows:
        candidates.append(
            {
                "id": r.id,
                "candidatename": r.candidatename,
                "designation": r.designation,
                "category": r.category,
                "mastercategory": r.mastercategory,
                "jobrole": r.jobrole,
                "domain": r.domain,
                "experience": r.experience,
                "skillset": r.skillset,
                "education": r.education,
                "location": r.location,
                "email": r.email,
                "mobile": r.mobile,
                "filename": r.filename,
                "resume_text": r.resume_text,
                "created_at": r.created_at,
                "updated_at": r.updated_at,
            }
        )
    return candidates


async def _run_test_context_pipeline() -> None:
    """Internal async runner for the end-to-end context pipeline test."""
    print("=== ATS Context Pipeline Test ===")

    sample_candidates = await _fetch_sample_candidates(limit=5)
    if not sample_candidates:
        print("No sample candidates found in database.")
        return

    print(f"Fetched sample candidates: {len(sample_candidates)}")

    indexer = ContextIndexer()
    indexer.ensure_index()
    indexed_count = indexer.upsert_candidates(sample_candidates)
    print(f"Indexed candidates into {CONTEXT_INDEX_NAME}: {indexed_count}")

    search_result = search_candidates(
        role="Python Developer",
        skills=["Django", "FastAPI"],
        experience="3+ years",
        top_k=5,
    )

    matches = _extract_matches(search_result)
    print(f"Search matches returned: {len(matches)}")
    for i, match in enumerate(matches, start=1):
        metadata = match.get("metadata", {}) or {}
        print(
            f"{i}. id={match.get('id')} score={match.get('score')} "
            f"name={metadata.get('name')} role={metadata.get('role')} skills={metadata.get('skills')}"
        )

    ats_response = search_context_ats_response(
        role="Python Developer",
        skills=["Django", "FastAPI"],
        experience="3+ years",
        top_k=20,
    )
    print(
        f"ATS formatted context results: {ats_response.get('total_results', 0)} "
        f"(results_from={ats_response.get('results_from')})"
    )


def test_context_pipeline() -> None:
    """
    End-to-end smoke test for ats_context_pipeline.

    Steps:
    1) Fetch sample candidates
    2) Build context text
    3) Generate Ollama embedding (context_embedding)
    4) Store in Pinecone
    5) Query and print results
    """
    asyncio.run(_run_test_context_pipeline())

