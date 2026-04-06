"""Context query search + end-to-end test helper for ats_context_pipeline."""
import asyncio
import re
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from app.database.connection import async_session_maker
from app.database.models import ResumeMetadata
from core.services.context_builder import build_context
from core.services.context_embedding import generate_embedding
from core.services.context_indexer import CONTEXT_INDEX_NAME, ContextIndexer


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


def format_context_results(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert context index matches into ATS-compatible response rows."""
    formatted_results: List[Dict[str, Any]] = []
    for match in matches:
        meta = match.get("metadata", {}) or {}
        raw_candidate_id = meta.get("candidate_id")
        score = float(match.get("score", 0) or 0)
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
                "experience_years": extract_years(meta.get("experience")),
                "skills": split_skills(meta.get("skills")),
                "location": meta.get("location", ""),
                "email": meta.get("email"),
                "mobile": meta.get("mobile"),
                "filename": meta.get("filename"),
                "score": round(score * 100, 2),
                "fit_tier": get_fit_tier(score),
            }
        )

    return formatted_results


def build_final_response(
    query: str,
    matches: List[Dict[str, Any]],
    results_from: str = "primary",
) -> Dict[str, Any]:
    """Build ATS-style final payload for context-search API."""
    formatted = format_context_results(matches)
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
    return build_final_response(query_text, matches, results_from="primary")


def search_context_ats_response_query(
    query: str,
    top_k: int = 20,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Natural-language query: parse into profile, then embed ONLY via build_context(profile).

    Raw request.query is never embedded directly.
    """
    profile = extract_profile_from_query(query or "")
    if not profile:
        query_text = _ensure_query_text("")
    else:
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
    return build_final_response(query_text, matches, results_from="primary")


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
    3) Generate OpenAI embedding (existing project config)
    4) Store in Pinecone
    5) Query and print results
    """
    asyncio.run(_run_test_context_pipeline())

