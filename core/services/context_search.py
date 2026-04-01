"""Context query search + end-to-end test helper for ats_context_pipeline."""
import asyncio
import re
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from app.database.connection import async_session_maker
from app.database.models import ResumeMetadata
from core.services.context_embedding import generate_embedding
from core.services.context_indexer import ContextIndexer


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


def build_context_query(
    role: str,
    skills: List[str],
    experience: Optional[str] = "",
) -> str:
    """Build structured context query instead of plain text query."""
    joined_skills = ", ".join([s.strip() for s in skills if s and s.strip()]) or "N/A"
    role_text = role.strip() if role and role.strip() else "N/A"
    exp_text = experience.strip() if experience and experience.strip() else "N/A"

    return (
        "Looking for Candidate\n"
        f"Role: {role_text}\n"
        f"Skills: {joined_skills}\n"
        f"Experience: {exp_text}"
    )


def search_candidates(
    role: str,
    skills: List[str],
    experience: Optional[str] = "",
    top_k: int = 5,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run context-aware semantic search on `ats-context`.

    Builds a structured context query, embeds it, and performs Pinecone similarity search.
    """
    query_text = build_context_query(role=role, skills=skills, experience=experience)
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

        formatted_results.append(
            {
                "candidate_id": f"C{raw_candidate_id}" if raw_candidate_id is not None else "C0",
                "resume_id": raw_candidate_id,
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


def build_final_response(query: str, matches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build ATS-style final payload for context-search API."""
    formatted = format_context_results(matches)
    return {
        "query": query,
        "total_results": len(formatted),
        "results": formatted,
        "search_type": "semantic",
        "results_from": "context_index",
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

    This is intentionally separate from existing ATS search endpoints.
    """
    query_text = build_context_query(role=role, skills=skills, experience=experience)
    query_embedding = generate_embedding(query_text)

    indexer = ContextIndexer()
    indexer.ensure_index()
    raw_result = indexer.query(
        query_embedding=query_embedding,
        top_k=top_k,
        metadata_filter=metadata_filter,
    )
    matches = _extract_matches(raw_result)
    return build_final_response(query_text, matches)


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
    print(f"Indexed candidates into ats-context: {indexed_count}")

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

