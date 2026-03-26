"""Standalone controller for ai-search-2 (query-only, no category/mastercategory identification)."""
import re
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.ai_search_2.ai_search_2_query_parser import AISearch2QueryParser
from app.ai_search_2.ai_search_2_repository import AISearch2Repository
from app.ai_search_2.ai_search_2_service import AISearch2Service
from app.config import settings
from app.repositories.resume_repo import ResumeRepository
from app.services.embedding_service import EmbeddingService
from app.services.pinecone_automation import PineconeAutomation
from app.utils.logging import get_logger

logger = get_logger(__name__)


def _normalize_text(text: Optional[str]) -> str:
    value = (text or "").strip().lower()
    value = re.sub(r"[^a-z0-9\s]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _is_strict_role_phrase_match(candidate: Dict[str, Any], role_phrase: Optional[str]) -> bool:
    """
    Strict phrase match for role-oriented queries.
    Example: query role "data analyst" should match titles containing that phrase,
    and exclude nearby-but-different roles like "data engineer".
    """
    normalized_role = _normalize_text(role_phrase)
    if not normalized_role:
        return False

    jobrole = _normalize_text(candidate.get("jobrole"))
    designation = _normalize_text(candidate.get("designation"))
    combined = f"{jobrole} {designation}".strip()
    if not combined:
        return False

    return normalized_role in combined


def _dynamic_role_match(
    candidate: Dict[str, Any],
    query_designation: Optional[str],
    designation_equivalent_list: Optional[List[str]] = None,
) -> bool:
    """
    Dynamic (query-driven) role matching without static role-family maps.
    Uses:
    1) exact role phrase in title
    2) LLM-provided equivalent titles from query parsing
    3) token overlap between query designation and title text
    """
    normalized_query = _normalize_text(query_designation)
    if not normalized_query:
        return False

    title_text = _normalize_text(
        f"{candidate.get('jobrole', '')} {candidate.get('designation', '')}"
    )
    if not title_text:
        return False

    # Query head-role guard (query-driven): prevent cross-role drift from LLM equivalents.
    # Example: query "data analyst" should not accept "data scientist".
    role_heads = ("analyst", "engineer", "developer", "scientist", "manager", "architect")
    query_heads = [h for h in role_heads if h in normalized_query.split()]
    if query_heads and not any(h in title_text.split() for h in query_heads):
        return False

    # 1) strict phrase
    if normalized_query in title_text:
        return True

    # 2) LLM-expanded equivalents from parser/matcher
    if designation_equivalent_list:
        for eq in designation_equivalent_list:
            normalized_eq = _normalize_text(eq)
            if normalized_eq and normalized_eq in title_text:
                return True

    # 3) token overlap fallback (query-driven, no static families)
    query_tokens = [t for t in normalized_query.split() if len(t) >= 3]
    if not query_tokens:
        return False

    matched_tokens = sum(1 for token in query_tokens if token in title_text)
    # For multi-word roles, require stronger overlap.
    min_required = 2 if len(query_tokens) >= 2 else 1
    return matched_tokens >= min_required


def _dynamic_family_match(
    candidate: Dict[str, Any],
    query_designation: Optional[str],
    designation_equivalent_list: Optional[List[str]] = None,
) -> bool:
    """
    AI-family expansion (query-driven):
    - Uses LLM-generated equivalent titles
    - Allows broader but still relevant matches across title/skills
    - No static role-family dictionaries
    """
    normalized_query = _normalize_text(query_designation)
    if not normalized_query:
        return False

    title_text = _normalize_text(
        f"{candidate.get('jobrole', '')} {candidate.get('designation', '')}"
    )
    skills = candidate.get("skills", []) or []
    if isinstance(skills, list):
        skills_text = _normalize_text(" ".join(str(s) for s in skills))
    else:
        skills_text = _normalize_text(str(skills))
    corpus = f"{title_text} {skills_text}".strip()
    if not corpus:
        return False

    equivalents = []
    if designation_equivalent_list:
        equivalents = [_normalize_text(eq) for eq in designation_equivalent_list if _normalize_text(eq)]

    # Strong equivalent phrase hit in title or skills
    for eq in equivalents:
        if len(eq) >= 4 and eq in corpus:
            return True

    # Soft token overlap with query + equivalents
    seed_phrases = [normalized_query] + equivalents
    tokens = set()
    for phrase in seed_phrases:
        for t in phrase.split():
            if len(t) >= 3:
                tokens.add(t)
    if not tokens:
        return False

    overlap = sum(1 for t in tokens if t in corpus)
    # Family-level threshold: require decent overlap to avoid noisy matches.
    return overlap >= 3


def _strict_query_role_only_match(candidate: Dict[str, Any], query_designation: Optional[str]) -> bool:
    """
    Strict role-only matching:
    - If query includes a role head token (analyst/engineer/developer/...), candidate title
      must include the same head token.
    - Also requires meaningful token overlap from query in candidate title.
    This keeps results role-precise (e.g., "data analyst" -> analyst-only titles).
    """
    normalized_query = _normalize_text(query_designation)
    if not normalized_query:
        return False

    title_text = _normalize_text(
        f"{candidate.get('jobrole', '')} {candidate.get('designation', '')}"
    )
    if not title_text:
        return False

    query_tokens = [t for t in normalized_query.split() if len(t) >= 3]
    if not query_tokens:
        return False

    role_heads = ("analyst", "engineer", "developer", "scientist", "manager", "architect")
    query_heads = [h for h in role_heads if h in query_tokens]
    if query_heads and not any(h in title_text.split() for h in query_heads):
        return False

    matched_tokens = sum(1 for token in query_tokens if token in title_text)
    min_required = 2 if len(query_tokens) >= 2 else 1
    return matched_tokens >= min_required


def _title_contains_any_token(candidate: Dict[str, Any], tokens: List[str]) -> bool:
    """Check if designation/jobrole contains at least one token (title-only matching)."""
    if not tokens:
        return True
    title_text = _normalize_text(
        f"{candidate.get('jobrole', '')} {candidate.get('designation', '')}"
    )
    if not title_text:
        return False
    title_tokens = set(title_text.split())
    return any(str(t).lower() in title_tokens for t in tokens if str(t).strip())


class AISearch2Controller:
    """Query-only AI search across indexes from query intent."""

    def __init__(
        self,
        session: AsyncSession,
        embedding_service: EmbeddingService,
        pinecone_automation: PineconeAutomation,
        resume_repo: ResumeRepository,
    ):
        self.session = session
        self.query_parser = AISearch2QueryParser()
        self.search_service = AISearch2Service(
            embedding_service=embedding_service,
            pinecone_automation=pinecone_automation,
            resume_repo=resume_repo,
        )
        self.repository = AISearch2Repository(session)

    async def search(
        self,
        query: str,
        user_id: Optional[int] = None,
        top_k: int = 100,
    ) -> Dict[str, Any]:
        top_k_max = getattr(settings, "ai_search_top_k_max", 200)
        top_k = min(top_k or 100, top_k_max)

        search_query = await self.repository.create_query(query_text=query, user_id=user_id)
        search_query_id = search_query.id

        try:
            try:
                parsed_query = await self.query_parser.parse_query(query, skip_category_inference=True)
            except Exception as e:
                logger.error(f"ai-search-2 parse failed: {e}", extra={"query_id": search_query_id, "error": str(e)})
                raise RuntimeError(f"Query parsing failed: {str(e)}")

            # ai-search-2 does not classify category/mastercategory; always broad from query.
            parsed_query["mastercategory"] = None
            parsed_query["category"] = None

            search_type = parsed_query.get("search_type", "semantic")
            if search_type == "name":
                candidate_name = parsed_query.get("filters", {}).get("candidate_name")
                results = await self.search_service.search_name(candidate_name, self.session) if candidate_name else []
            else:
                results = await self.search_service.search_semantic(
                    parsed_query=parsed_query,
                    top_k=top_k,
                    explicit_category_mode=False,
                )

            query_designation = parsed_query.get("filters", {}).get("designation")
            slash_role_tokens = parsed_query.get("slash_role_tokens") or []
            formatted_results: List[Dict[str, Any]] = []
            for result in results:
                score_decimal = result.get("score", 0.0)
                fit_tier = result.get("fit_tier", "Partial Match")
                score_pct = round(score_decimal * 100.0, 2)
                formatted_results.append(
                    {
                        "candidate_id": result.get("candidate_id", ""),
                        "resume_id": result.get("resume_id"),
                        "name": result.get("name", ""),
                        "category": result.get("category", ""),
                        "mastercategory": result.get("mastercategory", ""),
                        "designation": result.get("designation", ""),
                        "jobrole": result.get("jobrole", ""),
                        "experience_years": result.get("experience_years"),
                        "skills": result.get("skills", []),
                        "location": result.get("location"),
                        "score": score_pct,
                        "fit_tier": fit_tier,
                    }
                )

            # Comma-separated AND: drop any row missing a required skill (defense in depth).
            if search_type != "name" and parsed_query.get("comma_strict_and"):
                formatted_results = self.search_service._apply_comma_strict_must_have_all_filter(
                    formatted_results, parsed_query
                )

            # Query-driven strict role filtering for role-style queries.
            # Return only role-matching profiles for queries like "Data Analyst".
            if query_designation and str(query_designation).strip():
                strict_role_matches = [
                    row for row in formatted_results
                    if _strict_query_role_only_match(row, query_designation)
                ]

                if strict_role_matches:
                    formatted_results = strict_role_matches
                    logger.info(
                        "ai-search-2 strict query-role filter applied",
                        extra={
                            "query_designation": query_designation,
                            "before": len(results),
                            "after": len(strict_role_matches),
                            "mode": "query_role_only",
                        },
                    )

            # Slash-role title narrowing: e.g., "sql/etl developer" should keep only
            # developer titles containing one of ["sql","etl"] in title/jobrole.
            if slash_role_tokens:
                before = len(formatted_results)
                formatted_results = [
                    row for row in formatted_results
                    if _title_contains_any_token(row, slash_role_tokens)
                ]
                logger.info(
                    "ai-search-2 slash-role title filter applied",
                    extra={
                        "tokens": slash_role_tokens,
                        "before": before,
                        "after": len(formatted_results),
                    },
                )

            try:
                await self.repository.create_result(
                    search_query_id=search_query_id,
                    results_json={"total_results": len(formatted_results), "results": formatted_results},
                )
            except Exception as e:
                logger.warning(f"ai-search-2 result save failed: {e}", extra={"query_id": search_query_id})

            return {
                "query": parsed_query.get("effective_query", query),
                "total_results": len(formatted_results),
                "results": formatted_results,
                "search_type": search_type,
                "results_from": "name" if search_type == "name" else "primary",
            }
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"ai-search-2 failed: {e}", extra={"query": query[:100], "error": str(e)}, exc_info=True)
            raise

