"""Controller for AI search v1: query-driven IT/NON_IT + category, then same semantic/name pipeline as ai_search."""
import re
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from app.ai_search_1.ai_search_1_category import (
    infer_category_from_query,
    infer_category_from_token_match,
    match_override_category,
)
from app.ai_search_1.ai_search_1_mastercategory import infer_mastercategory_from_query
from app.ai_search_1.ai_search_1_query_parser import AISearch1QueryParser
from app.ai_search_1.ai_search_1_repository import AISearch1Repository
from app.ai_search_1.ai_search_1_service import AISearch1Service
from app.config import settings
from app.models.ai_search_models import _normalize_mastercategory
from app.repositories.resume_repo import ResumeRepository
from app.services.embedding_service import EmbeddingService
from app.services.pinecone_automation import PineconeAutomation
from app.utils.logging import get_logger

logger = get_logger(__name__)


class AISearch1Controller:
    """Orchestrates parse → IT/NON_IT + category classification → AISearch1Service (local stack)."""

    def __init__(
        self,
        session: AsyncSession,
        embedding_service: EmbeddingService,
        pinecone_automation: PineconeAutomation,
        resume_repo: ResumeRepository,
    ):
        self.session = session
        self.query_parser = AISearch1QueryParser()
        self.search_service = AISearch1Service(
            embedding_service=embedding_service,
            pinecone_automation=pinecone_automation,
            resume_repo=resume_repo,
        )
        self.repository = AISearch1Repository(session)
        self._pinecone = pinecone_automation

    async def search(
        self,
        query: str,
        mastercategory: Optional[str] = None,
        category: Optional[str] = None,
        user_id: Optional[int] = None,
        top_k: int = 100,
    ) -> Dict[str, Any]:
        top_k_max = getattr(settings, "ai_search_top_k_max", 200)
        top_k = min(top_k or 100, top_k_max)

        search_query = await self.repository.create_query(query_text=query, user_id=user_id)
        search_query_id = search_query.id

        inferred_mastercategory: Optional[str] = None
        inferred_category: Optional[str] = None
        classification_mode = "broad"

        try:
            try:
                parsed_query = await self.query_parser.parse_query(query, skip_category_inference=True)
            except Exception as e:
                logger.error(f"Query parsing failed: {e}", extra={"query_id": search_query_id, "error": str(e)})
                raise RuntimeError(f"Query parsing failed: {str(e)}")

            def _extract_focus_query(original_query: str, pq: Dict[str, Any]) -> str:
                """
                Pick a clean focus phrase for mastercategory/category inference.
                Priority:
                1) Query parser filters (designation, must_have_all, must_have_one_of_groups first item)
                2) First non-empty comma-separated query token
                3) First technical keyword in raw query
                4) Original query
                """
                filters = (pq.get("filters") or {}) if isinstance(pq, dict) else {}
                if isinstance(filters, dict):
                    designation = (filters.get("designation") or "").strip()
                    if designation:
                        return designation

                    must_have_all = filters.get("must_have_all") or []
                    if isinstance(must_have_all, list):
                        for item in must_have_all:
                            s = str(item).strip()
                            if s:
                                return s

                    groups = filters.get("must_have_one_of_groups") or []
                    if isinstance(groups, list):
                        for group in groups:
                            if isinstance(group, list):
                                for item in group:
                                    s = str(item).strip()
                                    if s:
                                        return s

                if isinstance(original_query, str) and "," in original_query:
                    for part in original_query.split(","):
                        token = (part or "").strip()
                        if token:
                            return token

                text = (original_query or "").strip().lower()
                if text:
                    tech_tokens = [
                        "python", "java", "javascript", "html", "css", "react", "angular",
                        "node", "sql", "mysql", "postgres", "aws", "azure", "gcp",
                        "devops", "selenium", "qa", "testing", "machine learning", "ai",
                    ]
                    for t in tech_tokens:
                        if re.search(rf"\b{re.escape(t)}\b", text):
                            return t

                return original_query

            focus_query = _extract_focus_query(query, parsed_query)
            comma_focus_terms: List[str] = []
            if isinstance(query, str) and "," in query:
                for part in query.split(","):
                    token = (part or "").strip()
                    if token:
                        comma_focus_terms.append(token)

            # Domain-term heuristic (same as legacy AI search controller)
            filters = parsed_query.get("filters") or {}
            if isinstance(filters, dict) and not filters.get("domain"):
                domain_keywords = {
                    "financial",
                    "finance",
                    "fintech",
                    "banking",
                    "healthcare",
                    "pharma",
                    "pharmaceutical",
                    "retail",
                    "ecommerce",
                    "e-commerce",
                    "insurance",
                    "telecom",
                    "telecommunications",
                    "manufacturing",
                    "logistics",
                }
                new_domain_terms = []
                must_have_all = filters.get("must_have_all") or []
                kept_all = []
                for term in must_have_all:
                    t = str(term).strip().lower()
                    if t and t in domain_keywords:
                        new_domain_terms.append(t)
                    else:
                        kept_all.append(term)
                filters["must_have_all"] = kept_all
                must_have_one_of_groups = filters.get("must_have_one_of_groups") or []
                new_groups = []
                for group in must_have_one_of_groups:
                    if not group:
                        continue
                    kept_group = []
                    for term in group:
                        t = str(term).strip().lower()
                        if t and t in domain_keywords:
                            new_domain_terms.append(t)
                        else:
                            kept_group.append(term)
                    if kept_group:
                        new_groups.append(kept_group)
                filters["must_have_one_of_groups"] = new_groups
                if new_domain_terms:
                    filters["domain"] = new_domain_terms[0]
                    parsed_query["filters"] = filters

            search_type = parsed_query.get("search_type", "semantic")
            effective_mc: Optional[str] = None
            effective_cat: Optional[str] = None

            async def _infer_mc_and_category() -> Tuple[Optional[str], Optional[str]]:
                # Requirement: for comma-separated queries, evaluate terms in order.
                # Try first term category first; only if category is None, move to next term.
                if comma_focus_terms:
                    first_mc: Optional[str] = None
                    first_term_for_fallback: Optional[str] = None
                    for term in comma_focus_terms:
                        if not first_term_for_fallback:
                            first_term_for_fallback = term
                        parsed_for_term: Dict[str, Any] = {
                            "filters": {},
                            "text_for_embedding": term,
                        }
                        mc = await infer_mastercategory_from_query(term, parsed_for_term)
                        if not first_mc and mc:
                            first_mc = mc
                        if not mc:
                            continue
                        # Requirement: first token priority without static skill maps.
                        # Try direct lexical token->category match from allowed list first.
                        token_cat = infer_category_from_token_match(mc, term)
                        if token_cat:
                            return mc, token_cat
                    # No direct token match from ordered list:
                    # fallback to AI category inference using the first token only.
                    if first_mc and first_term_for_fallback:
                        cat = await infer_category_from_query(first_term_for_fallback, first_mc, self._pinecone)
                        if cat:
                            return first_mc, cat
                    # If still no category matched, return first inferred mastercategory (if any).
                    return first_mc, None

                mc = await infer_mastercategory_from_query(focus_query, parsed_query)
                if not mc:
                    return None, None
                cat = await infer_category_from_query(focus_query, mc, self._pinecone)
                return mc, cat

            if mastercategory and category:
                mc_norm = _normalize_mastercategory(mastercategory)
                matched = match_override_category(mc_norm or "", category, self._pinecone) if mc_norm else None
                if mc_norm and matched:
                    effective_mc, effective_cat = mc_norm, matched
                    classification_mode = "explicit_payload"
                else:
                    logger.warning(
                        f"ai_search_1: override rejected or unknown category; falling back to inference: "
                        f"mastercategory={mastercategory!r}, category={category!r}",
                        extra={"query_id": search_query_id},
                    )
                    inferred_mastercategory, inferred_category = await _infer_mc_and_category()
                    effective_mc = inferred_mastercategory
                    effective_cat = inferred_category
                    if effective_mc:
                        classification_mode = "inferred"
                    else:
                        classification_mode = "broad"
            elif search_type == "name":
                classification_mode = "name"
                effective_mc, effective_cat = None, None
            else:
                inferred_mastercategory, inferred_category = await _infer_mc_and_category()
                effective_mc = inferred_mastercategory
                effective_cat = inferred_category
                if effective_mc:
                    classification_mode = "inferred"
                else:
                    classification_mode = "broad"

            explicit_mode = bool(effective_mc and effective_cat)
            if explicit_mode:
                parsed_query["mastercategory"] = effective_mc
                parsed_query["category"] = effective_cat
            else:
                # If only mastercategory is inferred, keep broad mode constrained to that mastercategory.
                parsed_query["mastercategory"] = effective_mc if effective_mc else None
                parsed_query["category"] = None

            if search_type == "name":
                candidate_name = parsed_query.get("filters", {}).get("candidate_name")
                if not candidate_name:
                    results = []
                else:
                    results = await self.search_service.search_name(candidate_name, self.session)
            elif search_type in ("semantic", "hybrid"):
                results = await self.search_service.search_semantic(
                    parsed_query=parsed_query,
                    top_k=top_k,
                    explicit_category_mode=explicit_mode,
                )
            else:
                results = await self.search_service.search_semantic(
                    parsed_query=parsed_query,
                    top_k=top_k,
                    explicit_category_mode=explicit_mode,
                )

            role_only_mode = any(r.get("role_only_mode") for r in results) if results else False
            formatted_results: List[Dict[str, Any]] = []
            for result in results:
                score_decimal = result.get("score", 0.0)
                if not role_only_mode and score_decimal <= 0.0:
                    continue
                score_percentage = round(score_decimal * 100.0, 2)
                fit_tier = result.get("fit_tier", "Partial Match")
                if not role_only_mode and fit_tier == "Low Match":
                    continue
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
                        "score": score_percentage,
                        "fit_tier": fit_tier,
                    }
                )

            try:
                await self.repository.create_result(
                    search_query_id=search_query_id,
                    results_json={
                        "total_results": len(formatted_results),
                        "results": formatted_results,
                        "classification_mode": classification_mode,
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to save ai_search_1 results: {e}", extra={"query_id": search_query_id})

            results_from = "name" if search_type == "name" else "primary"
            response_mastercategory = effective_mc
            response_category = effective_cat

            return {
                "query": query,
                "mastercategory": response_mastercategory,
                "category": response_category,
                "total_results": len(formatted_results),
                "results": formatted_results,
                "search_type": search_type,
                "results_from": results_from,
                "inferred_mastercategory": inferred_mastercategory,
                "inferred_category": inferred_category,
                "classification_mode": classification_mode,
            }

        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"ai_search_1 failed: {e}", extra={"query": query[:100], "error": str(e)}, exc_info=True)
            raise
