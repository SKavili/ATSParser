"""Controller for AI search operations."""
from typing import Optional, Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai_search.ai_search_query_parser import AISearchQueryParser
from app.ai_search.ai_search_service import AISearchService
from app.ai_search.ai_search_repository import AISearchRepository
from app.services.embedding_service import EmbeddingService
from app.services.pinecone_automation import PineconeAutomation
from app.repositories.resume_repo import ResumeRepository
from app.utils.logging import get_logger

logger = get_logger(__name__)


class AISearchController:
    """Controller for AI search operations."""
    
    def __init__(
        self,
        session: AsyncSession,
        embedding_service: EmbeddingService,
        pinecone_automation: PineconeAutomation,
        resume_repo: ResumeRepository
    ):
        self.session = session
        self.query_parser = AISearchQueryParser()
        self.search_service = AISearchService(
            embedding_service=embedding_service,
            pinecone_automation=pinecone_automation,
            resume_repo=resume_repo
        )
        self.repository = AISearchRepository(session)
    
    async def search(
        self,
        query: str,
        mastercategory: Optional[str] = None,
        category: Optional[str] = None,
        user_id: Optional[int] = None,
        top_k: int = 20
    ) -> Dict[str, Any]:
        """
        Perform AI-powered search for candidates.
        
        Supports two modes:
        1. Explicit mode: If mastercategory and category are provided, searches only that namespace
        2. Broad mode: If not provided, uses smart filtering to search relevant namespaces
        
        Args:
            query: Natural language search query
            mastercategory: Mastercategory (IT/NON_IT) - optional
            category: Category namespace - optional
            user_id: Optional user ID for tracking
            top_k: Number of results to return (default: 20)
        
        Returns:
            Dict with search results and metadata
        
        Raises:
            RuntimeError: If query parsing fails
            Exception: If search execution fails
        """
        try:
            # Step 1: Save query to database
            search_query = await self.repository.create_query(
                query_text=query,
                user_id=user_id
            )
            search_query_id = search_query.id
            
            # Determine search mode
            explicit_mode = bool(mastercategory and category)
            
            logger.info(
                f"Starting AI search: query_id={search_query_id}, mode={'explicit' if explicit_mode else 'broad'}",
                extra={
                    "query_id": search_query_id,
                    "query": query[:100],
                    "mastercategory": mastercategory,
                    "category": category,
                    "mode": "explicit" if explicit_mode else "broad"
                }
            )
            
            # Step 2: Parse query using OLLAMA (skip category inference)
            try:
                parsed_query = await self.query_parser.parse_query(
                    query,
                    skip_category_inference=True
                )
                
                if explicit_mode:
                    # Explicit mode: use provided category
                    parsed_query["mastercategory"] = mastercategory
                    parsed_query["category"] = category
                    
                    # Override: If category/mastercategory provided, force semantic search (skip name detection)
                    original_search_type = parsed_query.get("search_type", "semantic")
                    if original_search_type == "name":
                        parsed_query["search_type"] = "semantic"
                        # Clear candidate_name from filters since we're doing semantic search
                        if "filters" in parsed_query and "candidate_name" in parsed_query["filters"]:
                            parsed_query["filters"]["candidate_name"] = None
                        logger.info(
                            f"Category provided, forcing semantic search (overriding name detection). "
                            f"Original search_type: {original_search_type} → semantic",
                            extra={
                                "query_id": search_query_id,
                                "original_search_type": original_search_type,
                                "override_reason": "category_provided"
                            }
                        )
                    
                    logger.info(
                        f"Query parsed successfully: search_type={parsed_query['search_type']}, "
                        f"using explicit category: mastercategory={mastercategory}, category={category}",
                        extra={
                            "query_id": search_query_id,
                            "search_type": parsed_query["search_type"],
                            "mastercategory": mastercategory,
                            "category": category
                        }
                    )
                else:
                    # Broad mode: no category filtering
                    parsed_query["mastercategory"] = None
                    parsed_query["category"] = None
                    logger.info(
                        f"Query parsed successfully: search_type={parsed_query['search_type']}, "
                        f"using broad search mode (no category filtering)",
                        extra={
                            "query_id": search_query_id,
                            "search_type": parsed_query["search_type"]
                        }
                    )
                # Log parsed query details for debugging
                logger.info(
                    f"Parsed query details: designation={parsed_query.get('filters', {}).get('designation')}, "
                    f"must_have_all={parsed_query.get('filters', {}).get('must_have_all')}, "
                    f"text_for_embedding={parsed_query.get('text_for_embedding', '')[:100]}",
                    extra={
                        "query_id": search_query_id,
                        "parsed_query": parsed_query
                    }
                )
            except Exception as e:
                logger.error(
                    f"Query parsing failed: {e}",
                    extra={"query_id": search_query_id, "error": str(e)}
                )
                raise RuntimeError(f"Query parsing failed: {str(e)}")
            
            # Step 3: Execute search based on search_type
            search_type = parsed_query.get("search_type", "semantic")
            
            if search_type == "name":
                # Name search - SQL only
                candidate_name = parsed_query.get("filters", {}).get("candidate_name")
                if not candidate_name:
                    logger.warning("Name search type but no candidate_name in filters")
                    results = []
                else:
                    results = await self.search_service.search_name(candidate_name, self.session)
            
            elif search_type in ["semantic", "hybrid"]:
                # Semantic search - Pinecone with mode based on category presence
                # Note: "hybrid" is treated as semantic (Pinecone only)
                results = await self.search_service.search_semantic(
                    parsed_query=parsed_query,
                    top_k=top_k,
                    explicit_category_mode=explicit_mode
                )
            
            else:
                # Default to semantic
                logger.warning(f"Unknown search_type: {search_type}, defaulting to semantic")
                results = await self.search_service.search_semantic(
                    parsed_query=parsed_query,
                    top_k=top_k,
                    explicit_category_mode=explicit_mode
                )
            
            # Step 4: Format results (exclude profiles with score 0.0 or below)
            formatted_results = []
            for result in results:
                # Convert score from decimal (0.0-1.0) to percentage (0-100)
                score_decimal = result.get("score", 0.0)
                if score_decimal <= 0.0:
                    continue  # Do not return profiles with minimal/zero score
                score_percentage = round(score_decimal * 100.0, 2)
                
                formatted_results.append({
                    "candidate_id": result.get("candidate_id", ""),
                    "resume_id": result.get("resume_id"),
                    "name": result.get("name", ""),
                    "category": result.get("category", ""),
                    "mastercategory": result.get("mastercategory", ""),
                    "designation": result.get("designation", ""),  # Add designation to response
                    "jobrole": result.get("jobrole", ""),  # Add jobrole to response
                    "experience_years": result.get("experience_years"),
                    "skills": result.get("skills", []),
                    "location": result.get("location"),
                    "score": score_percentage,  # Score as percentage (0-100)
                    "fit_tier": result.get("fit_tier", "Partial Match")
                })
            
            # Step 5: Save results to database
            try:
                await self.repository.create_result(
                    search_query_id=search_query_id,
                    results_json={
                        "total_results": len(formatted_results),
                        "results": formatted_results
                    }
                )
            except Exception as e:
                logger.warning(
                    f"Failed to save results to database: {e}",
                    extra={"query_id": search_query_id, "error": str(e)}
                )
                # Continue even if save fails
            
            # Step 6: Return response
            response = {
                "query": query,
                "mastercategory": mastercategory if explicit_mode else None,
                "category": category if explicit_mode else None,
                "total_results": len(formatted_results),
                "results": formatted_results
            }
            
            logger.info(
                f"AI search completed: query_id={search_query_id}, results={len(formatted_results)}",
                extra={
                    "query_id": search_query_id,
                    "total_results": len(formatted_results)
                }
            )
            
            return response
            
        except RuntimeError:
            # Re-raise parsing errors
            raise
        except Exception as e:
            logger.error(
                f"AI search failed: {e}",
                extra={"query": query[:100], "error": str(e)},
                exc_info=True
            )
            raise
