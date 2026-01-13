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
        user_id: Optional[int] = None,
        top_k: int = 20
    ) -> Dict[str, Any]:
        """
        Perform AI-powered search for candidates.
        
        Args:
            query: Natural language search query
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
            
            logger.info(
                f"Starting AI search: query_id={search_query_id}",
                extra={"query_id": search_query_id, "query": query[:100]}
            )
            
            # Step 2: Parse query using OLLAMA
            try:
                parsed_query = await self.query_parser.parse_query(query)
                logger.info(
                    f"Query parsed successfully: search_type={parsed_query['search_type']}",
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
                # Semantic search - Pinecone
                # Note: "hybrid" is treated as semantic (Pinecone only)
                results = await self.search_service.search_semantic(
                    parsed_query=parsed_query,
                    top_k=top_k
                )
            
            else:
                # Default to semantic
                logger.warning(f"Unknown search_type: {search_type}, defaulting to semantic")
                results = await self.search_service.search_semantic(
                    parsed_query=parsed_query,
                    top_k=top_k
                )
            
            # Step 4: Format results
            formatted_results = []
            for result in results:
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
                    "score": result.get("score", 0.0),
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
                "identified_mastercategory": parsed_query.get("mastercategory"),
                "identified_category": parsed_query.get("category"),
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
