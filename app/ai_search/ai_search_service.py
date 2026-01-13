"""AI Search service implementing semantic search, filtering, and ranking."""
from typing import Dict, List, Optional, Any
from app.services.embedding_service import EmbeddingService
from app.services.vector_db_service import VectorDBService
from app.repositories.resume_repo import ResumeRepository
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Location alias mapping (optional, can be enhanced later)
LOCATION_MAP = {
    "nyc": "new york",
    "blr": "bangalore",
    "bombay": "mumbai"
}

# System Prompt for AI Search (documentation/reference)
# This prompt defines the principles implemented as code logic in this service
SYSTEM_PROMPT = """
You are an AI-powered ATS search assistant for recruiters.
Interpret recruiter queries to identify role, seniority, mandatory skills, experience, and constraints.
Match candidates using semantic understanding, not keyword matching.
Enforce mandatory requirements strictly; exclude profiles that do not meet them.
Infer skills only when clearly evidenced in role or project descriptions.
Prioritize recent, hands-on experience over total years or theory.
Rank candidates by relevance and return results in clear fit tiers.
Never invent or assume skills, experience, or qualifications.
"""


class AISearchService:
    """
    AI Search service implementing system prompt principles as code logic.
    
    System Prompt Principles (implemented as code):
    1. Enforce mandatory requirements strictly
    2. Prioritize recent, hands-on experience
    3. Rank by relevance
    4. Categorize into fit tiers
    5. Never invent or assume skills
    
    See SYSTEM_PROMPT constant above for the full prompt text.
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_db: VectorDBService,
        resume_repo: ResumeRepository
    ):
        self.embedding_service = embedding_service
        self.vector_db = vector_db
        self.resume_repo = resume_repo
    
    def normalize_location(self, location: str) -> str:
        """Normalize location to lowercase and apply alias mapping."""
        location_lower = location.lower().strip()
        return LOCATION_MAP.get(location_lower, location_lower)
    
    def build_pinecone_filter(self, parsed_query: Dict) -> Optional[Dict[str, Any]]:
        """
        Build Pinecone filter from parsed query.
        Principle: "Enforce mandatory requirements strictly"
        
        Args:
            parsed_query: Parsed query with filters
        
        Returns:
            Pinecone filter dictionary or None
        """
        filters = parsed_query.get("filters", {})
        pinecone_filter = {}
        
        # Handle must_have_all (mandatory skills - strict enforcement)
        must_have_all = filters.get("must_have_all", [])
        if must_have_all:
            # Normalize to lowercase
            normalized_skills = [s.lower().strip() for s in must_have_all if s]
            if normalized_skills:
                pinecone_filter["skills"] = {"$all": normalized_skills}
        
        # Handle must_have_one_of_groups (OR groups)
        must_have_one_of_groups = filters.get("must_have_one_of_groups", [])
        if must_have_one_of_groups:
            # If we have both AND and OR, combine with $or at top level
            if "skills" in pinecone_filter:
                # Both AND and OR exist - use $or to combine
                or_conditions = []
                # Add existing AND condition
                or_conditions.append({"skills": pinecone_filter["skills"]})
                # Add OR groups
                for group in must_have_one_of_groups:
                    if group:
                        normalized_group = [s.lower().strip() for s in group if s]
                        if normalized_group:
                            or_conditions.append({"skills": {"$in": normalized_group}})
                
                if len(or_conditions) > 1:
                    pinecone_filter = {"$or": or_conditions}
                elif len(or_conditions) == 1:
                    pinecone_filter = or_conditions[0]
            else:
                # Only OR groups exist
                if len(must_have_one_of_groups) == 1:
                    # Single OR group
                    normalized_group = [s.lower().strip() for s in must_have_one_of_groups[0] if s]
                    if normalized_group:
                        pinecone_filter["skills"] = {"$in": normalized_group}
                else:
                    # Multiple OR groups - combine with $or
                    or_conditions = []
                    for group in must_have_one_of_groups:
                        if group:
                            normalized_group = [s.lower().strip() for s in group if s]
                            if normalized_group:
                                or_conditions.append({"skills": {"$in": normalized_group}})
                    if or_conditions:
                        if len(or_conditions) == 1:
                            pinecone_filter = or_conditions[0]
                        else:
                            pinecone_filter = {"$or": or_conditions}
        
        # Handle min_experience (mandatory - strict)
        min_experience = filters.get("min_experience")
        if min_experience is not None:
            try:
                min_exp = int(min_experience)
                if min_exp > 0:
                    pinecone_filter["experience_years"] = {"$gte": min_exp}
            except (ValueError, TypeError):
                logger.warning(f"Invalid min_experience value: {min_experience}")
        
        # Handle location (optional - preference, not requirement)
        location = filters.get("location")
        if location:
            normalized_location = self.normalize_location(location)
            pinecone_filter["location"] = {"$eq": normalized_location}
        
        return pinecone_filter if pinecone_filter else None
    
    def check_mandatory_requirements(self, candidate: Dict, parsed_query: Dict) -> bool:
        """
        Check if candidate meets all mandatory requirements.
        Principle: "Enforce mandatory requirements strictly"
        
        Args:
            candidate: Candidate metadata from Pinecone
            parsed_query: Parsed query with filters
        
        Returns:
            True if all mandatory requirements are met, False otherwise
        """
        filters = parsed_query.get("filters", {})
        
        # Check must_have_all (mandatory skills)
        must_have_all = filters.get("must_have_all", [])
        if must_have_all:
            candidate_skills = [s.lower() for s in candidate.get("skills", [])]
            required_skills = [s.lower().strip() for s in must_have_all if s]
            if not all(skill in candidate_skills for skill in required_skills):
                return False
        
        # Check min_experience (mandatory)
        min_experience = filters.get("min_experience")
        if min_experience is not None:
            try:
                min_exp = int(min_experience)
                candidate_exp = candidate.get("experience_years", 0)
                if candidate_exp < min_exp:
                    return False
            except (ValueError, TypeError):
                pass
        
        return True
    
    def categorize_fit_tier(
        self,
        candidate: Dict,
        parsed_query: Dict,
        semantic_score: float
    ) -> str:
        """
        Categorize candidate into fit tier.
        Principle: "Rank candidates by relevance and return results in clear fit tiers"
        
        Args:
            candidate: Candidate metadata
            parsed_query: Parsed query
            semantic_score: Semantic similarity score (0-1)
        
        Returns:
            Fit tier: "Perfect Match", "Good Match", "Partial Match", "Low Match"
        """
        # Check mandatory requirements
        mandatory_met = self.check_mandatory_requirements(candidate, parsed_query)
        
        if not mandatory_met:
            return "Low Match"
        
        # Categorize based on score
        if semantic_score >= 0.85:
            return "Perfect Match"
        elif semantic_score >= 0.70:
            return "Good Match"
        elif semantic_score >= 0.50:
            return "Partial Match"
        else:
            return "Low Match"
    
    async def search_name(self, candidate_name: str, session) -> List[Dict[str, Any]]:
        """
        Search candidates by name using SQL.
        Principle: "Never invent or assume" - exact/partial name match only
        
        Args:
            candidate_name: Name to search for
            session: Database session
        
        Returns:
            List of candidate results with same format as Pinecone results
        """
        try:
            # SQL search - case insensitive, partial match
            from sqlalchemy import select, func
            from app.database.models import ResumeMetadata
            
            query = select(ResumeMetadata).where(
                func.LOWER(ResumeMetadata.candidatename).like(
                    f"%{candidate_name.lower()}%"
                )
            )
            
            result = await session.execute(query)
            resumes = result.scalars().all()
            
            # Format results to match Pinecone format
            results = []
            for resume in resumes:
                # Parse skillset (comma-separated) to array
                skills = []
                if resume.skillset:
                    skills = [s.strip() for s in resume.skillset.split(",") if s.strip()]
                
                # Extract experience_years from experience string
                experience_years = None
                if resume.experience:
                    import re
                    match = re.search(r'(\d+(?:\.\d+)?)', resume.experience)
                    if match:
                        experience_years = int(float(match.group(1)))
                
                results.append({
                    "resume_id": resume.id,
                    "candidate_id": f"C{resume.id}",  # Generate candidate_id
                    "name": resume.candidatename or "",
                    "category": resume.category or "",
                    "mastercategory": resume.mastercategory or "",
                    "experience_years": experience_years,
                    "skills": skills,
                    "location": None,  # Not stored in resume_metadata
                    "score": 1.0,  # Name match = perfect score
                    "fit_tier": "Perfect Match"
                })
            
            logger.info(
                f"Name search found {len(results)} candidates",
                extra={"candidate_name": candidate_name, "result_count": len(results)}
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Name search failed: {e}", extra={"error": str(e)})
            raise
    
    async def search_semantic(
        self,
        parsed_query: Dict,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Search candidates using semantic search with Pinecone.
        Principles implemented:
        - Semantic understanding (not keyword matching)
        - Mandatory requirement enforcement
        - Ranking by relevance
        
        Args:
            parsed_query: Parsed query with filters
            top_k: Number of results to return
        
        Returns:
            List of candidate results with fit tiers
        """
        try:
            # Get text for embedding
            text_for_embedding = parsed_query.get("text_for_embedding", "")
            
            # If empty, use original query (fallback)
            if not text_for_embedding or not text_for_embedding.strip():
                # This shouldn't happen if parser works correctly, but handle it
                logger.warning("text_for_embedding is empty, using filters for search")
                text_for_embedding = " ".join(
                    parsed_query.get("filters", {}).get("must_have_all", [])
                )
            
            # Generate embedding (semantic understanding)
            embedding = await self.embedding_service.generate_embedding(text_for_embedding)
            
            # Build Pinecone filter (mandatory requirements)
            pinecone_filter = self.build_pinecone_filter(parsed_query)
            
            # Query Pinecone
            raw_results = await self.vector_db.query_vectors(
                query_vector=embedding,
                top_k=top_k * 2,  # Get more results for filtering/ranking
                filter_dict=pinecone_filter
            )
            
            # Process and rank results
            processed_results = []
            for match in raw_results:
                metadata = match.get("metadata", {})
                score = match.get("score", 0.0)
                
                # Format candidate data
                candidate = {
                    "resume_id": metadata.get("resume_id"),
                    "candidate_id": metadata.get("candidate_id", ""),
                    "name": metadata.get("name", ""),
                    "category": metadata.get("category", ""),
                    "mastercategory": metadata.get("mastercategory", ""),
                    "experience_years": metadata.get("experience_years"),
                    "skills": metadata.get("skills", []),
                    "location": metadata.get("location"),
                    "score": score
                }
                
                # Categorize fit tier
                fit_tier = self.categorize_fit_tier(candidate, parsed_query, score)
                candidate["fit_tier"] = fit_tier
                
                # Only include if not "Low Match" (or include all and let client filter)
                # For now, include all but mark Low Match
                processed_results.append(candidate)
            
            # Rank by score (relevance)
            processed_results.sort(key=lambda x: x["score"], reverse=True)
            
            # Limit to top_k
            processed_results = processed_results[:top_k]
            
            logger.info(
                f"Semantic search found {len(processed_results)} candidates",
                extra={"result_count": len(processed_results)}
            )
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}", extra={"error": str(e)})
            raise
