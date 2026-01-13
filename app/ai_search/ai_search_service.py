"""AI Search service implementing semantic search, filtering, and ranking."""
import re
from typing import Dict, List, Optional, Any
from app.services.embedding_service import EmbeddingService
from app.services.pinecone_automation import PineconeAutomation
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
        pinecone_automation: PineconeAutomation,
        resume_repo: ResumeRepository
    ):
        self.embedding_service = embedding_service
        self.pinecone_automation = pinecone_automation
        self.resume_repo = resume_repo
    
    def _normalize_namespace(self, category: str) -> str:
        """
        Normalize category string to valid Pinecone namespace format.
        
        Namespace normalization rules (same as PineconeAutomation):
        - Convert to lowercase
        - Replace spaces, slashes, dots, parentheses with underscores
        - Remove all characters except [a-z0-9_]
        - Collapse multiple underscores into one
        
        Example:
        "Full Stack Development (Java)" → "full_stack_development_java"
        
        Args:
            category: Category string
            
        Returns:
            Normalized namespace string
        """
        if not category or not category.strip():
            return ""
        
        # Convert to lowercase
        normalized = category.lower().strip()
        
        # Replace spaces, slashes, dots, parentheses, and all other special chars with underscores
        normalized = re.sub(r'[^a-z0-9_]+', '_', normalized)
        
        # Collapse multiple consecutive underscores into one
        normalized = re.sub(r'_+', '_', normalized)
        
        # Remove leading/trailing underscores
        normalized = normalized.strip('_')
        
        return normalized
    
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
        # Note: Pinecone doesn't support $all operator, so we use $and with multiple $in conditions
        # Each skill must be present in the skills array
        must_have_all = filters.get("must_have_all", [])
        if must_have_all:
            # Normalize to lowercase
            normalized_skills = [s.lower().strip() for s in must_have_all if s]
            if normalized_skills:
                if len(normalized_skills) == 1:
                    # Single skill - use $in
                    pinecone_filter["skills"] = {"$in": normalized_skills}
                else:
                    # Multiple skills - use $and to require all
                    skill_conditions = [{"skills": {"$in": [skill]}} for skill in normalized_skills]
                    pinecone_filter["$and"] = skill_conditions
        
        # Handle must_have_one_of_groups (OR groups)
        must_have_one_of_groups = filters.get("must_have_one_of_groups", [])
        if must_have_one_of_groups:
            # Build OR conditions for skills
            or_skill_conditions = []
            for group in must_have_one_of_groups:
                if group:
                    normalized_group = [s.lower().strip() for s in group if s]
                    if normalized_group:
                        or_skill_conditions.append({"skills": {"$in": normalized_group}})
            
            if or_skill_conditions:
                # If we already have skills filter ($and or single skill), combine with $and
                if "$and" in pinecone_filter:
                    # Add OR skills condition to existing $and
                    if len(or_skill_conditions) == 1:
                        pinecone_filter["$and"].append(or_skill_conditions[0])
                    else:
                        pinecone_filter["$and"].append({"$or": or_skill_conditions})
                elif "skills" in pinecone_filter:
                    # Single skill condition exists, combine with $and
                    existing_skill_filter = {"skills": pinecone_filter.pop("skills")}
                    if len(or_skill_conditions) == 1:
                        pinecone_filter["$and"] = [existing_skill_filter, or_skill_conditions[0]]
                    else:
                        pinecone_filter["$and"] = [existing_skill_filter, {"$or": or_skill_conditions}]
                else:
                    # No existing skills filter
                    if len(or_skill_conditions) == 1:
                        pinecone_filter["skills"] = or_skill_conditions[0]["skills"]
                    else:
                        # Multiple OR groups - use $or at top level
                        pinecone_filter = {"$or": or_skill_conditions}
        
        # Handle designation/jobrole filtering (exact match)
        designation = filters.get("designation")
        if designation:
            # Normalize to lowercase for case-insensitive matching
            normalized_designation = designation.lower().strip()
            if normalized_designation:
                # Match either designation or jobrole field (case-insensitive)
                designation_filter = {
                    "$or": [
                        {"designation": {"$eq": normalized_designation}},
                        {"jobrole": {"$eq": normalized_designation}}
                    ]
                }
                
                # Combine with existing filters using $and
                if pinecone_filter:
                    if "$and" in pinecone_filter:
                        pinecone_filter["$and"].append(designation_filter)
                    else:
                        # Wrap existing filter and designation filter in $and
                        existing_filter = pinecone_filter.copy()
                        pinecone_filter = {
                            "$and": [existing_filter, designation_filter]
                        }
                else:
                    # No existing filters, just use designation filter
                    pinecone_filter = designation_filter
        
        # Handle min_experience (mandatory - strict)
        min_experience = filters.get("min_experience")
        if min_experience is not None:
            try:
                min_exp = int(min_experience)
                if min_exp > 0:
                    exp_filter = {"experience_years": {"$gte": min_exp}}
                    # Combine with existing filters
                    if pinecone_filter:
                        if "$and" in pinecone_filter:
                            pinecone_filter["$and"].append(exp_filter)
                        else:
                            existing_filter = pinecone_filter.copy()
                            pinecone_filter = {"$and": [existing_filter, exp_filter]}
                    else:
                        pinecone_filter = exp_filter
            except (ValueError, TypeError):
                logger.warning(f"Invalid min_experience value: {min_experience}")
        
        # Handle location (optional - preference, not requirement)
        location = filters.get("location")
        if location:
            normalized_location = self.normalize_location(location)
            location_filter = {"location": {"$eq": normalized_location}}
            # Combine with existing filters
            if pinecone_filter:
                if "$and" in pinecone_filter:
                    pinecone_filter["$and"].append(location_filter)
                else:
                    existing_filter = pinecone_filter.copy()
                    pinecone_filter = {"$and": [existing_filter, location_filter]}
            else:
                pinecone_filter = location_filter
        
        # Log the final Pinecone filter for debugging
        if pinecone_filter:
            logger.info(
                f"Pinecone filter created: {pinecone_filter}",
                extra={"pinecone_filter": pinecone_filter}
            )
        else:
            logger.info("No Pinecone filter applied (semantic search only)")
        
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
            # Handle both array and string formats for skills
            candidate_skills_raw = candidate.get("skills", [])
            if isinstance(candidate_skills_raw, str):
                # If skills is a string (comma-separated), parse it
                candidate_skills = [s.strip().lower() for s in candidate_skills_raw.split(",") if s.strip()]
            elif isinstance(candidate_skills_raw, list):
                candidate_skills = [s.lower() if isinstance(s, str) else str(s).lower() for s in candidate_skills_raw]
            else:
                candidate_skills = []
            
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
        Queries both IT and Non-IT indexes and merges results.
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
            # Initialize PineconeAutomation if not already done
            if not self.pinecone_automation.pc:
                await self.pinecone_automation.initialize_pinecone()
                await self.pinecone_automation.create_indexes()
            
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
            
            # Log search parameters for debugging
            logger.info(
                f"Starting semantic search: text_for_embedding='{text_for_embedding[:100] if text_for_embedding else ''}', "
                f"has_filter={pinecone_filter is not None}, top_k={top_k}",
                extra={
                    "text_for_embedding_preview": text_for_embedding[:100] if text_for_embedding else "",
                    "has_filter": pinecone_filter is not None,
                    "top_k": top_k,
                    "pinecone_filter": pinecone_filter
                }
            )
            
            # Query both IT and Non-IT indexes across all namespaces
            # Get more results from each namespace to allow for filtering/ranking
            per_namespace_k = max(10, top_k // 5)  # Get fewer per namespace, but from all namespaces
            
            # Check if category was identified from query
            identified_mastercategory = parsed_query.get("mastercategory")
            identified_category = parsed_query.get("category")
            target_namespace = None
            
            if identified_category:
                # Normalize category to namespace format (same as PineconeAutomation does)
                target_namespace = self._normalize_namespace(identified_category)
                logger.info(
                    f"Category identified from query: {identified_category} → namespace: {target_namespace}",
                    extra={
                        "category": identified_category,
                        "namespace": target_namespace,
                        "mastercategory": identified_mastercategory
                    }
                )
            
            # Get all namespaces for IT index
            it_namespaces = await self.pinecone_automation.get_all_namespaces("IT")
            if not it_namespaces:
                # Fallback: use default namespace if no namespaces found
                it_namespaces = [""]
                logger.warning("No namespaces found in IT index, querying default namespace")
            
            # Get all namespaces for Non-IT index
            non_it_namespaces = await self.pinecone_automation.get_all_namespaces("NON_IT")
            if not non_it_namespaces:
                # Fallback: use default namespace if no namespaces found
                non_it_namespaces = [""]
                logger.warning("No namespaces found in Non-IT index, querying default namespace")
            
            # If category was identified, prioritize the target namespace
            # Still query other namespaces but with lower priority (fewer results)
            if target_namespace and identified_mastercategory:
                target_index_name = "IT" if identified_mastercategory.upper() == "IT" else "NON_IT"
                target_namespaces_list = it_namespaces if target_index_name == "IT" else non_it_namespaces
                
                if target_namespace in target_namespaces_list:
                    # Move target namespace to front for priority querying
                    target_namespaces_list.remove(target_namespace)
                    target_namespaces_list.insert(0, target_namespace)
                    logger.info(
                        f"Prioritizing identified namespace '{target_namespace}' in {target_index_name} index",
                        extra={"namespace": target_namespace, "index": target_index_name}
                    )
                else:
                    logger.warning(
                        f"Identified namespace '{target_namespace}' not found in {target_index_name} index, querying all namespaces",
                        extra={"namespace": target_namespace, "index": target_index_name}
                    )
            
            logger.info(
                f"Querying {len(it_namespaces)} IT namespaces and {len(non_it_namespaces)} Non-IT namespaces",
                extra={
                    "it_namespace_count": len(it_namespaces),
                    "non_it_namespace_count": len(non_it_namespaces),
                    "identified_category": identified_category,
                    "target_namespace": target_namespace,
                    "identified_mastercategory": identified_mastercategory
                }
            )
            
            # Query IT index - all namespaces (prioritized if category identified)
            it_results = []
            for idx, namespace in enumerate(it_namespaces):
                try:
                    # If this is the prioritized namespace (first in list), get more results
                    is_priority = (idx == 0 and target_namespace and 
                                 identified_mastercategory and 
                                 identified_mastercategory.upper() == "IT" and
                                 namespace == target_namespace)
                    namespace_k = per_namespace_k * 2 if is_priority else per_namespace_k
                    
                    namespace_results = await self.pinecone_automation.query_vectors(
                        query_vector=embedding,
                        mastercategory="IT",
                        namespace=namespace if namespace else None,
                        top_k=namespace_k,
                        filter_dict=pinecone_filter
                    )
                    it_results.extend(namespace_results)
                    
                    if is_priority:
                        logger.info(
                            f"Priority query on IT namespace '{namespace}' returned {len(namespace_results)} results",
                            extra={"namespace": namespace, "result_count": len(namespace_results)}
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to query IT index namespace '{namespace}': {e}",
                        extra={"namespace": namespace, "error": str(e)}
                    )
            
            logger.info(f"IT index query returned {len(it_results)} results from {len(it_namespaces)} namespaces")
            
            # Query Non-IT index - all namespaces (prioritized if category identified)
            non_it_results = []
            for idx, namespace in enumerate(non_it_namespaces):
                try:
                    # If this is the prioritized namespace (first in list), get more results
                    is_priority = (idx == 0 and target_namespace and 
                                 identified_mastercategory and 
                                 identified_mastercategory.upper() == "NON_IT" and
                                 namespace == target_namespace)
                    namespace_k = per_namespace_k * 2 if is_priority else per_namespace_k
                    
                    namespace_results = await self.pinecone_automation.query_vectors(
                        query_vector=embedding,
                        mastercategory="NON_IT",
                        namespace=namespace if namespace else None,
                        top_k=namespace_k,
                        filter_dict=pinecone_filter
                    )
                    non_it_results.extend(namespace_results)
                    
                    if is_priority:
                        logger.info(
                            f"Priority query on Non-IT namespace '{namespace}' returned {len(namespace_results)} results",
                            extra={"namespace": namespace, "result_count": len(namespace_results)}
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to query Non-IT index namespace '{namespace}': {e}",
                        extra={"namespace": namespace, "error": str(e)}
                    )
            
            logger.info(f"Non-IT index query returned {len(non_it_results)} results from {len(non_it_namespaces)} namespaces")
            
            # Merge results from both indexes
            all_results = it_results + non_it_results
            
            # Process and rank results
            processed_results = []
            seen_resume_ids = set()  # Deduplicate by resume_id
            
            for match in all_results:
                metadata = match.get("metadata", {})
                score = match.get("score", 0.0)
                resume_id = metadata.get("resume_id")
                
                # Skip if we've already seen this resume (deduplicate)
                if resume_id and resume_id in seen_resume_ids:
                    continue
                if resume_id:
                    seen_resume_ids.add(resume_id)
                
                # Parse skills from skillset string if needed
                skills = metadata.get("skills", [])
                if isinstance(skills, str):
                    # If skills is a string (comma-separated), parse it
                    skills = [s.strip() for s in skills.split(",") if s.strip()]
                elif not isinstance(skills, list):
                    skills = []
                
                # Extract experience_years if not already in metadata
                experience_years = metadata.get("experience_years")
                if not experience_years and metadata.get("experience"):
                    import re
                    exp_str = str(metadata.get("experience", ""))
                    match_exp = re.search(r'(\d+(?:\.\d+)?)', exp_str)
                    if match_exp:
                        experience_years = int(float(match_exp.group(1)))
                
                # Format candidate data
                candidate = {
                    "resume_id": resume_id,
                    "candidate_id": metadata.get("candidate_id", f"C{resume_id}" if resume_id else ""),
                    "name": metadata.get("candidate_name") or metadata.get("name", ""),
                    "category": metadata.get("category", ""),
                    "mastercategory": metadata.get("mastercategory", ""),
                    "designation": metadata.get("designation", ""),  # Add designation to response
                    "jobrole": metadata.get("jobrole", ""),  # Add jobrole to response
                    "experience_years": experience_years,
                    "skills": skills,
                    "location": metadata.get("location"),
                    "score": score
                }
                
                # Categorize fit tier
                fit_tier = self.categorize_fit_tier(candidate, parsed_query, score)
                candidate["fit_tier"] = fit_tier
                
                processed_results.append(candidate)
            
            # Rank by score (relevance)
            processed_results.sort(key=lambda x: x["score"], reverse=True)
            
            # Limit to top_k
            processed_results = processed_results[:top_k]
            
            logger.info(
                f"Semantic search found {len(processed_results)} candidates (IT: {len(it_results)}, Non-IT: {len(non_it_results)})",
                extra={
                    "result_count": len(processed_results),
                    "it_results": len(it_results),
                    "non_it_results": len(non_it_results)
                }
            )
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}", extra={"error": str(e)})
            raise
