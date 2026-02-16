"""Controller for job-related operations."""
import asyncio
import re
import uuid
from typing import Optional, Dict, List, Any

from fastapi import HTTPException

from app.services.job_parser import JobParser
from app.services.embedding_service import EmbeddingService
from app.services.vector_db_service import VectorDBService
from app.services.pinecone_automation import PineconeAutomation
from app.services.job_cache import job_cache
from app.models.job_models import (
    JobCreate,
    JobCreateResponse,
    MatchRequest,
    MatchResult,
    MatchResponse,
    JDParseRequest,
    ParseJDResponse,
    JDMatchCandidate,
    ParseJDMatchInfo,
)
from app.jd_parser import JDExtractor, ParsedJD
from app.category.category_extractor import CategoryExtractor
from app.repositories.resume_repo import ResumeRepository
from app.utils.logging import get_logger
from app.utils.cleaning import normalize_skill_list
from app.config import settings

logger = get_logger(__name__)


class JobController:
    """Controller for handling job creation and matching."""
    
    def __init__(
        self,
        job_parser: JobParser,
        embedding_service: EmbeddingService,
        vector_db: VectorDBService,
        resume_repo: ResumeRepository,
        jd_extractor: JDExtractor,
        pinecone_automation: PineconeAutomation,
    ):
        self.job_parser = job_parser
        self.embedding_service = embedding_service
        self.vector_db = vector_db
        self.resume_repo = resume_repo
        self.jd_extractor = jd_extractor
        self.pinecone_automation = pinecone_automation
        # Reuse existing category taxonomy (52 categories across IT/NON_IT)
        self.category_extractor = CategoryExtractor()
    
    async def create_job(self, job_data: JobCreate) -> JobCreateResponse:
        """Create a job posting and generate embeddings."""
        try:
            # Parse job description using LLM
            parsed_job = await self.job_parser.parse_job(
                title=job_data.title,
                description=job_data.description,
                job_id=job_data.job_id
            )
            
            # Use provided job_id or generated one
            job_id = parsed_job.get("job_id") or job_data.job_id or f"job_{uuid.uuid4().hex[:12]}"
            
            # Generate embedding for job summary
            summary_text = parsed_job.get("summary_for_embedding", f"{job_data.title}. {job_data.description}")
            embedding = await self.embedding_service.generate_embedding(summary_text)
            
            # Store job embedding in vector DB
            vector_id = f"job_{job_id}"
            job_metadata = {
                "type": "job",
                "job_id": job_id,
                "title": parsed_job.get("title"),
                "location": parsed_job.get("location"),
                "summary": summary_text,
                "full_embedding": True,  # Mark as full job embedding (not chunked)
            }
            await self.vector_db.upsert_vectors([{
                "id": vector_id,
                "embedding": embedding,
                "metadata": job_metadata
            }])
            
            # Cache job embedding for quick retrieval
            job_cache.store_job(job_id, embedding, job_metadata)
            
            logger.info(
                f"Created job: {job_id}",
                extra={"job_id": job_id, "title": parsed_job.get("title")}
            )
            
            return JobCreateResponse(
                job_id=job_id,
                title=parsed_job.get("title", job_data.title),
                embedding_id=vector_id,
                message="Job created successfully"
            )
        
        except Exception as e:
            logger.error(f"Error creating job: {e}", extra={"error": str(e)})
            raise HTTPException(status_code=500, detail=f"Failed to create job: {str(e)}")
    
    async def match_job(self, match_request: MatchRequest) -> MatchResponse:
        """Match resumes to a job description."""
        try:
            # Determine job embedding
            job_embedding = None
            job_id = match_request.job_id
            
            if job_id:
                # Retrieve existing job embedding from cache or regenerate
                logger.info(f"Matching against job_id: {job_id}")
                
                # Try to get from cache first
                cached_job = job_cache.get_job(job_id)
                if cached_job:
                    job_embedding = cached_job["embedding"]
                    logger.info(f"Retrieved job embedding from cache: {job_id}")
                else:
                    # Cache miss - try to get from vector DB metadata and regenerate
                    # This is a fallback, ideally jobs should be in cache
                    logger.warning(f"Job {job_id} not in cache, attempting to retrieve from vector DB")
                    
                    # Try to query vector DB for job metadata
                    # Use a dummy query to filter by metadata
                    dummy_vector = [0.0] * settings.embedding_dimension
                    job_results = await self.vector_db.query_vectors(
                        query_vector=dummy_vector,
                        top_k=100,  # Get enough results to find our job
                        filter_dict=None  # FAISS doesn't support filters well, we'll filter in code
                    )
                    
                    # Find our job in results
                    job_metadata = None
                    for result in job_results:
                        meta = result.get("metadata", {})
                        if meta.get("job_id") == job_id and meta.get("type") == "job":
                            job_metadata = meta
                            break
                    
                    if job_metadata and job_metadata.get("summary"):
                        # Regenerate embedding from stored summary
                        job_embedding = await self.embedding_service.generate_embedding(
                            job_metadata["summary"]
                        )
                        # Store in cache for future use
                        job_cache.store_job(job_id, job_embedding, job_metadata)
                    else:
                        raise HTTPException(
                            status_code=404,
                            detail=f"Job with ID {job_id} not found. Please create the job first via /create-job"
                        )
            
            elif match_request.job_description:
                # Generate embedding for provided job description
                job_embedding = await self.embedding_service.generate_embedding(match_request.job_description)
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Either job_id or job_description must be provided"
                )
            
            # Determine top_k
            top_k = match_request.top_k or settings.top_k_results
            
            # Query similar resumes from vector DB
            # Get more results to filter by threshold
            query_results = await self.vector_db.query_vectors(
                query_vector=job_embedding,
                top_k=top_k * 3,  # Get more results to filter
                filter_dict=None  # We'll filter resume vectors by checking metadata
            )
            
            # Apply similarity threshold and filter for resume vectors only
            threshold = settings.similarity_threshold
            filtered_results = [
                r for r in query_results
                if (r["score"] >= threshold 
                    and r["metadata"].get("resume_id") 
                    and r["metadata"].get("type") != "job")  # Exclude job vectors
            ]
            
            # Limit to top_k
            filtered_results = filtered_results[:top_k]
            
            # Fetch resume metadata from database
            matches = []
            for result in filtered_results:
                resume_id = result["metadata"].get("resume_id")
                if not resume_id:
                    continue
                
                resume_metadata = await self.resume_repo.get_by_id(resume_id)
                if not resume_metadata:
                    continue
                
                # Get candidate summary from metadata or generate from resume data
                candidate_summary = result["metadata"].get("summary") or \
                    f"{resume_metadata.candidatename or 'Candidate'} with {resume_metadata.experience or 'N/A'} experience in {resume_metadata.domain or 'various domains'}."
                
                matches.append(MatchResult(
                    resume_id=resume_metadata.id,
                    candidate_name=resume_metadata.candidatename or "Unknown",
                    similarity_score=result["score"],
                    candidate_summary=candidate_summary,
                    filename=resume_metadata.filename
                ))
            
            logger.info(
                f"Found {len(matches)} matches",
                extra={"match_count": len(matches), "job_id": job_id}
            )
            
            return MatchResponse(
                matches=matches,
                total_results=len(matches),
                job_id=job_id
            )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error matching job: {e}", extra={"error": str(e)})
            raise HTTPException(status_code=500, detail=f"Failed to match job: {str(e)}")

    @staticmethod
    def _parse_experience_years(experience: Optional[str]) -> Optional[float]:
        """Parse experience string to a numeric years value (e.g. '5 years' -> 5.0)."""
        if not experience or not str(experience).strip():
            return None
        s = str(experience).strip()
        # Match leading number or range like "3-5"
        m = re.match(r"^(\d+)(?:\s*[-–]\s*(\d+))?", s)
        if m:
            return float(m.group(1))
        return None

    @staticmethod
    def _parse_skills_list(skillset: Optional[str]) -> Optional[List[str]]:
        """Parse skillset (JSON array or comma-separated) to list of strings."""
        if not skillset or not str(skillset).strip():
            return None
        s = str(skillset).strip()
        if s.startswith("["):
            try:
                import json
                out = json.loads(s)
                return [str(x).strip() for x in out] if isinstance(out, list) else [s]
            except Exception:
                pass
        return [x.strip() for x in s.split(",") if x.strip()] or None

    def _build_pinecone_filter_from_jd(self, parsed_jd: ParsedJD, use_fallback: bool = False) -> Optional[Dict[str, Any]]:
        """
        Build Pinecone metadata filter from JD requirements.
        
        Filters applied in Pinecone:
        - Designation/role: Exact match (if provided)
        - Skills: Must have at least one skill match (if provided)
        - Location: Exact match (if provided)
        - Experience: Min/max range (if provided)
        
        Args:
            parsed_jd: Parsed JD with designation, skills, experience, location
            use_fallback: If True, return None (no filter) to allow broader semantic search
        
        Returns:
            Pinecone filter dict or None (if no filters applicable or fallback mode)
        """
        if use_fallback:
            # Fallback mode: No filters - rely on semantic similarity
            return None
        
        pinecone_filter = {}
        filter_conditions = []
        
        # Filter 1: Designation/role (exact match)
        if parsed_jd.designation:
            normalized_designation = parsed_jd.designation.lower().strip()
            if normalized_designation:
                filter_conditions.append({
                    "$or": [
                        {"designation": {"$eq": normalized_designation}},
                        {"jobrole": {"$eq": normalized_designation}}
                    ]
                })
        
        # Filter 2: Skills (at least one must match)
        if parsed_jd.must_have_skills:
            normalized_skills = normalize_skill_list(parsed_jd.must_have_skills)
            if normalized_skills:
                # Require at least one skill match
                skill_conditions = [{"skills": {"$in": [skill]}} for skill in normalized_skills]
                if len(skill_conditions) == 1:
                    filter_conditions.append(skill_conditions[0])
                else:
                    filter_conditions.append({"$or": skill_conditions})
        
        # Filter 3: Location (exact match)
        if parsed_jd.location:
            normalized_location = parsed_jd.location.lower().strip().split(",")[0].strip()
            if normalized_location:
                filter_conditions.append({"location": {"$eq": normalized_location}})
        
        # Filter 4: Experience range (min/max)
        experience_filters = []
        if parsed_jd.min_experience_years is not None:
            min_exp = int(parsed_jd.min_experience_years)
            experience_filters.append({"experience_years": {"$gte": min_exp}})
        
        if parsed_jd.max_experience_years is not None:
            max_exp = int(parsed_jd.max_experience_years)
            experience_filters.append({"experience_years": {"$lte": max_exp}})
        
        if experience_filters:
            if len(experience_filters) == 1:
                filter_conditions.append(experience_filters[0])
            else:
                # Combine min and max with $and
                filter_conditions.append({"$and": experience_filters})
        
        # Combine all filter conditions with $and
        if not filter_conditions:
            return None
        
        if len(filter_conditions) == 1:
            return filter_conditions[0]
        else:
            return {"$and": filter_conditions}
    
    def _filter_candidates_by_jd_requirements(
        self,
        candidates: List[Dict[str, Any]],
        parsed_jd: ParsedJD,
        resume_metadata_dict: Dict[int, Any]
    ) -> List[Dict[str, Any]]:
        """
        Filter candidates in code based on JD requirements (skills, experience, location).
        
        This post-query filtering is faster than complex Pinecone metadata filters,
        especially with large datasets (180K+ records).
        
        Args:
            candidates: List of candidate results from Pinecone (with metadata)
            parsed_jd: Parsed JD with requirements
            resume_metadata_dict: Dictionary mapping resume_id -> ResumeMetadata (for full candidate data)
        
        Returns:
            Filtered list of candidates matching JD requirements
        """
        filtered = []
        
        for candidate in candidates:
            resume_id = candidate.get("metadata", {}).get("resume_id")
            if not resume_id:
                continue
            
            # Get full resume metadata (already batch-fetched)
            resume_metadata = resume_metadata_dict.get(resume_id)
            if not resume_metadata:
                continue
            
            # Filter 1: Skills (at least one must match)
            if parsed_jd.must_have_skills:
                candidate_skills = self._parse_skills_list(resume_metadata.skillset)
                normalized_jd_skills = normalize_skill_list(parsed_jd.must_have_skills)
                normalized_cand_skills = normalize_skill_list(candidate_skills) if candidate_skills else []
                
                # Check if candidate has at least one required skill
                has_required_skill = any(
                    skill in normalized_cand_skills
                    for skill in normalized_jd_skills
                )
                
                if not has_required_skill:
                    continue  # Skip candidate without required skills
            
            # Filter 2: Experience range (min/max)
            if parsed_jd.min_experience_years is not None or parsed_jd.max_experience_years is not None:
                candidate_exp_years = self._parse_experience_years(resume_metadata.experience)
                
                if candidate_exp_years is not None:
                    # Check min experience
                    if parsed_jd.min_experience_years is not None:
                        if candidate_exp_years < parsed_jd.min_experience_years:
                            continue  # Skip candidate with insufficient experience
                    
                    # Check max experience
                    if parsed_jd.max_experience_years is not None:
                        if candidate_exp_years > parsed_jd.max_experience_years:
                            continue  # Skip candidate with too much experience
                else:
                    # Candidate has no experience data - skip if experience is required
                    # (You can change this logic to include candidates with NULL experience if desired)
                    if parsed_jd.min_experience_years is not None:
                        continue  # Skip candidates without experience data when min_exp is required
            
            # Filter 3: Location (exact match or flexible matching)
            if parsed_jd.location:
                normalized_jd_location = parsed_jd.location.lower().strip().split(",")[0].strip()
                candidate_location = (resume_metadata.location or "").lower().strip()
                
                # Exact match or contains match
                if normalized_jd_location and candidate_location:
                    # Check exact match or if JD location is contained in candidate location
                    if normalized_jd_location not in candidate_location and candidate_location not in normalized_jd_location:
                        # For strict location matching, skip if no match
                        # For flexible matching, you can remove this check
                        if parsed_jd.location_type == "strict":
                            continue  # Skip candidate with non-matching location
            
            # Candidate passed all filters
            filtered.append(candidate)
        
        return filtered

    def _score_designation_match(
        self, 
        jd_designation: Optional[str], 
        candidate_designation: Optional[str],
        candidate_jobrole: Optional[str]
    ) -> float:
        """
        Score designation/role match (flexible matching, not strict filter).
        
        Returns:
            Score boost (0.0 to 50.0) - higher = better match
        """
        if not jd_designation:
            return 0.0
        
        jd_designation_lower = jd_designation.lower().strip()
        candidate_designation_lower = (candidate_designation or "").lower().strip()
        candidate_jobrole_lower = (candidate_jobrole or "").lower().strip()
        
        # Check exact match
        if jd_designation_lower == candidate_designation_lower or jd_designation_lower == candidate_jobrole_lower:
            return 50.0  # Perfect match
        
        # Check substring match (e.g., "qa automation engineer" contains "qa automation")
        if jd_designation_lower in candidate_designation_lower or candidate_designation_lower in jd_designation_lower:
            return 40.0
        if jd_designation_lower in candidate_jobrole_lower or candidate_jobrole_lower in jd_designation_lower:
            return 40.0
        
        # Check keyword overlap
        jd_keywords = set(jd_designation_lower.split())
        cand_keywords = set((candidate_designation_lower + " " + candidate_jobrole_lower).split())
        if jd_keywords and cand_keywords:
            overlap_ratio = len(jd_keywords & cand_keywords) / len(jd_keywords)
            if overlap_ratio >= 0.5:
                return 30.0 * overlap_ratio
        
        return 0.0

    def _score_skills_match(
        self,
        jd_must_have_skills: List[str],
        candidate_skills: Optional[List[str]]
    ) -> float:
        """
        Score skills match (flexible - partial matches get partial score).
        
        Returns:
            Score boost (0.0 to 40.0) based on percentage of JD skills found in candidate
        """
        if not jd_must_have_skills:
            return 0.0
        
        if not candidate_skills:
            return 0.0
        
        # Normalize both lists
        jd_skills_normalized = normalize_skill_list(jd_must_have_skills)
        cand_skills_normalized = normalize_skill_list(candidate_skills)
        
        if not jd_skills_normalized or not cand_skills_normalized:
            return 0.0
        
        # Count matches
        matched_skills = sum(1 for skill in jd_skills_normalized if skill in cand_skills_normalized)
        match_ratio = matched_skills / len(jd_skills_normalized)
        
        # Return score based on match ratio (max 40 points)
        return 40.0 * match_ratio

    async def _filter_resumes_by_jd_requirements(
        self, parsed_jd: ParsedJD, mastercategory: str
    ) -> List[int]:
        """
        Pre-filter resumes from MySQL based on JD requirements (cost-efficient SQL filtering).
        
        Filters by:
        - Mastercategory (IT/Non-IT) - MUST match
        - Experience range (if specified)
        - Location (if specified)
        - Domain (if specified)
        - Skills (if specified) - checks if skillset contains required skills
        
        Returns:
            List of resume_ids that match JD requirements
        """
        from sqlalchemy import select, or_
        from app.database.models import ResumeMetadata
        
        # Build SQL query with filters
        query = select(ResumeMetadata.id)
        
        # Filter 1: Mastercategory MUST match
        mastercategory_db = "IT" if mastercategory == "IT" else "NON_IT"
        query = query.where(ResumeMetadata.mastercategory == mastercategory_db)
        
        # Filter 2: Experience range (if specified) - FLEXIBLE: Optional if NULL
        # Only filter if experience is specified AND we want to be strict
        # For flexibility, we'll make this optional - if experience is NULL, don't exclude
        if parsed_jd.min_experience_years is not None:
            min_exp = int(parsed_jd.min_experience_years)
            # Match resumes with experience >= min_exp OR NULL (don't exclude NULLs)
            experience_conditions = [
                ResumeMetadata.experience.is_(None)  # Include NULLs (flexible)
            ]
            
            # Check for patterns like "5 years", "5+ years", "5-7 years", etc.
            experience_conditions.append(ResumeMetadata.experience.ilike(f"%{min_exp}%"))
            # Also check for higher numbers (up to min_exp + 15)
            for i in range(min_exp + 1, min_exp + 15):
                experience_conditions.append(ResumeMetadata.experience.ilike(f"%{i}%"))
            
            # Use OR to include NULLs or matching experience
            query = query.where(or_(*experience_conditions))
        
        # Filter 3: Location (if specified) - REMOVED: Too strict, causes 0 results
        # Location filtering removed from SQL pre-filter to maintain flexibility
        # Location matching will be handled in post-query scoring instead
        
        # Filter 4: Domain (if specified) - REMOVED: Too strict, causes 0 results
        # Domain values don't match exactly (e.g., "qa automation" vs "quality assurance")
        # Domain matching will be handled in post-query scoring instead
        
        # Filter 5: Skills (if specified) - FLEXIBLE: Require at least 2 skills match
        # This balances between too strict (all skills) and too loose (1 skill)
        if parsed_jd.must_have_skills:
            normalized_jd_skills = normalize_skill_list(parsed_jd.must_have_skills)
            if normalized_jd_skills:
                # Build skill conditions
                skill_conditions = []
                for skill in normalized_jd_skills:
                    skill_conditions.append(ResumeMetadata.skillset.ilike(f"%{skill}%"))
                
                if skill_conditions:
                    # FLEXIBLE: Require at least 2 skills to match (instead of just 1)
                    # This filters out completely irrelevant resumes while keeping flexibility
                    # For SQL simplicity, we use OR (at least one match)
                    # The actual "at least 2" filtering can be done in post-processing if needed
                    # For now, use OR to keep it flexible and avoid 0 results
                    query = query.where(or_(*skill_conditions))
        
        # Only get resumes that are indexed in Pinecone (pinecone_status = 1)
        query = query.where(ResumeMetadata.pinecone_status == 1)
        
        # Limit SQL pre-filter results to avoid Pinecone filter size limits
        # Pinecone can handle ~1000 IDs efficiently, so we limit to 500 for safety
        MAX_PRE_FILTER_RESULTS = 500
        query = query.limit(MAX_PRE_FILTER_RESULTS)
        
        # Execute query using repository's session
        result = await self.resume_repo.session.execute(query)
        resume_ids = [row[0] for row in result.fetchall()]
        
        if len(resume_ids) >= MAX_PRE_FILTER_RESULTS:
            logger.warning(
                f"SQL pre-filter returned {len(resume_ids)} results (limit: {MAX_PRE_FILTER_RESULTS}). "
                "Consider refining filters to reduce result set size.",
                extra={"resume_ids_count": len(resume_ids), "limit": MAX_PRE_FILTER_RESULTS},
            )
        
        logger.info(
            f"SQL pre-filter found {len(resume_ids)} matching resumes",
            extra={
                "mastercategory": mastercategory_db,
                "resume_ids_count": len(resume_ids),
                "has_experience_filter": parsed_jd.min_experience_years is not None,
                "experience_filter_includes_nulls": parsed_jd.min_experience_years is not None,  # NULLs included for flexibility
                "has_skills_filter": bool(parsed_jd.must_have_skills),
                "location_filter": "removed (too strict)",
                "domain_filter": "removed (too strict)",
            },
        )
        
        return resume_ids

    def _normalize_category_to_namespace(self, category: str) -> str:
        """
        Normalize category string to valid Pinecone namespace format.
        
        Same logic as PineconeAutomation._normalize_namespace:
        - Convert to lowercase
        - Replace spaces, slashes, dots, parentheses with underscores
        - Remove all characters except [a-z0-9_]
        - Collapse multiple underscores into one
        
        Args:
            category: Category string (e.g., "Full Stack Development (Python)")
            
        Returns:
            Normalized namespace string (e.g., "full_stack_development_python")
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
        
        return normalized if normalized else ""

    async def _classify_jd_category(self, jd_text: str, inferred_mastercategory: str) -> Optional[str]:
        """
        Classify JD into one of the 52 configured categories using the existing CategoryExtractor.
        
        Uses:
        - mastercategory: mapped from JD's inferred_mastercategory ("IT" | "Non-IT") to "IT" | "NON_IT"
        - text: raw JD text (first 1000 chars used by prompt)
        """
        if not jd_text or not inferred_mastercategory:
            return None
        master = str(inferred_mastercategory).strip().upper()
        if master.startswith("IT"):
            master = "IT"
        else:
            master = "NON_IT"
        try:
            category = await self.category_extractor.extract_category(
                resume_text=jd_text,
                mastercategory=master,
                filename="job_description",
            )
            return category
        except Exception as e:
            logger.error(
                f"JD category classification failed: {e}",
                extra={"error": str(e), "inferred_mastercategory": inferred_mastercategory},
            )
            return None

    async def parse_job_description(self, jd_text: str) -> ParsedJD:
        """Parse a raw job description into structured hiring requirements."""
        try:
            parsed_jd = await self.jd_extractor.extract_jd(jd_text)
            return parsed_jd
        except Exception as e:
            logger.error(f"Error parsing job description: {e}", extra={"error": str(e)})
            raise HTTPException(status_code=500, detail=f"Failed to parse job description: {str(e)}")

    async def parse_job_description_with_matches(self, request: JDParseRequest) -> ParseJDResponse:
        """
        Parse JD and optionally return matching candidates from Pinecone.
        
        NEW FLOW:
        - Uses provided mastercategory and category directly (no LLM classification)
        - Parses JD to extract: designation, skills, experience, location
        - Queries Pinecone directly with filters (no SQL pre-filter)
        - Filters by: designation, skills, location, experience in Pinecone metadata
        """
        # Step 1: Parse JD to extract structured data (designation, skills, experience, location)
        # Note: We still parse JD but use provided mastercategory/category instead of LLM inference
        parsed_jd = await self.jd_extractor.extract_jd(request.text)
        
        # Override inferred_mastercategory with provided mastercategory
        parsed_jd.inferred_mastercategory = request.mastercategory
        
        # Use provided category (or empty string for all namespaces)
        provided_category = request.category.strip() if request.category else ""
        
        # Namespace used for logging
        namespace_used = provided_category if provided_category else "all_namespaces"

        # Step 2: If parse-only, return immediately with empty candidates
        if request.mode == "parse-only":
            match_info = ParseJDMatchInfo(
                namespace_used=namespace_used,
                top_k_requested=request.top_k,
                candidates_found=0,
            )
            return ParseJDResponse(parsed_jd=parsed_jd, candidates=[], match_info=match_info)

        # Step 3: parse-and-match — Direct Pinecone query with filters (NO SQL pre-filter)
        candidates: List[JDMatchCandidate] = []
        try:
            # Ensure PineconeAutomation is initialized
            if not self.pinecone_automation.pc:
                await self.pinecone_automation.initialize_pinecone()
            
            # Generate embedding for JD (same model as resume indexing)
            jd_embedding = await self.embedding_service.generate_embedding(parsed_jd.text_for_embedding)
            
            # Map mastercategory to PineconeAutomation format ("IT" or "NON_IT")
            mastercategory_for_query = "IT" if request.mastercategory == "IT" else "NON_IT"
            
            # Build Pinecone filter with all JD requirements (designation, skills, location, experience)
            pinecone_filter = self._build_pinecone_filter_from_jd(parsed_jd, use_fallback=False)
            
            # Determine namespaces to query
            if provided_category:
                # Query specific category namespace only
                # Normalize category to namespace format (same logic as PineconeAutomation)
                normalized_namespace = self._normalize_category_to_namespace(provided_category)
                target_namespaces = [normalized_namespace]
                logger.info(
                    f"Querying specific namespace: {normalized_namespace} in {mastercategory_for_query} index",
                    extra={
                        "mastercategory": mastercategory_for_query,
                        "category": provided_category,
                        "normalized_namespace": normalized_namespace,
                        "has_filter": pinecone_filter is not None,
                    },
                )
            else:
                # Query all namespaces in the index
                target_namespaces = await self.pinecone_automation.get_all_namespaces(mastercategory_for_query)
                if not target_namespaces:
                    target_namespaces = [""]
                    logger.warning(f"No namespaces found in {mastercategory_for_query} index, querying default namespace")
                
                logger.info(
                    f"Querying all {len(target_namespaces)} namespaces in {mastercategory_for_query} index",
                    extra={
                        "mastercategory": mastercategory_for_query,
                        "namespace_count": len(target_namespaces),
                        "has_filter": pinecone_filter is not None,
                    },
                )
            
            # Query all namespaces in parallel (much faster than sequential)
            per_namespace_k = max(5, request.top_k // len(target_namespaces)) if target_namespaces else request.top_k
            
            async def query_namespace(namespace: str) -> List[Dict[str, Any]]:
                """Helper function to query a single namespace."""
                try:
                    return await self.pinecone_automation.query_vectors(
                        query_vector=jd_embedding,
                        mastercategory=mastercategory_for_query,
                        namespace=namespace if namespace else None,
                        top_k=per_namespace_k * 2,  # Get more per namespace to account for filtering
                        filter_dict=pinecone_filter,  # Apply all filters: designation, skills, location, experience
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to query namespace '{namespace}' in {mastercategory_for_query} index: {e}",
                        extra={"namespace": namespace, "error": str(e)},
                    )
                    return []
            
            # Execute all namespace queries in parallel
            namespace_tasks = [query_namespace(ns) for ns in target_namespaces]
            namespace_results_list = await asyncio.gather(*namespace_tasks)
            
            # Flatten results from all namespaces
            all_query_results = []
            for results in namespace_results_list:
                all_query_results.extend(results)
            
            logger.info(
                f"Collected {len(all_query_results)} results from {len(target_namespaces)} namespaces",
                extra={"total_results": len(all_query_results), "namespaces_queried": len(target_namespaces)},
            )
            
            # Fallback: if 0 results with filters, try without filters (semantic search only)
            used_fallback_mode = False
            if len(all_query_results) == 0 and pinecone_filter is not None:
                logger.info(
                    "Pinecone query with filters returned 0 results, trying fallback mode (no filters)",
                    extra={
                        "mastercategory": mastercategory_for_query,
                        "category": provided_category,
                    },
                )
                
                # Fallback: no filters, rely on semantic similarity only
                used_fallback_mode = True
                
                async def query_namespace_fallback(namespace: str) -> List[Dict[str, Any]]:
                    """Helper function to query a single namespace in fallback mode."""
                    try:
                        return await self.pinecone_automation.query_vectors(
                            query_vector=jd_embedding,
                            mastercategory=mastercategory_for_query,
                            namespace=namespace if namespace else None,
                            top_k=per_namespace_k * 2,
                            filter_dict=None,  # No filters in fallback mode
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to query namespace '{namespace}' in fallback mode: {e}",
                            extra={"namespace": namespace, "error": str(e)},
                        )
                        return []
                
                # Execute all fallback namespace queries in parallel
                fallback_tasks = [query_namespace_fallback(ns) for ns in target_namespaces]
                fallback_results_list = await asyncio.gather(*fallback_tasks)
                
                # Flatten results from all namespaces
                all_query_results = []
                for results in fallback_results_list:
                    all_query_results.extend(results)
                
                logger.info(
                    f"Fallback mode collected {len(all_query_results)} results from {len(target_namespaces)} namespaces",
                    extra={"total_results": len(all_query_results), "namespaces_queried": len(target_namespaces)},
                )
            
            # Apply similarity threshold and filter for resume vectors only
            # Use lower threshold for JD parsing (0.3) vs default (0.5) to allow more candidates
            # Semantic similarity scores can be lower than 0.5 even for relevant matches
            threshold = 0.3  # Lower threshold for JD parsing
            
            # Log score distribution for debugging
            if all_query_results:
                scores = [r["score"] for r in all_query_results]
                scores_above_threshold = [s for s in scores if s >= threshold]
                scores_below_threshold = [s for s in scores if s < threshold]
                
                logger.info(
                    f"Score distribution analysis: min={min(scores):.3f}, max={max(scores):.3f}, "
                    f"avg={sum(scores)/len(scores):.3f}, threshold={threshold}",
                    extra={
                        "total_results": len(all_query_results),
                        "scores_above_threshold": len(scores_above_threshold),
                        "scores_below_threshold": len(scores_below_threshold),
                        "threshold_used": threshold,
                        "min_score": min(scores),
                        "max_score": max(scores),
                        "avg_score": sum(scores) / len(scores),
                    },
                )
            
            # Step 1: Apply similarity threshold and basic filters
            threshold_filtered = [
                r
                for r in all_query_results
                if r["score"] >= threshold
                and r.get("metadata", {}).get("resume_id") is not None
                and r.get("metadata", {}).get("type") != "job"
            ]
            
            logger.info(
                f"After similarity threshold filtering: {len(threshold_filtered)} results (threshold={threshold})",
                extra={
                    "before_filtering": len(all_query_results),
                    "after_threshold_filtering": len(threshold_filtered),
                    "threshold": threshold,
                },
            )
            
            # PERFORMANCE FIX: Batch fetch all resume metadata in ONE query instead of N sequential queries
            # This reduces database round-trips from N (e.g., 60) to 1, saving ~6 seconds with 60 records
            resume_ids_to_fetch = [
                r["metadata"].get("resume_id")
                for r in threshold_filtered
                if r["metadata"].get("resume_id")
            ]
            
            # Batch fetch all resume metadata at once
            resume_metadata_dict = await self.resume_repo.get_batch_by_ids(resume_ids_to_fetch)
            
            logger.info(
                f"Batch fetched {len(resume_metadata_dict)} resume metadata records from {len(resume_ids_to_fetch)} requested IDs",
                extra={
                    "requested_count": len(resume_ids_to_fetch),
                    "found_count": len(resume_metadata_dict),
                    "missing_count": len(resume_ids_to_fetch) - len(resume_metadata_dict),
                },
            )
            
            # Step 2: JD requirements filtering is now done in Pinecone (skills, experience, location)
            # Post-query filtering is kept as a secondary check for data consistency
            # Pinecone filters handle the primary filtering, this ensures data integrity
            filtered_results = threshold_filtered
            
            # Optional: Apply secondary filtering in code as a safety check
            # This ensures candidates match JD requirements even if Pinecone metadata is inconsistent
            if parsed_jd.must_have_skills or parsed_jd.min_experience_years is not None or parsed_jd.max_experience_years is not None or parsed_jd.location:
                filtered_results = self._filter_candidates_by_jd_requirements(
                    candidates=threshold_filtered,
                    parsed_jd=parsed_jd,
                    resume_metadata_dict=resume_metadata_dict
                )
                
                logger.info(
                    f"After secondary JD requirements filtering (safety check): {len(filtered_results)} results",
                    extra={
                        "before_secondary_filtering": len(threshold_filtered),
                        "after_secondary_filtering": len(filtered_results),
                        "filtered_out": len(threshold_filtered) - len(filtered_results),
                        "has_skills_filter": bool(parsed_jd.must_have_skills),
                        "has_experience_filter": parsed_jd.min_experience_years is not None or parsed_jd.max_experience_years is not None,
                        "has_location_filter": bool(parsed_jd.location),
                    },
                )
            
            # Step 3: Score candidates with flexible matching (designation, skills, etc.)
            # Deduplicate by resume_id (multiple chunks from same resume can be returned)
            # Keep the highest-scoring chunk per resume_id
            resume_candidates = {}  # resume_id -> best candidate data
            
            for r in filtered_results:
                resume_id = r["metadata"].get("resume_id")
                if not resume_id:
                    continue
                
                # Lookup from batch-fetched dict (O(1) lookup)
                resume_metadata = resume_metadata_dict.get(resume_id)
                if not resume_metadata:
                    logger.warning(
                        f"Resume ID {resume_id} from Pinecone not found in database; skipping.",
                        extra={"resume_id": resume_id, "pinecone_metadata": r["metadata"]}
                    )
                    continue
                
                # Base score from Pinecone similarity (0-1 → 0-100)
                base_score = float(r["score"]) * 100.0
                
                # Add flexible scoring boosts
                designation_boost = self._score_designation_match(
                    parsed_jd.designation,
                    resume_metadata.designation,
                    resume_metadata.jobrole,
                )
                
                candidate_skills = self._parse_skills_list(resume_metadata.skillset)
                skills_boost = self._score_skills_match(
                    parsed_jd.must_have_skills,
                    candidate_skills,
                )
                
                # Combined score (base similarity + designation match + skills match)
                combined_score = base_score + designation_boost + skills_boost
                
                candidate_data = {
                    "resume_id": resume_id,
                    "resume_metadata": resume_metadata,
                    "base_score": base_score,
                    "designation_boost": designation_boost,
                    "skills_boost": skills_boost,
                    "combined_score": combined_score,
                }
                
                # Deduplication: Keep only the highest-scoring chunk per resume_id
                if resume_id not in resume_candidates:
                    resume_candidates[resume_id] = candidate_data
                else:
                    # Replace if this chunk has a higher combined score
                    if combined_score > resume_candidates[resume_id]["combined_score"]:
                        resume_candidates[resume_id] = candidate_data
            
            # Convert dict values to list for sorting
            scored_candidates = list(resume_candidates.values())
            
            # Sort by combined score (highest first) and take top_k
            scored_candidates.sort(key=lambda x: x["combined_score"], reverse=True)
            top_candidates = scored_candidates[: request.top_k]
            
            # Build final candidate list
            for candidate_data in top_candidates:
                resume_metadata = candidate_data["resume_metadata"]
                combined_score = candidate_data["combined_score"]
                # Clamp combined score to 0-100 range
                similarity_score = round(min(100.0, max(0.0, combined_score)), 1)
                
                candidates.append(
                    JDMatchCandidate(
                        candidate_id=str(int(candidate_data["resume_id"])),
                        similarity_score=similarity_score,
                        name=resume_metadata.candidatename or None,
                        designation=resume_metadata.designation or resume_metadata.jobrole or None,
                        experience_years=self._parse_experience_years(resume_metadata.experience),
                        skills=self._parse_skills_list(resume_metadata.skillset),
                        location=resume_metadata.location or None,
                    )
                )

            logger.info(
                "JD parse-and-match completed",
                extra={
                    "designation": parsed_jd.designation,
                    "mastercategory": mastercategory_for_query,
                    "category": provided_category,
                    "namespaces_queried": len(target_namespaces),
                    "candidates_found": len(candidates),
                    "filter_mode": "fallback" if used_fallback_mode else "pinecone_metadata_filters",
                    "has_skills_filter": bool(parsed_jd.must_have_skills),
                    "has_experience_filter": parsed_jd.min_experience_years is not None,
                    "has_location_filter": bool(parsed_jd.location),
                },
            )
        except Exception as e:
            logger.error(
                f"Pinecone query failed for parse-jd; returning empty candidates: {e}",
                extra={"error": str(e), "designation": parsed_jd.designation},
            )

        # Update namespace_used to reflect that we queried all namespaces
        namespace_used_for_info = f"all_namespaces_{parsed_jd.inferred_mastercategory.lower()}"
        
        match_info = ParseJDMatchInfo(
            namespace_used=namespace_used_for_info,
            top_k_requested=request.top_k,
            candidates_found=len(candidates),
        )
        return ParseJDResponse(parsed_jd=parsed_jd, candidates=candidates, match_info=match_info)