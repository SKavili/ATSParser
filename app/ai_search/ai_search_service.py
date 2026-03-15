"""AI Search service implementing semantic search, filtering, and ranking."""
import re
from typing import Dict, List, Optional, Any, Tuple
from app.config import settings
from app.services.embedding_service import EmbeddingService
from app.services.pinecone_automation import PineconeAutomation
from app.repositories.resume_repo import ResumeRepository
from app.utils.logging import get_logger
from app.utils.cleaning import normalize_skill_list
from app.utils.nlp_utils import (
    stemmed_skill_overlap_ratio,
    expand_query_for_embedding,
    keyword_score_for_candidate,
)
from app.ai_search.designation_matcher import DesignationMatcher

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
You are an AI-powered ATS SEARCH assistant for recruiters.

PURPOSE:
Help recruiters discover relevant candidates using semantic understanding.
Optimize for recall and relevance, not strict exclusion.

CORE BEHAVIOR:
- Interpret recruiter queries to understand intent, role, skills, experience, and preferences.
- Treat extracted information as relevance signals, not absolute rules.
- Prefer semantic similarity and ranking over hard filtering.
- Enforce strict exclusion ONLY when the recruiter explicitly states it
  (e.g., "must have", "only", "mandatory", "exclude", "do not include").

SEARCH PRINCIPLES:
- Ranking is more important than filtering.
- Partial matches are acceptable and should be ranked lower, not excluded.
- Designation and skills can coexist and both contribute to relevance.
- Experience is a guideline, not a hard cutoff, unless explicitly stated.
- Boolean logic (AND / OR) represents preference and alternatives.

SKILL & EXPERIENCE RULES:
- Extract only explicitly mentioned skills.
- Do not invent or assume skills.
- Do not exclude candidates due to small experience gaps.

NAME SEARCH:
- If the query clearly refers to a person name, treat it as a name search.

OUTPUT GOAL:
Support semantic search, soft filtering, and relevance ranking.
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
    
    # NEW APPROACH: Role-family to namespace mapping for fallback when category is None
    ROLE_FAMILY_NAMESPACES = {
        "qa": [
            "full_stack_development_java",
            "full_stack_development_python",
            "full_stack_development_selenium",
            "full_stack_development_dotnet",
            "programming_scripting",
            "web_mobile_development",
        ],
        "software_engineer": [
            "full_stack_development_java",
            "full_stack_development_python",
            "full_stack_development_dotnet",
            "full_stack_development_net",
            "web_mobile_development",
            "programming_scripting",
        ],
        "developer": [
            "full_stack_development_java",
            "full_stack_development_python",
            "full_stack_development_dotnet",
            "full_stack_development_net",
            "web_mobile_development",
            "programming_scripting",
        ],
        "data_engineer": [
            "data_science",
            "data_analysis_business_intelligence",
            "databases_data_technologies",
            "programming_scripting",
            "cloud_platforms_aws",
            "cloud_platforms_azure",
            "cloud_platforms_gcp",
        ],
        "devops": [
            "devops_platform_engineering",
            "cloud_platforms_aws",
            "cloud_platforms_azure",
            "programming_scripting",
        ],
    }

    # NEW: Canonical role normalization for exact role matching
    # This is used to implement hard role gating when the query specifies a role.
    ROLE_NORMALIZATION = {
        # QA Automation Engineer family
        "qa_automation_engineer": [
            "qa",
            "qa automation",
            "qa automation engineer",
            "automation qa engineer",
            "qa engineer automation",
            "qa engineer - automation",
            "automation test engineer",
            "test automation engineer",
            "software test automation engineer",
            "qa engineer – automation",
            "sdet",  # Software Development Engineer in Test
            "tester",  # Generic tester role
            "qa tester",
            "test engineer",
            "qa test engineer",
            "quality assurance tester",
            "qa engineer",
            "quality assurance engineer",
            "quality assurance analyst",
            "dynamic quality assurance analyst",
            "senior quality assurance analyst",
            "lead quality assurance analyst",
            "qa automation test engineer",
            "qa automation tester",
            "test engineer automation",
            "automation engineer (qa)",
        ],
        # Generic Software Engineer family
        "software_engineer": [
            "software engineer",
            "software developer",
            "application developer",
            "software development engineer",
            "sde",
            "swe",
            "software engineer i",
            "software engineer ii",
            "senior software engineer",
        ],
        # Scrum Master family (for 180k+ resumes optimization)
        "scrum_master": [
            "scrum master",
            "agile scrum master",
            "certified scrummaster",
            "certified scrum master",
            "scrummaster",
            "scrum master/agile coach",
            "scrum lead",
            "agile coach",
        ],
        # Project / Program Manager family
        "project_manager": [
            "project manager",
            "project/program manager",
            "technical project manager",
            "it project manager",
            "senior project manager",
            "delivery manager",
        ],
        "program_manager": [
            "program manager",
            "senior program manager",
            "it program manager",
        ],
        # Change Manager family
        "change_manager": [
            "change manager",
            "organizational change manager",
            "ocm consultant",
            "change management consultant",
            "change management lead",
            "change management specialist",
        ],
        # Data Professional family (separate from data_analyst; e.g. designation=Data Professional, jobrole=Data Analyst)
        "data_professional": [
            "data professional",
            "data science professional",
        ],
        # Data Analyst / Data Scientist family
        "data_analyst": [
            "data analyst",
            "data scientist",
            "business data analyst",
            "bi analyst",
            "business intelligence analyst",
            "reporting analyst",
            "mis analyst",
            "data analytics analyst",
            "data analytics specialist",
            "insights analyst",
            "marketing data analyst",
            "financial data analyst",
        ],
        # Database / SQL / SSIS / SSRS / BI developer family (SQL Server stack; distinct from data_engineer)
        "database_developer": [
            "sql developer",
            "sql/ssis developer",
            "ssis developer",
            "ssrs developer",
            "ssas developer",
            "sql server developer",
            "sql server (ssis/ssrs/bi) developer",
            "sr. sql server (ssis/ssrs/bi) developer",
            "bi developer",
            "business intelligence developer",
            "power bi developer",
            "etl developer (ssis)",
            "msbi developer",
        ],
        # Data Engineer / Big Data Engineer family (big data, pipelines; not SQL/SSIS-specific)
        "data_engineer": [
            "data engineer",
            "big data engineer",
            "sr data engineer",
            "sr. data engineer",
            "senior data engineer",
            "sr big data engineer",
            "etl developer",
            "etl engineer",
            "big data developer",
            "data engineering specialist",
        ],
        # Backend Developer family
        "backend_developer": [
            "backend developer",
            "back end developer",
            "backend engineer",
            "back end engineer",
            "api developer",
            "server side developer",
        ],
        # Frontend Developer family
        "frontend_developer": [
            "frontend developer",
            "front end developer",
            "frontend engineer",
            "front end engineer",
            "ui developer",
            "web developer",
        ],
        # Full Stack Developer family
        "fullstack_developer": [
            "full stack developer",
            "fullstack developer",
            "full stack engineer",
            "fullstack engineer",
            "python full stack developer",
            "java full stack developer",
        ],
        # DevOps Engineer family
        "devops_engineer": [
            "devops engineer",
            "devops",
            "site reliability engineer",
            "sre",
            "platform engineer",
        ],
        # Business Analyst family
        "business_analyst": [
            "business analyst",
            "it business analyst",
            "ba",
            "functional consultant",
        ],
        # Product Manager family
        "product_manager": [
            "product manager",
            "technical product manager",
            "digital product manager",
            "senior product manager",
        ],
        # Logistics / Operations Manager family (NON-IT)
        "logistics_manager": [
            "distribution logistics",
            "logistics manager",
            "distribution manager",
            "warehouse and logistics",
            "logistics coordinator",
        ],
        # UI/UX Engineer family
        "ui_ux_engineer": [
            "ui developer",
            "ui engineer",
            "ux engineer",
            "ux designer",
            "ui ux designer",
            "ui ux developer",
        ],
    }

    # Canonical role -> family for equivalent-title filtering (same as in categorize_fit_tier)
    ROLE_FAMILIES = {
        "product_manager": "product_management",
        "project_manager": "project_program_management",
        "program_manager": "project_program_management",
        "scrum_master": "agile_delivery",
        "business_analyst": "business_analysis",
        "data_analyst": "data_analytics",
        "data_professional": "data_analytics",
        "database_developer": "database_development",  # SQL/SSIS/SSRS/BI; separate from data_engineer
        "data_engineer": "data_engineering",
    }
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        pinecone_automation: PineconeAutomation,
        resume_repo: ResumeRepository
    ):
        self.embedding_service = embedding_service
        self.pinecone_automation = pinecone_automation
        self.resume_repo = resume_repo
        self.designation_matcher = DesignationMatcher()
    
    def _strip_seniority_words(self, title: Optional[str]) -> Optional[str]:
        """
        Remove seniority adjectives (senior/lead/principal/junior/etc.) from a title
        for role-family based matching and filtering.
        """
        if not title:
            return None
        words = str(title).lower().split()
        seniority_words = {
            "senior",
            "sr",
            "lead",
            "principal",
            "junior",
            "jr",
            "staff",
            "associate",
        }
        filtered = [w for w in words if w not in seniority_words]
        if not filtered:
            # If everything was stripped, fall back to original words
            return " ".join(words)
        return " ".join(filtered)
    
    def _normalize_role(self, title: Optional[str]) -> Optional[str]:
        """
        Normalize a job title to a canonical role key, if known.
        
        This is used for HARD role gating when the query specifies a role.
        Example:
            "QA Automation Engineer" and "SDET" both → "qa_automation_engineer"
        """
        if not title:
            return None
        
        # Strip seniority words first, then normalize whitespace
        base = self._strip_seniority_words(title)
        normalized = " ".join(str(base).lower().split()) if base else ""
        if not normalized:
            return None

        # First try exact/variant list matches
        for canonical, variants in self.ROLE_NORMALIZATION.items():
            for variant in variants:
                v_norm = " ".join(variant.lower().split())
                if normalized == v_norm:
                    return canonical

        # Then try substring-based match (e.g., "senior qa automation engineer" contains "qa automation engineer")
        for canonical, variants in self.ROLE_NORMALIZATION.items():
            for variant in variants:
                v_norm = " ".join(variant.lower().split())
                if v_norm and v_norm in normalized:
                    return canonical

        return None

    def _get_ai_search_weights(self) -> Dict[str, float]:
        """Return relevance scoring weights from config (with defaults)."""
        return {
            "skill_weight": getattr(settings, "ai_search_skill_weight", 40.0),
            "group_skill_weight": getattr(settings, "ai_search_group_skill_weight", 30.0),
            "designation_match_weight": getattr(settings, "ai_search_designation_match_weight", 50.0),
            "designation_mismatch_penalty": getattr(settings, "ai_search_designation_mismatch_penalty", 40.0),
            "experience_exact_weight": getattr(settings, "ai_search_experience_exact_weight", 18.0),
            "domain_match_weight": getattr(settings, "ai_search_domain_match_weight", 12.0),
        }

    def _compute_hybrid_combined_score(
        self,
        semantic_score: float,
        keyword_score: float,
        relevance_score: float,
    ) -> float:
        """
        Blend semantic (0-1), keyword (0-1), and relevance (0-100) into a final 0-100 score.

        Design goals:
        - Recruiter-facing score is always in [0, 100]
        - Weights are explicit and easy to tune
        - Negative relevance penalties cannot collapse everything to 0
        """
        # Clamp inputs to expected ranges
        semantic_clamped = max(0.0, min(1.0, semantic_score))
        keyword_clamped = max(0.0, min(1.0, keyword_score))
        relevance_clamped = max(0.0, min(100.0, relevance_score))

        semantic_pct = semantic_clamped * 100.0
        keyword_pct = keyword_clamped * 100.0

        # Explicit, interpretable weights (can be moved to config later if needed)
        semantic_weight = 0.6
        relevance_weight = 0.3
        keyword_weight = 0.1

        final_score = (
            semantic_pct * semantic_weight
            + relevance_clamped * relevance_weight
            + keyword_pct * keyword_weight
        )

        # Bound the final score to [0, 100]
        return max(0.0, min(100.0, final_score))

    def _candidate_matches_query_role(
        self,
        candidate: Dict[str, Any],
        query_designation: Optional[str],
        designation_equivalent_list: Optional[List[str]] = None,
    ) -> bool:
        """
        Return True if the candidate's role matches the query designation.
        
        Priority:
        1) If we have an LLM-expanded designation_equivalent_list, treat any exact
           match on candidate designation/jobrole against that list as a match.
        2) Otherwise (or in addition), fall back to canonical role-family matching
           via _normalize_role and ROLE_NORMALIZATION.
        
        If query_designation is missing, returns True (no role-based filter).
        If we cannot normalize the role and have no equivalents, returns True
        to avoid over-filtering.
        """
        if not query_designation or not str(query_designation).strip():
            return True
        
        # Compute canonical family for the query, if known
        normalized_query = self._normalize_role(query_designation)
        multi_role = self._is_multi_role_query(query_designation)
        role_keywords = self._role_keywords_from_query(query_designation) if multi_role else set()
        
        # 1) LLM-expanded equivalent titles (dynamic, query-specific)
        # Always include the query designation itself so "sql/etl developer" matches candidates with that exact title
        q_normalized = str(query_designation).strip().lower() if query_designation else ""
        equivalents = {q_normalized} if q_normalized else set()
        if designation_equivalent_list:
            try:
                for s in designation_equivalent_list:
                    if not isinstance(s, str):
                        continue
                    title = s.strip().lower()
                    if not title:
                        continue
                    # Multi-role query (e.g. "sql/etl developers"): only add titles that contain query role keywords
                    # so we don't match whole family (e.g. exclude "data engineer" when query is sql/etl).
                    if multi_role and role_keywords:
                        if not any(kw in title for kw in role_keywords):
                            continue
                    elif normalized_query:
                        # Single-role: only same canonical (exact/same role), not same family
                        fam = self._normalize_role(title)
                        if fam is not None and fam != normalized_query:
                            continue  # different canonical, skip
                    else:
                        # Query not in static map: only add equivalents that overlap the query (exact/same intent)
                        q_lower = str(query_designation).strip().lower()
                        if q_lower and q_lower not in title and title not in q_lower:
                            continue  # no overlap, skip (e.g. don't add "operations manager" for "gdc transportation sme")
                    equivalents.add(title)
            except Exception:
                equivalents = {q_normalized} if q_normalized else set()
        
        # Check candidate against equivalents (always includes query designation, plus LLM titles when available)
        if equivalents:
            cand_designation = (candidate.get("designation") or "").strip().lower()
            cand_jobrole = (candidate.get("jobrole") or "").strip().lower()
            for raw in (cand_jobrole, cand_designation):
                if raw and raw in equivalents:
                    return True
        
        # 2) Canonical role matching only (exact/same role; no same-family broadening)
        cand_designation = candidate.get("designation") or ""
        cand_jobrole = candidate.get("jobrole") or ""
        if normalized_query is not None:
            # Prefer jobrole over designation for canonical role matching
            for raw in (cand_jobrole, cand_designation):
                if not raw:
                    continue
                cand_normalized = self._normalize_role(raw)
                if cand_normalized == normalized_query:
                    # Multi-role: require candidate's raw role to contain a query keyword (e.g. "sql" or "etl")
                    if multi_role and role_keywords:
                        raw_lower = raw.strip().lower()
                        if not any(kw in raw_lower for kw in role_keywords):
                            continue  # same canonical but wrong role (e.g. data engineer for sql/etl query)
                    return True
            # No match: same-family not allowed (only exact/same canonical)
            return False
        
        # 3) Fallback for roles without a known family:
        #    - If the query designation is actually a tool-like phrase (e.g. "power bi"),
        #      don't enforce strict role gating – treat it as skill-only.
        #    - Multi-role: only match if candidate role contains at least one query role keyword.
        #    - Otherwise, consider substring matches in designation/jobrole before giving up.
        from app.utils import is_tool_like_phrase
        q = str(query_designation).strip().lower()
        if q:
            # Tool-like "designation" (e.g. "power bi") → do not gate by role
            if is_tool_like_phrase(q):
                return True
            # Multi-role: candidate must contain at least one role keyword (e.g. sql or etl)
            if multi_role and role_keywords:
                for raw in (cand_jobrole, cand_designation):
                    if not raw:
                        continue
                    raw_lower = raw.strip().lower()
                    if any(kw in raw_lower for kw in role_keywords):
                        return True
                return False  # strict: no keyword in candidate role => no match
            # If the query role phrase appears inside the candidate designation/jobrole,
            for raw in (cand_jobrole, cand_designation):
                if not raw:
                    continue
                raw_lower = raw.strip().lower()
                if raw_lower == q or q in raw_lower:
                    return True
            # No match: only exact/same or equivalents pass; do not pass everyone for unknown roles
            return False
        
        # Should not reach here (empty query handled above), but be safe
        return True
    
    def _exact_role_match(self, candidate: Dict[str, Any], query_designation: str) -> bool:
        """True if candidate's jobrole or designation equals query_designation (normalized: lower, strip)."""
        if not query_designation or not str(query_designation).strip():
            return False
        q = str(query_designation).strip().lower()
        jobrole = (candidate.get("jobrole") or "").strip().lower()
        designation = (candidate.get("designation") or "").strip().lower()
        return jobrole == q or designation == q
    
    def _candidate_from_resume_metadata(self, resume: Any) -> Dict[str, Any]:
        """Build a candidate dict from ResumeMetadata ORM (for DB backfill of exact-role matches)."""
        skills = []
        if getattr(resume, "skillset", None) and str(resume.skillset).strip():
            skills = [s.strip() for s in str(resume.skillset).split(",") if s.strip()]
        experience_years = None
        if getattr(resume, "experience", None) and str(resume.experience).strip():
            exp_str = str(resume.experience).strip()
            m = re.search(r"(\d+(?:\.\d+)?)", exp_str)
            if m:
                experience_years = int(float(m.group(1)))
        return {
            "resume_id": getattr(resume, "id", None),
            "candidate_id": f"C{getattr(resume, 'id', 0)}",
            "name": (getattr(resume, "candidatename", None) or "").strip() or "Unknown",
            "category": getattr(resume, "category", "") or "",
            "mastercategory": getattr(resume, "mastercategory", "") or "",
            "designation": getattr(resume, "designation", "") or "",
            "jobrole": getattr(resume, "jobrole", "") or "",
            "experience_years": experience_years,
            "skills": skills,
            "location": getattr(resume, "location", None) or "",
            "domain": getattr(resume, "domain", None),
            "score": 0.0,
            "semantic_score": 0.0,
            "keyword_score": 0.0,
            "relevance_score": 0.0,
        }
    
    def _is_multi_role_query(self, query_designation: Optional[str]) -> bool:
        """True if query lists multiple specific roles (e.g. 'sql/etl developers') so we match only those, not whole family."""
        if not query_designation or not str(query_designation).strip():
            return False
        q = str(query_designation).strip().lower()
        return "/" in q or (len(q.split()) >= 2 and any(
            t in q for t in (" sql ", " etl ", " ssis ", " ssrs ", " and ", " or ")
        ))
    
    def _role_keywords_from_query(self, query_designation: str) -> set:
        """Extract role keywords from query for strict matching (e.g. 'sql/etl developers' -> {'sql', 'etl'})."""
        if not query_designation or not str(query_designation).strip():
            return set()
        q = str(query_designation).strip().lower()
        # Split by / and space; drop common suffixes so we keep role-specific terms
        stop = {"developer", "developers", "engineer", "engineers", "and", "or", "the"}
        tokens = set()
        for part in q.replace("/", " ").split():
            t = part.strip()
            if t and t not in stop:
                tokens.add(t)
        return tokens
    
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
    
    def _detect_role_family(self, parsed_query: Dict) -> Optional[str]:
        """
        Detect role family from query for namespace fallback.
        
        Args:
            parsed_query: Parsed query with filters and designation
        
        Returns:
            Role family string ("qa", "software_engineer", "developer", etc.) or None
        """
        designation = (parsed_query.get("filters", {}).get("designation") or "").lower()
        text_for_embedding = parsed_query.get("text_for_embedding", "").lower()
        query_text = f"{designation} {text_for_embedding}".lower()
        
        # QA role family
        if any(kw in query_text for kw in ["qa", "quality assurance", "test", "testing", "automation", "selenium"]):
            if "qa" in query_text or "quality assurance" in query_text:
                return "qa"
            elif any(kw in query_text for kw in ["test", "testing", "automation", "selenium"]):
                return "qa"
        
        # Software Engineer role family
        if any(kw in query_text for kw in ["software engineer", "software developer", "sde", "swe"]):
            return "software_engineer"
        
        # Generic Developer role family
        if any(kw in query_text for kw in ["developer", "programmer", "coder"]):
            if "software" not in query_text:  # Avoid duplicate with software_engineer
                return "developer"
        
        # Data Engineer role family
        if any(kw in query_text for kw in ["data engineer", "data engineering"]):
            return "data_engineer"
        
        # DevOps role family
        if any(kw in query_text for kw in ["devops", "sre", "site reliability", "platform engineer"]):
            return "devops"
        
        return None
    
    def normalize_location(self, location: str) -> str:
        """
        Normalize location string for strict filtering.
        
        Rules:
        - Lowercase and trim whitespace
        - If there is a comma, keep only the part before the first comma
          (e.g. "Hyderabad, Telangana" -> "hyderabad")
        - Apply alias mapping to the final value (e.g. "nyc" -> "new york")
        """
        if not location:
            return ""
        
        location_lower = str(location).lower().strip()
        if not location_lower:
            return ""
        
        # Keep only the primary part before the first comma
        primary_part = location_lower.split(",", 1)[0].strip()
        
        # Apply alias mapping on the primary part
        return LOCATION_MAP.get(primary_part, primary_part)
    
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
            # Normalize skills to canonical forms (e.g., "react.js" → "react", "angularjs" → "angular")
            normalized_skills = normalize_skill_list(must_have_all)
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
                    # Normalize skills to canonical forms (e.g., "react.js" → "react", "angularjs" → "angular")
                    normalized_group = normalize_skill_list(group)
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
        
        # Handle experience filtering
        # Single number (e.g. "10 years"): band filter min±2 (8-12) for recruiter-friendly relevance
        # Range (e.g. "5-7 years"): use min and max as given
        min_experience = filters.get("min_experience")
        max_experience = filters.get("max_experience")
        if min_experience is not None:
            try:
                min_exp = int(min_experience)
                if max_experience is None:
                    # Single number: band (min_exp - 2, min_exp + 2) e.g. "10 years" -> 8-12
                    band_low = max(0, min_exp - 2)
                    band_high = min_exp + 2
                    experience_filter = {
                        "$and": [
                            {"experience_years": {"$gte": band_low}},
                            {"experience_years": {"$lte": band_high}}
                        ]
                    }
                    logger.debug(
                        f"Added experience band filter: {band_low}-{band_high} (min {min_exp} ±2)",
                        extra={"min_experience": min_exp, "band_low": band_low, "band_high": band_high}
                    )
                else:
                    # Range query: enforce >= min (max handled below)
                    experience_filter = {"experience_years": {"$gte": min_exp}}
                    logger.debug(
                        f"Added experience filter: experience_years >= {min_exp}",
                        extra={"min_experience": min_exp}
                    )
                
                # Combine with existing filters
                if pinecone_filter:
                    if "$and" in pinecone_filter:
                        pinecone_filter["$and"].append(experience_filter)
                    else:
                        existing_filter = pinecone_filter.copy()
                        pinecone_filter = {"$and": [existing_filter, experience_filter]}
                else:
                    pinecone_filter = experience_filter
            except (ValueError, TypeError):
                logger.warning(f"Invalid min_experience value: {min_experience}")
        
        # Handle max_experience (for range queries like "5-7 years")
        if max_experience is not None:
            try:
                max_exp = int(max_experience)
                # Add experience_years <= max_experience filter to Pinecone
                max_experience_filter = {"experience_years": {"$lte": max_exp}}
                
                # Combine with existing filters
                if pinecone_filter:
                    if "$and" in pinecone_filter:
                        pinecone_filter["$and"].append(max_experience_filter)
                    else:
                        existing_filter = pinecone_filter.copy()
                        pinecone_filter = {"$and": [existing_filter, max_experience_filter]}
                else:
                    pinecone_filter = max_experience_filter
                
                logger.debug(
                    f"Added max experience filter: experience_years <= {max_exp}",
                    extra={"max_experience": max_exp}
                )
            except (ValueError, TypeError):
                logger.warning(f"Invalid max_experience value: {max_experience}")
        
        # Handle location (optional - preference, not requirement)
        location = filters.get("location")
        if location:
            # Accept both list and string from parser; take primary element
            if isinstance(location, list):
                location = location[0] if location else None
            normalized_location = self.normalize_location(location) if location else ""
            if normalized_location:
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
    
    async def calculate_relevance_score(
        self, 
        candidate: Dict, 
        parsed_query: Dict,
        strict_mastercategory: Optional[str] = None,
        strict_category: Optional[str] = None
    ) -> float:
        """
        Calculate relevance score for a candidate based on query filters.
        Uses strict matching when explicit category is provided.
        
        Args:
            candidate: Candidate metadata from Pinecone
            parsed_query: Parsed query with filters
            strict_mastercategory: If provided, enforce strict mastercategory matching
            strict_category: If provided, enforce strict category matching
        
        Returns:
            Relevance score (0.0 to 100.0) - higher is better
        """
        score = 0.0
        filters = parsed_query.get("filters", {})
        
        # Strict mastercategory enforcement (when explicit category provided)
        if strict_mastercategory:
            candidate_mastercategory = candidate.get("mastercategory", "")
            if candidate_mastercategory.upper() != strict_mastercategory.upper():
                # Hard exclusion for explicit category mode
                logger.debug(
                    f"Mastercategory mismatch: expected={strict_mastercategory}, "
                    f"got={candidate_mastercategory}, excluding candidate"
                )
                return -100.0  # Return very low score (will be filtered out)
        
        # Strict category enforcement (when explicit category provided)
        if strict_category:
            candidate_category = candidate.get("category", "")
            normalized_candidate_category = self._normalize_namespace(candidate_category)
            normalized_strict_category = self._normalize_namespace(strict_category)
            if normalized_candidate_category != normalized_strict_category:
                # Heavy penalty for category mismatch
                score -= 30.0
                logger.debug(
                    f"Category mismatch penalty: expected={normalized_strict_category}, "
                    f"got={normalized_candidate_category}, penalty=-30"
                )
        
        # Parse candidate skills and normalize to canonical forms
        candidate_skills_raw = candidate.get("skills", [])
        if isinstance(candidate_skills_raw, str):
            raw_skills = [s.strip() for s in candidate_skills_raw.split(",") if s.strip()]
            candidate_skills = normalize_skill_list(raw_skills)
        elif isinstance(candidate_skills_raw, list):
            candidate_skills = normalize_skill_list([str(s) for s in candidate_skills_raw if s])
        else:
            candidate_skills = []
        
        weights = self._get_ai_search_weights()
        skill_w = weights["skill_weight"]
        group_w = weights["group_skill_weight"]
        
        # Score must_have_all skills (soft scoring - partial matches get partial score)
        must_have_all = filters.get("must_have_all", [])
        if must_have_all:
            # Normalize required skills to canonical forms for matching
            required_skills = normalize_skill_list(must_have_all)
            matched_skills = sum(1 for skill in required_skills if skill in candidate_skills)
            if matched_skills > 0:
                # Partial match: score based on percentage of skills matched
                skill_match_ratio = matched_skills / len(required_skills)
                score += skill_match_ratio * skill_w
            else:
                # NLP: stemmed overlap (e.g. "testing" vs "test", "selenium" vs "selenium webdriver")
                stemmed_ratio = stemmed_skill_overlap_ratio(must_have_all, candidate_skills)
                if stemmed_ratio > 0:
                    score += stemmed_ratio * skill_w
        
        # Score must_have_one_of_groups (OR logic - any group match gets points)
        must_have_one_of_groups = filters.get("must_have_one_of_groups", [])
        if must_have_one_of_groups:
            max_group_score = 0.0
            for group in must_have_one_of_groups:
                if not group:
                    continue
                # Normalize group skills to canonical forms for matching
                group_skills = normalize_skill_list([str(s) for s in group if s])
                matched_in_group = sum(1 for skill in group_skills if skill in candidate_skills)
                if matched_in_group > 0:
                    group_ratio = matched_in_group / len(group_skills)
                    group_score = group_ratio * group_w
                    max_group_score = max(max_group_score, group_score)
                else:
                    # NLP: stemmed overlap for this OR group
                    stemmed_ratio = stemmed_skill_overlap_ratio(group, candidate_skills)
                    if stemmed_ratio > 0:
                        max_group_score = max(max_group_score, stemmed_ratio * group_w)
            score += max_group_score
        
        # FIX 5: Domain-specific skill boosts (QA/Automation)
        query_text = parsed_query.get("text_for_embedding", "").lower()
        designation_filter = (filters.get("designation") or "").lower()
        
        qa_keywords = ["qa", "automation", "selenium", "webdriver", "test", "testing", "testng", "cucumber"]
        is_qa_query = any(kw in query_text or kw in designation_filter for kw in qa_keywords)
        
        if is_qa_query:
            # Count QA-specific skills in candidate
            qa_skill_matches = sum(1 for skill in candidate_skills if any(
                qa_kw in skill for qa_kw in qa_keywords
            ))
            if qa_skill_matches > 0:
                qa_boost = qa_skill_matches * 5.0  # +5 per QA skill
                score += qa_boost
                logger.debug(
                    f"QA skill boost: {qa_skill_matches} skills matched, boost=+{qa_boost}"
                )
        
        # Domain alignment boost (soft ranking signal only, never a hard filter)
        domain_filter = (filters.get("domain") or "").lower().strip()
        domain_w = weights.get("domain_match_weight", 12.0)
        if domain_filter:
            candidate_domain = str(candidate.get("domain") or "").lower().strip()
            if candidate_domain:
                if candidate_domain == domain_filter:
                    score += domain_w
                    logger.debug(
                        f"Domain match: query={domain_filter}, candidate={candidate_domain}, boost=+{domain_w}"
                    )
                elif domain_filter in candidate_domain or candidate_domain in domain_filter:
                    score += 6.0
                    logger.debug(
                        f"Domain partial match: query={domain_filter}, candidate={candidate_domain}, boost=+6"
                    )

        # Title similarity / exact-title boost (applies to jobrole/designation)
        designation_query = (filters.get("designation") or "").strip().lower()
        if designation_query:
            cand_jobrole = (candidate.get("jobrole") or "").strip().lower()
            cand_designation = (candidate.get("designation") or "").strip().lower()
            # Exact match on jobrole or designation → strong boost
            if cand_jobrole == designation_query or cand_designation == designation_query:
                score += 25.0
                logger.debug(
                    f"Exact title match boost: query='{designation_query}', "
                    f"candidate_jobrole='{cand_jobrole}', candidate_designation='{cand_designation}', "
                    f"boost=+25"
                )
            else:
                # Simple token-overlap similarity: if most words match in order, give a small boost
                q_tokens = designation_query.split()
                jr_tokens = cand_jobrole.split() if cand_jobrole else []
                des_tokens = cand_designation.split() if cand_designation else []
                def _overlap_ratio(a, b):
                    if not a or not b:
                        return 0.0
                    set_a, set_b = set(a), set(b)
                    inter = len(set_a & set_b)
                    return inter / len(set_a)
                title_sim = max(_overlap_ratio(q_tokens, jr_tokens), _overlap_ratio(q_tokens, des_tokens))
                if title_sim >= 0.75:
                    score += 10.0
                    logger.debug(
                        f"Title similarity boost: query='{designation_query}', "
                        f"candidate_jobrole='{cand_jobrole}', candidate_designation='{cand_designation}', "
                        f"similarity={title_sim:.2f}, boost=+10"
                    )
        
        # OPTIMIZATION for 180k+ resumes: Rule-based designation matching first, LLM only for top candidates
        # Prefer jobrole (normalized) over designation (raw from resume) for matching.
        designation = filters.get("designation")
        if designation:
            candidate_designation = candidate.get("designation", "")
            candidate_jobrole = candidate.get("jobrole", "")
            
            # STEP 1: Fast rule-based matching (no LLM call) - O(1) lookup
            is_match = False
            confidence = 0.0
            
            # Normalize query role once
            normalized_query_role = self._normalize_role(designation)
            normalized_designation_query = designation.lower().strip()
            
            # Try matching with candidate jobrole first (normalized), then designation (raw)
            if candidate_jobrole:
                normalized_cand_jobrole = self._normalize_role(candidate_jobrole)
                if normalized_query_role and normalized_cand_jobrole:
                    if normalized_query_role == normalized_cand_jobrole:
                        is_match, confidence = True, 1.0
                    elif normalized_query_role in normalized_cand_jobrole or normalized_cand_jobrole in normalized_query_role:
                        is_match, confidence = True, 0.85
                if not is_match:
                    candidate_jobrole_lower = candidate_jobrole.lower()
                    if normalized_designation_query in candidate_jobrole_lower or candidate_jobrole_lower in normalized_designation_query:
                        is_match, confidence = True, 0.85
            
            if not is_match and candidate_designation:
                normalized_cand_role = self._normalize_role(candidate_designation)
                if normalized_query_role and normalized_cand_role:
                    if normalized_query_role == normalized_cand_role:
                        is_match, confidence = True, 1.0
                    elif normalized_query_role in normalized_cand_role or normalized_cand_role in normalized_query_role:
                        is_match, confidence = True, 0.85
                if not is_match:
                    candidate_designation_lower = candidate_designation.lower()
                    if normalized_designation_query in candidate_designation_lower or candidate_designation_lower in normalized_designation_query:
                        is_match, confidence = True, 0.8
            
            # Role-only mode: same role family = match (align with _candidate_matches_query_role); never -40
            # Skip same-family for multi-role queries (e.g. "sql/etl" => only sql/etl, not data engineer)
            role_only = parsed_query.get("role_only", False)
            multi_role = self._is_multi_role_query(designation)
            role_keywords = self._role_keywords_from_query(designation) if multi_role else set()
            if not is_match and role_only and normalized_query_role and not multi_role:
                query_family = self.ROLE_FAMILIES.get(normalized_query_role)
                for raw in (candidate_jobrole, candidate_designation):
                    if not raw:
                        continue
                    cand_norm = self._normalize_role(raw)
                    if cand_norm and query_family and self.ROLE_FAMILIES.get(cand_norm) == query_family:
                        is_match, confidence = True, 0.85
                        break
            # Multi-role: match only if candidate role contains at least one query role keyword
            if not is_match and role_only and multi_role and role_keywords:
                for raw in (candidate_jobrole, candidate_designation):
                    if not raw:
                        continue
                    raw_lower = str(raw).strip().lower()
                    if any(kw in raw_lower for kw in role_keywords):
                        is_match, confidence = True, 0.85
                        break
            
            # Apply scoring based on rule-based match result
            # Note: LLM matching will be called later for top candidates only (see two-stage processing below)
            des_match_w = weights.get("designation_match_weight", 50.0)
            des_penalty = weights.get("designation_mismatch_penalty", 40.0)
            if is_match:
                # Strong positive score based on confidence
                if confidence >= 0.9:
                    score += des_match_w  # Very high confidence match
                elif confidence >= 0.7:
                    score += 40.0  # High confidence match
                elif confidence >= 0.5:
                    score += 25.0  # Moderate confidence match
                else:
                    score += 15.0  # Low confidence match
                
                logger.debug(
                    f"Designation match (rule-based): query='{designation}', candidate='{candidate_designation or candidate_jobrole}', "
                    f"match=True, confidence={confidence}, boost=+{score}"
                )
            else:
                # Penalty only when not role_only (in role_only mode we use inclusion gate; no penalty)
                if not role_only:
                    score -= des_penalty
                    logger.debug(
                        f"Designation mismatch (rule-based): query='{designation}', candidate='{candidate_designation or candidate_jobrole}', "
                        f"match=False, penalty=-{des_penalty}"
                    )
        
        # Experience score tiers: exact = best, ±1 = next, ±2 = next, then others (recruiter-friendly ordering)
        min_experience = filters.get("min_experience")
        max_experience = filters.get("max_experience")
        exp_exact_w = weights.get("experience_exact_weight", 18.0)
        
        if min_experience is not None:
            try:
                min_exp = int(min_experience)
                candidate_exp = candidate.get("experience_years")
                if candidate_exp is None:
                    candidate_exp = 0
                
                diff = abs(candidate_exp - min_exp)
                if diff == 0:
                    score += exp_exact_w  # Exact match (best)
                elif diff == 1:
                    score += 14.0  # ±1 year (next)
                elif diff == 2:
                    score += 10.0  # ±2 years (next)
                elif candidate_exp >= min_exp:
                    score += 4.0   # Above band (e.g. 13 when query was 10; still show but rank lower)
                elif candidate_exp >= min_exp - 2:
                    score += 3.0   # Within 2 below (edge case if band not applied)
                else:
                    score -= 15.0  # Too little experience
                    logger.debug(
                        f"Experience penalty: required={min_exp}, candidate={candidate_exp}, penalty=-15"
                    )
            except (ValueError, TypeError):
                pass
        
        # FIX 3: Handle max_experience (for range queries like "5-7 years")
        if max_experience is not None:
            try:
                max_exp = int(max_experience)
                candidate_exp = candidate.get("experience_years", 0)
                
                if candidate_exp > max_exp:
                    # Small penalty for exceeding max experience
                    score -= 5.0
                    logger.debug(
                        f"Max experience penalty: max={max_exp}, candidate={candidate_exp}, penalty=-5"
                    )
                elif candidate_exp <= max_exp and candidate_exp >= (filters.get("min_experience") or 0):
                    # Bonus for being within range
                    score += 5.0
                    logger.debug(
                        f"Experience range match: candidate={candidate_exp} within range, boost=+5"
                    )
            except (ValueError, TypeError):
                pass
        
        # Mastercategory alignment (only if not using strict matching)
        # Strict matching is handled at the beginning of the function
        if not strict_mastercategory:
            identified_mastercategory = parsed_query.get("mastercategory")
            candidate_mastercategory = candidate.get("mastercategory", "")
            
            if identified_mastercategory and candidate_mastercategory:
                if candidate_mastercategory.upper() != identified_mastercategory.upper():
                    # Strong penalty for wrong mastercategory
                    score -= 50.0  # Heavy penalty
                    logger.debug(
                        f"Mastercategory mismatch: query={identified_mastercategory}, "
                        f"candidate={candidate_mastercategory}, penalty=-50"
                    )
                else:
                    # Boost for correct mastercategory
                    score += 10.0  # Small boost for alignment
                    logger.debug(
                        f"Mastercategory match: {identified_mastercategory}, boost=+10"
                    )
        
        return score
    
    def categorize_fit_tier(
        self,
        candidate: Dict,
        parsed_query: Dict,
        combined_score: float
    ) -> str:
        """
        Categorize candidate into fit tier based on combined score.
        Principle: "Rank candidates by relevance and return results in clear fit tiers"
        
        Args:
            candidate: Candidate metadata
            parsed_query: Parsed query
            combined_score: Combined score (0-100, recruiter-facing)
        
        Returns:
            Fit tier: "Perfect Match", "Good Match", "Partial Match", "Low Match"
        """
        # Normalize combined score to 0-1 for categorization
        normalized_score = max(0.0, min(1.0, combined_score / 100.0))
        
        # FIX 6: Additional checks for fit tier (domain/role alignment + exact role gating)
        identified_mastercategory = parsed_query.get("mastercategory")
        candidate_mastercategory = candidate.get("mastercategory", "")
        
        # Hard exclusion for mastercategory mismatch
        if identified_mastercategory and candidate_mastercategory:
            if candidate_mastercategory.upper() != identified_mastercategory.upper():
                return "Low Match"  # Force low match for wrong domain
        
        # Check for student/intern roles when query wants professional
        filters = parsed_query.get("filters", {})
        designation = (filters.get("designation") or "").lower()
        candidate_role_for_check = (candidate.get("jobrole") or candidate.get("designation") or "").lower()
        
        if designation and (
            "student" in candidate_role_for_check
            or "intern" in candidate_role_for_check
            or "trainee" in candidate_role_for_check
        ):
            if (
                "student" not in designation
                and "intern" not in designation
                and "trainee" not in designation
            ):
                return "Low Match"  # Force low match for students when query wants professionals

        # NEW: Functional role-family gating when query specifies a role/designation.
        # We want to allow close functional roles (e.g. product manager vs product ops)
        # but push clearly different families (e.g. scrum master vs product ops) to Low Match.
        # Prefer jobrole (normalized) over designation (raw) for candidate role.
        query_role = filters.get("designation")
        candidate_role_raw = candidate.get("jobrole") or candidate.get("designation") or ""

        normalized_query_role = self._normalize_role(query_role) if query_role else None
        normalized_candidate_role = (
            self._normalize_role(candidate_role_raw) if candidate_role_raw else None
        )

        # Use class-level ROLE_FAMILIES for consistent family gating
        query_family = self.ROLE_FAMILIES.get(normalized_query_role) if normalized_query_role else None
        candidate_family = (
            self.ROLE_FAMILIES.get(normalized_candidate_role) if normalized_candidate_role else None
        )

        # If both roles map to families and the families differ, force Low Match
        if query_family and candidate_family and query_family != candidate_family:
            return "Low Match"

            # If roles match exactly, we can promote the fit tier based on experience
            min_experience = filters.get("min_experience")
            candidate_exp = candidate.get("experience_years")
            experience_match = False
            try:
                if min_experience is None or candidate_exp is None:
                    experience_match = True
                else:
                    min_exp_int = int(min_experience)
                    experience_match = candidate_exp >= min_exp_int
            except (ValueError, TypeError):
                experience_match = True  # If parsing fails, don't block promotion

            if experience_match:
                # Exact role + experience match → Perfect Match
                return "Perfect Match"
            else:
                # Exact role but weaker experience → Good Match
                return "Good Match"
        
        # NEW: Skill-based promotion - if required skills match, promote tier
        must_have_all = filters.get("must_have_all", [])
        must_have_one_of_groups = filters.get("must_have_one_of_groups", [])
        # Normalize candidate skills to canonical forms
        candidate_skills_raw = candidate.get("skills", []) or []
        if isinstance(candidate_skills_raw, str):
            raw_skills = [s.strip() for s in candidate_skills_raw.split(",") if s.strip()]
            candidate_skills = normalize_skill_list(raw_skills)
        elif isinstance(candidate_skills_raw, list):
            candidate_skills = normalize_skill_list([str(s) for s in candidate_skills_raw if s])
        else:
            candidate_skills = []
        
        # Check if all required skills are present (exact then stemmed)
        has_all_required_skills = True
        if must_have_all:
            # Normalize required skills to canonical forms for matching
            required_skills = normalize_skill_list(must_have_all)
            for req_skill in required_skills:
                # Check if any candidate skill matches the required skill (exact match after normalization)
                skill_found = req_skill in candidate_skills
                if not skill_found:
                    has_all_required_skills = False
                    break
            # Stemmed fallback: e.g. "testing" vs "test"
            if not has_all_required_skills and stemmed_skill_overlap_ratio(must_have_all, candidate_skills) >= 1.0:
                has_all_required_skills = True
        
        # Check if at least one skill from any group is present (exact then stemmed)
        has_one_of_skills = True
        if must_have_one_of_groups:
            has_one_of_skills = False
            for group in must_have_one_of_groups:
                if not group:
                    continue
                # Normalize group skills to canonical forms for matching
                group_skills = normalize_skill_list([str(s) for s in group if s])
                for req_skill in group_skills:
                    if req_skill in candidate_skills:
                        has_one_of_skills = True
                        break
                if has_one_of_skills:
                    break
            # Stemmed fallback for OR groups
            if not has_one_of_skills:
                for group in must_have_one_of_groups:
                    if not group:
                        continue
                    if stemmed_skill_overlap_ratio(group, candidate_skills) > 0:
                        has_one_of_skills = True
                        break
        
        # Apply skill-based promotion
        skills_match = (not must_have_all or has_all_required_skills) and \
                      (not must_have_one_of_groups or has_one_of_skills)
        
        if skills_match and (must_have_all or must_have_one_of_groups):
            # Skills match + role is relevant (even if not exactly normalized) → promote tier
            # Check if role is at least somewhat relevant (fuzzy match)
            query_role_lower = (query_role or "").lower()
            candidate_role_lower = candidate_role_raw.lower()
            
            # Simple relevance check: if query role keywords appear in candidate role
            role_keywords = set(query_role_lower.split()) if query_role_lower else set()
            candidate_role_words = set(candidate_role_lower.split())
            role_relevant = False
            
            if role_keywords:
                # Remove common stop words
                stop_words = {"the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for", "with"}
                role_keywords = {w for w in role_keywords if w not in stop_words and len(w) > 2}
                candidate_role_words = {w for w in candidate_role_words if w not in stop_words and len(w) > 2}
                
                # If there's significant overlap, consider role relevant
                if role_keywords and candidate_role_words:
                    overlap = len(role_keywords.intersection(candidate_role_words))
                    role_relevant = overlap > 0 and (overlap / len(role_keywords)) >= 0.3
            
            # Promote based on skills + role relevance
            if role_relevant or not query_role:
                # Skills match + role relevant (or no role specified) → promote
                if normalized_score >= 0.40:  # Lower threshold when skills match
                    if normalized_score >= 0.65:
                        return "Good Match"
                    else:
                        return "Partial Match"
                else:
                    # Even with low score, if skills match, at least Partial Match
                    return "Partial Match"
        
        # Then apply score-based tiers
        if normalized_score >= 0.85:
            return "Perfect Match"
        elif normalized_score >= 0.70:
            return "Good Match"
        elif normalized_score >= 0.50:
            return "Partial Match"
        else:
            # Role-only mode: same role family → at least Partial Match (never drop as Low Match)
            role_only = parsed_query.get("role_only", False)
            if role_only and query_family and candidate_family and query_family == candidate_family:
                return "Partial Match"
            return "Low Match"
    
    async def search_name(self, candidate_name: str, session) -> List[Dict[str, Any]]:
        """
        Search candidates by name using SQL with token-based matching, phonetic matching (Soundex), and ranking.
        Principle: "Never invent or assume" - exact/partial/phonetic name match
        
        Args:
            candidate_name: Name to search for
            session: Database session
        
        Returns:
            List of candidate results with same format as Pinecone results, ranked by match quality
        """
        try:
            # FIX 1: Token-based name search - split name and use OR conditions
            from sqlalchemy import select, func, or_, and_
            from app.database.models import ResumeMetadata
            
            # Split name into tokens (normalize and remove empty strings)
            tokens = [token.strip().lower() for token in candidate_name.split() if token.strip()]

            if not tokens:
                logger.warning(f"Empty name tokens from query: '{candidate_name}'")
                return []

            # Ignore very short tokens (1–2 characters) for matching to avoid overly broad results
            filtered_tokens = [token for token in tokens if len(token) >= 3]
            
            # DEBUG: Check sample database values before querying
            try:
                sample_query = select(ResumeMetadata).where(
                    ResumeMetadata.candidatename.isnot(None)
                ).limit(10)
                sample_result = await session.execute(sample_query)
                sample_resumes = sample_result.scalars().all()
                sample_names = [
                    {
                        "id": r.id,
                        "name": r.candidatename,
                        "name_lower": r.candidatename.lower() if r.candidatename else None,
                        "name_length": len(r.candidatename) if r.candidatename else 0,
                        "name_repr": repr(r.candidatename) if r.candidatename else None
                    }
                    for r in sample_resumes[:5]
                ]
                logger.info(
                    f"DEBUG: Sample database names (first 5 non-null): {sample_names}",
                    extra={
                        "candidate_name": candidate_name,
                        "sample_names": sample_names
                    }
                )
                
                # DEBUG: Check if record id=15 exists (the one mentioned by user)
                test_record_query = select(ResumeMetadata).where(ResumeMetadata.id == 15)
                test_result = await session.execute(test_record_query)
                test_record = test_result.scalar_one_or_none()
                if test_record:
                    logger.info(
                        f"DEBUG: Record id=15 found",
                        extra={
                            "id": test_record.id,
                            "candidatename": test_record.candidatename,
                            "candidatename_repr": repr(test_record.candidatename),
                            "candidatename_lower": test_record.candidatename.lower() if test_record.candidatename else None,
                            "candidatename_length": len(test_record.candidatename) if test_record.candidatename else 0,
                            "candidatename_is_none": test_record.candidatename is None,
                            "candidatename_is_empty": test_record.candidatename == "" if test_record.candidatename else None,
                            "contains_andrey": "andrey" in test_record.candidatename.lower() if test_record.candidatename else False
                        }
                    )
                else:
                    logger.warning("DEBUG: Record id=15 NOT FOUND in database")
            except Exception as e:
                logger.warning(f"DEBUG: Failed to fetch sample names: {e}")
            
            # Build OR conditions for each significant token (substring matching - simplified for better compatibility)
            conditions = []
            for token in filtered_tokens:
                # Use LOWER with LIKE for case-insensitive matching (more reliable than TRIM)
                conditions.append(
                    func.LOWER(ResumeMetadata.candidatename).like(f"%{token}%")
                )
            
            # NEW: Add Soundex phonetic matching for the full query name
            # MySQL SOUNDEX() converts names to phonetic codes (e.g., "sasha" -> "S200")
            # This helps match similar-sounding names like "sasha" and "seshareddy"
            normalized_query_name = candidate_name.strip().lower()
            if normalized_query_name:
                # Soundex matching: match if Soundex codes are similar
                # Using SOUNDEX() function from MySQL
                soundex_condition = func.SOUNDEX(func.LOWER(ResumeMetadata.candidatename)) == func.SOUNDEX(normalized_query_name)
                conditions.append(soundex_condition)
                
                # Also try Soundex matching for individual tokens (for multi-word names)
                # This helps with cases like "John Smith" where "John" might be "Jon"
                for token in tokens:
                    if len(token) > 2:  # Only use Soundex for tokens longer than 2 characters
                        token_soundex = func.SOUNDEX(func.LOWER(ResumeMetadata.candidatename)) == func.SOUNDEX(token)
                        conditions.append(token_soundex)
            
            # DEBUG: Log the conditions being built
            logger.info(
                f"DEBUG: Name search conditions built",
                extra={
                    "candidate_name": candidate_name,
                    "tokens": tokens,
                    "filtered_tokens": filtered_tokens,
                    "normalized_query_name": normalized_query_name,
                    "condition_count": len(conditions),
                    "condition_types": [
                        "LIKE" if "like" in str(c).lower() else "SOUNDEX" if "soundex" in str(c).lower() else "OTHER"
                        for c in conditions
                    ]
                }
            )
            
            # Query with NULL/empty filtering and OR conditions (matches if any token is found OR Soundex matches)
            query = select(ResumeMetadata).where(
                and_(
                    ResumeMetadata.candidatename.isnot(None),
                    ResumeMetadata.candidatename != "",
                    or_(*conditions)
                )
            )
            
            # DEBUG: Compile and log the actual SQL query
            try:
                from sqlalchemy.dialects import mysql
                compiled_query = query.compile(dialect=mysql.dialect(), compile_kwargs={"literal_binds": False})
                sql_str = str(compiled_query)
                params = compiled_query.params if hasattr(compiled_query, 'params') else {}
                logger.info(
                    f"DEBUG: Generated SQL query for name search",
                    extra={
                        "candidate_name": candidate_name,
                        "sql_query": sql_str,
                        "sql_params": params,
                        "query_repr": repr(query)
                    }
                )
            except Exception as e:
                logger.warning(f"DEBUG: Failed to compile SQL query: {e}")
            
            result = await session.execute(query)
            resumes = result.scalars().all()
            
            # DEBUG: Test with a simple LIKE query to verify database connection
            try:
                simple_test_query = select(ResumeMetadata).where(
                    ResumeMetadata.candidatename.like(f"%{normalized_query_name}%")
                ).limit(5)
                simple_test_result = await session.execute(simple_test_query)
                simple_test_resumes = simple_test_result.scalars().all()
                logger.info(
                    f"DEBUG: Simple LIKE test query (case-sensitive): found {len(simple_test_resumes)} results",
                    extra={
                        "candidate_name": candidate_name,
                        "test_pattern": f"%{normalized_query_name}%",
                        "test_results": [
                            {
                                "id": r.id,
                                "name": r.candidatename,
                                "name_lower": r.candidatename.lower() if r.candidatename else None
                            }
                            for r in simple_test_resumes
                        ]
                    }
                )
            except Exception as e:
                logger.warning(f"DEBUG: Simple LIKE test query failed: {e}")
            
            # Debug logging to help diagnose issues (using INFO level so it's visible)
            logger.info(
                f"Name search query executed: found {len(resumes)} raw results",
                extra={
                    "candidate_name": candidate_name,
                    "query_tokens": tokens,
                    "filtered_tokens": [token for token in tokens if len(token) >= 3],
                    "normalized_query_name": normalized_query_name,
                    "raw_result_count": len(resumes),
                    "sample_names": [r.candidatename for r in resumes[:5]] if resumes else [],
                    "all_result_ids": [r.id for r in resumes[:10]] if resumes else []
                }
            )
            
            # FIX 4: Rank results by match quality (exact > substring > token > Soundex)
            normalized_query = candidate_name.lower().strip()
            results_with_scores = []
            
            # Pre-compute Soundex code for query name (for comparison and scoring)
            # Get Soundex code from database for the query name
            query_soundex_code = None
            try:
                soundex_query_result = await session.execute(
                    select(func.SOUNDEX(normalized_query_name))
                )
                query_soundex_code = soundex_query_result.scalar()
            except Exception as e:
                logger.warning(f"Failed to compute Soundex for query name: {e}")
            
            # Pre-compute Soundex codes for all candidate names (batch for efficiency)
            candidate_soundex_map = {}
            if query_soundex_code and resumes:
                try:
                    # Compute Soundex codes for all candidate names
                    for resume in resumes:
                        if resume.candidatename:
                            try:
                                soundex_result = await session.execute(
                                    select(func.SOUNDEX(resume.candidatename))
                                )
                                candidate_soundex_map[resume.candidatename.lower()] = soundex_result.scalar()
                            except Exception:
                                pass
                except Exception as e:
                    logger.warning(f"Failed to compute Soundex codes for candidates: {e}")
            
            for resume in resumes:
                candidate_name_lower = (resume.candidatename or "").lower()
                
                # Calculate match score based on match type (prioritize exact matches)
                if candidate_name_lower == normalized_query:
                    # Exact full name match (highest priority)
   
                    match_score = 1.0
                    match_type = "exact"
                elif normalized_query in candidate_name_lower or candidate_name_lower in normalized_query:
                    # Partial match (query is substring or vice versa)
                    match_score = 0.8
                    match_type = "partial"
                else:
                    # Check token-based match (ignore very short tokens for scoring as well)
                    effective_tokens = [token for token in tokens if len(token) >= 3]
                    matched_tokens = sum(1 for token in effective_tokens if token in candidate_name_lower)
                    if matched_tokens > 0 and effective_tokens:
                        # Token-based match (at least one meaningful token matched)
                        match_ratio = matched_tokens / len(effective_tokens)
                        match_score = 0.6 * match_ratio  # Base score for token match
                        match_type = "token"
                    else:
                        # Likely a Soundex match (phonetic similarity)
                        candidate_soundex_code = candidate_soundex_map.get(candidate_name_lower)
                        
                        # Compare Soundex codes
                        if query_soundex_code and candidate_soundex_code:
                            if query_soundex_code == candidate_soundex_code:
                                # Exact Soundex match
                                match_score = 0.5
                                match_type = "soundex_exact"
                            elif len(query_soundex_code) >= 2 and len(candidate_soundex_code) >= 2 and query_soundex_code[:2] == candidate_soundex_code[:2]:
                                # Similar Soundex (first 2 chars match)
                                match_score = 0.4
                                match_type = "soundex_similar"
                            else:
                                # Weak Soundex match
                                match_score = 0.3
                                match_type = "soundex_weak"
                        else:
                            # Fallback: very low score if we can't compute Soundex
                            match_score = 0.2
                            match_type = "unknown"
                
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
                
                results_with_scores.append({
                    "resume_id": resume.id,
                    "candidate_id": f"C{resume.id}",  # Generate candidate_id
                    "name": resume.candidatename or "",
                    "category": resume.category or "",
                    "mastercategory": resume.mastercategory or "",
                    "experience_years": experience_years,
                    "skills": skills,
                    "location": None,  # Not stored in resume_metadata
                    "score": match_score,
                    "match_type": match_type,  # For debugging
                    "fit_tier": self._get_fit_tier_from_score(match_score)
                })
            
            # Sort by score (highest first), then by number of matched tokens (more tokens = better), then by name length (shorter = closer match)
            def name_sort_key(r):
                s = r["score"]
                # Secondary: prefer more matched tokens (approximate via score tier)
                token_bonus = 1.0 if s >= 0.6 else (0.5 if s >= 0.4 else 0.0)
                # Tertiary: shorter name length when scores equal (exact/substring tend to be shorter)
                name_len = len(r.get("name", "") or "")
                return (s, token_bonus, -name_len)
            
            results_with_scores.sort(key=name_sort_key, reverse=True)
            
            # Cap results for name search (configurable)
            name_cap = getattr(settings, "ai_search_name_result_cap", 100)
            if len(results_with_scores) > name_cap:
                results_with_scores = results_with_scores[:name_cap]
                logger.info(
                    f"Name search capped to {name_cap} results",
                    extra={"candidate_name": candidate_name, "cap": name_cap}
                )
            
            # Remove match_type from final results (internal only)
            results = [{k: v for k, v in r.items() if k != "match_type"} for r in results_with_scores]
            
            # Get match types for logging (before removing from results)
            match_types_summary = [r.get("match_type", "unknown") for r in results_with_scores[:5]] if results_with_scores else []
            
            logger.info(
                f"Name search found {len(results)} candidates (tokens: {tokens}, soundex: {query_soundex_code})",
                extra={
                    "candidate_name": candidate_name,
                    "tokens": tokens,
                    "query_soundex_code": query_soundex_code,
                    "result_count": len(results),
                    "match_types": match_types_summary
                }
            )
            
            return results
            
        except Exception as e:
            logger.error(
                f"Name search failed: {e}",
                extra={"candidate_name": candidate_name, "error": str(e)}
            )
            raise
    
    def _get_fit_tier_from_score(self, score: float) -> str:
        """
        Convert score to fit tier for name search results.
        
        Args:
            score: Match score (0.0 to 1.0)
        
        Returns:
            Fit tier string
        """
        if score >= 0.9:
            return "Perfect Match"
        elif score >= 0.7:
            return "Good Match"
        elif score >= 0.5:
            return "Partial Match"
        else:
            return "Low Match"
    
    async def search_semantic(
        self,
        parsed_query: Dict,
        top_k: int,
        explicit_category_mode: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search candidates using semantic search with Pinecone.
        
        Args:
            parsed_query: Parsed query with filters
            top_k: Number of results to return
            explicit_category_mode: If True, use ONLY the provided category (no inference, no fallbacks)
        
        Returns:
            List of candidate results with fit tiers
        """
        try:
            # Initialize PineconeAutomation if not already done
            if not self.pinecone_automation.pc:
                await self.pinecone_automation.initialize_pinecone()
                await self.pinecone_automation.create_indexes()
            
            # If query has a designation, expand to equivalent roles for potential downstream use
            filters = parsed_query.get("filters", {})
            query_designation = filters.get("designation")
            if query_designation and str(query_designation).strip():
                try:
                    normalized_for_filter = self._strip_seniority_words(query_designation)
                    designation_equivalent_list = await self.designation_matcher.expand_designation_to_equivalent_roles(
                        normalized_for_filter or query_designation
                    )
                    if designation_equivalent_list:
                        parsed_query.setdefault("filters", {})["designation_equivalent_list"] = designation_equivalent_list
                except Exception as e:
                    logger.warning(
                        f"Designation expand failed, skipping designation filter: {e}",
                        extra={"designation": query_designation, "error": str(e)}
                    )
            
            # EXPLICIT CATEGORY MODE: Hard-constrained search
            if explicit_category_mode:
                mastercategory = parsed_query.get("mastercategory")
                category = parsed_query.get("category")
                
                if not mastercategory or not category:
                    logger.error("explicit_category_mode=True but mastercategory/category missing in parsed_query")
                    return []
                
                # Choose index STRICTLY from payload
                target_index_name = "IT" if mastercategory.upper() == "IT" else "NON_IT"
                
                # Normalize category to namespace format
                target_namespace = self._normalize_namespace(category)
                
                logger.info(
                    f"Explicit category mode: index={target_index_name}, namespace={target_namespace}",
                    extra={
                        "mastercategory": mastercategory,
                        "category": category,
                        "normalized_namespace": target_namespace
                    }
                )
                
                # Get text for embedding
                text_for_embedding = parsed_query.get("text_for_embedding", "")
                if not text_for_embedding or not text_for_embedding.strip():
                    # Fallback to query if text_for_embedding is empty
                    filters = parsed_query.get("filters", {})
                    filter_parts = []
                    if filters.get("designation"):
                        filter_parts.append(filters.get("designation"))
                    if filters.get("must_have_all"):
                        filter_parts.extend(filters.get("must_have_all", []))
                    if filter_parts:
                        text_for_embedding = " ".join(filter_parts)
                    else:
                        text_for_embedding = "candidate resume"
                
                # Query expansion: add synonyms for better embedding recall
                text_to_embed = expand_query_for_embedding(text_for_embedding) or text_for_embedding
                
                # Generate embedding
                embedding = await self.embedding_service.generate_embedding(text_to_embed)
                
                # Build Pinecone filter
                pinecone_filter = self.build_pinecone_filter(parsed_query)
                
                # When we have a designation, fetch more from Pinecone so role filter has a larger pool
                # (exact-role candidates may be ranked 101+ by similarity and would otherwise be missed)
                pinecone_fetch_k = top_k
                if query_designation and str(query_designation).strip():
                    pinecone_fetch_k = min(getattr(settings, "ai_search_top_k_max", 200) * 2, max(top_k * 3, 200))
                
                # Query ONLY the specified namespace (NO fallbacks)
                all_results = []
                try:
                    namespace_results = await self.pinecone_automation.query_vectors(
                        query_vector=embedding,
                        mastercategory=target_index_name,
                        namespace=target_namespace if target_namespace else None,
                        top_k=pinecone_fetch_k,
                        filter_dict=pinecone_filter
                    )
                    all_results.extend(namespace_results)
                    logger.info(
                        f"Explicit category search returned {len(all_results)} results from namespace '{target_namespace}'",
                        extra={
                            "namespace": target_namespace,
                            "index": target_index_name,
                            "result_count": len(all_results)
                        }
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to query explicit namespace '{target_namespace}': {e}",
                        extra={"namespace": target_namespace, "error": str(e)}
                    )
                
                if len(all_results) == 0:
                    logger.warning(
                        f"Explicit category search returned 0 results. "
                        f"Namespace: {target_namespace}, Index: {target_index_name}",
                        extra={
                            "namespace": target_namespace,
                            "index": target_index_name,
                            "mastercategory": mastercategory,
                            "category": category
                        }
                    )
                    # Return empty (no fallback to other namespaces)
                
                # Process and rank results
                processed_results = []
                seen_resume_ids = set()
                
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
                        "designation": metadata.get("designation", ""),
                        "jobrole": metadata.get("jobrole", ""),
                        "experience_years": experience_years,
                        "skills": skills,
                        "location": metadata.get("location"),
                    "domain": metadata.get("domain"),
                        "score": score
                    }
                    
                    # Calculate relevance score with strict matching
                    relevance_score = await self.calculate_relevance_score(
                        candidate,
                        parsed_query,
                        strict_mastercategory=mastercategory,
                        strict_category=category
                    )
                    
                    # Hybrid: blend semantic + keyword overlap + relevance
                    cand_skills = candidate.get("skills", []) or []
                    if isinstance(cand_skills, str):
                        cand_skills = [s.strip() for s in cand_skills.split(",") if s.strip()]
                    keyword_score = keyword_score_for_candidate(
                        text_for_embedding,
                        cand_skills,
                        candidate.get("designation"),
                        candidate.get("domain"),
                    )
                    combined_score = self._compute_hybrid_combined_score(
                        score, keyword_score, relevance_score
                    )
                    # Store per-signal scores
                    candidate["keyword_score"] = keyword_score
                    candidate["score"] = max(0.0, min(1.0, combined_score / 100.0))
                    candidate["semantic_score"] = score
                    candidate["relevance_score"] = relevance_score
                    
                    # Filter out invalid candidates
                    if not candidate.get("candidate_id") or not candidate.get("candidate_id").strip():
                        continue
                    if not candidate.get("resume_id"):
                        continue
                    if not candidate.get("name") or not candidate.get("name").strip():
                        continue
                    
                    # Categorize fit tier
                    fit_tier = self.categorize_fit_tier(candidate, parsed_query, combined_score)
                    candidate["fit_tier"] = fit_tier
                    
                    processed_results.append(candidate)
                
                # Hard filter by role when query specifies a designation (exclude wrong roles e.g. Python dev for QA query)
                # Always keep candidates with exact role match (100% same) even if they'd fail family/equivalents
                filters_for_role = parsed_query.get("filters", {})
                query_designation = filters_for_role.get("designation")
                if query_designation:
                    designation_equivalent_list = filters_for_role.get("designation_equivalent_list") or None
                    before_count = len(processed_results)
                    processed_results = [
                        c
                        for c in processed_results
                        if self._candidate_matches_query_role(
                            c,
                            query_designation,
                            designation_equivalent_list=designation_equivalent_list,
                        )
                        or self._exact_role_match(c, query_designation)
                    ]
                    logger.info(
                        f"Role filter (designation={query_designation!r}): {before_count} -> {len(processed_results)} candidates",
                        extra={
                            "query_designation": query_designation,
                            "before": before_count,
                            "after": len(processed_results)
                        }
                    )
                
                # Hard filter by category when explicit category mode is used
                if category:
                    normalized_target_category = self._normalize_namespace(category)
                    before_cat_count = len(processed_results)
                    processed_results = [
                        c for c in processed_results
                        if self._normalize_namespace(c.get("category", "")) == normalized_target_category
                    ]
                    logger.info(
                        f"Category filter (category={category!r}): {before_cat_count} -> {len(processed_results)} candidates",
                        extra={
                            "category": category,
                            "before": before_cat_count,
                            "after": len(processed_results)
                        }
                    )
                
                # Backfill: include exact-role matches from DB that may be outside Pinecone top_k
                seen_resume_ids = {c.get("resume_id") for c in processed_results if c.get("resume_id")}
                if query_designation and str(query_designation).strip():
                    try:
                        backfill_ids = await self.resume_repo.get_resume_ids_by_exact_role(
                            mastercategory, category, query_designation, limit=50
                        )
                        logger.info(
                            "Exact-role backfill query: DB returned %s ids for role=%r, category=%r",
                            len(backfill_ids), query_designation, category,
                            extra={"backfill_ids": backfill_ids[:20], "mastercategory": mastercategory}
                        )
                        backfill_ids = [i for i in backfill_ids if i not in seen_resume_ids]
                        if backfill_ids:
                            resumes_map = await self.resume_repo.get_batch_by_ids(backfill_ids)
                            for rid, resume in resumes_map.items():
                                candidate = self._candidate_from_resume_metadata(resume)
                                relevance_score = await self.calculate_relevance_score(
                                    candidate, parsed_query,
                                    strict_mastercategory=mastercategory,
                                    strict_category=category
                                )
                                cand_skills = candidate.get("skills") or []
                                keyword_score = keyword_score_for_candidate(
                                    text_for_embedding,
                                    cand_skills,
                                    candidate.get("designation"),
                                    candidate.get("domain"),
                                )
                                combined_score = self._compute_hybrid_combined_score(
                                    0.0, keyword_score, relevance_score
                                )
                                candidate["keyword_score"] = keyword_score
                                candidate["relevance_score"] = relevance_score
                                candidate["score"] = max(0.35, min(1.0, combined_score / 100.0))
                                candidate["semantic_score"] = 0.0
                                candidate["fit_tier"] = self.categorize_fit_tier(candidate, parsed_query, combined_score)
                                candidate["_exact_role_match"] = True
                                if parsed_query.get("role_only"):
                                    candidate["role_only_mode"] = True
                                processed_results.append(candidate)
                                seen_resume_ids.add(rid)
                            logger.info(
                                f"Backfilled {len(resumes_map)} exact-role candidates from DB",
                                extra={"count": len(resumes_map), "resume_ids": list(resumes_map.keys())}
                            )
                    except Exception as e:
                        logger.warning(
                            f"Exact-role backfill failed: {e}",
                            extra={"error": str(e), "designation": query_designation}
                        )
                
                # Role-only mode: guarantee minimum score so role-matched candidates are not dropped by controller
                if parsed_query.get("role_only"):
                    for c in processed_results:
                        c["role_only_mode"] = True
                        if c.get("score", 0) < 0.35:
                            c["score"] = 0.35
                
                # Mark exact role match (100% same) so we sort them first and never drop by top_k
                q_des = filters_for_role.get("designation") if query_designation else None
                if q_des:
                    for c in processed_results:
                        c["_exact_role_match"] = self._exact_role_match(c, q_des)
                else:
                    for c in processed_results:
                        c["_exact_role_match"] = False
                
                # Sort: exact role matches first, then by score (so exact matches are never pushed out by top_k)
                processed_results.sort(
                    key=lambda x: (not x.get("_exact_role_match", False), -x.get("score", 0.0))
                )
                
                # Limit to top_k
                processed_results = processed_results[:top_k]
                
                logger.info(
                    f"Explicit category search completed: {len(processed_results)} results",
                    extra={
                        "result_count": len(processed_results),
                        "namespace": target_namespace,
                        "index": target_index_name
                    }
                )
                
                return processed_results
            
            # BROAD SEARCH MODE: Smart filtering when category not provided
            return await self._search_broad_mode(parsed_query, top_k)
            
            # If empty, use original query as fallback (better than empty or just skills)
            if not text_for_embedding or not text_for_embedding.strip():
                logger.warning("text_for_embedding is empty, using original query as fallback")
                # Try to reconstruct from filters first
                filters = parsed_query.get("filters", {})
                filter_parts = []
                if filters.get("designation"):
                    filter_parts.append(filters.get("designation"))
                if filters.get("must_have_all"):
                    filter_parts.extend(filters.get("must_have_all", []))
                if filter_parts:
                    text_for_embedding = " ".join(filter_parts)
                else:
                    # Last resort: use a generic search term
                    text_for_embedding = "candidate resume"
                    logger.warning("No filters available, using generic embedding text")
            
            # Detect minimal queries (like "5 years") that might have low semantic similarity
            # Check if text_for_embedding is very short or contains only numbers/years
            filters = parsed_query.get("filters", {})
            has_filters = bool(
                filters.get("min_experience") or 
                filters.get("must_have_all") or 
                filters.get("must_have_one_of_groups") or
                filters.get("designation")
            )
            
            is_minimal_query = (
                len(text_for_embedding.strip().split()) <= 3 and  # Very short query
                has_filters and  # Has filters to apply
                not any(keyword in text_for_embedding.lower() for keyword in 
                       ["developer", "engineer", "manager", "analyst", "specialist", 
                        "python", "java", "software", "candidate", "professional"])
            )
            
            if is_minimal_query:
                logger.info(
                    f"Detected minimal query: '{text_for_embedding}'. "
                    "Will use fallback strategy if semantic search returns 0 results.",
                    extra={"text_for_embedding": text_for_embedding, "has_filters": has_filters}
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
            
            # OPTIMIZATION for 180k+ resumes: Limit results per namespace based on query type
            # Role queries are very targeted, so we need fewer results per namespace
            has_designation = bool(parsed_query.get("filters", {}).get("designation"))
            if has_designation:
                # Role-based queries: very targeted, 3-5 results per namespace is enough
                per_namespace_k = max(3, min(5, top_k // 3))
            else:
                # General queries: need more results for diversity
                per_namespace_k = max(5, top_k // 5)
            
            logger.info(
                f"Per-namespace k set to {per_namespace_k} (has_designation={has_designation}, top_k={top_k})",
                extra={"per_namespace_k": per_namespace_k, "has_designation": has_designation, "top_k": top_k}
            )
            
            # Check if category was identified from query
            identified_mastercategory = parsed_query.get("mastercategory")
            identified_category = parsed_query.get("category")
            target_namespace = None
            role_family_namespaces = []  # NEW: Role-family fallback namespaces
            
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
            else:
                # NEW APPROACH: Role-family namespace fallback when category is None
                role_family = self._detect_role_family(parsed_query)
                if role_family and role_family in self.ROLE_FAMILY_NAMESPACES:
                    role_family_namespaces = self.ROLE_FAMILY_NAMESPACES[role_family]
                    logger.info(
                        f"Using role-family namespace fallback: role_family={role_family}, namespaces={role_family_namespaces}",
                        extra={
                            "role_family": role_family,
                            "namespaces": role_family_namespaces,
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
            
            # OPTIMIZATION for 180k+ resumes: Restrict namespaces for role queries
            # Role queries should only search relevant namespaces, not all 55
            query_designation = (parsed_query.get("filters", {}).get("designation") or "").lower()
            
            # NEW APPROACH: Prioritize namespaces (category-based or role-family fallback)
            # Track if we're restricting to category namespace
            restricted_to_category_namespace = False
            
            if identified_mastercategory:
                target_index_name = "IT" if identified_mastercategory.upper() == "IT" else "NON_IT"
                target_namespaces_list = it_namespaces if target_index_name == "IT" else non_it_namespaces
                
                # NEW: Restrict to identified category namespace when category is identified AND role exists
                if target_namespace and target_namespace in target_namespaces_list and has_designation:
                    # Restrict to ONLY the identified category namespace
                    target_namespaces_list[:] = [target_namespace]
                    restricted_to_category_namespace = True
                    logger.info(
                        f"Restricted to identified category namespace '{target_namespace}' (category identified + role present)",
                        extra={
                            "namespace": target_namespace,
                            "index": target_index_name,
                            "category": identified_category,
                            "role": query_designation,
                            "source": "category_restriction"
                        }
                    )
                # OPTIMIZATION: For role queries without identified category, restrict to 2-3 most relevant namespaces
                elif has_designation and query_designation:
                    # Map common roles to their most relevant namespaces
                    role_namespace_map = {
                        # NON-IT roles
                        "scrum master": ["business_management", "project_management_non_it"],
                        "scrummaster": ["business_management", "project_management_non_it"],
                        "project manager": ["project_management_non_it", "business_management"],
                        "change manager": ["business_management"],
                        "organizational change manager": ["business_management"],
                        # IT roles
                        "qa automation engineer": ["full_stack_development_selenium", "full_stack_development_net"],
                        "automation": ["full_stack_development_selenium"],
                        "qa": ["full_stack_development_selenium"],
                    }
                    
                    # Find matching namespaces for this role
                    restricted_namespaces = []
                    for role_key, namespaces in role_namespace_map.items():
                        if role_key in query_designation:
                            restricted_namespaces = [ns for ns in namespaces if ns in target_namespaces_list]
                            break
                    
                    # If we found restricted namespaces, use only those
                    if restricted_namespaces:
                        target_namespaces_list[:] = restricted_namespaces
                        logger.info(
                            f"Restricted to {len(restricted_namespaces)} namespaces for role query: {restricted_namespaces}",
                            extra={
                                "role": query_designation,
                                "restricted_namespaces": restricted_namespaces,
                                "index": target_index_name,
                                "source": "role_restriction"
                            }
                        )
                
                # Priority 1: Category-based namespace (if available, but no role)
                elif target_namespace and target_namespace in target_namespaces_list:
                    # Move target namespace to front for priority querying
                    target_namespaces_list.remove(target_namespace)
                    target_namespaces_list.insert(0, target_namespace)
                    logger.info(
                        f"Prioritizing identified namespace '{target_namespace}' in {target_index_name} index",
                        extra={"namespace": target_namespace, "index": target_index_name, "source": "category"}
                    )
                # Priority 2: Role-family namespace fallback (if category is None)
                elif role_family_namespaces:
                    # Prioritize role-family namespaces
                    prioritized = []
                    remaining = []
                    
                    for ns in target_namespaces_list:
                        if ns in role_family_namespaces:
                            prioritized.append(ns)
                        else:
                            remaining.append(ns)
                    
                    # Reorder: prioritized namespaces first, then others
                    target_namespaces_list[:] = prioritized + remaining
                    logger.info(
                        f"Prioritizing role-family namespaces in {target_index_name} index: {prioritized}",
                        extra={
                            "role_family_namespaces": role_family_namespaces,
                            "prioritized": prioritized,
                            "index": target_index_name,
                            "source": "role_family"
                        }
                    )
                elif target_namespace:
                    logger.warning(
                        f"Identified namespace '{target_namespace}' not found in {target_index_name} index, querying all namespaces",
                        extra={"namespace": target_namespace, "index": target_index_name}
                    )
            
            # Determine which indexes to query based on identified mastercategory
            # Query IT index only if:
            # 1. Mastercategory is IT (identified), OR
            # 2. Mastercategory is not identified (fallback - query both), OR
            # 3. Restricted to category namespace and it's IT
            should_query_it = (
                (identified_mastercategory and identified_mastercategory.upper() == "IT") or
                (not identified_mastercategory) or
                (restricted_to_category_namespace and identified_mastercategory and identified_mastercategory.upper() == "IT")
            )
            
            # Query Non-IT index only if:
            # 1. Mastercategory is NON_IT (identified), OR
            # 2. Mastercategory is not identified (fallback - query both), OR
            # 3. Restricted to category namespace and it's NON_IT
            should_query_non_it = (
                (identified_mastercategory and identified_mastercategory.upper() == "NON_IT") or
                (not identified_mastercategory) or
                (restricted_to_category_namespace and identified_mastercategory and identified_mastercategory.upper() == "NON_IT")
            )
            
            # Log which indexes will be queried
            indexes_to_query = []
            if should_query_it:
                indexes_to_query.append("IT")
            if should_query_non_it:
                indexes_to_query.append("Non-IT")
            
            logger.info(
                f"Querying {len(it_namespaces) if should_query_it else 0} IT namespaces and {len(non_it_namespaces) if should_query_non_it else 0} Non-IT namespaces",
                extra={
                    "it_namespace_count": len(it_namespaces) if should_query_it else 0,
                    "non_it_namespace_count": len(non_it_namespaces) if should_query_non_it else 0,
                    "indexes_to_query": indexes_to_query,
                    "identified_category": identified_category,
                    "target_namespace": target_namespace,
                    "identified_mastercategory": identified_mastercategory,
                    "restricted_to_category": restricted_to_category_namespace
                }
            )
            
            # Query IT index - all namespaces (or restricted if category identified + role present)
            it_results = []
            if should_query_it:
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
            
            # Query Non-IT index - all namespaces (or restricted if category identified + role present)
            non_it_results = []
            if should_query_non_it:
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
            
            # FALLBACK: If we restricted to category namespace and got 0 results, broaden the search
            was_restricted_to_category = (
                restricted_to_category_namespace and
                len(all_results) == 0
            )
            
            if was_restricted_to_category:
                logger.info(
                    f"Restricted category namespace search returned 0 results. "
                    f"Falling back to broader search (role-family namespaces or all namespaces).",
                    extra={
                        "restricted_namespace": target_namespace,
                        "category": identified_category,
                        "role": query_designation
                    }
                )
                
                # Get all namespaces again for fallback
                it_namespaces_fallback = await self.pinecone_automation.get_all_namespaces("IT")
                non_it_namespaces_fallback = await self.pinecone_automation.get_all_namespaces("NON_IT")
                
                if not it_namespaces_fallback:
                    it_namespaces_fallback = [""]
                if not non_it_namespaces_fallback:
                    non_it_namespaces_fallback = [""]
                
                # Try role-family namespaces first as fallback
                role_family = self._detect_role_family(parsed_query)
                if role_family and role_family in self.ROLE_FAMILY_NAMESPACES:
                    role_family_namespaces_fallback = self.ROLE_FAMILY_NAMESPACES[role_family]
                    if identified_mastercategory:
                        target_index_fallback = "IT" if identified_mastercategory.upper() == "IT" else "NON_IT"
                        target_namespaces_fallback = it_namespaces_fallback if target_index_fallback == "IT" else non_it_namespaces_fallback
                        
                        # Filter to only role-family namespaces that exist
                        fallback_namespaces = [ns for ns in role_family_namespaces_fallback if ns in target_namespaces_fallback]
                        if fallback_namespaces:
                            target_namespaces_fallback[:] = fallback_namespaces
                            logger.info(
                                f"Fallback: Using role-family namespaces: {fallback_namespaces}",
                                extra={"role_family": role_family, "namespaces": fallback_namespaces}
                            )
                
                # Query with fallback namespaces
                it_results_fallback = []
                for namespace in (it_namespaces_fallback if identified_mastercategory and identified_mastercategory.upper() == "IT" else []):
                    try:
                        namespace_results = await self.pinecone_automation.query_vectors(
                            query_vector=embedding,
                            mastercategory="IT",
                            namespace=namespace if namespace else None,
                            top_k=per_namespace_k * 2,  # Get more results in fallback
                            filter_dict=pinecone_filter
                        )
                        it_results_fallback.extend(namespace_results)
                    except Exception as e:
                        logger.warning(f"Failed to query IT namespace '{namespace}' (fallback): {e}")
                
                non_it_results_fallback = []
                for namespace in (non_it_namespaces_fallback if identified_mastercategory and identified_mastercategory.upper() == "NON_IT" else []):
                    try:
                        namespace_results = await self.pinecone_automation.query_vectors(
                            query_vector=embedding,
                            mastercategory="NON_IT",
                            namespace=namespace if namespace else None,
                            top_k=per_namespace_k * 2,  # Get more results in fallback
                            filter_dict=pinecone_filter
                        )
                        non_it_results_fallback.extend(namespace_results)
                    except Exception as e:
                        logger.warning(f"Failed to query Non-IT namespace '{namespace}' (fallback): {e}")
                
                if it_results_fallback or non_it_results_fallback:
                    logger.info(
                        f"Fallback search returned {len(it_results_fallback)} IT and {len(non_it_results_fallback)} Non-IT results",
                        extra={
                            "it_results": len(it_results_fallback),
                            "non_it_results": len(non_it_results_fallback)
                        }
                    )
                    all_results = it_results_fallback + non_it_results_fallback
                
                # OPTION 5: Third-level fallback - Search ALL namespaces when filters exist
                # This ensures no false negatives when filters are present (skills/designation)
                # Filters make this efficient even with 180k+ resumes
                if len(all_results) == 0 and has_filters and pinecone_filter:
                    logger.info(
                        "Role-family fallback returned 0 results but filters exist. "
                        "Searching ALL namespaces with filters to ensure no false negatives.",
                        extra={
                            "filters": pinecone_filter,
                            "designation": filters.get("designation"),
                            "must_have_all": filters.get("must_have_all"),
                            "must_have_one_of_groups": filters.get("must_have_one_of_groups")
                        }
                    )
                    
                    # Get all namespaces for comprehensive search
                    all_it_namespaces = await self.pinecone_automation.get_all_namespaces("IT")
                    all_non_it_namespaces = await self.pinecone_automation.get_all_namespaces("NON_IT")
                    
                    if not all_it_namespaces:
                        all_it_namespaces = [""]
                    if not all_non_it_namespaces:
                        all_non_it_namespaces = [""]
                    
                    # Search all IT namespaces with filters (filters limit results, so it's efficient)
                    it_results_all = []
                    for namespace in (all_it_namespaces if identified_mastercategory and identified_mastercategory.upper() == "IT" else []):
                        try:
                            namespace_results = await self.pinecone_automation.query_vectors(
                                query_vector=embedding,
                                mastercategory="IT",
                                namespace=namespace if namespace else None,
                                top_k=per_namespace_k,  # Smaller k per namespace since we're searching all
                                filter_dict=pinecone_filter
                            )
                            it_results_all.extend(namespace_results)
                        except Exception as e:
                            logger.warning(f"Failed to query IT namespace '{namespace}' (all-namespace fallback): {e}")
                    
                    # Search all Non-IT namespaces with filters
                    non_it_results_all = []
                    for namespace in (all_non_it_namespaces if identified_mastercategory and identified_mastercategory.upper() == "NON_IT" else []):
                        try:
                            namespace_results = await self.pinecone_automation.query_vectors(
                                query_vector=embedding,
                                mastercategory="NON_IT",
                                namespace=namespace if namespace else None,
                                top_k=per_namespace_k,
                                filter_dict=pinecone_filter
                            )
                            non_it_results_all.extend(namespace_results)
                        except Exception as e:
                            logger.warning(f"Failed to query Non-IT namespace '{namespace}' (all-namespace fallback): {e}")
                    
                    if it_results_all or non_it_results_all:
                        logger.info(
                            f"All-namespace fallback (with filters) returned {len(it_results_all)} IT and {len(non_it_results_all)} Non-IT results",
                            extra={
                                "it_results": len(it_results_all),
                                "non_it_results": len(non_it_results_all),
                                "total_namespaces_searched": len(all_it_namespaces) + len(all_non_it_namespaces)
                            }
                        )
                        all_results = it_results_all + non_it_results_all
            
            # FALLBACK STRATEGY: If semantic search returned 0 results but we have filters,
            # retry with a more generic embedding that better matches resume content
            if len(all_results) == 0 and has_filters and is_minimal_query:
                logger.info(
                    "Semantic search returned 0 results with minimal query. "
                    "Retrying with generic embedding for filter-based search.",
                    extra={
                        "original_text": text_for_embedding,
                        "filters": filters
                    }
                )
                
                # Use a generic embedding that matches resume content better
                # This will allow filters to work even when semantic similarity is low
                generic_text = "professional candidate resume experience skills"
                generic_embedding = await self.embedding_service.generate_embedding(generic_text)
                
                logger.info("Retrying search with generic embedding for filter-based matching")
                
                # Retry IT index queries with generic embedding
                it_results_fallback = []
                for idx, namespace in enumerate(it_namespaces):
                    try:
                        namespace_k = per_namespace_k * 2  # Get more results for fallback
                        namespace_results = await self.pinecone_automation.query_vectors(
                            query_vector=generic_embedding,
                            mastercategory="IT",
                            namespace=namespace if namespace else None,
                            top_k=namespace_k,
                            filter_dict=pinecone_filter
                        )
                        it_results_fallback.extend(namespace_results)
                    except Exception as e:
                        logger.warning(
                            f"Failed to query IT index namespace '{namespace}' (fallback): {e}",
                            extra={"namespace": namespace, "error": str(e)}
                        )
                
                # Retry Non-IT index queries with generic embedding
                non_it_results_fallback = []
                for idx, namespace in enumerate(non_it_namespaces):
                    try:
                        namespace_k = per_namespace_k * 2  # Get more results for fallback
                        namespace_results = await self.pinecone_automation.query_vectors(
                            query_vector=generic_embedding,
                            mastercategory="NON_IT",
                            namespace=namespace if namespace else None,
                            top_k=namespace_k,
                            filter_dict=pinecone_filter
                        )
                        non_it_results_fallback.extend(namespace_results)
                    except Exception as e:
                        logger.warning(
                            f"Failed to query Non-IT index namespace '{namespace}' (fallback): {e}",
                            extra={"namespace": namespace, "error": str(e)}
                        )
                
                # Use fallback results if we got any
                if it_results_fallback or non_it_results_fallback:
                    logger.info(
                        f"Fallback search returned {len(it_results_fallback)} IT and {len(non_it_results_fallback)} Non-IT results",
                        extra={
                            "it_results": len(it_results_fallback),
                            "non_it_results": len(non_it_results_fallback)
                        }
                    )
                    all_results = it_results_fallback + non_it_results_fallback
                else:
                    # Final fallback: retry with NO filters (pure semantic search)
                    logger.info(
                        "Fallback with filters returned 0 results. "
                        "Retrying with pure semantic search (no filters).",
                        extra={"original_filters": pinecone_filter}
                    )
                    
                    # Retry IT index with no filters
                    it_results_semantic = []
                    for namespace in it_namespaces:
                        try:
                            namespace_results = await self.pinecone_automation.query_vectors(
                                query_vector=generic_embedding,
                                mastercategory="IT",
                                namespace=namespace if namespace else None,
                                top_k=per_namespace_k * 3,  # Get more for pure semantic
                                filter_dict=None  # NO FILTERS
                            )
                            it_results_semantic.extend(namespace_results)
                        except Exception as e:
                            logger.warning(f"Failed semantic-only search on IT namespace '{namespace}': {e}")
                    
                    # Retry Non-IT index with no filters
                    non_it_results_semantic = []
                    for namespace in non_it_namespaces:
                        try:
                            namespace_results = await self.pinecone_automation.query_vectors(
                                query_vector=generic_embedding,
                                mastercategory="NON_IT",
                                namespace=namespace if namespace else None,
                                top_k=per_namespace_k * 3,  # Get more for pure semantic
                                filter_dict=None  # NO FILTERS
                            )
                            non_it_results_semantic.extend(namespace_results)
                        except Exception as e:
                            logger.warning(f"Failed semantic-only search on Non-IT namespace '{namespace}': {e}")
                    
                    if it_results_semantic or non_it_results_semantic:
                        logger.info(
                            f"Pure semantic search returned {len(it_results_semantic)} IT and {len(non_it_results_semantic)} Non-IT results",
                            extra={
                                "it_results": len(it_results_semantic),
                                "non_it_results": len(non_it_results_semantic)
                            }
                        )
                        all_results = it_results_semantic + non_it_results_semantic
                    else:
                        logger.warning(
                            "All search attempts returned 0 results. "
                            "This may indicate no data is indexed in Pinecone.",
                            extra={"filters": pinecone_filter}
                        )
            
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
                    "domain": metadata.get("domain"),
                    "score": score  # Semantic similarity score from Pinecone
                }
                
                # Calculate relevance score based on filters (soft scoring)
                relevance_score = await self.calculate_relevance_score(candidate, parsed_query)
                
                # Hybrid: blend semantic + keyword overlap + relevance
                text_for_embedding = parsed_query.get("text_for_embedding", "") or ""
                keyword_score = keyword_score_for_candidate(
                    text_for_embedding,
                    skills,
                    candidate.get("designation"),
                    metadata.get("domain"),
                )
                combined_score = self._compute_hybrid_combined_score(
                    score, keyword_score, relevance_score
                )
                normalized_score = max(0.0, min(1.0, combined_score / 100.0))
                
                # Update candidate with normalized score (0-1 range for Pydantic model)
                candidate["score"] = normalized_score
                candidate["semantic_score"] = score  # Keep original semantic score (0-1)
                candidate["relevance_score"] = relevance_score  # Keep relevance score (0-100)
                candidate["keyword_score"] = keyword_score  # Keep keyword score (0-1)
                
                # FIX 1: Hard-gate empty/category results - filter out invalid candidates
                if not candidate.get("candidate_id") or not candidate.get("candidate_id").strip():
                    logger.warning(
                        f"Skipping candidate with missing candidate_id: resume_id={candidate.get('resume_id')}"
                    )
                    continue
                
                if not candidate.get("resume_id"):
                    logger.warning(
                        f"Skipping candidate with missing resume_id: candidate_id={candidate.get('candidate_id')}"
                    )
                    continue
                
                # Also filter candidates with no name
                if not candidate.get("name") or not candidate.get("name").strip():
                    logger.warning(
                        f"Skipping candidate with no name: resume_id={candidate.get('resume_id')}, "
                        f"candidate_id={candidate.get('candidate_id')}"
                    )
                    continue
                
                # Categorize fit tier based on combined score (pass raw 0-200 range)
                fit_tier = self.categorize_fit_tier(candidate, parsed_query, combined_score)
                candidate["fit_tier"] = fit_tier
                
                processed_results.append(candidate)
            
            # OPTIMIZATION for 180k+ resumes: Two-stage processing
            # Stage 1: Sort by semantic score first (fast, no LLM)
            processed_results.sort(key=lambda x: x.get("semantic_score", 0.0), reverse=True)
            
            # Stage 2: LLM designation matching only for top candidates (limit LLM calls)
            # This dramatically reduces LLM calls from 500+ to ~20-50
            designation = parsed_query.get("filters", {}).get("designation")
            if designation:
                # Only call LLM for top 50 candidates (or all if less than 50)
                llm_candidate_limit = min(50, len(processed_results))
                top_candidates_for_llm = processed_results[:llm_candidate_limit]
                
                logger.info(
                    f"Running LLM designation matching for top {llm_candidate_limit} candidates (out of {len(processed_results)} total)",
                    extra={
                        "total_candidates": len(processed_results),
                        "llm_candidate_limit": llm_candidate_limit,
                        "designation": designation
                    }
                )
                
                # Re-score top candidates with LLM if rule-based didn't match
                for candidate in top_candidates_for_llm:
                    candidate_designation = candidate.get("designation", "")
                    candidate_jobrole = candidate.get("jobrole", "")
                    
                    # Only call LLM if rule-based matching didn't find a strong match
                    # Check if we already have a high confidence match from rule-based
                    current_relevance = candidate.get("relevance_score", 0.0)
                    has_strong_match = current_relevance >= 40.0  # Rule-based gives 40-50 for matches
                    
                    if not has_strong_match and (candidate_designation or candidate_jobrole):
                        try:
                            # Call LLM for ambiguous cases
                            cand_role = candidate_designation or candidate_jobrole
                            is_match, confidence = await self.designation_matcher.is_designation_match(
                                query_designation=designation,
                                candidate_designation=cand_role
                            )
                            
                            if is_match:
                                # Recalculate relevance score with LLM match
                                if confidence >= 0.9:
                                    llm_boost = 50.0
                                elif confidence >= 0.7:
                                    llm_boost = 40.0
                                elif confidence >= 0.5:
                                    llm_boost = 30.0
                                else:
                                    llm_boost = 20.0
                                
                                # Update relevance score
                                candidate["relevance_score"] = current_relevance + llm_boost
                                
                                # Recalculate combined score
                                semantic_score = candidate.get("semantic_score", 0.0)
                                keyword_score = candidate.get("keyword_score", 0.0)
                                new_combined_score = self._compute_hybrid_combined_score(
                                    semantic_score,
                                    keyword_score,
                                    candidate["relevance_score"],
                                )
                                candidate["score"] = max(
                                    0.0, min(1.0, new_combined_score / 100.0)
                                )
                                
                                logger.debug(
                                    f"LLM match found: {designation} vs {cand_role}, confidence={confidence}, boost=+{llm_boost}",
                                    extra={
                                        "query_designation": designation,
                                        "candidate_role": cand_role,
                                        "confidence": confidence,
                                        "boost": llm_boost
                                    }
                                )
                        except Exception as e:
                            logger.warning(
                                f"LLM designation matching failed for candidate {candidate.get('candidate_id')}: {e}",
                                extra={"candidate_id": candidate.get("candidate_id"), "error": str(e)}
                            )
                            # Continue with rule-based score
            
            # Rank by final combined score (relevance)
            processed_results.sort(key=lambda x: x["score"], reverse=True)

            # NEW: Optional hard filtering based on mastercategory and exact role
            filters_for_post = parsed_query.get("filters", {})

            # 1) Prefer candidates that match the identified mastercategory (IT / NON_IT)
            # But don't filter out if we have very few results
            if identified_mastercategory and len(processed_results) > 3:
                preferred_by_mc = [
                    c
                    for c in processed_results
                    if (c.get("mastercategory") or "").upper() == identified_mastercategory.upper()
                ]
                # Only narrow if we still have at least 2 results (to avoid over-filtering)
                if len(preferred_by_mc) >= 2:
                    processed_results = preferred_by_mc
                elif len(preferred_by_mc) == 1 and len(processed_results) <= 3:
                    # If we only have a few results total, keep the mastercategory match
                    processed_results = preferred_by_mc

            # 2) If query specifies a recognizable role, prefer only that exact role
            # But be less aggressive - only filter if we have many results
            query_role = filters_for_post.get("designation")
            normalized_query_role = (
                self._normalize_role(query_role) if query_role else None
            )

            if normalized_query_role and len(processed_results) > 5:
                role_matched = []
                for c in processed_results:
                    cand_role_raw = c.get("jobrole") or c.get("designation") or ""
                    normalized_cand_role = (
                        self._normalize_role(cand_role_raw) if cand_role_raw else None
                    )
                    if normalized_cand_role == normalized_query_role:
                        role_matched.append(c)

                # Only narrow to exact-role candidates if we still have at least 2 results
                # This prevents over-filtering when we have few candidates
                if len(role_matched) >= 2:
                    processed_results = role_matched
                elif len(role_matched) == 1 and len(processed_results) <= 5:
                    # If we only have a few results total, keep the role match
                    processed_results = role_matched
            
            # OPTIMIZATION: Early termination if we have enough perfect matches
            perfect_matches = [c for c in processed_results if c.get("fit_tier") == "Perfect Match"]
            if len(perfect_matches) >= top_k:
                logger.info(
                    f"Early termination: Found {len(perfect_matches)} perfect matches, returning top {top_k}",
                    extra={"perfect_matches": len(perfect_matches), "top_k": top_k}
                )
                processed_results = perfect_matches[:top_k]
            else:
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
    
    async def _search_broad_mode(
        self,
        parsed_query: Dict,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Broad search mode: Smart filtering when category/mastercategory not provided.
        Uses role-based and skill-based filtering to reduce namespace queries.
        
        Args:
            parsed_query: Parsed query with filters
            top_k: Number of results to return
        
        Returns:
            List of candidate results with fit tiers
        """
        import asyncio
        
        try:
            # Get text for embedding
            text_for_embedding = parsed_query.get("text_for_embedding", "")
            if not text_for_embedding or not text_for_embedding.strip():
                filters = parsed_query.get("filters", {})
                filter_parts = []
                if filters.get("designation"):
                    filter_parts.append(filters.get("designation"))
                if filters.get("must_have_all"):
                    filter_parts.extend(filters.get("must_have_all", []))
                if filter_parts:
                    text_for_embedding = " ".join(filter_parts)
                else:
                    text_for_embedding = "candidate resume"
            
            # Query expansion for better embedding recall
            text_to_embed = expand_query_for_embedding(text_for_embedding) or text_for_embedding
            
            # Generate embedding
            embedding = await self.embedding_service.generate_embedding(text_to_embed)
            
            # Build Pinecone filter (only query filters, no category)
            pinecone_filter = self.build_pinecone_filter(parsed_query)
            
            # Smart namespace filtering
            namespaces_to_query = self._get_smart_namespaces(parsed_query)
            
            logger.info(
                f"Broad search mode: querying {len(namespaces_to_query)} namespaces using smart filtering",
                extra={
                    "namespace_count": len(namespaces_to_query),
                    "namespaces": namespaces_to_query[:10]  # Log first 10
                }
            )
            
            # Query namespaces in parallel with timeout
            try:
                results = await asyncio.wait_for(
                    self._query_namespaces_parallel(
                        namespaces_to_query,
                        embedding,
                        pinecone_filter,
                        top_k
                    ),
                    timeout=10.0  # 10 second timeout
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Broad search timed out, returning partial results",
                    extra={"namespaces_queried": len(namespaces_to_query)}
                )
                results = []
            
            # Process, deduplicate, and rank results
            processed_results = self._process_broad_search_results(results, parsed_query, top_k)
            
            logger.info(
                f"Broad search completed: {len(processed_results)} results from {len(namespaces_to_query)} namespaces",
                extra={
                    "result_count": len(processed_results),
                    "namespace_count": len(namespaces_to_query)
                }
            )
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Broad search mode failed: {e}", extra={"error": str(e)})
            raise
    
    def _get_smart_namespaces(self, parsed_query: Dict) -> List[tuple]:
        """
        Get namespaces to query using smart filtering.
        Returns list of (mastercategory, namespace) tuples.
        
        Strategy:
        1. Role-based filtering (if designation present)
        2. Skill-based mastercategory inference (if skills present)
        3. Fallback: top 10 most common namespaces
        """
        filters = parsed_query.get("filters", {})
        designation = filters.get("designation", "").lower() if filters.get("designation") else ""
        skills = filters.get("must_have_all", [])
        
        # Get all categories
        it_categories = self.pinecone_automation._get_all_it_categories()
        non_it_categories = self.pinecone_automation._get_all_non_it_categories()
        
        namespaces = []
        
        # Strategy 1: Role-based namespace filtering
        if designation:
            for role_key, role_namespaces in self.ROLE_FAMILY_NAMESPACES.items():
                if role_key in designation:
                    # Map role namespaces to (mastercategory, namespace) tuples
                    for ns in role_namespaces:
                        # Determine mastercategory from namespace
                        normalized_ns = self._normalize_namespace(ns)
                        # Check if it's IT or NON_IT namespace
                        it_normalized = [self._normalize_namespace(c) for c in it_categories]
                        if normalized_ns in it_normalized:
                            namespaces.append(("IT", normalized_ns))
                        else:
                            non_it_normalized = [self._normalize_namespace(c) for c in non_it_categories]
                            if normalized_ns in non_it_normalized:
                                namespaces.append(("NON_IT", normalized_ns))
                    
                    if namespaces:
                        # Deduplicate
                        namespaces = list(set(namespaces))
                        logger.info(
                            f"Role-based filtering: found {len(namespaces)} namespaces for role '{designation}'",
                            extra={"role": designation, "namespaces": namespaces}
                        )
                        return namespaces
        
        # Strategy 2: Skill-based mastercategory inference
        if skills:
            likely_mastercategory = self._infer_mastercategory_from_skills(skills)
            if likely_mastercategory:
                categories = it_categories if likely_mastercategory == "IT" else non_it_categories
                # Limit to top 10 most relevant namespaces
                for category in categories[:10]:
                    namespace = self._normalize_namespace(category)
                    namespaces.append((likely_mastercategory, namespace))
                
                logger.info(
                    f"Skill-based filtering: querying {likely_mastercategory} index ({len(namespaces)} namespaces)",
                    extra={"mastercategory": likely_mastercategory, "skills": skills}
                )
                return namespaces
        
        # Strategy 3: Fallback - query top 5 IT + top 5 NON_IT namespaces
        for category in it_categories[:5]:
            namespace = self._normalize_namespace(category)
            namespaces.append(("IT", namespace))
        
        for category in non_it_categories[:5]:
            namespace = self._normalize_namespace(category)
            namespaces.append(("NON_IT", namespace))
        
        logger.info(
            f"Fallback filtering: querying top 10 namespaces (5 IT + 5 NON_IT)",
            extra={"namespaces": namespaces}
        )
        
        return namespaces
    
    def _infer_mastercategory_from_skills(self, skills: List[str]) -> Optional[str]:
        """
        Infer mastercategory (IT/NON_IT) from skills list.
        Returns "IT", "NON_IT", or None if ambiguous.
        """
        it_keywords = [
            "python", "java", "javascript", "react", "angular", "node", "sql", "database",
            "aws", "azure", "gcp", "docker", "kubernetes", "devops", "ci/cd", "git",
            "machine learning", "ai", "data science", "tensorflow", "pytorch",
            "spring", "django", "flask", "express", "mongodb", "postgresql", "mysql"
        ]
        
        non_it_keywords = [
            "accounting", "finance", "hr", "human resources", "marketing", "sales",
            "project management", "business analysis", "scrum", "agile", "pmp"
        ]
        
        skills_lower = [s.lower() for s in skills]
        
        it_score = sum(1 for skill in skills_lower if any(kw in skill for kw in it_keywords))
        non_it_score = sum(1 for skill in skills_lower if any(kw in skill for kw in non_it_keywords))
        
        if it_score > non_it_score and it_score > 0:
            return "IT"
        elif non_it_score > it_score and non_it_score > 0:
            return "NON_IT"
        else:
            return None
    
    async def _query_namespaces_parallel(
        self,
        namespaces: List[tuple],
        embedding: List[float],
        filter_dict: Dict,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Query multiple namespaces in parallel.
        
        Args:
            namespaces: List of (mastercategory, namespace) tuples
            embedding: Query embedding vector
            filter_dict: Pinecone filter dictionary
            top_k: Total results needed (distributed across namespaces)
        
        Returns:
            Combined results from all namespaces
        """
        import asyncio
        
        # Distribute top_k across namespaces
        top_k_per_namespace = max(1, top_k // len(namespaces)) if namespaces else top_k
        
        # Create tasks for parallel execution
        tasks = []
        for mastercategory, namespace in namespaces:
            task = self.pinecone_automation.query_vectors(
                query_vector=embedding,
                mastercategory=mastercategory,
                namespace=namespace,
                top_k=top_k_per_namespace,
                filter_dict=filter_dict
            )
            tasks.append(task)
        
        # Execute all queries in parallel
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results and handle exceptions
        all_results = []
        for i, result in enumerate(results_list):
            if isinstance(result, Exception):
                logger.warning(
                    f"Namespace query failed: {result}",
                    extra={"namespace": namespaces[i], "error": str(result)}
                )
            elif isinstance(result, list):
                all_results.extend(result)
        
        return all_results
    
    def _process_broad_search_results(
        self,
        all_results: List[Dict[str, Any]],
        parsed_query: Dict,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Process, deduplicate, and rank results from broad search.
        """
        # Derive a text representation for keyword scoring
        text_for_embedding = parsed_query.get("text_for_embedding", "") or ""
        if not text_for_embedding.strip():
            filters = parsed_query.get("filters", {})
            filter_parts: List[str] = []
            designation = filters.get("designation")
            if designation:
                filter_parts.append(str(designation))
            must_have_all = filters.get("must_have_all") or []
            if must_have_all:
                filter_parts.extend([str(s) for s in must_have_all])
            if filter_parts:
                text_for_embedding = " ".join(filter_parts)
            else:
                text_for_embedding = "candidate resume"

        # Deduplicate by resume_id
        seen_resume_ids = set()
        unique_results = []
        
        for match in all_results:
            metadata = match.get("metadata", {})
            score = match.get("score", 0.0)
            resume_id = metadata.get("resume_id")
            
            if not resume_id or resume_id in seen_resume_ids:
                continue
            
            seen_resume_ids.add(resume_id)
            
            # Parse skills
            skills = metadata.get("skills", [])
            if isinstance(skills, str):
                skills = [s.strip() for s in skills.split(",") if s.strip()]
            elif not isinstance(skills, list):
                skills = []
            
            # Extract experience_years
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
                "designation": metadata.get("designation", ""),
                "jobrole": metadata.get("jobrole", ""),
                "experience_years": experience_years,
                "skills": skills,
                "location": metadata.get("location"),
                "score": score
            }
            
            # Calculate relevance score (no strict category matching in broad mode)
            relevance_score = self._calculate_relevance_score_sync(
                candidate,
                parsed_query
            )
            
            # Hybrid: blend semantic + keyword overlap + relevance
            keyword_score = keyword_score_for_candidate(
                text_for_embedding,
                skills,
                candidate.get("designation"),
                metadata.get("domain"),
            )
            combined_score = self._compute_hybrid_combined_score(
                score, keyword_score, relevance_score
            )
            normalized_score = max(0.0, min(1.0, combined_score / 100.0))
            
            candidate["score"] = normalized_score
            candidate["semantic_score"] = score
            candidate["relevance_score"] = relevance_score
            candidate["keyword_score"] = keyword_score
            
            # Filter invalid candidates
            if not candidate.get("candidate_id") or not candidate.get("candidate_id").strip():
                continue
            if not candidate.get("resume_id"):
                continue
            if not candidate.get("name") or not candidate.get("name").strip():
                continue
            
            # Categorize fit tier
            fit_tier = self.categorize_fit_tier(candidate, parsed_query, combined_score)
            candidate["fit_tier"] = fit_tier
            
            unique_results.append(candidate)
        
        # Sort by final combined score
        unique_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Limit to top_k
        return unique_results[:top_k]
    
    def _calculate_relevance_score_sync(
        self,
        candidate: Dict,
        parsed_query: Dict
    ) -> float:
        """
        Synchronous version of calculate_relevance_score for broad search.
        Simplified version without strict category matching.
        """
        score = 0.0
        filters = parsed_query.get("filters", {})
        
        # Skills matching (same as before)
        must_have_all = filters.get("must_have_all", [])
        if must_have_all:
            candidate_skills = [s.lower() for s in candidate.get("skills", [])]
            matched_skills = sum(1 for skill in must_have_all if skill.lower() in " ".join(candidate_skills))
            if matched_skills > 0:
                skill_match_ratio = matched_skills / len(must_have_all)
                score += 30.0 * skill_match_ratio
        
        # Experience matching (same as before)
        min_exp = filters.get("min_experience")
        max_exp = filters.get("max_experience")
        candidate_exp = candidate.get("experience_years")
        
        if candidate_exp is not None:
            if min_exp and candidate_exp >= min_exp:
                score += 20.0
            elif min_exp and candidate_exp >= min_exp - 1:
                score += 10.0
            
            if max_exp and candidate_exp <= max_exp:
                score += 10.0
        
        # Designation matching (soft, no strict penalty)
        query_designation = filters.get("designation", "").lower() if filters.get("designation") else ""
        candidate_designation = candidate.get("designation", "").lower() if candidate.get("designation") else ""
        
        if query_designation and candidate_designation:
            if query_designation in candidate_designation or candidate_designation in query_designation:
                score += 15.0
        
        return min(100.0, max(0.0, score))