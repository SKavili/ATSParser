"""Query parser for AI search using OLLAMA LLM."""
import json
import re
from typing import Dict, Optional
import httpx
from httpx import Timeout

from app.config import settings
from app.utils.logging import get_logger
# QueryCategoryIdentifier removed - category is now provided explicitly in payload

logger = get_logger(__name__)

# Try to import OLLAMA Python client
try:
    import ollama
    OLLAMA_CLIENT_AVAILABLE = True
except ImportError:
    OLLAMA_CLIENT_AVAILABLE = False
    logger.warning("OLLAMA Python client not available, using HTTP API directly")

AI_SEARCH_PROMPT = """
IMPORTANT:
This is a FRESH, ISOLATED, SINGLE-TASK operation.
Ignore all previous instructions, memory, or conversations.
 
ROLE:
You are an ATS AI search query parser used in a resume search system.
 
TASK:
Convert the user's natural language search query into a structured search intent
that can be used for candidate filtering and semantic search.
 
EXTRACTION PRIORITY (highest → lowest):
1. Name search detection (if person name detected)
   - If query contains only person-like tokens (2-3 words) and no skills/roles
   - Example: "John Smith" → name search
   
2. Designation / Role
   - Extract job title/designation first
   - Example: "Python Developer" → designation="python developer"
   
3. Skills
   - Extract after designation is identified
   - Example: "Python Developer with Django" → designation="python developer", skill="django"
   
4. Experience
   - Extract numeric experience values
   - Example: "Python Developer with 5 years" → min_experience: 5
   
5. Location
   - Extract location last (optional preference)
   - Example: "Python Developer in Bangalore" → location="bangalore"
 
PRIORITY RULES:
- If name detected AND no skills/roles → name search (stop here)
- If designation detected → extract it first, then skills, then experience, then location
- If only skills detected → extract skills, then experience, then location
- If ambiguous (e.g., "John Python Developer"), check if "John" is a name or part of designation
 
INSTRUCTIONS:
- Detect job role/designation (e.g., "QA Automation Engineer", "Software Engineer", "Business Analyst"), skills, boolean logic (AND / OR), experience, location, and person name.
- If the query appears to be a job role/designation (job title), extract it as designation. Common patterns: "X Engineer", "X Developer", "X Manager", "X Analyst", etc.
- Designation and skills can BOTH exist - they are not mutually exclusive.
- Normalize skill names and designation to lowercase.
 
SKILLS EXTRACTION:
- Skills may be separated by commas, spaces, "and", or "or"
- Treat space-separated technical terms as individual skills
- Example: "python django postgres" → must_have_all: ["python", "django", "postgres"]
- Example: "Python, Django, PostgreSQL" → must_have_all: ["python", "django", "postgresql"]
- Example: "Java and Spring Boot" → must_have_all: ["java", "spring boot"]
 
- If a person name is detected, treat it as a name search. If query contains only person-like tokens (2-3 words) and no skills/roles, classify as name search. Support partial names and phonetic variations.
- If experience is mentioned, extract minimum experience in years. Support ranges like "5-7 years" → min_experience: 5, max_experience: 7.
 
EXPERIENCE EXTRACTION PATTERNS (STRICT):
- "with X years" → min_experience: X
- "having X years" → min_experience: X
- "X years experience" → min_experience: X
- "X years of experience" → min_experience: X
- "minimum X years" → min_experience: X
- "at least X years" → min_experience: X
- "X+ years" → min_experience: X
- "X-Y years" → min_experience: X, max_experience: Y
- "between X and Y years" → min_experience: X, max_experience: Y
- "X to Y years" → min_experience: X, max_experience: Y
 
LOCATION HANDLING:
- If location is mentioned, extract city or place (normalize to lowercase)
- Support multiple locations using OR
- Example: "Bangalore OR Hyderabad" → location: ["bangalore", "hyderabad"]
- Example: "in Bangalore" → location: "bangalore"
- If single location, use string. If multiple locations, use array.
- Preserve logical intent.
- If no filters are detected, use semantic-only search.
 
SEMANTIC TEXT (VERY IMPORTANT - CRITICAL):
- ALWAYS include role, skills, AND experience in text_for_embedding.
- If experience is mentioned, text_for_embedding MUST include it (e.g., "software engineer with 5 years experience").
- NEVER return empty text_for_embedding.
- Do NOT remove role, skills, or experience from semantic text - they help semantic search.
- Keep the full query context for better semantic understanding.
- Example: "Software Engineer with 5 years" → text_for_embedding: "software engineer with 5 years experience"
 
SEMANTIC TEXT ORDER (STRICT - CRITICAL):
- ALWAYS use this exact order: designation → skills → experience → location
- This ensures embedding consistency for Pinecone/FAISS search
- Do NOT vary the order - it affects search accuracy
 
EXAMPLES:
- "Software Engineer with 5 years"
  → text_for_embedding: "software engineer 5 years experience"
  (order: designation → experience)
 
- "Python Developer with Django having 5 years in Bangalore"
  → text_for_embedding: "python developer django 5 years experience bangalore"
  (order: designation → skills → experience → location)
 
- "QA Engineer selenium 5 years"
  → text_for_embedding: "qa engineer selenium 5 years experience"
  (order: designation → skills → experience)
 
BOOLEAN LOGIC HANDLING:
- Parentheses group OR conditions: ("A" OR "B")
- AND connects different groups: (A OR B) AND C AND (D OR E)
- Terms without parentheses are AND conditions: A AND B AND C
- Example: (python AND django) OR (java OR spring boot)
  → must_have_one_of_groups: [["python", "django"], ["java", "spring boot"]]
 
BOOLEAN OUTPUT RULES (CRITICAL):
- Each OR option MUST be its own group
- Example: "Python OR Java"
  → must_have_one_of_groups: [["python"], ["java"]]
  (NOT [["python", "java"]] which means python AND java)
- Example: "React OR Angular OR Vue"
  → must_have_one_of_groups: [["react"], ["angular"], ["vue"]]
- Example: "(Python AND Django) OR (Java AND Spring)"
  → must_have_one_of_groups: [["python", "django"], ["java", "spring"]]
 
SEARCH TYPE RULES:
- name → when name search detected and no role/skills
  Example: "John Smith" → search_type: "name"
- semantic → when only free-text or filters are weak
  Example: "Python Developer" → search_type: "semantic"
- hybrid → when BOTH semantic text AND strong filters are present
  Example: "Python Developer with Django having 5 years" → search_type: "hybrid"
  (has designation + skills + experience = strong filters + semantic text)
 
OUTPUT FORMAT (STRICT JSON ONLY):
{
  "search_type": "semantic | name | hybrid",
  "text_for_embedding": "",
  "filters": {
    "designation": null,
    "must_have_all": [],
    "must_have_one_of_groups": [],
    "min_experience": null,
    "max_experience": null,
    "location": null,
    "candidate_name": null
  }
}
 
DO NOT:
- Do not explain anything
- Do not add extra keys
- Do not return text outside JSON
- Do not invent or assume skills, experience, or qualifications
 
DO NOT GUESS (STRICT - CRITICAL):
- Do NOT infer experience numbers unless explicitly stated
  Example: "Senior Developer" → min_experience: null (NOT 5 years)
 
- Do NOT infer skills from company names or context
  Example: "Google Engineer" → designation="engineer" only (NOT skills=["google cloud", "kubernetes"])
 
- Do NOT infer seniority/experience from job titles alone
  Example: "Senior Python Developer" → designation="senior python developer" (NOT min_experience: 5)
 
- Do NOT extract skills that are part of job title
  Example: "Python Developer" → designation="python developer" (NOT skill="python" + designation="developer")
 
- Only extract what is EXPLICITLY mentioned in the query
- If unsure, leave it as null - do NOT guess
"""


class AISearchQueryParser:
    """Service for parsing search queries using OLLAMA LLM."""
    
    def __init__(self):
        self.ollama_host = settings.ollama_host
        self.model = "llama3.1"
        # QueryCategoryIdentifier removed - category is now provided explicitly in payload
    
    async def _check_ollama_connection(self) -> tuple[bool, Optional[str]]:
        """Check if OLLAMA is accessible and running. Returns (is_connected, available_model)."""
        try:
            async with httpx.AsyncClient(timeout=Timeout(5.0)) as client:
                response = await client.get(f"{self.ollama_host}/api/tags")
                if response.status_code == 200:
                    models_data = response.json()
                    models = models_data.get("models", [])
                    for model in models:
                        model_name = model.get("name", "")
                        if "llama3.1" in model_name.lower() or "llama3" in model_name.lower():
                            return True, model_name
                    if models:
                        return True, models[0].get("name", "")
                    return True, None
                return False, None
        except Exception as e:
            logger.warning(f"Failed to check OLLAMA connection: {e}", extra={"error": str(e)})
            return False, None
    
    def _extract_json(self, text: str) -> Dict:
        """Extract JSON object from LLM response."""
        if not text:
            logger.warning("Empty response from LLM")
            return self._default_response()
        
        # Clean the text - remove markdown code blocks if present
        cleaned_text = text.strip()
        cleaned_text = re.sub(r'```json\s*', '', cleaned_text)
        cleaned_text = re.sub(r'```\s*', '', cleaned_text)
        
        # Try to find JSON object
        json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                return self._validate_response(parsed)
            except json.JSONDecodeError:
                pass
        
        # Try parsing the entire cleaned text
        try:
            parsed = json.loads(cleaned_text)
            return self._validate_response(parsed)
        except json.JSONDecodeError:
            logger.warning(
                f"Failed to parse JSON from LLM response. Response preview: {cleaned_text[:500]}"
            )
            return self._default_response()
    
    def _validate_response(self, parsed: Dict) -> Dict:
        """Validate and normalize parsed response."""
        # Ensure required fields exist
        if "search_type" not in parsed:
            parsed["search_type"] = "semantic"
        
        if "text_for_embedding" not in parsed:
            parsed["text_for_embedding"] = ""
        
        if "filters" not in parsed:
            parsed["filters"] = {}
        
        # Ensure filter fields exist
        filters = parsed["filters"]
        if "designation" not in filters:
            filters["designation"] = None
        if "must_have_all" not in filters:
            filters["must_have_all"] = []
        if "must_have_one_of_groups" not in filters:
            filters["must_have_one_of_groups"] = []
        if "min_experience" not in filters:
            filters["min_experience"] = None
        if "max_experience" not in filters:
            filters["max_experience"] = None
        if "location" not in filters:
            filters["location"] = None
        if "candidate_name" not in filters:
            filters["candidate_name"] = None
        
        # Category fields are not part of LLM output - they are provided explicitly in payload
        # Set to None (will be overridden by controller with explicit values)
            parsed["mastercategory"] = None
            parsed["category"] = None
        
        # Normalize search_type
        search_type = parsed["search_type"].lower()
        if search_type not in ["semantic", "name", "hybrid"]:
            parsed["search_type"] = "semantic"
        
        # Normalize designation: convert list to string if needed, then lowercase
        if filters["designation"]:
            if isinstance(filters["designation"], list):
                # If LLM returns a list, take the first element
                filters["designation"] = filters["designation"][0] if filters["designation"] else None
            if filters["designation"]:
                filters["designation"] = str(filters["designation"]).lower().strip()
                # Set to None if empty string after normalization
                if not filters["designation"]:
                    filters["designation"] = None
        
        # Normalize location to lowercase if present
        if filters["location"]:
            filters["location"] = str(filters["location"]).lower().strip()
        
        # Normalize candidate_name: convert list to string if needed, then strip
        if filters["candidate_name"]:
            if isinstance(filters["candidate_name"], list):
                # If LLM returns a list, take the first element
                filters["candidate_name"] = filters["candidate_name"][0] if filters["candidate_name"] else None
            if filters["candidate_name"]:
                filters["candidate_name"] = str(filters["candidate_name"]).strip()
                # Set to None if empty string after normalization
                if not filters["candidate_name"]:
                    filters["candidate_name"] = None
        
        return parsed
    
    def _default_response(self) -> Dict:
        """Return default response structure."""
        return {
            "search_type": "semantic",
            "text_for_embedding": "",
            "filters": {
                "designation": None,
                "must_have_all": [],
                "must_have_one_of_groups": [],
                "min_experience": None,
                "max_experience": None,
                "location": None,
                "candidate_name": None
            },
            # Category fields are not part of LLM output - set to None (will be overridden by controller)
            "mastercategory": None,
            "category": None
        }
    
    def _infer_mastercategory_from_query(self, query: str, parsed_data: Dict) -> Optional[str]:
        """
        Infer mastercategory from query when LLM identification fails.
        Uses keyword-based heuristics for common IT/NON-IT indicators.
        
        Args:
            query: Original search query
            parsed_data: Parsed query data (may contain designation, skills, etc.)
        
        Returns:
            "IT" or "NON_IT" or None if cannot infer
        """
        query_lower = query.lower()
        designation = (parsed_data.get("filters", {}).get("designation") or "").lower()
        text_for_embedding = parsed_data.get("text_for_embedding", "").lower()
        
        # Combine all text for analysis
        combined_text = f"{query_lower} {designation} {text_for_embedding}".lower()
        
        # IT domain indicators (strong signals)
        it_keywords = [
            # Job titles
            "engineer", "developer", "programmer", "architect", "qa", "automation",
            "devops", "sre", "sdet", "sde", "data engineer", "data scientist",
            "software", "backend", "frontend", "full stack", "fullstack",
            # Technologies
            "python", "java", "javascript", "typescript", "c#", "c++", "go", "rust",
            "sql", "database", "api", "microservices", "kubernetes", "docker",
            "aws", "azure", "gcp", "cloud", "ai", "ml", "machine learning",
            "selenium", "testing", "test automation", "ci/cd", "jenkins", "git"
        ]
        
        # NON-IT domain indicators (strong signals)
        non_it_keywords = [
            # Job titles (generic - check context)
            "manager", "director", "executive", "consultant",
            "sales", "marketing", "hr", "human resources", "recruiter",
            "finance", "accounting", "accountant", "cfo", "controller",
            "operations", "supply chain", "procurement", "vendor",
            # Functions
            "business development", "customer service", "support"
        ]
        
        # Context-aware NON-IT keywords (only if IT context is weak)
        if "it" not in combined_text and "software" not in combined_text:
            # Add context-specific NON-IT keywords
            if "business" in combined_text and "analyst" in combined_text:
                non_it_keywords.append("business analyst")
            if "project" in combined_text and "manager" in combined_text:
                non_it_keywords.append("project manager")
        
        # Count IT indicators
        it_count = sum(1 for keyword in it_keywords if keyword in combined_text)
        
        # Count NON-IT indicators (but exclude if IT indicators are strong)
        non_it_count = 0
        if it_count < 2:  # Only count NON-IT if IT signals are weak
            non_it_count = sum(1 for keyword in non_it_keywords if keyword in combined_text)
        
        # Decision logic
        if it_count >= 2:
            return "IT"
        elif non_it_count >= 2:
            return "NON_IT"
        elif it_count == 1:
            # Single IT indicator - likely IT
            return "IT"
        else:
            # Cannot infer - return None
            return None
    
    async def parse_query(self, query: str, skip_category_inference: bool = False) -> Dict:
        """
        Parse natural language search query into structured format.
        
        Args:
            query: Natural language search query
            skip_category_inference: If True, skip category identification (use explicit category from payload)
        
        Returns:
            Dict with structured search intent
        
        Raises:
            RuntimeError: If OLLAMA is not available or parsing fails
        """
        # Check OLLAMA connection
        is_connected, available_model = await self._check_ollama_connection()
        if not is_connected:
            raise RuntimeError(
                f"OLLAMA is not accessible at {self.ollama_host}. "
                "Please ensure OLLAMA is running."
            )
        
        # Use available model if llama3.1 not found
        model_to_use = self.model
        if available_model and "llama3.1" not in available_model.lower():
            logger.warning(f"llama3.1 not found, using available model: {available_model}")
            model_to_use = available_model
        
        # Prepare prompt
        full_prompt = f"{AI_SEARCH_PROMPT}\n\nInput Query: {query}\n\nOutput:"
        
        # Try using OLLAMA Python client first
        result = None
        last_error = None
        
        if OLLAMA_CLIENT_AVAILABLE:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                
                def _generate():
                    client = ollama.Client(
                        host=self.ollama_host.replace("http://", "").replace("https://", "")
                    )
                    response = client.generate(
                        model=model_to_use,
                        prompt=full_prompt,
                        options={
                            "temperature": 0.1,
                            "top_p": 0.9,
                        }
                    )
                    return {"response": response.get("response", "")}
                
                result = await loop.run_in_executor(None, _generate)
                logger.debug("Successfully used OLLAMA Python client for query parsing")
            except Exception as e:
                logger.warning(f"OLLAMA Python client failed, falling back to HTTP API: {e}")
                result = None
        
        # Fallback to HTTP API
        if result is None:
            async with httpx.AsyncClient(timeout=Timeout(300.0)) as client:
                # Try /api/generate endpoint
                try:
                    response = await client.post(
                        f"{self.ollama_host}/api/generate",
                        json={
                            "model": model_to_use,
                            "prompt": full_prompt,
                            "stream": False,
                            "options": {
                                "temperature": 0.1,
                                "top_p": 0.9,
                            }
                        }
                    )
                    response.raise_for_status()
                    result = response.json()
                    logger.debug("Successfully used /api/generate endpoint for query parsing")
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 404:
                        # Try /api/chat endpoint
                        logger.debug("OLLAMA /api/generate not found, trying /api/chat endpoint")
                        try:
                            response = await client.post(
                                f"{self.ollama_host}/api/chat",
                                json={
                                    "model": model_to_use,
                                    "messages": [
                                        {"role": "system", "content": AI_SEARCH_PROMPT},
                                        {"role": "user", "content": query}
                                    ],
                                    "stream": False,
                                    "options": {
                                        "temperature": 0.1,
                                        "top_p": 0.9,
                                    }
                                }
                            )
                            response.raise_for_status()
                            result = response.json()
                            if "message" in result and "content" in result["message"]:
                                result = {"response": result["message"]["content"]}
                            else:
                                raise ValueError("Unexpected response format from OLLAMA chat API")
                            logger.debug("Successfully used /api/chat endpoint for query parsing")
                        except Exception as e2:
                            last_error = e2
                            logger.error(f"OLLAMA /api/chat also failed: {e2}")
                    else:
                        raise
        
        if result is None:
            raise RuntimeError(
                f"All OLLAMA API endpoints failed. "
                f"OLLAMA is running at {self.ollama_host} but endpoints return errors. "
                f"Last error: {last_error}"
            )
        
        # Extract JSON from response
        raw_output = result.get("response", "")
        parsed_data = self._extract_json(raw_output)
        
        # Retry once if parsing failed
        if parsed_data == self._default_response() and raw_output:
            logger.warning("Initial JSON parsing failed, retrying with cleaned text")
            # Try to fix common JSON issues
            cleaned = re.sub(r'[^\{\}\[\]",:\s\w]', '', raw_output)
            try:
                parsed_data = json.loads(cleaned)
                parsed_data = self._validate_response(parsed_data)
            except:
                pass
        
        # Skip category identification if flag is set (explicit category provided in payload)
        if skip_category_inference:
            logger.debug("Skipping category inference (explicit category provided in payload)")
        else:
            # Category identification removed - always use explicit category from payload
            logger.debug("Category inference is disabled - using explicit category from payload")
        
        # Do not set mastercategory or category - they will be provided from payload
        parsed_data["mastercategory"] = None
        parsed_data["category"] = None
        
        logger.info(
            f"Query parsed successfully: search_type={parsed_data['search_type']}",
            extra={
                "search_type": parsed_data["search_type"],
                "has_filters": bool(parsed_data["filters"]),
                "skip_category_inference": skip_category_inference
            }
        )
        
        return parsed_data
