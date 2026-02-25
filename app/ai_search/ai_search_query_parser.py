"""Query parser for AI search using OLLAMA or OpenAI LLM (via LLM_MODEL)."""
import json
import re
from typing import Dict, Optional
import httpx
from httpx import Timeout

from app.config import settings
from app.utils.logging import get_logger
from app.services.llm_service import use_openai, generate as openai_generate
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
You are an ATS query parser for recruiters. Your ONLY job is to convert the EXACT text of the natural language or boolean query into structured JSON. You must be extremely literal and conservative.

CORE RULES – YOU MUST FOLLOW THESE STRICTLY:

1. Extract ONLY what is EXPLICITLY WRITTEN in the query.
   - Never assume, infer, guess, expand, normalize meaning, or add anything that is not directly in the text.
   - Never use knowledge from examples to decide what something "probably means".
   - Never decide that something is a "domain", "company", "acronym", "location", etc. unless it is very clearly used that way in the sentence.

2. Designation (job title / role)
   - Set designation for ANY phrase that is clearly a job title or role, including in AND queries.
   - Do NOT split a job title into skills. Example: "Python Developer" → designation = "python developer" (NOT skills = ["python"])
   - "Senior QA Automation Engineer" → designation = "senior qa automation engineer"
   - In "X AND Y" style: if one term is a role/title (e.g. QA, PM, Developer, Manager, Tester, Analyst) and another is a skill/tech, put the ROLE in designation and the SKILL in must_have_all.
   - Examples: "Python AND QA" → designation = "qa", must_have_all = ["python"]. "sql AND python full stack developer" → designation = "python full stack developer", must_have_all = ["sql"]
   - Single-token roles: "QA", "PM", "BA", "Dev", "Tester", "Manager", "Analyst" etc. in AND with skills → set as designation, NOT in must_have_all.
   - Multi-word role phrases (e.g. "python full stack developer", "front end python developer") → always designation, never split into skills.

3. Skills / must_have_all
   - Only include words or short phrases that are clearly listed as required skills/technologies/tools.
   - Do NOT interpret nouns as skills unless they are in a clear skills list context.
   - "in pharmaceutical domain" → do NOT add "pharmaceutical domain" to skills unless the query explicitly says it is a skill/requirement.

4. Experience
   - ONLY extract when there are clear numeric patterns:
     - "15+ years", "5 years", "5-8 years", "at least 10 years", "more than 7 years"
   - Do NOT infer experience from words like: senior, lead, principal, experienced, expert, veteran, etc.

5. Location
   - ONLY when clearly used as a place: "in Bangalore", "located in Texas", "New York"
   - Do NOT treat company names, project names, or product names as locations.

6. Boolean logic – be literal
   - Parse AND / OR / parentheses / quotes exactly as written.
   - "A OR B" → must_have_one_of_groups = [["a"], ["b"]]
   - "(A AND B) OR C" → must_have_one_of_groups = [["a", "b"], ["c"]]
   - Terms without operators → treat as AND → must_have_all
   - Quoted phrases stay together as one term.

7. text_for_embedding – mandatory
   - ALWAYS create it.
   - Use ONLY words that appear in the original query.
   - Order preference: main role phrase → listed skills/tools → experience phrase → location phrase
   - Keep it natural but lowercase.

8. search_type
   - "name" → only when query is clearly just a full name with no other content
   - "hybrid" → when there is clear designation/role + at least one of: numeric experience OR list of skills/tools
   - "semantic" → everything else

9. Absolute prohibitions – YOU MUST NOT:
   - Assume any word is a skill just because it often is in resumes
   - Split job titles into skills
   - Infer experience level from seniority words
   - Decide something is a "domain" or "industry" unless explicitly phrased that way
   - Use meaning or context from the examples below to interpret the current query
   - Add, remove, or rephrase terms that do not appear in the query

Output format – STRICT JSON only – nothing else:

{
  "search_type": "semantic" | "name" | "hybrid",
  "text_for_embedding": "lowercase text built only from query words",
  "filters": {
    "designation": null | "exact lowercase phrase",
    "must_have_all": ["exact lowercase term", "another term"],
    "must_have_one_of_groups": [["term1"], ["term2 and phrase"]],
    "min_experience": null | integer,
    "max_experience": null | integer,
    "location": null | "city" | ["city1", "city2"],
    "candidate_name": null | "full name lowercase"
  }
}

Examples are ONLY for understanding the output format — NEVER use their logic, assumptions or categorizations on a new query.
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
        
        # Heuristic: if designation is missing but we have a single term in must_have_all
        # that is a role (multi-word phrase or known role keyword), treat as designation.
        if not filters["designation"]:
            must_all = filters.get("must_have_all") or []
            if isinstance(must_all, list) and len(must_all) >= 1:
                # Single multi-word term → designation (e.g. "computer technician", "python full stack developer")
                if len(must_all) == 1 and isinstance(must_all[0], str):
                    term = must_all[0].strip().lower()
                    if term and " " in term:
                        filters["designation"] = term
                        filters["must_have_all"] = []
                # Single-token role keyword in must_have_all (e.g. "qa" in ["python", "qa"]) → move "qa" to designation
                else:
                    role_keywords = {"qa", "pm", "ba", "dev", "tester", "analyst", "manager", "developer", "engineer", "sre", "sdet"}
                    role_terms = [t for t in must_all if isinstance(t, str) and t.strip().lower() in role_keywords]
                    skill_terms = [t for t in must_all if isinstance(t, str) and t.strip().lower() not in role_keywords]
                    if len(role_terms) == 1 and skill_terms:
                        filters["designation"] = role_terms[0].strip().lower()
                        filters["must_have_all"] = [s.strip().lower() for s in skill_terms if s.strip()]
                    elif len(role_terms) == 1 and not skill_terms:
                        filters["designation"] = role_terms[0].strip().lower()
                        filters["must_have_all"] = []
        
        # Category fields are not part of LLM output - they are provided explicitly in payload
        parsed["mastercategory"] = None
        parsed["category"] = None
        
        # Normalize search_type
        search_type = parsed["search_type"].lower()
        if search_type not in ["semantic", "name", "hybrid"]:
            parsed["search_type"] = "semantic"
        
        # Normalize designation: convert list to string if needed, then lowercase
        if filters["designation"]:
            designation_value = filters["designation"]
            if isinstance(designation_value, list):
                # If LLM returns a list, take the first element
                designation_value = designation_value[0] if designation_value else None
            if designation_value:
                designation_norm = str(designation_value).lower().strip()
                filters["designation"] = designation_norm or None
        
        # Normalize location: handle both list and string forms and lowercase
        if filters["location"]:
            loc_value = filters["location"]
            if isinstance(loc_value, list):
                # Take the first non-empty element
                primary = None
                for item in loc_value:
                    if item:
                        primary = str(item).lower().strip()
                        if primary:
                            break
                filters["location"] = primary or None
            else:
                loc_norm = str(loc_value).lower().strip()
                filters["location"] = loc_norm or None
        
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
        # Prepare prompt (used for both OpenAI and OLLAMA)
        full_prompt = f"{AI_SEARCH_PROMPT}\n\nInput Query: {query}\n\nOutput:"

        # LLM_MODEL=OpenAI: use OpenAI for query parsing
        if use_openai():
            if not getattr(settings, "openai_api_key", None) or not str(settings.openai_api_key).strip():
                raise RuntimeError(
                    "LLM_MODEL=OpenAI but OPENAI_API_KEY is not set. "
                    "Set OPENAI_API_KEY in your .env file."
                )
            try:
                raw_output = await openai_generate(
                    prompt=f"Input Query: {query}\n\nOutput:",
                    system_prompt=AI_SEARCH_PROMPT,
                    temperature=0.1,
                    timeout_seconds=300.0,
                )
                parsed_data = self._extract_json(raw_output or "")
                if parsed_data == self._default_response() and raw_output:
                    cleaned = re.sub(r'[^\{\}\[\]",:\s\w]', '', raw_output)
                    try:
                        parsed_data = json.loads(cleaned)
                        parsed_data = self._validate_response(parsed_data)
                    except Exception:
                        pass
                parsed_data["mastercategory"] = None
                parsed_data["category"] = None
                logger.info(
                    f"Query parsed successfully (OpenAI): search_type={parsed_data['search_type']}",
                    extra={"search_type": parsed_data["search_type"], "has_filters": bool(parsed_data["filters"])}
                )
                return parsed_data
            except Exception as e:
                logger.error(f"OpenAI query parsing failed: {e}", extra={"error": str(e)})
                raise RuntimeError(f"OpenAI query parsing failed: {e}") from e

        # LLM_MODEL=OLLAMA (default): use OLLAMA
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
