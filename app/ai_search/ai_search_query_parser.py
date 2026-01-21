"""Query parser for AI search using OLLAMA LLM."""
import json
import re
from typing import Dict, Optional
import httpx
from httpx import Timeout

from app.config import settings
from app.utils.logging import get_logger
from app.ai_search.query_category_identifier import QueryCategoryIdentifier

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

INSTRUCTIONS:
- Detect job role/designation (e.g., "QA Automation Engineer", "Software Engineer", "Business Analyst"), skills, boolean logic (AND / OR), experience, location, and person name.
- If the query appears to be a job role/designation (job title), extract it as designation. Common patterns: "X Engineer", "X Developer", "X Manager", "X Analyst", etc.
- Designation and skills can BOTH exist - they are not mutually exclusive.
- Normalize skill names and designation to lowercase.
- If a person name is detected, treat it as a name search. If query contains only person-like tokens (2-3 words) and no skills/roles, classify as name search. Support partial names and phonetic variations.
- If experience is mentioned, extract minimum experience in years. Support ranges like "5-7 years" → min_experience: 5, max_experience: 7.
- If location is mentioned, extract city or place (normalize to lowercase).
- Preserve logical intent.
- If no filters are detected, use semantic-only search.

SEMANTIC TEXT (VERY IMPORTANT - CRITICAL):
- ALWAYS include role, skills, AND experience in text_for_embedding.
- If experience is mentioned, text_for_embedding MUST include it (e.g., "software engineer with 5 years experience").
- NEVER return empty text_for_embedding.
- Do NOT remove role, skills, or experience from semantic text - they help semantic search.
- Keep the full query context for better semantic understanding.
- Example: "Software Engineer with 5 years" → text_for_embedding: "software engineer with 5 years experience"

BOOLEAN LOGIC HANDLING:
- Parentheses group OR conditions: ("A" OR "B")
- AND connects different groups: (A OR B) AND C AND (D OR E)
- Terms without parentheses are AND conditions: A AND B AND C
- Example: (python AND django) OR (java OR spring boot)
  → must_have_one_of_groups: [["python", "django"], ["java", "spring boot"]]

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
  },
  "mastercategory": null,
  "category": null
}

DO NOT:
- Do not explain anything
- Do not add extra keys
- Do not return text outside JSON
- Do not invent or assume skills, experience, or qualifications
"""


class AISearchQueryParser:
    """Service for parsing search queries using OLLAMA LLM."""
    
    def __init__(self):
        self.ollama_host = settings.ollama_host
        self.model = "llama3.1"
        self.category_identifier = QueryCategoryIdentifier()
    
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
        
        # Ensure category fields exist
        if "mastercategory" not in parsed:
            parsed["mastercategory"] = None
        if "category" not in parsed:
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
    
    async def parse_query(self, query: str) -> Dict:
        """
        Parse natural language search query into structured format.
        
        Args:
            query: Natural language search query
        
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
        
        # Try to identify category from query (optional, non-blocking with timeout)
        # Use asyncio.wait_for to prevent category identification from blocking too long
        try:
            import asyncio
            # Set a timeout of 30 seconds for category identification
            # If it takes longer, skip it and proceed without category
            mastercategory, category = await asyncio.wait_for(
                self.category_identifier.identify_category_from_query(query),
                timeout=30.0
            )
            if mastercategory:
                parsed_data["mastercategory"] = mastercategory
                logger.info(
                    f"Identified mastercategory from query: {mastercategory}",
                    extra={"query": query, "mastercategory": mastercategory}
                )
            if category:
                parsed_data["category"] = category
                logger.info(
                    f"Identified category from query: {category}",
                    extra={"query": query, "mastercategory": mastercategory, "category": category}
                )
        except asyncio.TimeoutError:
            # Category identification took too long, skip it
            logger.warning(
                f"Category identification timed out after 30 seconds, proceeding without category",
                extra={"query": query}
            )
        except Exception as e:
            # Category identification is optional, don't fail the whole query parsing
            logger.warning(
                f"Category identification failed (non-blocking): {e}",
                extra={"query": query, "error": str(e)}
            )
        
        # NEW APPROACH: Infer mastercategory from query if LLM failed
        if not parsed_data.get("mastercategory"):
            inferred_mastercategory = self._infer_mastercategory_from_query(query, parsed_data)
            if inferred_mastercategory:
                parsed_data["mastercategory"] = inferred_mastercategory
                logger.info(
                    f"Inferred mastercategory from query: {inferred_mastercategory}",
                    extra={"query": query, "mastercategory": inferred_mastercategory, "source": "inferred"}
                )
        
        # OPTIMIZATION: Hard keyword fallback for role-based queries (for 180k+ resumes)
        # This ensures role queries never stay null, improving namespace targeting
        if not parsed_data.get("mastercategory"):
            query_l = query.lower()
            designation = (parsed_data.get("filters", {}).get("designation") or "").lower()
            combined_role_text = f"{query_l} {designation}".lower()
            
            # NON-IT roles (explicit list for common roles)
            non_it_roles = [
                "scrum master", "scrummaster", "agile scrum master",
                "project manager", "program manager", "product owner",
                "business analyst", "change manager", "organizational change manager",
                "procurement manager", "vendor manager", "sourcing manager"
            ]
            
            # IT roles (explicit list)
            it_roles = [
                "developer", "engineer", "programmer", "architect",
                "qa", "automation", "selenium", "sdet", "test automation",
                "devops", "sre", "data engineer", "data scientist",
                "software", "backend", "frontend", "full stack", "fullstack"
            ]
            
            # Check for explicit role matches first (fastest path)
            if any(role in combined_role_text for role in non_it_roles):
                parsed_data["mastercategory"] = "NON_IT"
                # Set default category for common roles
                if "scrum" in combined_role_text:
                    parsed_data["category"] = "Business & Management"
                elif "project" in combined_role_text or "program" in combined_role_text:
                    parsed_data["category"] = "Project Management (Non-IT)"
                elif "change" in combined_role_text:
                    parsed_data["category"] = "Business & Management"
                logger.info(
                    f"Mastercategory inferred via role keyword: NON_IT",
                    extra={"query": query, "role_match": "non_it_role", "source": "role_keyword"}
                )
            elif any(role in combined_role_text for role in it_roles):
                parsed_data["mastercategory"] = "IT"
                logger.info(
                    f"Mastercategory inferred via role keyword: IT",
                    extra={"query": query, "role_match": "it_role", "source": "role_keyword"}
                )
        
        logger.info(
            f"Query parsed successfully: search_type={parsed_data['search_type']}",
            extra={
                "search_type": parsed_data["search_type"],
                "has_filters": bool(parsed_data["filters"]),
                "mastercategory": parsed_data.get("mastercategory"),
                "category": parsed_data.get("category")
            }
        )
        
        return parsed_data
