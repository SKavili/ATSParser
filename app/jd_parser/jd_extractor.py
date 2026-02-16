"""Service for extracting structured data from Job Descriptions using LLM."""
import json
import re
from typing import Any, Dict, Optional
import httpx
from httpx import Timeout

from app.config import settings
from app.utils.logging import get_logger
from app.services.llm_service import use_openai, generate as openai_generate
from app.jd_parser.jd_models import ParsedJD

logger = get_logger(__name__)

# Try to import OLLAMA Python client
try:
    import ollama
    OLLAMA_CLIENT_AVAILABLE = True
except ImportError:
    OLLAMA_CLIENT_AVAILABLE = False
    logger.warning("OLLAMA Python client not available, using HTTP API directly")

JD_EXTRACTION_PROMPT = """You are an expert ATS and recruitment data extraction specialist with 15+ years of experience parsing thousands of job descriptions. 
Your sole task is to extract structured hiring requirements **accurately and conservatively** from the provided JD text ONLY. 
Never invent, assume, or add information not explicitly supported by the text.

Core Rules (Strict Contract):
- Output MUST be **valid JSON only** — no explanations, no markdown, no comments, no extra text, no apologies, no prefixes/suffixes.
- If a field has no clear evidence in the JD → return null (except inferred_mastercategory, which is always required) or empty array [] for lists.
- Be conservative: err on the side of null/[] rather than guessing.
- Normalize terms: lowercase, no punctuation, standard spelling (e.g., "postgresql" not "PostgreSQL", "react" not "React.js").
- Handle negations: phrases like "no Java", "without experience in X", "not required" → do NOT put them in must_have_skills.
- If uncertain about any field (especially category), set low confidence and prefer null.
- For inferred_mastercategory: always choose "IT" or "Non-IT" based on dominant keywords (never null).
- Arrays (skills, education, other_requirements) MUST be [] if empty — never null.

### Extraction Guidelines (follow exactly):

1. designation: Primary job title (most prominent/senior one). Normalize to lowercase (e.g., "senior python backend developer").
2. must_have_skills: ONLY explicitly required skills ("must have", "required", "essential", "mandatory", "need", "proficiency in"). If "or" alternatives in required context → include both/all as separate items.
3. nice_to_have_skills: Preferred/optional skills ("preferred", "nice to have", "advantage", "good to have", "plus", "desirable").
4. min_experience_years & max_experience_years: Numeric years only. Map qualifiers:
   - "freshers", "entry-level", "0–1 year" → min=0
   - "3+ years" → min=3, max=null
   - "3–5 years" → min=3, max=5
   - Vague ("experienced", "senior-level" without years) → null
5. education_requirements: ONLY explicitly required degrees/certifications (e.g., ["b.tech", "mba finance"]). Normalize to lowercase.
6. location: Explicit location if mentioned. Null otherwise.
7. location_type: "strict" (must be based in, onsite only), "preferred" (preferred location), "remote" (remote ok), null.
8. domain_focus: Short functional domain (e.g., "backend development", "digital marketing"). Null if unclear.
9. inferred_mastercategory: "IT" if software/dev/data/cloud/QA/ML roles dominant; "Non-IT" otherwise (sales/HR/finance/operations/healthcare/education). Always set.
10. inferred_category: Snake_case granular namespace hint (e.g., "backend_development_python", "qa_automation", "digital_marketing", "human_resources"). Null if confidence low.
11. inferred_category_confidence: 0.0–1.0 (how certain you are of the category).
12. other_requirements: Soft skills / non-tech (e.g., ["communication", "teamwork"]). [] if none.
13. text_for_embedding: Concise keyword summary: "designation + must-have skills + domain + experience level". No full sentences.

### Think step-by-step internally (do NOT output thinking):
- Read entire JD carefully.
- Identify explicit vs implicit requirements.
- Classify skills strictly by wording.
- Determine IT/Non-IT first (keyword dominance).
- Assign category only if clear match to common namespaces.
- Check for contradictions/negations.

### Few-Shot Examples (follow this style exactly):

Example 1:
JD: "We are looking for a Python Backend Developer with 3+ years experience. Must have Python, Django, REST APIs. Preferred: PostgreSQL, AWS. Freshers not considered."

Output:
{
  "designation": "python backend developer",
  "must_have_skills": ["python", "django", "rest apis"],
  "nice_to_have_skills": ["postgresql", "aws"],
  "min_experience_years": 3,
  "max_experience_years": null,
  "education_requirements": [],
  "domain_focus": "backend development",
  "inferred_mastercategory": "IT",
  "inferred_category": "backend_development_python",
  "inferred_category_confidence": 0.92,
  "location": null,
  "location_type": null,
  "other_requirements": [],
  "text_for_embedding": "python backend developer python django rest apis 3+ years backend development"
}

Example 2:
JD: "Sales Manager role. 5 years sales experience preferred. Good communication skills. Remote possible."

Output:
{
  "designation": "sales manager",
  "must_have_skills": [],
  "nice_to_have_skills": ["communication"],
  "min_experience_years": null,
  "max_experience_years": null,
  "education_requirements": [],
  "domain_focus": "sales",
  "inferred_mastercategory": "Non-IT",
  "inferred_category": "sales_management",
  "inferred_category_confidence": 0.85,
  "location": null,
  "location_type": "remote",
  "other_requirements": [],
  "text_for_embedding": "sales manager sales 5 years preferred sales"
}

Now analyze this Job Description:

### Job Description:
{JD_TEXT}

Return ONLY the valid JSON object matching the schema above. No other text."""


class JDExtractor:
    """Service for extracting structured data from Job Descriptions using LLM."""
    
    def __init__(self):
        self.ollama_host = settings.ollama_host
        self.model = "llama3.1"
    
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
            return {}
        
        # Clean the text - remove markdown code blocks if present
        cleaned_text = text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()
        
        # Remove any leading/trailing text before/after JSON
        # Find the first { and last }
        start_idx = cleaned_text.find('{')
        end_idx = cleaned_text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            cleaned_text = cleaned_text[start_idx:end_idx + 1]
        
        # Try parsing the cleaned text
        try:
            parsed = json.loads(cleaned_text)
            if isinstance(parsed, dict):
                logger.debug(f"Successfully extracted JSON: {parsed}")
                return parsed
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON with balanced braces
        try:
            start_idx = cleaned_text.find('{')
            if start_idx != -1:
                brace_count = 0
                end_idx = start_idx
                for i in range(start_idx, len(cleaned_text)):
                    if cleaned_text[i] == '{':
                        brace_count += 1
                    elif cleaned_text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                if brace_count == 0:
                    json_str = cleaned_text[start_idx:end_idx]
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict):
                        logger.debug(f"Successfully extracted JSON with balanced braces: {parsed}")
                        return parsed
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON with balanced braces: {e}")
        
        # If all parsing fails, log the issue
        logger.error(
            "ERROR: Failed to parse JSON from LLM response", 
            extra={
                "response_preview": text[:500],
                "response_length": len(text),
                "cleaned_preview": cleaned_text[:500]
            }
        )
        return {}
    
    def _normalize_location_type(self, value: Any) -> Optional[str]:
        """
        Map LLM location_type values to schema literals: strict, preferred, remote.
        ParsedJD only accepts these three; LLM may return e.g. 'onsite', 'in-office', 'hybrid'.
        """
        if value is None:
            return None
        s = str(value).strip().lower()
        if not s:
            return None
        if s in ("strict", "preferred", "remote"):
            return s
        if s in ("onsite", "on-site", "in-office", "in office", "office"):
            return "strict"
        if s in ("work from home", "wfh"):
            return "remote"
        if s in ("hybrid", "preferred"):
            return "preferred"
        # Default unknown values to strict (location required) to be safe
        return "strict"

    def _normalize_response(self, parsed_data: Dict) -> Dict:
        """Normalize and validate parsed data to match schema."""
        # Ensure arrays are lists, not null
        normalized = {
            "designation": parsed_data.get("designation"),
            "must_have_skills": parsed_data.get("must_have_skills", []) or [],
            "nice_to_have_skills": parsed_data.get("nice_to_have_skills", []) or [],
            "min_experience_years": parsed_data.get("min_experience_years"),
            "max_experience_years": parsed_data.get("max_experience_years"),
            "education_requirements": parsed_data.get("education_requirements", []) or [],
            "domain_focus": parsed_data.get("domain_focus"),
            "inferred_mastercategory": parsed_data.get("inferred_mastercategory", "IT"),  # Default to IT if missing
            "inferred_category": parsed_data.get("inferred_category"),
            "inferred_category_confidence": parsed_data.get("inferred_category_confidence", 0.5),
            "location": parsed_data.get("location"),
            "location_type": self._normalize_location_type(parsed_data.get("location_type")),
            "other_requirements": parsed_data.get("other_requirements", []) or [],
            "text_for_embedding": parsed_data.get("text_for_embedding", "")
        }
        
        # Ensure arrays are lists
        for key in ["must_have_skills", "nice_to_have_skills", "education_requirements", "other_requirements"]:
            if normalized[key] is None:
                normalized[key] = []
            elif not isinstance(normalized[key], list):
                normalized[key] = []
        
        # Ensure inferred_mastercategory is valid
        if normalized["inferred_mastercategory"] not in ["IT", "Non-IT"]:
            # Try to infer from other fields
            skills = normalized["must_have_skills"] + normalized["nice_to_have_skills"]
            it_keywords = ["python", "java", "javascript", "react", "node", "sql", "aws", "docker", "kubernetes", 
                          "developer", "engineer", "programmer", "software", "backend", "frontend", "devops"]
            if any(kw in str(skills).lower() or kw in str(normalized.get("designation", "")).lower() 
                   for kw in it_keywords):
                normalized["inferred_mastercategory"] = "IT"
            else:
                normalized["inferred_mastercategory"] = "Non-IT"
        
        # Ensure confidence is in valid range
        conf = normalized["inferred_category_confidence"]
        if not isinstance(conf, (int, float)) or conf < 0.0 or conf > 1.0:
            normalized["inferred_category_confidence"] = 0.5
        
        # Generate text_for_embedding if missing
        if not normalized["text_for_embedding"]:
            parts = []
            if normalized["designation"]:
                parts.append(normalized["designation"])
            if normalized["must_have_skills"]:
                parts.extend(normalized["must_have_skills"])
            if normalized["domain_focus"]:
                parts.append(normalized["domain_focus"])
            if normalized["min_experience_years"] is not None:
                parts.append(f"{normalized['min_experience_years']}+ years")
            normalized["text_for_embedding"] = " ".join(parts) or "job description"
        
        return normalized
    
    async def extract_jd(self, jd_text: str) -> ParsedJD:
        """
        Extract structured data from Job Description text using LLM.
        
        Args:
            jd_text: The text content of the job description
        
        Returns:
            ParsedJD: Validated Pydantic model with extracted data
        
        Raises:
            RuntimeError: If LLM extraction fails
            ValueError: If extracted data doesn't match schema
        """
        try:
            model_to_use = self.model
            prompt = JD_EXTRACTION_PROMPT.replace("{JD_TEXT}", jd_text[:15000])  # Limit JD text length
            
            raw_output = ""
            if use_openai() and getattr(settings, "openai_api_key", None) and str(settings.openai_api_key).strip():
                try:
                    logger.info("Using OpenAI for JD extraction")
                    raw_output = await openai_generate(
                        prompt=prompt,
                        system_prompt=JD_EXTRACTION_PROMPT[:800],
                        temperature=0.1,
                        timeout_seconds=120.0,
                    )
                except Exception as e:
                    logger.error(
                        "OpenAI JD extraction failed and OLLAMA fallback is disabled when LLM_MODEL=OpenAI",
                        extra={"error": str(e)},
                    )
                    raise

            if not raw_output or not raw_output.strip():
                # If OpenAI is configured, do not attempt OLLAMA fallback
                if use_openai() and getattr(settings, "openai_api_key", None) and str(settings.openai_api_key).strip():
                    raise RuntimeError(
                        "OpenAI JD extraction returned empty response and "
                        "OLLAMA fallback is disabled when LLM_MODEL=OpenAI"
                    )

                is_connected, available_model = await self._check_ollama_connection()
                if not is_connected:
                    raise RuntimeError(
                        f"OLLAMA is not accessible at {self.ollama_host}. "
                        "Please ensure OLLAMA is running. Start it with: ollama serve"
                    )
                model_to_use = self.model
                if available_model and "llama3.1" not in available_model.lower():
                    model_to_use = available_model
                
                logger.info(
                    f"📤 CALLING OLLAMA API for JD extraction",
                    extra={"model": model_to_use, "ollama_host": self.ollama_host, "jd_text_length": len(jd_text)}
                )
                
                result = None
                last_error = None
                async with httpx.AsyncClient(timeout=Timeout(600.0)) as client:
                    try:
                        response = await client.post(
                            f"{self.ollama_host}/api/generate",
                            json={
                                "model": model_to_use,
                                "prompt": prompt,
                                "stream": False,
                                "options": {"temperature": 0.1, "top_p": 0.9}
                            }
                        )
                        response.raise_for_status()
                        result = response.json()
                        response_text = result.get("response", "") or result.get("text", "")
                        if not response_text and "message" in result:
                            response_text = result.get("message", {}).get("content", "")
                        result = {"response": response_text}
                        logger.info("✅ Successfully used /api/generate endpoint for JD extraction")
                    except httpx.HTTPStatusError as e:
                        if e.response.status_code != 404:
                            raise
                        last_error = e
                        logger.warning("OLLAMA /api/generate returned 404, trying /api/chat endpoint")
                    
                    if result is None:
                        try:
                            response = await client.post(
                                f"{self.ollama_host}/api/chat",
                                json={
                                    "model": model_to_use,
                                    "messages": [
                                        {"role": "system", "content": "You are a fresh, isolated extraction agent."},
                                        {"role": "user", "content": prompt}
                                    ],
                                    "stream": False,
                                    "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 2000}
                                }
                            )
                            response.raise_for_status()
                            result = response.json()
                            if "message" in result and "content" in result["message"]:
                                result = {"response": result["message"]["content"]}
                            else:
                                raise ValueError("Unexpected response format from OLLAMA chat API")
                            logger.info("✅ Successfully used /api/chat endpoint for JD extraction")
                        except Exception as e2:
                            last_error = e2
                            logger.error(f"OLLAMA /api/chat also failed: {e2}", extra={"error": str(e2)})
                    
                    if result is None:
                        raise RuntimeError(
                            f"All OLLAMA API endpoints failed. "
                            f"OLLAMA is running at {self.ollama_host} but endpoints return errors. "
                            f"Last error: {last_error}"
                        )
                
                if isinstance(result, dict):
                    if "response" in result:
                        raw_output = str(result["response"])
                    elif "text" in result:
                        raw_output = str(result["text"])
                    elif "content" in result:
                        raw_output = str(result["content"])
                    elif "message" in result and isinstance(result.get("message"), dict):
                        raw_output = str(result["message"].get("content", ""))
                else:
                    raw_output = str(result)
            
            # Log raw output for debugging
            logger.info(
                f"Raw LLM response for JD extraction",
                extra={
                    "raw_output_preview": raw_output[:500], 
                    "raw_output_length": len(raw_output)
                }
            )
            
            # Extract JSON
            logger.info(f"🔍 PARSING JSON from LLM response")
            parsed_data = self._extract_json(raw_output)
            
            if not parsed_data:
                raise ValueError("Failed to extract JSON from LLM response")
            
            # Normalize response
            normalized_data = self._normalize_response(parsed_data)
            
            # Validate with Pydantic
            try:
                parsed_jd = ParsedJD(**normalized_data)
                logger.info(
                    f"✅ JD EXTRACTED successfully",
                    extra={
                        "designation": parsed_jd.designation,
                        "mastercategory": parsed_jd.inferred_mastercategory,
                        "must_have_skills_count": len(parsed_jd.must_have_skills),
                        "status": "success"
                    }
                )
                return parsed_jd
            except Exception as e:
                logger.error(
                    f"Validation error for extracted JD data: {e}",
                    extra={"parsed_data": normalized_data, "error": str(e)},
                    exc_info=True
                )
                raise ValueError(f"Extracted data doesn't match schema: {e}")
            
        except httpx.HTTPError as e:
            error_details = {
                "error": str(e),
                "error_type": type(e).__name__,
                "ollama_host": self.ollama_host,
                "model": model_to_use,
            }
            if hasattr(e, "response"):
                error_details["response_status"] = e.response.status_code if e.response else None
                error_details["response_text"] = e.response.text[:500] if e.response else None
            logger.error(
                f"HTTP error calling LLM for JD extraction: {e}",
                extra=error_details,
                exc_info=True
            )
            raise RuntimeError(f"Failed to extract JD with LLM: {e}")
        except Exception as e:
            logger.error(
                f"Error extracting JD: {e}",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "ollama_host": self.ollama_host,
                    "model": model_to_use,
                },
                exc_info=True
            )
            raise
