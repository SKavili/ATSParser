"""Service for extracting skills from resumes using OLLAMA LLM."""
import json
import re
from typing import Dict, List, Optional
import httpx
from httpx import Timeout

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Try to import OLLAMA Python client
try:
    import ollama
    OLLAMA_CLIENT_AVAILABLE = True
except ImportError:
    OLLAMA_CLIENT_AVAILABLE = False
    logger.warning("OLLAMA Python client not available, using HTTP API directly")

SKILLS_PROMPT = """
IMPORTANT: This is a FRESH, ISOLATED extraction task. Ignore any previous context or conversations.

ROLE:
You are an ATS resume parsing expert specializing in NON-IT and NON-TECH professional profiles.

CONTEXT:
Candidate profiles and resumes may be unstructured and inconsistently formatted.
Skills refer ONLY to practical, applied, and demonstrable professional capabilities,
domain knowledge areas, tools, techniques, methodologies, or certifications
that a candidate can actively use or perform.

TASK:
Extract all NON-IT professional skills from the profile text.

SELECTION RULES:
1. Extract ONLY practical, functional, or domain skills that represent WHAT the candidate CAN DO.
2. Include domain skills (e.g., Financial Accounting, Cost Accounting, HR Operations, Sales Management).
3. Include tools and software used in non-IT roles (e.g., MS Excel, MS Word, Tally, SAP, QuickBooks, CRM).
4. Include techniques, methodologies, standards, or processes explicitly mentioned (e.g., Research Methodology, Audit Planning, Financial Analysis).
5. Include certifications or professional qualifications ONLY as skills (e.g., CA Foundation, GST Certification).
6. Include subject areas ONLY if they represent applied professional knowledge (e.g., Financial Management, Management Accounting).
7. Do NOT include programming languages, coding technologies, frameworks, or development libraries.
8. Do NOT include soft skills (e.g., Communication, Leadership, Teamwork).
9. Do NOT include company names, organization names, institutions, colleges, universities, banks, or employers.
10. Do NOT include conference names, seminar names, workshop names, journal names, or event titles.
11. Do NOT include research paper titles, thesis titles, presentation titles, or publication headings.
12. Do NOT include locations, cities, states, or countries.
13. Do NOT include job titles, roles, or designations.

CONSTRAINTS:
- Extract ONLY relevant NON-IT professional skills.
- Preserve skill names exactly as written (case-sensitive).
- Remove duplicates.
- Limit to a maximum of 50 skills.

ANTI-HALLUCINATION RULES:
- Extract skills ONLY if they are explicitly mentioned in the resume.
- Never guess, infer, or assume skills.
- Do NOT convert topics, events, or titles into skills.
- Do NOT derive skills from job titles or organization names.

OUTPUT FORMAT:
Return only valid JSON.
No additional text.
No explanations.
No markdown formatting.

JSON SCHEMA:
{
  "skills": ["skill1", "skill2", "skill3", ...]
}

VALID OUTPUT EXAMPLES:
{"skills": ["Financial Accounting", "Cost and Management Accounting", "Business Statistics", "Research Methodology", "Financial Management", "MS Excel", "Tally ERP"]}
{"skills": []}


"""


class SkillsExtractor:
    """Service for extracting skills from resume text using OLLAMA LLM."""
    
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
            return {"skills": []}
        
        # Clean the text - remove markdown code blocks if present
        cleaned_text = text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()
        
        # Find the first { and last }
        start_idx = cleaned_text.find('{')
        end_idx = cleaned_text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            cleaned_text = cleaned_text[start_idx:end_idx + 1]
        
        # Try parsing the cleaned text
        try:
            parsed = json.loads(cleaned_text)
            if isinstance(parsed, dict) and "skills" in parsed:
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
                    if isinstance(parsed, dict) and "skills" in parsed:
                        logger.debug(f"Successfully extracted JSON with balanced braces: {parsed}")
                        return parsed
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON with balanced braces: {e}")
        
        logger.error(
            "ERROR: Failed to parse JSON from LLM response", 
            extra={
                "response_preview": text[:500],
                "response_length": len(text),
                "cleaned_preview": cleaned_text[:500]
            }
        )
        return {"skills": []}
    
    async def extract_skills(self, resume_text: str, filename: str = "resume") -> List[str]:
        """
        Extract skills from resume text using OLLAMA LLM.
        
        Args:
            resume_text: The text content of the resume
            filename: Name of the resume file (for logging)
        
        Returns:
            List of extracted skills
        """
        try:
            is_connected, available_model = await self._check_ollama_connection()
            if not is_connected:
                raise RuntimeError(
                    f"OLLAMA is not accessible at {self.ollama_host}. "
                    "Please ensure OLLAMA is running. Start it with: ollama serve"
                )
            
            model_to_use = self.model
            if available_model and "llama3.1" not in available_model.lower():
                logger.warning(
                    f"llama3.1 not found, using available model: {available_model}",
                    extra={"available_model": available_model}
                )
                model_to_use = available_model
            
            # ========== DEBUG: Check what's being sent to LLM ==========
            text_to_send = resume_text[:10000]
            print("\n" + "="*80)
            print("[DEBUG] TEXT BEING SENT TO LLM FOR SKILLS EXTRACTION")
            print("="*80)
            print(f"Full resume text length: {len(resume_text)} characters")
            print(f"Text being sent to LLM: {len(text_to_send)} characters (first 10,000)")
            print(f"Text truncated: {'YES' if len(resume_text) > 10000 else 'NO'}")
            if len(resume_text) > 10000:
                print(f"⚠️  WARNING: {len(resume_text) - 10000} characters are being CUT OFF!")
            print(f"\nFirst 2000 characters being sent:")
            print("-"*80)
            print(text_to_send[:2000])
            print("-"*80)
            print(f"Last 1000 characters being sent:")
            print("-"*80)
            print(text_to_send[-1000:] if len(text_to_send) > 1000 else text_to_send)
            print("="*80 + "\n")
            # ========== END DEBUG ==========
            
            prompt = f"""{SKILLS_PROMPT}

Input resume text:
{text_to_send}

Output (JSON only, no other text, no explanations):"""
            
            logger.info(
                f"📤 CALLING OLLAMA API for skills extraction",
                extra={
                    "file_name": filename,
                    "model": model_to_use,
                    "ollama_host": self.ollama_host,
                    "resume_text_length": len(resume_text),
                }
            )
            
            result = None
            last_error = None
            
            async with httpx.AsyncClient(timeout=Timeout(3600.0)) as client:
                try:
                    response = await client.post(
                        f"{self.ollama_host}/api/generate",
                        json={
                            "model": model_to_use,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": 0.1,
                                "top_p": 0.9,
                            }
                        }
                    )
                    response.raise_for_status()
                    result = response.json()
                    response_text = result.get("response", "") or result.get("text", "")
                    if not response_text and "message" in result:
                        response_text = result.get("message", {}).get("content", "")
                    result = {"response": response_text}
                    logger.info("✅ Successfully used /api/generate endpoint for skills extraction")
                except httpx.HTTPStatusError as e:
                    if e.response.status_code != 404:
                        raise
                    last_error = e
                    logger.warning("OLLAMA /api/generate returned 404, trying /api/chat endpoint")
                
                if result is None:
                    try:
                        # Use /api/chat with fresh conversation (no history)
                        # System message ensures complete session isolation
                        response = await client.post(
                            f"{self.ollama_host}/api/chat",
                            json={
                                "model": model_to_use,
                                "messages": [
                                    {"role": "system", "content": "You are a fresh, isolated extraction agent. This is a new, independent task with no previous context. Ignore any previous conversations."},
                                    {"role": "user", "content": prompt}
                                ],
                                "stream": False,
                                "options": {
                                    "temperature": 0.1,
                                    "top_p": 0.9,
                                    "num_predict": 500,  # Limit response length for isolation
                                }
                            }
                        )
                        response.raise_for_status()
                        result = response.json()
                        if "message" in result and "content" in result["message"]:
                            result = {"response": result["message"]["content"]}
                        else:
                            raise ValueError("Unexpected response format from OLLAMA chat API")
                        logger.info("Successfully used /api/chat endpoint for skills extraction")
                    except Exception as e2:
                        last_error = e2
                        logger.error(f"OLLAMA /api/chat also failed: {e2}", extra={"error": str(e2)})
                
                if result is None:
                    raise RuntimeError(
                        f"All OLLAMA API endpoints failed. "
                        f"OLLAMA is running at {self.ollama_host} but endpoints return errors. "
                        f"Last error: {last_error}"
                    )
            
            raw_output = ""
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
            
            # ========== DEBUG: Check raw LLM response ==========
            print("\n" + "="*80)
            print("[DEBUG] RAW LLM RESPONSE")
            print("="*80)
            print(f"Response length: {len(raw_output)} characters")
            print(f"Full raw response:")
            print("-"*80)
            print(raw_output)
            print("-"*80)
            print("="*80 + "\n")
            # ========== END DEBUG ==========
            
            parsed_data = self._extract_json(raw_output)
            
            # ========== DEBUG: Check parsed data ==========
            print("\n" + "="*80)
            print("[DEBUG] PARSED JSON DATA")
            print("="*80)
            print(f"Parsed data: {parsed_data}")
            print(f"Skills found: {parsed_data.get('skills', [])}")
            print(f"Number of skills: {len(parsed_data.get('skills', []))}")
            print("="*80 + "\n")
            # ========== END DEBUG ==========
            
            skills = parsed_data.get("skills", [])
            
            # Validate and clean skills
            if skills and isinstance(skills, list):
                skills = [str(skill).strip() for skill in skills if skill and str(skill).strip()]
                skills = list(dict.fromkeys(skills))  # Remove duplicates while preserving order
                skills = skills[:50]  # Limit to 50 skills
            else:
                skills = []
            
            logger.info(
                f"✅ SKILLS EXTRACTED from {filename}",
                extra={
                    "file_name": filename,
                    "skills_count": len(skills),
                    "skills": skills[:10]  # Log first 10
                }
            )
            
            return skills
            
        except httpx.HTTPError as e:
            error_details = {
                "error": str(e),
                "error_type": type(e).__name__,
                "ollama_host": self.ollama_host,
                "model": model_to_use,
            }
            logger.error(
                f"HTTP error calling OLLAMA for skills extraction: {e}",
                extra=error_details,
                exc_info=True
            )
            raise RuntimeError(f"Failed to extract skills with LLM: {e}")
        except Exception as e:
            logger.error(
                f"Error extracting skills: {e}",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "ollama_host": self.ollama_host,
                    "model": model_to_use,
                },
                exc_info=True
            )
            raise

