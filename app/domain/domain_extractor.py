"""Service for extracting industry domain from resumes using OLLAMA LLM."""
import json
import re
from typing import Dict, Optional
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

DOMAIN_PROMPT = """
IMPORTANT: This is a FRESH, ISOLATED extraction task. Ignore any previous context or conversations.

ROLE:
You are an ATS resume parsing expert specializing in US IT staffing profiles.

CONTEXT:
Candidate profiles and resumes may be unstructured and inconsistently formatted.
Domain refers to the primary industry domain or business domain the candidate has worked in (e.g., Healthcare, Finance, E-commerce, Banking, Insurance, Retail, etc.).

TASK:
Extract the candidate's primary industry domain from the profile text.

SELECTION RULES:
1. Look for domain information in work experience sections.
2. Identify the industry domain from company names or project descriptions.
3. If multiple domains are mentioned, select the most prominent or recent one.
4. Common domains: Healthcare, Finance, Banking, Insurance, E-commerce, Retail, Manufacturing, Education, Government, etc.

CONSTRAINTS:
- Extract only one primary domain.
- Preserve the domain name exactly as written or use standard industry terminology.
- Return as a string (e.g., "Healthcare", "Finance", "E-commerce").

ANTI-HALLUCINATION RULES:
- If no explicit domain information is found, return null.
- Never guess or infer a domain.
- Do not derive domain from skills alone.

OUTPUT FORMAT:
Return only valid JSON. No additional text. No explanations. No markdown formatting.

JSON SCHEMA:
{
  "domain": "string | null"
}

Example valid outputs:
{"domain": "Healthcare"}
{"domain": "Finance"}
{"domain": null}
"""


class DomainExtractor:
    """Service for extracting industry domain from resume text using OLLAMA LLM."""
    
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
            return {"domain": None}
        
        cleaned_text = text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()
        
        start_idx = cleaned_text.find('{')
        end_idx = cleaned_text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            cleaned_text = cleaned_text[start_idx:end_idx + 1]
        
        try:
            parsed = json.loads(cleaned_text)
            if isinstance(parsed, dict) and "domain" in parsed:
                logger.debug(f"Successfully extracted JSON: {parsed}")
                return parsed
        except json.JSONDecodeError:
            pass
        
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
                    if isinstance(parsed, dict) and "domain" in parsed:
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
        return {"domain": None}
    
    async def extract_domain(self, resume_text: str, filename: str = "resume") -> Optional[str]:
        """
        Extract industry domain from resume text using OLLAMA LLM.
        
        Args:
            resume_text: The text content of the resume
            filename: Name of the resume file (for logging)
        
        Returns:
            The extracted domain string or None if not found
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
            
            prompt = f"""{DOMAIN_PROMPT}

Input resume text:
{resume_text[:10000]}

Output (JSON only, no other text, no explanations):"""
            
            logger.info(
                f"📤 CALLING OLLAMA API for domain extraction",
                extra={
                    "file_name": filename,
                    "model": model_to_use,
                    "ollama_host": self.ollama_host,
                }
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
                    logger.info("✅ Successfully used /api/generate endpoint for domain extraction")
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
                        logger.info("Successfully used /api/chat endpoint for domain extraction")
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
            
            parsed_data = self._extract_json(raw_output)
            domain = parsed_data.get("domain")
            
            if domain:
                domain = str(domain).strip()
                if not domain or domain.lower() in ["null", "none", ""]:
                    domain = None
            
            logger.info(
                f"✅ DOMAIN EXTRACTED from {filename}",
                extra={
                    "file_name": filename,
                    "domain": domain,
                    "status": "success" if domain else "not_found"
                }
            )
            
            return domain
            
        except httpx.HTTPError as e:
            error_details = {
                "error": str(e),
                "error_type": type(e).__name__,
                "ollama_host": self.ollama_host,
                "model": model_to_use,
            }
            logger.error(
                f"HTTP error calling OLLAMA for domain extraction: {e}",
                extra=error_details,
                exc_info=True
            )
            raise RuntimeError(f"Failed to extract domain with LLM: {e}")
        except Exception as e:
            logger.error(
                f"Error extracting domain: {e}",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "ollama_host": self.ollama_host,
                    "model": model_to_use,
                },
                exc_info=True
            )
            raise

