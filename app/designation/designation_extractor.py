"""Service for extracting designation (job title) from resumes using OLLAMA LLM."""
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

DESIGNATION_PROMPT = """
IMPORTANT: This is a FRESH, ISOLATED extraction task. Ignore any previous context or conversations.

ROLE:
You are an ATS resume parsing expert specializing in US IT staffing profiles.

CONTEXT:
Candidate profiles and resumes may be unstructured and inconsistently formatted.
Designation refers to the candidate's explicitly stated current or most recent job title.

TASK:
Extract the candidate's designation (job title) from the profile text.

SELECTION RULES (IN ORDER OF PRIORITY):
1. If a title is explicitly marked as "current", "present", or equivalent, select that.
2. Else, select the title associated with the most recent experience entry.
3. Else, select the designation mentioned in the resume headline or summary.
4. If multiple titles appear at the same level, select the first occurrence.

CONSTRAINTS:
- Extract only one designation.
- Preserve the designation exactly as written.
- Do not infer or normalize titles.
- Do not include company names, skills, durations, or locations.
- Ignore aspirational, desired, or target roles.

ANTI-HALLUCINATION RULES:
- If no explicit designation is found, return null (not "Other" or any placeholder).
- Never guess or infer a designation.
- Do not derive designation from skills, certifications, or projects.

OUTPUT FORMAT:
Return only valid JSON. No additional text. No explanations. No markdown formatting.

JSON SCHEMA:
{
  "designation": "string | null"
}

Example valid outputs:
{"designation": "Senior Software Engineer"}
{"designation": null}
"""


class DesignationExtractor:
    """Service for extracting designation from resume text using OLLAMA LLM."""
    
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
            return {"designation": None}
        
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
            if isinstance(parsed, dict) and "designation" in parsed:
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
                    if isinstance(parsed, dict) and "designation" in parsed:
                        logger.debug(f"Successfully extracted JSON with balanced braces: {parsed}")
                        return parsed
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON with balanced braces: {e}")
        
        # If all parsing fails, log the issue with more detail
        logger.error(
            "ERROR: Failed to parse JSON from LLM response", 
            extra={
                "response_preview": text[:500],
                "response_length": len(text),
                "cleaned_preview": cleaned_text[:500]
            }
        )
        # Return default structure with only designation
        return {"designation": None}
    
    async def extract_designation(self, resume_text: str, filename: str = "resume") -> Optional[str]:
        """
        Extract designation from resume text using OLLAMA LLM.
        
        Args:
            resume_text: The text content of the resume
            filename: Name of the resume file (for logging)
        
        Returns:
            The extracted designation string or None if not found
        """
        try:
            # Check OLLAMA connection first
            is_connected, available_model = await self._check_ollama_connection()
            if not is_connected:
                raise RuntimeError(
                    f"OLLAMA is not accessible at {self.ollama_host}. "
                    "Please ensure OLLAMA is running. Start it with: ollama serve"
                )
            
            # Use available model if llama3.1 not found
            model_to_use = self.model
            if available_model and "llama3.1" not in available_model.lower():
                logger.warning(
                    f"llama3.1 not found, using available model: {available_model}",
                    extra={"available_model": available_model}
                )
                model_to_use = available_model
            
            # Prepare prompt - be very explicit about JSON-only output
            # Add explicit context clearing to ensure no session bleeding
            prompt = f"""{DESIGNATION_PROMPT}

Input resume text:
{resume_text[:10000]}

Output (JSON only, no other text, no explanations):"""
            
            # Clear any potential context by using a fresh, isolated request
            # Each extraction is completely independent
            
            logger.info(
                f"📤 CALLING OLLAMA API for designation extraction",
                extra={
                    "file_name": filename,  # Use file_name instead of filename (reserved in LogRecord)
                    "model": model_to_use,
                    "ollama_host": self.ollama_host,
                    "resume_text_length": len(resume_text),
                    "prompt_length": len(prompt)
                }
            )
            print(f"\n📤 CALLING OLLAMA API")
            print(f"   Model: {model_to_use}")
            print(f"   Host: {self.ollama_host}")
            print(f"   Resume text length: {len(resume_text)} characters")
            print(f"   Prompt length: {len(prompt)} characters")
            
            result = None
            last_error = None
            
            # Use HTTP API directly (more reliable than Python client)
            # Try /api/generate endpoint first
            async with httpx.AsyncClient(timeout=Timeout(600.0)) as client:
                # Try /api/generate endpoint
                try:
                    logger.debug(f"Sending request to {self.ollama_host}/api/generate")
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
                    # Extract response text from /api/generate format
                    # OLLAMA /api/generate returns response in "response" field
                    response_text = result.get("response", "")
                    if not response_text:
                        # Sometimes it's in a different format
                        response_text = result.get("text", "")
                    if not response_text and "message" in result:
                        response_text = result.get("message", {}).get("content", "")
                    
                    logger.info(
                        f"📥 OLLAMA API RESPONSE RECEIVED",
                        extra={
                            "result_keys": list(result.keys()), 
                            "has_response": "response" in result,
                            "response_text_length": len(response_text),
                            "response_preview": response_text[:200]
                        }
                    )
                    print(f"📥 OLLAMA API RESPONSE RECEIVED")
                    print(f"   Response keys: {list(result.keys())}")
                    print(f"   Response text length: {len(response_text)} characters")
                    print(f"   Response preview: {response_text[:150]}...")
                    
                    result = {"response": response_text}
                    logger.info("✅ Successfully used /api/generate endpoint for designation extraction")
                except httpx.HTTPStatusError as e:
                    if e.response.status_code != 404:
                        raise
                    last_error = e
                    logger.warning("OLLAMA /api/generate returned 404, trying /api/chat endpoint")
                
                # Try /api/chat endpoint
                if result is None:
                    try:
                        # Use /api/chat with fresh conversation (no history)
                        # This ensures complete session isolation
                        response = await client.post(
                                f"{self.ollama_host}/api/chat",
                                json={
                                    "model": model_to_use,
                                    "messages": [
                                        {"role": "system", "content": "You are a fresh, isolated extraction agent. This is a new task with no previous context."},
                                        {"role": "user", "content": prompt}
                                    ],
                                    "stream": False,
                                    "options": {
                                        "temperature": 0.1,
                                        "top_p": 0.9,
                                        "num_predict": 500,  # Limit response length
                                    }
                                }
                            )
                        response.raise_for_status()
                        result = response.json()
                        # Extract response from chat format
                        if "message" in result and "content" in result["message"]:
                            result = {"response": result["message"]["content"]}
                        else:
                            raise ValueError("Unexpected response format from OLLAMA chat API")
                        logger.info("Successfully used /api/chat endpoint for designation extraction")
                    except Exception as e2:
                        last_error = e2
                        logger.error(f"OLLAMA /api/chat also failed: {e2}", extra={"error": str(e2)})
                
                if result is None:
                    raise RuntimeError(
                        f"All OLLAMA API endpoints failed. "
                        f"OLLAMA is running at {self.ollama_host} but endpoints return errors. "
                        f"Last error: {last_error}"
                    )
            
            # Extract JSON from response - handle different OLLAMA response formats
            raw_output = ""
            if isinstance(result, dict):
                # Directly get response - check keys explicitly to avoid issues with 'or' chain
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
                f"Raw OLLAMA response for designation extraction from {filename}",
                extra={
                    "raw_output_preview": raw_output[:500], 
                    "raw_output_length": len(raw_output),
                    "file_name": filename  # Use file_name instead of filename (reserved in LogRecord)
                }
            )
            
            # Extract JSON and get designation
            logger.info(f"🔍 PARSING JSON from OLLAMA response")
            print(f"\n🔍 PARSING JSON from OLLAMA response...")
            parsed_data = self._extract_json(raw_output)
            designation = parsed_data.get("designation")
            
            logger.info(
                f"📊 JSON PARSED: {parsed_data}",
                extra={"parsed_data": parsed_data, "designation": designation}
            )
            print(f"📊 JSON PARSED: {parsed_data}")
            
            # Clean and validate designation
            if designation is not None:
                designation = str(designation).strip()
                # Convert invalid placeholder values to None
                if not designation or designation.lower() in ["null", "none", "", "other"]:
                    designation = None
                    logger.warning("Designation was empty/null/invalid placeholder after cleaning")
                    print(f"⚠️  Designation was empty/null/invalid placeholder after cleaning")
            
            # Log the extraction result clearly
            if designation:
                logger.info(
                    f"✅ DESIGNATION EXTRACTED from {filename}",
                    extra={
                        "file_name": filename,
                        "designation": designation,
                        "raw_output_length": len(raw_output),
                        "status": "success"
                    }
                )
                print(f"\n{'='*60}")
                print(f"✅ DESIGNATION EXTRACTED: '{designation}'")
                print(f"   File: {filename}")
                print(f"   Raw OLLAMA response length: {len(raw_output)} chars")
                print(f"{'='*60}\n")
            else:
                logger.warning(
                    f"⚠️  NO DESIGNATION FOUND in {filename}",
                    extra={
                        "file_name": filename,
                        "designation": None,
                        "raw_output_length": len(raw_output),
                        "status": "not_found"
                    }
                )
                print(f"\n{'='*60}")
                print(f"⚠️  NO DESIGNATION FOUND in {filename}")
                print(f"   Raw OLLAMA response: {raw_output[:200]}")
                print(f"{'='*60}\n")
            
            return designation
            
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
                f"HTTP error calling OLLAMA for designation extraction: {e}",
                extra=error_details,
                exc_info=True
            )
            raise RuntimeError(f"Failed to extract designation with LLM: {e}")
        except Exception as e:
            logger.error(
                f"Error extracting designation: {e}",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "ollama_host": self.ollama_host,
                    "model": model_to_use,
                },
                exc_info=True
            )
            raise

