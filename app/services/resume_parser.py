"""Service for parsing resumes using LLM."""
import json
import re
from typing import Dict, Optional, Tuple
import httpx
from httpx import Timeout
from docx import Document
import PyPDF2
from io import BytesIO

from app.config import settings
from app.utils.logging import get_logger
from app.utils.cleaning import normalize_phone, normalize_email, extract_skills, normalize_text

logger = get_logger(__name__)

# Try to import OLLAMA Python client
try:
    import ollama
    OLLAMA_CLIENT_AVAILABLE = True
except ImportError:
    OLLAMA_CLIENT_AVAILABLE = False
    logger.warning("OLLAMA Python client not available, using HTTP API directly")

MASTER_PROMPT = """
You are a strict JSON-output resume parser. Input: the raw text of a resume (or OCR text).

Output: a single-line JSON object with these keys:

{
  "candidateName": string | null,
  "mobile": string | null,
  "email": string | null,
  "education": string | null,
  "experience": string | null,           # e.g., "5 years" or "3y 6m"
  "domain": string | null,               # e.g., "Fintech, Healthcare"
  "jobrole": string | null,              # best-matched role title
  "skillset": [string],                  # list of normalized skills
  "filename": string | null,
  "summary": string | null
}

Rules:
- Always return valid JSON and nothing else.
- If a field is missing, set it to null or [] (for skillset).
- Normalize phone numbers to E.164 if possible; otherwise return cleaned digits.
- Lowercase emails but return original case for names.
- Extract skills as an array of single tokens or short phrases.
- Provide a short 1-2 sentence summary of the candidate in `summary`.

Example:
Input: <resume_text_here>
Output: {"candidateName":"Jane Doe", "mobile":"+14155552671", ...}
"""


class ResumeParser:
    """Service for parsing resume files using OLLAMA LLM."""
    
    def __init__(self):
        self.ollama_host = settings.ollama_host
        self.model = "llama3.1"
    
    async def extract_text(self, file_content: bytes, filename: str) -> str:
        """Extract text from uploaded file based on extension."""
        try:
            if filename.lower().endswith('.pdf'):
                return self._extract_pdf_text(file_content)
            elif filename.lower().endswith(('.docx', '.doc')):
                return self._extract_docx_text(file_content)
            elif filename.lower().endswith('.txt'):
                return file_content.decode('utf-8', errors='ignore')
            else:
                # Try as text
                return file_content.decode('utf-8', errors='ignore')
        except Exception as e:
            logger.error(f"Error extracting text from {filename}: {e}", extra={"error": str(e)})
            raise ValueError(f"Failed to extract text from file: {e}")
    
    def _extract_pdf_text(self, file_content: bytes) -> str:
        """Extract text from PDF file."""
        pdf_file = BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text_parts = []
        for page in pdf_reader.pages:
            text_parts.append(page.extract_text())
        return "\n".join(text_parts)
    
    def _extract_docx_text(self, file_content: bytes) -> str:
        """Extract text from DOCX file."""
        doc_file = BytesIO(file_content)
        doc = Document(doc_file)
        text_parts = []
        for paragraph in doc.paragraphs:
            text_parts.append(paragraph.text)
        return "\n".join(text_parts)
    
    async def _check_ollama_connection(self) -> tuple[bool, Optional[str]]:
        """Check if OLLAMA is accessible and running. Returns (is_connected, available_model)."""
        try:
            # Create fresh client for each check
            async with httpx.AsyncClient(timeout=Timeout(5.0)) as client:
                # Try to access OLLAMA API tags endpoint
                response = await client.get(f"{self.ollama_host}/api/tags")
                if response.status_code == 200:
                    models_data = response.json()
                    models = models_data.get("models", [])
                    # Check if llama3.1 is available
                    for model in models:
                        model_name = model.get("name", "")
                        if "llama3.1" in model_name.lower() or "llama3" in model_name.lower():
                            return True, model_name
                    # Return first available model if llama3.1 not found
                    if models:
                        return True, models[0].get("name", "")
                    return True, None
                return False, None
        except Exception as e:
            logger.warning(f"Failed to check OLLAMA connection: {e}", extra={"error": str(e)})
            return False, None
    
    async def parse_resume(self, resume_text: str, filename: str) -> Dict:
        """Parse resume text using LLM and return structured data."""
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
            
            # Prepare prompt
            prompt = f"{MASTER_PROMPT}\n\nInput: {resume_text[:10000]}\n\nOutput:"
            
            # Try using OLLAMA Python client first (handles API differences automatically)
            if OLLAMA_CLIENT_AVAILABLE:
                try:
                    import asyncio
                    # OLLAMA client is synchronous, so we'll run it in executor
                    loop = asyncio.get_event_loop()
                    def _generate():
                        # Create fresh OLLAMA client for each request
                        client = ollama.Client(host=self.ollama_host)
                        response = client.generate(
                            model=model_to_use,
                            prompt=prompt,
                            options={
                                "temperature": 0.1,
                                "top_p": 0.9,
                            }
                        )
                        return {"response": response.get("response", "")}
                    
                    result = await loop.run_in_executor(None, _generate)
                    logger.info("Successfully used OLLAMA Python client")
                except Exception as e:
                    logger.warning(f"OLLAMA Python client failed, falling back to HTTP API: {e}", extra={"error": str(e)})
                    result = None
            else:
                result = None
            
            # Fallback to HTTP API if Python client not available or failed
            if result is None:
                # Call OLLAMA API - create fresh client for each request
                async with httpx.AsyncClient(timeout=Timeout(600.0)) as client:
                    result = None
                    last_error = None
                    
                    # Try 1: /api/generate (standard OLLAMA endpoint)
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
                        logger.info("Successfully used /api/generate endpoint")
                    except httpx.HTTPStatusError as e:
                        if e.response.status_code != 404:
                            raise
                        last_error = e
                        logger.warning("OLLAMA /api/generate returned 404, trying alternative endpoints")
                
                    # Try 2: /api/chat (newer OLLAMA versions)
                    if result is None:
                        try:
                            logger.debug(
                                f"Attempting OLLAMA /api/chat with model: {model_to_use}",
                                extra={"model": model_to_use, "endpoint": "/api/chat"}
                            )
                            response = await client.post(
                                f"{self.ollama_host}/api/chat",
                                json={
                                    "model": model_to_use,
                                    "messages": [
                                        {"role": "user", "content": prompt}
                                    ],
                                    "stream": False,
                                    "options": {
                                        "temperature": 0.1,
                                        "top_p": 0.9,
                                    }
                                }
                            )
                            logger.debug(
                                f"OLLAMA /api/chat response status: {response.status_code}",
                                extra={"status_code": response.status_code}
                            )
                            response.raise_for_status()
                            result = response.json()
                            # Extract response from chat format
                            if "message" in result and "content" in result["message"]:
                                result = {"response": result["message"]["content"]}
                            else:
                                raise ValueError("Unexpected response format from OLLAMA chat API")
                            logger.info("Successfully used /api/chat endpoint")
                        except httpx.HTTPStatusError as e:
                            error_details = {
                                "status_code": e.response.status_code if e.response else None,
                                "response_text": e.response.text[:500] if e.response else None,
                                "endpoint": "/api/chat",
                                "model": model_to_use,
                            }
                            if e.response.status_code != 404:
                                logger.error(
                                    f"OLLAMA /api/chat failed with status {e.response.status_code}: {e}",
                                    extra=error_details,
                                    exc_info=True
                                )
                                raise
                            last_error = e
                            logger.warning(
                                "OLLAMA /api/chat returned 404, trying alternative endpoints",
                                extra=error_details
                            )
                        except httpx.RequestError as e:
                            last_error = e
                            logger.error(
                                f"Request error calling OLLAMA /api/chat: {e}",
                                extra={
                                    "error": str(e),
                                    "error_type": type(e).__name__,
                                    "endpoint": "/api/chat",
                                    "model": model_to_use,
                                },
                                exc_info=True
                            )
                
                    # Try 3: Direct model endpoint (some OLLAMA setups)
                    if result is None:
                        try:
                            logger.debug(
                                f"Attempting simplified OLLAMA /api/generate with model: {model_to_use}",
                                extra={"model": model_to_use, "endpoint": "/api/generate (simplified)"}
                            )
                            # Some OLLAMA setups use /api/generate with different structure
                            response = await client.post(
                                f"{self.ollama_host}/api/generate",
                                json={
                                    "model": model_to_use,
                                    "prompt": prompt,
                                    "stream": False
                                }
                            )
                            logger.debug(
                                f"OLLAMA simplified /api/generate response status: {response.status_code}",
                                extra={"status_code": response.status_code}
                            )
                            response.raise_for_status()
                            result = response.json()
                            logger.info("Successfully used simplified /api/generate endpoint")
                        except Exception as e:
                            last_error = e
                            logger.warning(
                                f"Simplified /api/generate also failed: {e}",
                                extra={
                                    "error": str(e),
                                    "error_type": type(e).__name__,
                                    "endpoint": "/api/generate (simplified)",
                                    "model": model_to_use,
                                },
                                exc_info=True
                            )
                
                    if result is None:
                        # Final attempt: try using OLLAMA Python client if available
                        if OLLAMA_CLIENT_AVAILABLE:
                            try:
                                import asyncio
                                loop = asyncio.get_event_loop()
                                def _generate():
                                    # Remove http:// or https:// from host for OLLAMA client
                                    host = self.ollama_host.replace("http://", "").replace("https://", "")
                                    client = ollama.Client(host=host)
                                    response = client.generate(
                                        model=model_to_use,
                                        prompt=prompt,
                                        options={"temperature": 0.1, "top_p": 0.9}
                                    )
                                    return {"response": response.get("response", "")}
                                result = await loop.run_in_executor(None, _generate)
                                logger.info("Successfully used OLLAMA Python client as fallback")
                            except Exception as e:
                                logger.error(f"OLLAMA Python client also failed: {e}", extra={"error": str(e)})
                                last_error = e
                        
                        if result is None:
                            error_summary = {
                                "ollama_host": self.ollama_host,
                                "model_attempted": model_to_use,
                                "available_model": available_model or "unknown",
                                "last_error_type": type(last_error).__name__ if last_error else None,
                                "last_error": str(last_error) if last_error else None,
                            }
                            if last_error and hasattr(last_error, "response"):
                                error_summary["last_status_code"] = last_error.response.status_code if last_error.response else None
                                error_summary["last_response_text"] = last_error.response.text[:500] if last_error.response else None
                            
                            logger.error(
                                "All OLLAMA API endpoints failed",
                                extra=error_summary,
                                exc_info=True
                            )
                            raise RuntimeError(
                                f"OLLAMA API endpoints not found. Tried /api/generate and /api/chat. "
                                f"OLLAMA is running at {self.ollama_host} but endpoints return 404. "
                                f"Please install OLLAMA Python client: pip install ollama "
                                f"or check your OLLAMA installation and version. "
                                f"Available model: {available_model or 'unknown'}. "
                                f"Last error: {last_error}"
                            )
                
                # Extract JSON from response
                raw_output = result.get("response", "")
                parsed_data = self._extract_json(raw_output)
                
                # Add filename if missing
                if not parsed_data.get("filename"):
                    parsed_data["filename"] = filename
                
                # Normalize fields
                parsed_data["mobile"] = normalize_phone(parsed_data.get("mobile"))
                parsed_data["email"] = normalize_email(parsed_data.get("email"))
                
                # Normalize skillset
                skills = parsed_data.get("skillset", [])
                if isinstance(skills, str):
                    skills = extract_skills(skills)
                parsed_data["skillset"] = skills
                
                # Normalize text fields
                for field in ["candidateName", "jobrole", "experience", "domain", "education"]:
                    if field in parsed_data:
                        parsed_data[field] = normalize_text(parsed_data[field])
                
                logger.info(
                    f"Parsed resume: {filename}",
                    extra={"file_name": filename, "candidate": parsed_data.get("candidateName")}
                )
                
                return parsed_data
                
        except httpx.HTTPError as e:
            error_details = {
                "error": str(e),
                "error_type": type(e).__name__,
                "ollama_host": self.ollama_host,
                "model": model_to_use,
                "prompt_length": len(prompt),
            }
            if hasattr(e, "request"):
                error_details["request_url"] = str(e.request.url) if e.request else None
                error_details["request_method"] = e.request.method if e.request else None
            if hasattr(e, "response"):
                error_details["response_status"] = e.response.status_code if e.response else None
                error_details["response_text"] = e.response.text[:500] if e.response else None
            logger.error(
                f"HTTP error calling OLLAMA: {e}",
                extra=error_details,
                exc_info=True
            )
            raise RuntimeError(f"Failed to parse resume with LLM: {e}")
        except Exception as e:
            logger.error(
                f"Error parsing resume: {e}",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "ollama_host": self.ollama_host,
                    "model": model_to_use,
                },
                exc_info=True
            )
            raise
    
    def _extract_json(self, text: str) -> Dict:
        """Extract JSON object from LLM response."""
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Try parsing the whole text
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from LLM response", extra={"response": text[:500]})
            # Return default structure
            return {
                "candidateName": None,
                "mobile": None,
                "email": None,
                "education": None,
                "experience": None,
                "domain": None,
                "jobrole": None,
                "skillset": [],
                "filename": None,
                "summary": None,
            }


