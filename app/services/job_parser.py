"""Service for parsing job descriptions and creating embeddings."""
import json
import re
from typing import Dict, Optional
import httpx
from httpx import Timeout

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

JOB_PROMPT = """
You are a job-description normalizer. Given a job title and description, extract:

{
  "job_id": "<string or null>",
  "title": "<cleaned title>",
  "location": "<location or null>",
  "required_skills": ["..."],
  "responsibilities": ["..."],
  "summary_for_embedding": "<one paragraph summary for embedding>"
}

Rules:
- Return JSON only.
- `summary_for_embedding` should be ~40-70 words, combining title, top skills, and responsibilities — this is what will be embedded.
"""


class JobParser:
    """Service for parsing job descriptions using OLLAMA LLM."""
    
    def __init__(self):
        self.ollama_host = settings.ollama_host
        self.model = "llama3.1"
    
    async def _check_ollama_connection(self) -> bool:
        """Check if OLLAMA is accessible and running."""
        try:
            async with httpx.AsyncClient(timeout=Timeout(5.0)) as client:
                response = await client.get(f"{self.ollama_host}/api/tags")
                return response.status_code == 200
        except Exception:
            return False
    
    async def parse_job(self, title: str, description: str, job_id: Optional[str] = None) -> Dict:
        """Parse job description and return structured data with embedding summary."""
        try:
            # Check OLLAMA connection first
            if not await self._check_ollama_connection():
                raise RuntimeError(
                    f"OLLAMA is not accessible at {self.ollama_host}. "
                    "Please ensure OLLAMA is running. Start it with: ollama serve"
                )
            
            # Prepare prompt
            prompt = f"{JOB_PROMPT}\n\nTitle: {title}\nDescription: {description}\n\nOutput:"
            
            # Call OLLAMA API - create fresh client for each request
            async with httpx.AsyncClient(timeout=Timeout(600.0)) as client:
                # Try /api/generate first (older OLLAMA versions)
                try:
                    response = await client.post(
                        f"{self.ollama_host}/api/generate",
                        json={
                            "model": self.model,
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
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 404:
                        # Try /api/chat endpoint (newer OLLAMA versions)
                        logger.warning("OLLAMA /api/generate not found, trying /api/chat endpoint")
                        response = await client.post(
                            f"{self.ollama_host}/api/chat",
                            json={
                                "model": self.model,
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
                        response.raise_for_status()
                        result = response.json()
                        # Extract response from chat format
                        if "message" in result and "content" in result["message"]:
                            result = {"response": result["message"]["content"]}
                        else:
                            raise ValueError("Unexpected response format from OLLAMA chat API")
                    else:
                        raise
                except httpx.ConnectError:
                    raise RuntimeError(
                        f"Cannot connect to OLLAMA at {self.ollama_host}. "
                        "Please ensure OLLAMA is running. Start it with: ollama serve"
                    )
                
                # Extract JSON from response
                raw_output = result.get("response", "")
                parsed_data = self._extract_json(raw_output)
                
                # Override job_id if provided
                if job_id:
                    parsed_data["job_id"] = job_id
                
                # Ensure job_id exists
                if not parsed_data.get("job_id"):
                    import uuid
                    parsed_data["job_id"] = f"job_{uuid.uuid4().hex[:12]}"
                
                # Ensure summary_for_embedding exists
                if not parsed_data.get("summary_for_embedding"):
                    parsed_data["summary_for_embedding"] = f"{title}. {description[:200]}"
                
                logger.info(
                    f"Parsed job: {parsed_data.get('job_id')}",
                    extra={"job_id": parsed_data.get("job_id"), "title": title}
                )
                
                return parsed_data
                
        except httpx.HTTPError as e:
            logger.error(f"HTTP error calling OLLAMA: {e}", extra={"error": str(e)})
            raise RuntimeError(f"Failed to parse job with LLM: {e}")
        except Exception as e:
            logger.error(f"Error parsing job: {e}", extra={"error": str(e)})
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
            import uuid
            return {
                "job_id": f"job_{uuid.uuid4().hex[:12]}",
                "title": "",
                "location": None,
                "required_skills": [],
                "responsibilities": [],
                "summary_for_embedding": "",
            }


