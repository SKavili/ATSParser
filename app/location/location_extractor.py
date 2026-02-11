"""Service for extracting candidate location from resumes using OpenAI or OLLAMA LLM."""
import json
from typing import Dict, Optional
import httpx
from httpx import Timeout

from app.config import settings
from app.utils.logging import get_logger
from app.services.llm_service import use_openai, generate as openai_generate

logger = get_logger(__name__)

LOCATION_PROMPT = """IMPORTANT: This is a FRESH, ISOLATED extraction task. Ignore any previous context, memory, or conversations.
ROLE: You are an ATS resume parsing expert specializing in candidate profiles.
CONTEXT:
- Resumes may be unstructured, multi-line, or poorly formatted.
- Location refers to the candidate's current location, address, or place of residence (city, state, country, or region).
- It often appears in the header, contact section, or near name/email/phone.
TASK: Extract the candidate's location from the profile text.
SELECTION RULES (IN ORDER):
1. Prefer location in the resume header or contact/address section.
2. Prefer text that is clearly labeled (e.g. "Location:", "Address:", "Based in", "City:", city name next to contact details).
3. If multiple locations appear (e.g. current vs past), prefer the one that looks like current residence or "Current location" / "Based in".
4. Prefer a single, concise location string (e.g. "Bangalore, India" or "San Francisco, CA") over a full street address.
LOCATION NORMALIZATION:
- Return one string: city, state/region, and/or country as written (e.g. "Mumbai, Maharashtra", "Hyderabad, India", "New York, NY").
- You may normalize to "City, State" or "City, Country" if that is clearly what the resume indicates.
- Do NOT include full street address, PIN/postal code, or building names unless no city/region is given.
- Collapse extra whitespace; do not change spelling or add/remove parts of the location.
CONSTRAINTS:
- Do NOT invent or guess a location.
- Do NOT infer location from company names, university names, or project names.
- Do NOT use job location or "willing to relocate" as the candidate's location unless it is explicitly stated as their current location.
- If only a partial location is present (e.g. only "India" or only "Bangalore"), return that.
- Extract exactly ONE primary location string.
ANTI-HALLUCINATION RULES:
- If no explicit location or address is found, return null.
- Never infer location from email domain, phone country code, or language of the resume alone.
- Never correct or expand location names.
OUTPUT FORMAT: Return ONLY valid JSON. No additional text. No explanations. No markdown.
JSON SCHEMA: { "location": "string | null" }
VALID EXAMPLES:
{"location": "Bangalore, India"}
{"location": "Hyderabad, Telangana"}
{"location": "San Francisco, CA"}
{"location": "Mumbai"}
{"location": "New York, NY, USA"}
{"location": null}"""

LOCATION_EXTRACTION_TEXT_LIMIT = 6000  # Location often in header/contact
OLLAMA_TIMEOUT = 90.0
OLLAMA_MAX_TOKENS = 80


class LocationExtractor:
    """Service for extracting candidate location from resume text using OpenAI or OLLAMA LLM."""

    def __init__(self):
        self.ollama_host = settings.ollama_host
        self.model = "llama3.1"

    async def _check_ollama_connection(self) -> tuple[bool, Optional[str]]:
        """Check if OLLAMA is accessible and running."""
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
            return {"location": None}

        cleaned_text = text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()

        start_idx = cleaned_text.find("{")
        end_idx = cleaned_text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = cleaned_text[start_idx : end_idx + 1]
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict) and "location" in parsed:
                    return parsed
            except json.JSONDecodeError:
                pass

        # Balanced braces fallback
        start_idx = cleaned_text.find("{")
        if start_idx != -1:
            brace_count = 0
            for i in range(start_idx, len(cleaned_text)):
                if cleaned_text[i] == "{":
                    brace_count += 1
                elif cleaned_text[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        try:
                            parsed = json.loads(cleaned_text[start_idx : i + 1])
                            if isinstance(parsed, dict) and "location" in parsed:
                                return parsed
                        except json.JSONDecodeError:
                            pass
                        break

        logger.warning("Failed to parse JSON from LLM response", extra={"response_preview": text[:300]})
        return {"location": None}

    async def extract_location(self, resume_text: str, filename: str = "resume") -> Optional[str]:
        """
        Extract candidate location from resume text using OpenAI (when LLM_MODEL=OpenAI) or OLLAMA.

        Uses the first part of the resume (header/contact) where location usually appears.

        Args:
            resume_text: Full resume text.
            filename: Name of the resume file (for logging).

        Returns:
            Extracted location string or None if not found.
        """
        if not resume_text or len(resume_text.strip()) < 10:
            logger.warning(f"Resume text too short for location extraction: {filename}")
            return None

        limited_text = (
            resume_text[:LOCATION_EXTRACTION_TEXT_LIMIT]
            if len(resume_text) >= LOCATION_EXTRACTION_TEXT_LIMIT
            else resume_text
        )

        try:
            raw_output = ""
            # Use OpenAI when LLM_MODEL=OpenAI and API key is set
            if use_openai() and getattr(settings, "openai_api_key", None) and str(settings.openai_api_key).strip():
                try:
                    logger.info("Using OpenAI for location extraction", extra={"file_name": filename})
                    raw_output = await openai_generate(
                        prompt=f"Input resume text:\n{limited_text}\n\nOutput (JSON only, no other text):",
                        system_prompt=LOCATION_PROMPT[:2000],
                        temperature=0.0,
                        timeout_seconds=90.0,
                    )
                except Exception as e:
                    logger.error(
                        "OpenAI location extraction failed and OLLAMA fallback is disabled when LLM_MODEL=OpenAI",
                        extra={"file_name": filename, "error": str(e)},
                    )
                    raise

            if not raw_output or not raw_output.strip():
                if use_openai() and getattr(settings, "openai_api_key", None) and str(settings.openai_api_key).strip():
                    raise RuntimeError(
                        "OpenAI location extraction returned empty response and "
                        "OLLAMA fallback is disabled when LLM_MODEL=OpenAI"
                    )

                # OLLAMA path
                is_connected, available_model = await self._check_ollama_connection()
                if not is_connected:
                    logger.warning(f"OLLAMA not accessible for {filename}, skipping location extraction")
                    return None

                model_to_use = self.model
                if available_model and "llama3.1" not in available_model.lower():
                    model_to_use = available_model
                prompt = f"{LOCATION_PROMPT}\n\nInput resume text:\n{limited_text}\n\nOutput (JSON only, no other text):"

                async with httpx.AsyncClient(timeout=Timeout(OLLAMA_TIMEOUT)) as client:
                    result = None
                    try:
                        response = await client.post(
                            f"{self.ollama_host}/api/generate",
                            json={
                                "model": model_to_use,
                                "prompt": prompt,
                                "stream": False,
                                "options": {
                                    "temperature": 0.0,
                                    "top_p": 0.9,
                                    "num_predict": OLLAMA_MAX_TOKENS,
                                },
                            },
                        )
                        response.raise_for_status()
                        result = response.json()
                        response_text = result.get("response", "") or result.get("text", "")
                        if not response_text and "message" in result:
                            response_text = result.get("message", {}).get("content", "")
                        result = {"response": response_text}
                    except httpx.HTTPStatusError as e:
                        if e.response.status_code == 404:
                            response = await client.post(
                                f"{self.ollama_host}/api/chat",
                                json={
                                    "model": model_to_use,
                                    "messages": [{"role": "user", "content": prompt}],
                                    "stream": False,
                                    "options": {
                                        "temperature": 0.0,
                                        "top_p": 0.9,
                                        "num_predict": OLLAMA_MAX_TOKENS,
                                    },
                                },
                            )
                            response.raise_for_status()
                            result = response.json()
                            if "message" in result and "content" in result["message"]:
                                result = {"response": result["message"]["content"]}
                            else:
                                result = None
                        else:
                            raise

                    if result is None:
                        return None

                    raw_output = (
                        result.get("response") or result.get("text") or ""
                        or (result.get("message") or {}).get("content", "")
                    )
                    raw_output = str(raw_output) if raw_output else ""

            parsed = self._extract_json(raw_output)
            location = parsed.get("location")

            if location is not None and isinstance(location, str):
                location = location.strip()
                if not location or location.lower() in ("null", "none"):
                    location = None
                elif len(location) > 255:
                    location = location[:255].strip()

            if location:
                logger.info(
                    f"Location extracted from {filename}",
                    extra={"file_name": filename, "location": location},
                )
            else:
                logger.debug(f"No location found for {filename}")

            return location

        except httpx.TimeoutException:
            logger.warning(f"LLM timeout for location extraction: {filename}")
            return None
        except Exception as e:
            logger.warning(f"Location extraction failed for {filename}: {e}", extra={"file_name": filename})
            return None
