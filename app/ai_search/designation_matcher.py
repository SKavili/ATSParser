"""Service for matching candidate designations with query designations using OLLAMA LLM."""
import json
import re
from typing import Optional, Tuple
import httpx
from httpx import Timeout

from app.config import settings
from app.utils.logging import get_logger

# Timeout settings for designation matching
DESIGNATION_MATCH_TIMEOUT = 10.0  # 10 seconds for designation matching

logger = get_logger(__name__)

# Try to import OLLAMA Python client
try:
    import ollama
    OLLAMA_CLIENT_AVAILABLE = True
except ImportError:
    OLLAMA_CLIENT_AVAILABLE = False
    logger.warning("OLLAMA Python client not available, using HTTP API directly")

DESIGNATION_MATCH_PROMPT = """IMPORTANT: This is a FRESH, ISOLATED matching task.
Ignore all prior context, memory, or previous conversations.

ROLE:
You are a job title matching expert for an ATS (Applicant Tracking System).

TASK:
Determine if a candidate's job title/designation matches the required role from a search query.

CONTEXT:
- Job titles can have many variations and synonyms
- Senior/Lead variations of the same role should match
- Industry-specific abbreviations should be recognized (e.g., SDET = QA Automation Engineer)
- Completely different roles should NOT match

MATCHING RULES:
1. EXACT MATCHES: "QA Automation Engineer" = "QA Automation Engineer" → MATCH
2. SYNONYMS: "QA Automation Engineer" = "Automation Test Engineer" → MATCH
3. VARIATIONS: "QA Automation Engineer" = "Senior QA Automation Engineer" → MATCH
4. ABBREVIATIONS: "QA Automation Engineer" = "SDET" → MATCH (if SDET means Software Development Engineer in Test)
5. DIFFERENT ROLES: "QA Automation Engineer" ≠ "Software Engineer" (without QA context) → NO MATCH
6. DIFFERENT ROLES: "QA Automation Engineer" ≠ "UI/UX Specialist" → NO MATCH
7. DIFFERENT ROLES: "QA Automation Engineer" ≠ "Data Analyst" → NO MATCH
8. STUDENT/INTERN: "QA Automation Engineer" ≠ "Software Engineering Student" → NO MATCH

EXAMPLES:
Query: "QA Automation Engineer"
- "Automation Test Engineer" → MATCH (synonym)
- "Test Automation Engineer" → MATCH (synonym)
- "SDET" → MATCH (abbreviation)
- "Senior QA Automation Engineer" → MATCH (senior variation)
- "QA Engineer - Automation" → MATCH (variation)
- "Software Engineer" → NO MATCH (different role, no QA context)
- "UI/UX Specialist" → NO MATCH (completely different)
- "Software Engineering Student" → NO MATCH (student, not professional role)

OUTPUT FORMAT (STRICT JSON ONLY):
{
  "match": true/false,
  "confidence": 0.0-1.0,
  "reason": "Brief explanation"
}

OUTPUT RULES:
- Return ONLY valid JSON
- No markdown, no code blocks
- match: boolean (true if roles match, false otherwise)
- confidence: float between 0.0 and 1.0 (1.0 = perfect match, 0.0 = no match)
- reason: string explaining the decision

Now analyze this match:

Query Role: "{query_designation}"
Candidate Role: "{candidate_designation}"

Output:"""


class DesignationMatcher:
    """Service for matching candidate designations with query designations using LLM."""
    
    def __init__(self):
        self.ollama_host = getattr(settings, 'OLLAMA_HOST', 'http://localhost:11434')
        self.model = getattr(settings, 'OLLAMA_MODEL', 'llama3.1')
        self.cache = {}  # Simple in-memory cache for designation matches
    
    def _get_cache_key(self, query_designation: str, candidate_designation: str) -> str:
        """Generate cache key for designation pair."""
        return f"{query_designation.lower().strip()}|{candidate_designation.lower().strip()}"
    
    async def is_designation_match(
        self, 
        query_designation: str, 
        candidate_designation: str
    ) -> Tuple[bool, float]:
        """
        Use LLM to determine if candidate designation matches query designation.
        
        Args:
            query_designation: Required role from search query (e.g., "QA Automation Engineer")
            candidate_designation: Candidate's job title/designation (e.g., "Automation Test Engineer")
        
        Returns:
            Tuple of (is_match: bool, confidence: float 0.0-1.0)
        """
        if not query_designation or not candidate_designation:
            return False, 0.0
        
        # Normalize inputs
        query_designation = query_designation.strip()
        candidate_designation = candidate_designation.strip()
        
        if not query_designation or not candidate_designation:
            return False, 0.0
        
        # Check cache
        cache_key = self._get_cache_key(query_designation, candidate_designation)
        if cache_key in self.cache:
            logger.debug(
                f"Designation match cache hit: {query_designation} vs {candidate_designation}",
                extra={"query_designation": query_designation, "candidate_designation": candidate_designation}
            )
            return self.cache[cache_key]
        
        # Prepare prompt
        prompt = DESIGNATION_MATCH_PROMPT.format(
            query_designation=query_designation,
            candidate_designation=candidate_designation
        )
        
        try:
            # Try using OLLAMA Python client first
            result = None
            if OLLAMA_CLIENT_AVAILABLE:
                try:
                    import asyncio
                    loop = asyncio.get_event_loop()
                    
                    def _generate():
                        client = ollama.Client(
                            host=self.ollama_host.replace("http://", "").replace("https://", "")
                        )
                        # Request STRICT JSON output from OLLAMA if supported
                        response = client.generate(
                            model=self.model,
                            prompt=prompt,
                            format="json",  # Hint to OLLAMA to return strict JSON
                            options={
                                "temperature": 0.1,
                                "top_p": 0.9,
                            }
                        )
                        # When format=\"json\" is used, many OLLAMA models return JSON directly
                        raw = response.get("response", "") if isinstance(response, dict) else str(response)
                        return {"response": raw}
                    
                    result = await loop.run_in_executor(None, _generate)
                    logger.debug("Successfully used OLLAMA Python client for designation matching")
                except Exception as e:
                    logger.warning(f"OLLAMA Python client failed, falling back to HTTP API: {e}")
                    result = None
            
            # Fallback to HTTP API
            if result is None:
                async with httpx.AsyncClient(timeout=Timeout(DESIGNATION_MATCH_TIMEOUT)) as client:
                    try:
                        response = await client.post(
                            f"{self.ollama_host}/api/generate",
                            json={
                                "model": self.model,
                                "prompt": prompt,
                                "format": "json",  # Ask OLLAMA HTTP API for strict JSON
                                "stream": False,
                                "options": {
                                    "temperature": 0.1,
                                    "top_p": 0.9,
                                }
                            }
                        )
                        response.raise_for_status()
                        result = response.json()
                        logger.debug("Successfully used /api/generate endpoint for designation matching")
                    except httpx.HTTPStatusError as e:
                        if e.response.status_code == 404:
                            # Try /api/chat endpoint
                            logger.debug("OLLAMA /api/generate not found, trying /api/chat endpoint")
                            try:
                                response = await client.post(
                                    f"{self.ollama_host}/api/chat",
                                    json={
                                        "model": self.model,
                                        "messages": [
                                            {"role": "system", "content": "You are a job title matching expert."},
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
                                chat_result = response.json()
                                result = {"response": chat_result.get("message", {}).get("content", "")}
                                logger.debug("Successfully used /api/chat endpoint for designation matching")
                            except Exception as e2:
                                logger.error(f"Both OLLAMA endpoints failed: {e2}")
                                raise
                        else:
                            raise
            
            # Extract JSON from response
            raw_output = result.get("response", "")
            match_result = self._extract_json(raw_output)
            
            # Parse result
            is_match = match_result.get("match", False)
            confidence = float(match_result.get("confidence", 0.0))
            
            # Clamp confidence to [0.0, 1.0]
            confidence = max(0.0, min(1.0, confidence))
            
            # Cache result
            self.cache[cache_key] = (is_match, confidence)
            
            logger.info(
                f"Designation match: query='{query_designation}', candidate='{candidate_designation}', "
                f"match={is_match}, confidence={confidence}",
                extra={
                    "query_designation": query_designation,
                    "candidate_designation": candidate_designation,
                    "match": is_match,
                    "confidence": confidence,
                    "reason": match_result.get("reason", "")
                }
            )
            
            return is_match, confidence
            
        except Exception as e:
            logger.warning(
                f"Designation matching failed: {e}, falling back to keyword matching",
                extra={
                    "query_designation": query_designation,
                    "candidate_designation": candidate_designation,
                    "error": str(e)
                }
            )
            # Fallback to simple keyword matching
            return self._fallback_keyword_match(query_designation, candidate_designation)
    
    def _extract_json(self, text: str) -> dict:
        """Extract JSON from LLM response.

        The LLM is instructed to return strict JSON, but we still
        defensively handle extra text before/after the JSON block.
        """
        if not text:
            return {"match": False, "confidence": 0.0, "reason": "Empty LLM response"}

        # 1) Fast path: try parsing the whole text as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2) Try to locate the first '{' and the last '}' and parse that slice
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate_json = text[start : end + 1]
            try:
                return json.loads(candidate_json)
            except json.JSONDecodeError:
                # Fall through to regex/cleaning
                pass

        # 3) Fallback: try to find a JSON object containing the "match" key
        json_match = re.search(r"\{.*\"match\".*\}", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # 4) Last resort: clean non-JSON characters and try again
        cleaned = re.sub(r'[^\{\}\[\]",:\s\w\.\-]', "", text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # 5) Handle incomplete JSON fragments (e.g., '\n  "match"')
        # Try to extract just the "match" key value if present
        match_patterns = [
            r'"match"\s*:\s*true',
            r'"match"\s*:\s*false',
            r'match"\s*:\s*true',
            r'match"\s*:\s*false',
            r'"match"',  # Just the key name (incomplete)
        ]
        text_lower = text.lower()
        for pattern in match_patterns:
            match_found = re.search(pattern, text_lower, re.IGNORECASE)
            if match_found:
                # Try to extract confidence if present
                confidence_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', text_lower)
                confidence = 0.8 if confidence_match else 0.7
                if confidence_match:
                    try:
                        confidence = float(confidence_match.group(1))
                        confidence = max(0.0, min(1.0, confidence))
                    except (ValueError, TypeError):
                        pass
                
                # Determine match value
                if 'true' in match_found.group().lower():
                    return {"match": True, "confidence": confidence, "reason": "Parsed from incomplete JSON fragment (true detected)"}
                elif 'false' in match_found.group().lower():
                    return {"match": False, "confidence": 0.0, "reason": "Parsed from incomplete JSON fragment (false detected)"}
                else:
                    # If we only found "match" key without value, try to infer from context
                    # Check if there are any positive indicators
                    if any(word in text_lower for word in ['yes', 'match', 'similar', 'related', 'same']):
                        return {"match": True, "confidence": 0.6, "reason": "Inferred match from incomplete JSON with positive context"}
                    else:
                        return {"match": False, "confidence": 0.0, "reason": "Inferred no match from incomplete JSON"}

        # 6) OPTIMIZATION: Safety parser - check for true/false in raw text (for malformed JSON)
        if '"match": true' in text_lower or '"match":true' in text_lower or 'match": true' in text_lower:
            return {"match": True, "confidence": 0.8, "reason": "Parsed from malformed JSON (true detected)"}
        if '"match": false' in text_lower or '"match":false' in text_lower or 'match": false' in text_lower:
            return {"match": False, "confidence": 0.0, "reason": "Parsed from malformed JSON (false detected)"}

        # Default fallback if everything fails
        return {"match": False, "confidence": 0.0, "reason": "Failed to parse LLM response"}
    
    def _fallback_keyword_match(self, query_designation: str, candidate_designation: str) -> Tuple[bool, float]:
        """Conservative fallback keyword-based matching when LLM fails.

        Only very strong text similarity is treated as a match:
        - Exact same normalized title
        - Very high overlap on meaningful words
        This prevents unrelated roles like "software engineering student"
        from matching "QA Automation Engineer".
        """
        query_lower = query_designation.lower().strip()
        candidate_lower = candidate_designation.lower().strip()

        if not query_lower or not candidate_lower:
            return False, 0.0

        # 1) Exact match
        if query_lower == candidate_lower:
            return True, 1.0

        # 2) High-overlap word match on meaningful terms
        query_terms = set(query_lower.split())
        candidate_terms = set(candidate_lower.split())

        # Remove very common / uninformative words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "of",
            "in",
            "on",
            "at",
            "to",
            "for",
            "with",
            "senior",
            "lead",
            "jr",
            "sr",
            "engineer",  # too generic by itself
            "developer",
            "manager",
        }
        query_terms = {t for t in query_terms if t not in stop_words}
        candidate_terms = {t for t in candidate_terms if t not in stop_words}

        if not query_terms or not candidate_terms:
            return False, 0.0

        common_terms = query_terms.intersection(candidate_terms)
        overlap_ratio = len(common_terms) / len(query_terms)

        # Require very high overlap (>= 0.8) to consider it a match
        if overlap_ratio >= 0.8:
            return True, overlap_ratio

        return False, 0.0

