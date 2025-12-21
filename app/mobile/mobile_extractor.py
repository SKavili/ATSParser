"""Service for extracting mobile phone numbers from resumes using OLLAMA LLM."""
import json
import re
from typing import Dict, Optional
import httpx
from httpx import Timeout

from app.config import settings
from app.utils.logging import get_logger
from app.utils.cleaning import normalize_phone

logger = get_logger(__name__)

# Try to import OLLAMA Python client
try:
    import ollama
    OLLAMA_CLIENT_AVAILABLE = True
except ImportError:
    OLLAMA_CLIENT_AVAILABLE = False
    logger.warning("OLLAMA Python client not available, using HTTP API directly")

MOBILE_PROMPT = """
IMPORTANT: This is a FRESH, ISOLATED extraction task. Ignore any previous context or conversations.

ROLE:
You are an ATS resume parsing expert specializing in US IT staffing profiles.

CONTEXT:
Candidate profiles and resumes may be unstructured and inconsistently formatted.
Mobile refers to the candidate's phone number or mobile number.

TASK:
Extract the candidate's phone/mobile number from the profile text.

SELECTION RULES:
1. Look for phone numbers in contact information sections.
2. Look for phone numbers in header/footer sections.
3. Look for phone numbers near email addresses or addresses.
4. Extract only the primary phone number (first valid phone found).
5. Accept various formats: (123) 456-7890, 123-456-7890, 123.456.7890, +1-123-456-7890, etc.

CONSTRAINTS:
- Extract only one phone number.
- Preserve the phone number as written (will be normalized).

ANTI-HALLUCINATION RULES:
- If no explicit phone number is found, return null.
- Never guess or infer a phone number.
- Do not create phone numbers from other information.

OUTPUT FORMAT:
Return only valid JSON. No additional text. No explanations. No markdown formatting.

JSON SCHEMA:
{
  "mobile": "string | null"
}

Example valid outputs:
{"mobile": "+1-555-123-4567"}
{"mobile": "(555) 123-4567"}
{"mobile": null}
"""


class MobileExtractor:
    """Service for extracting mobile phone numbers from resume text using OLLAMA LLM."""
    
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
            return {"mobile": None}
        
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
            if isinstance(parsed, dict) and "mobile" in parsed:
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
                    if isinstance(parsed, dict) and "mobile" in parsed:
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
        return {"mobile": None}
    
    def _extract_mobile_from_header(self, text: str) -> Optional[str]:
        """
        Specialized extraction for phone numbers in header section.
        Handles formats like (757)606-0446 where there's no space after closing parenthesis.
        This is a common format in resume headers.
        
        Examples handled:
        - (757)606-0446
        - phone num:(757)606-0446
        - |phone num:(757)606-0446|Email:...|
        - (757) 606-0446 (with space)
        
        Args:
            text: The resume text (should be header section, first 2000 chars)
        
        Returns:
            Extracted phone number or None if not found
        """
        if not text:
            return None
        
        # Pattern for (757)606-0446 format - no space after closing parenthesis
        # Matches: (123)456-7890, (123) 456-7890, (123)4567890, etc.
        # Also handles pipes: |phone num:(757)606-0446| and bullet points: • (971) 282-1140 •
        header_patterns = [
            # Format: |phone num:(757)606-0446| or |phone num:(757)606 -0446| - with pipes and label
            # Handles spaces before dash: "|phone num: (757)606 -0446 |"
            re.compile(r'\|?\s*(?:phone|mobile|tel|cell)\s*(?:num|number)?\s*:?\s*\(([0-9]{3})\)\s*([0-9]{3})\s*[-.\s]?\s*([0-9]{4})\s*\|?', re.IGNORECASE),
            # Format: | ( 470 ) 315 -3949 | - with pipes, spaces inside parentheses and around dash
            # Handles: "Atlanta, GA 98052 | ( 470 ) 315 -3949 | carlottahayes14@gmail.com"
            re.compile(r'\|?\s*\(\s*([0-9]{3})\s*\)\s*([0-9]{3})\s+[-.\s]?\s*([0-9]{4})\s*\|?'),
            # Format: |(757)606-0446| or |(732) 798 -0702| or | 678 -982-0122 | - with pipes, no label
            # Handles: "dhanush29.putti@gmail.com |(732) 798 -0702 |" (space before dash)
            # Handles: "Armand.Kalunga@gmail.com | 678 -982-0122 |" (no parentheses, space before dash)
            re.compile(r'\|?\s*\(?([0-9]{3})\)?\s*([0-9]{3})\s*[-.\s]?\s*([0-9]{4})\s*\|?'),
            # Format: | (202) 820 -1280 | - with pipes and spaces before dash
            # Handles: "| (202) 820 -1280 | meazat29@gmail.com |"
            re.compile(r'\|?\s*\(([0-9]{3})\)\s+([0-9]{3})\s+[-.\s]?\s*([0-9]{4})\s*\|?'),
            # Format: • (971) 282-1140 • or • (971) 282 -1140 • - with bullet point separator
            # Handles: "Portland, OR 97229 • michael.sanganh.ho@gmail.com • (971) 282 -1140"
            re.compile(r'[•·]\s*\(([0-9]{3})\)\s*([0-9]{3})\s*[-.\s]?\s*([0-9]{4})\s*[•·]?'),
            # Format: phone num:(757)606-0446 - without pipes but with label
            re.compile(r'(?:phone|mobile|tel|cell)\s*(?:num|number)?\s*:\(([0-9]{3})\)([0-9]{3})[-.\s]?([0-9]{4})', re.IGNORECASE),
            # Format: (251) 243-3892 - parentheses with space after, then space before dash
            # Handles: "(251) 243-3892" from resume
            re.compile(r'\(([0-9]{3})\)\s+([0-9]{3})\s*[-.\s]?\s*([0-9]{4})\b'),
            # Format: ( 470 ) 315 -3949 - parentheses with spaces inside, then space before dash
            # Handles: "( 470 ) 315 -3949" (spaces inside parentheses, space before dash)
            re.compile(r'\(\s*([0-9]{3})\s*\)\s*([0-9]{3})\s+[-.\s]?\s*([0-9]{4})\b'),
            # Format: (678) -545-9293 - parentheses with space after, then space and dash
            # Handles: "Phone : (678) -545-9293" (space after closing parenthesis, space before dash)
            re.compile(r'\(([0-9]{3})\)\s+[-.\s]?\s*([0-9]{3})[-.\s]?([0-9]{4})\b'),
            # Format: (251) 243-3892 - parentheses with space after, then space before dash
            # Handles: "(251) 243-3892" from resume (space after parentheses, space before dash)
            re.compile(r'\(([0-9]{3})\)\s+([0-9]{3})\s+[-.\s]?\s*([0-9]{4})\b'),
            # Format: (757)606-0446 or (757)606 -0446 - parentheses with optional space before dash
            re.compile(r'\(([0-9]{3})\)\s*([0-9]{3})\s*[-.\s]?\s*([0-9]{4})\b'),
            # Format: (757) 606-0446 or (757) 606 -0446 - parentheses with space, optional space before dash
            re.compile(r'\(([0-9]{3})\)\s+([0-9]{3})\s*[-.\s]?\s*([0-9]{4})\b'),
            # Format: (517)-528-5480 - dash immediately after closing parenthesis
            # Handles: "(517)-528-5480" (dash after closing parenthesis, no space)
            re.compile(r'\(([0-9]{3})\)\s*[-]\s*([0-9]{3})[-.\s]?([0-9]{4})\b'),
            # Format: 404.644.4014 - phone with dots (no parentheses)
            # Handles: "404.644.4014" (dots instead of dashes)
            # Note: No word boundary at end to handle cases like "404.644.4014CHRISTOPHER"
            re.compile(r'\b([0-9]{3})\.([0-9]{3})\.([0-9]{4})(?![0-9])'),
            # Format: 678 -982-0122 - no parentheses, space before dash
            # Handles: "678 -982-0122" (space before dash, no parentheses)
            re.compile(r'\b([0-9]{3})\s+[-.\s]?\s*([0-9]{3})\s*[-.\s]?\s*([0-9]{4})\b'),
            # Format with "phone" or "mobile" label: phone num:(757)606-0446 (with space after colon)
            re.compile(r'(?:phone|mobile|tel|cell)\s*(?:num|number)?\s*:\s*\(?([0-9]{3})\)?\s*([0-9]{3})[-.\s]?([0-9]{4})\b', re.IGNORECASE),
            # Format: phone:(757)606-0446 (no space after colon)
            re.compile(r'(?:phone|mobile|tel|cell)\s*:\(([0-9]{3})\)([0-9]{3})[-.\s]?([0-9]{4})\b', re.IGNORECASE),
            # More flexible: any 10 consecutive digits in header (last resort)
            # Only use this if we're in the first 500 chars (header area)
            re.compile(r'\b([0-9]{3})[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b') if len(text) <= 500 else None,
        ]
        
        for pattern in header_patterns:
            if pattern is None:  # Skip None patterns (conditional patterns)
                continue
            try:
                matches = pattern.findall(text)
                if matches:
                    for match in matches:
                        if isinstance(match, tuple) and len(match) >= 3:
                            # Reconstruct: area code + 3 digits + 4 digits
                            area_code = str(match[0]).strip()
                            first_part = str(match[1]).strip()
                            second_part = str(match[2]).strip()
                            
                            # Combine to get full number
                            phone = f"{area_code}{first_part}{second_part}"
                            digits_only = re.sub(r'[^\d]', '', phone)
                            
                            # Must be exactly 10 digits for US number
                            if len(digits_only) == 10:
                                # Additional validation: area code should not start with 0 or 1
                                if area_code and area_code[0] not in ['0', '1']:
                                    normalized = normalize_phone(phone)
                                    if normalized:
                                        logger.info(
                                            f"Header extraction found phone: {normalized}",
                                            extra={
                                                "phone": normalized,
                                                "area_code": area_code,
                                                "pattern": pattern.pattern[:80]
                                            }
                                        )
                                        return normalized
                                else:
                                    logger.debug(f"Skipping phone with invalid area code: {area_code}")
                            else:
                                logger.debug(f"Skipping phone with wrong digit count: {len(digits_only)} digits")
            except Exception as e:
                logger.debug(f"Error in header pattern matching: {e}")
                continue
        
        return None
    
    def _extract_mobile_regex_fallback(self, text: str) -> Optional[str]:
        """
        Fast regex-based fallback for mobile/phone extraction.
        Extracts the first valid phone number found in the text.
        Uses comprehensive patterns to catch various phone formats.
        
        Args:
            text: The resume text
        
        Returns:
            Extracted phone number or None if not found
        """
        if not text:
            return None
        
        # First, try header-specific extraction (for formats like (757)606-0446)
        # This handles cases where phone is in header with no space after parenthesis
        # Check first 2000 chars (header area) - most common location
        header_text = text[:2000] if len(text) > 2000 else text
        header_phone = self._extract_mobile_from_header(header_text)
        if header_phone:
            logger.debug(f"Found phone using header-specific extraction: {header_phone}")
            return header_phone
        
        # Also check last 1000 chars (footer area) - contact info sometimes in footer
        if len(text) > 2000:
            footer_text = text[-1000:]
            footer_phone = self._extract_mobile_from_header(footer_text)
            if footer_phone:
                logger.debug(f"Found phone using footer-specific extraction: {footer_phone}")
                return footer_phone
        
        # Comprehensive phone number patterns (various formats)
        # Added patterns to handle "|" and "•" separators
        phone_patterns = [
            # US formats with country code: +1 (123) 456-7890, +1-123-456-7890
            re.compile(r'\+1\s*[-.\s]?\s*\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'),
            # Phone with pipes separator: |(732) 798-0702| or |(732) 798 -0702| or | ( 470 ) 315 -3949 |
            # Handles: "dhanush29.putti@gmail.com |(732) 798 -0702 |"
            # Handles: "Atlanta, GA 98052 | ( 470 ) 315 -3949 | carlottahayes14@gmail.com"
            re.compile(r'\|?\s*\(\s*([0-9]{3})\s*\)\s*([0-9]{3})\s+[-.\s]?\s*([0-9]{4})\s*\|?'),
            # Format: |(732) 798-0702| or |(732) 798 -0702| (no spaces inside parentheses)
            re.compile(r'\|?\s*\(([0-9]{3})\)\s*([0-9]{3})\s*[-.\s]?\s*([0-9]{4})\s*\|?'),
            # Phone with bullet point separator: • (971) 282-1140 • or • (971) 282 -1140 •
            # Handles: "Portland, OR 97229 • michael.sanganh.ho@gmail.com • (971) 282 -1140"
            re.compile(r'[•·]\s*\(([0-9]{3})\)\s*([0-9]{3})\s*[-.\s]?\s*([0-9]{4})\s*[•·]?'),
            # US formats without country code: (123) 456-7890, (123) 456 -7890, 123-456-7890, 123.456.7890
            # Updated to handle spaces before dash
            re.compile(r'\(?([0-9]{3})\)?\s*([0-9]{3})\s*[-.\s]?\s*([0-9]{4})\b'),
            # International formats: +91-1234567890, +44-20-1234-5678
            re.compile(r'\+(\d{1,3})[-.\s]?(\d{1,4})[-.\s]?(\d{1,4})[-.\s]?(\d{1,9})\b'),
            # Phone with labels: Phone: 123-456-7890, Mobile: 1234567890, Tel: +1-123-456-7890
            re.compile(r'(?:Phone|Mobile|Tel|Cell|Contact)\s*:?\s*[:\-]?\s*([+]?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})', re.IGNORECASE),
            # Phone with concatenated label (no space/colon): Phone123-456-7890 or Mobile1234567890
            # Handles cases where label is directly concatenated with phone number
            re.compile(r'(?:Phone|Mobile|Tel|Cell|Contact)([+]?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})', re.IGNORECASE),
            # Simple 10 digit numbers (US format)
            re.compile(r'\b(\d{10})\b'),
            # 11 digit numbers starting with 1 (US with country code)
            re.compile(r'\b(1\d{10})\b'),
            # Numbers with spaces: 123 456 7890
            re.compile(r'\b(\d{3})\s+(\d{3})\s+(\d{4})\b'),
            # Numbers with dots: 123.456.7890
            re.compile(r'\b(\d{3})\.(\d{3})\.(\d{4})\b'),
            # Numbers with dashes: 123-456-7890
            re.compile(r'\b(\d{3})-(\d{3})-(\d{4})\b'),
            # Extended format: 10-15 digits (international)
            re.compile(r'\b(\d{10,15})\b'),
        ]
        
        found_numbers = []
        
        # Try each pattern
        for pattern in phone_patterns:
            matches = pattern.findall(text)
            if matches:
                for match in matches:
                    # Handle tuple results (from groups)
                    if isinstance(match, tuple):
                        # Reconstruct phone from groups, filter out empty groups
                        phone = ''.join(str(g) for g in match if g and str(g).strip())
                    else:
                        phone = str(match).strip()
                    
                    # Clean up separators and extra spaces from phone number
                    phone = phone.replace('|', '').replace('•', '').replace('·', '').strip()
                    
                    # Skip if too short or looks like a date/year
                    digits_only = re.sub(r'[^\d]', '', phone)
                    if len(digits_only) < 10:
                        continue
                    
                    # Skip if it looks like a year (1900-2099)
                    if len(digits_only) == 4 and 1900 <= int(digits_only) <= 2099:
                        continue
                    
                    # Skip if it looks like a zip code (5 digits in US context)
                    if len(digits_only) == 5 and not phone.startswith('+'):
                        # Check if it's in a zip code context (near "zip", "postal", etc.)
                        match_pos = text.find(phone)
                        if match_pos > 0:
                            context = text[max(0, match_pos-20):match_pos+len(phone)+20].lower()
                            if any(word in context for word in ['zip', 'postal', 'code', 'address']):
                                continue
                    
                    # Normalize and validate
                    normalized = normalize_phone(phone)
                    if normalized and len(re.sub(r'[^\d]', '', normalized)) >= 10:
                        # Avoid duplicates
                        if normalized not in found_numbers:
                            found_numbers.append(normalized)
                            logger.debug(f"Regex fallback found mobile candidate: {normalized}")
        
        # Return the first valid phone number found
        if found_numbers:
            # Prefer numbers with country code (+1) for US numbers
            for num in found_numbers:
                if num.startswith('+1') and len(re.sub(r'[^\d]', '', num)) == 11:
                    logger.debug(f"Regex fallback extracted mobile (with country code): {num}")
                    return num
            
            # Otherwise return first found
            logger.debug(f"Regex fallback extracted mobile: {found_numbers[0]}")
            return found_numbers[0]
        
        return None
    
    async def extract_mobile(self, resume_text: str, filename: str = "resume") -> Optional[str]:
        """
        Extract mobile phone number from resume text using regex fallback first, then OLLAMA LLM.
        Scans full text multiple times with different strategies.
        
        Args:
            resume_text: The text content of the resume
            filename: Name of the resume file (for logging)
        
        Returns:
            The extracted mobile phone number or None if not found
        """
        if not resume_text or len(resume_text.strip()) < 5:
            logger.warning(f"Resume text too short for mobile extraction: {filename}")
            return None
        
        # Step 1: Try header-specific extraction first (handles formats like (757)606-0446)
        # This is critical for resumes where phone is in header with no space after parenthesis
        try:
            header_text = resume_text[:2000] if len(resume_text) > 2000 else resume_text
            header_mobile = self._extract_mobile_from_header(header_text)
            if header_mobile:
                logger.info(
                    f"✅ MOBILE EXTRACTED via header-specific extraction from {filename}",
                    extra={"file_name": filename, "mobile": header_mobile, "method": "header_specific"}
                )
                return header_mobile
        except Exception as e:
            logger.debug(f"Header-specific mobile extraction failed: {e}")
        
        # Step 2: Try fast regex extraction on full text
        try:
            regex_mobile = self._extract_mobile_regex_fallback(resume_text)
            if regex_mobile:
                logger.info(
                    f"✅ MOBILE EXTRACTED via regex from {filename}",
                    extra={"file_name": filename, "mobile": regex_mobile, "method": "regex"}
                )
                return regex_mobile
        except Exception as e:
            logger.debug(f"Regex mobile extraction failed: {e}")
        
        # Step 3: Try scanning different sections of the resume
        # Phone is often in header (first 2000 chars) or footer (last 1000 chars)
        try:
            # Scan header section with general patterns
            header_text = resume_text[:2000] if len(resume_text) > 2000 else resume_text
            regex_mobile = self._extract_mobile_regex_fallback(header_text)
            if regex_mobile:
                logger.info(
                    f"✅ MOBILE EXTRACTED via regex from header section of {filename}",
                    extra={"file_name": filename, "mobile": regex_mobile, "method": "regex_header"}
                )
                return regex_mobile
            
            # Scan footer section
            if len(resume_text) > 1000:
                footer_text = resume_text[-1000:]
                regex_mobile = self._extract_mobile_regex_fallback(footer_text)
                if regex_mobile:
                    logger.info(
                        f"✅ MOBILE EXTRACTED via regex from footer section of {filename}",
                        extra={"file_name": filename, "mobile": regex_mobile, "method": "regex_footer"}
                    )
                    return regex_mobile
        except Exception as e:
            logger.debug(f"Section-based regex mobile extraction failed: {e}")
        
        # Step 2: Try LLM extraction if regex didn't find anything
        try:
            is_connected, available_model = await self._check_ollama_connection()
            if not is_connected:
                logger.warning(f"OLLAMA not accessible for {filename}, skipping LLM extraction")
                return None
            
            model_to_use = self.model
            if available_model and "llama3.1" not in available_model.lower():
                logger.warning(
                    f"llama3.1 not found, using available model: {available_model}",
                    extra={"available_model": available_model}
                )
                model_to_use = available_model
            
            prompt = f"""{MOBILE_PROMPT}

Input resume text:
{resume_text[:10000]}

Output (JSON only, no other text, no explanations):"""
            
            logger.info(
                f"📤 CALLING OLLAMA API for mobile extraction",
                extra={
                    "file_name": filename,
                    "model": model_to_use,
                    "ollama_host": self.ollama_host,
                }
            )
            
            result = None
            last_error = None
            
            # Reduced timeout from 600s to 60s for faster processing
            async with httpx.AsyncClient(timeout=Timeout(60.0)) as client:
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
                    logger.info("✅ Successfully used /api/generate endpoint for mobile extraction")
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
                        logger.info("Successfully used /api/chat endpoint for mobile extraction")
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
            mobile = parsed_data.get("mobile")
            
            # Normalize phone number
            if mobile:
                mobile = normalize_phone(str(mobile).strip())
            
            logger.info(
                f"✅ MOBILE EXTRACTED from {filename}",
                extra={
                    "file_name": filename,
                    "mobile": mobile,
                    "status": "success" if mobile else "not_found"
                }
            )
            
            return mobile
            
        except httpx.TimeoutException:
            logger.warning(f"OLLAMA timeout for mobile extraction: {filename}, returning None")
            return None
        except httpx.HTTPError as e:
            logger.warning(
                f"HTTP error calling OLLAMA for mobile extraction: {e}",
                extra={"file_name": filename, "error": str(e)}
            )
            return None
        except Exception as e:
            logger.warning(
                f"Error extracting mobile with LLM: {e}",
                extra={"file_name": filename, "error": str(e)}
            )
            return None
        
        # If LLM extraction failed, return None (regex already tried)
        return None

