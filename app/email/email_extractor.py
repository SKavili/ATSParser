"""Service for extracting email addresses from resumes using OLLAMA LLM."""
import json
import re
from typing import Dict, Optional
import httpx
from httpx import Timeout

from app.config import settings
from app.utils.logging import get_logger
from app.utils.cleaning import normalize_email, remove_symbols_and_emojis

logger = get_logger(__name__)

# Try to import OLLAMA Python client
try:
    import ollama
    OLLAMA_CLIENT_AVAILABLE = True
except ImportError:
    OLLAMA_CLIENT_AVAILABLE = False
    logger.warning("OLLAMA Python client not available, using HTTP API directly")

EMAIL_PROMPT = """
IMPORTANT: This is a FRESH, ISOLATED extraction task. Ignore any previous context or conversations.

ROLE:
You are an ATS resume parsing expert specializing in US IT staffing profiles.

CONTEXT:
Candidate profiles and resumes may be unstructured and inconsistently formatted.
Email refers to the candidate's contact email address.

TASK:
Extract the candidate's email address from the profile text.

SELECTION RULES:
1. Look for email addresses in contact information sections.
2. Look for email addresses in header/footer sections.
3. Look for email addresses near phone numbers or addresses.
4. Extract only the primary email address (first valid email found).

CONSTRAINTS:
- Extract only one email address.
- Preserve the email exactly as written (will be normalized to lowercase).
- Email must be in valid format (user@domain.com).

ANTI-HALLUCINATION RULES:
- If no explicit email is found, return null.
- Never guess or infer an email address.
- Do not create email addresses from names or other information.

OUTPUT FORMAT:
Return only valid JSON. No additional text. No explanations. No markdown formatting.

JSON SCHEMA:
{
  "email": "string | null"
}

Example valid outputs:
{"email": "john.doe@example.com"}
{"email": null}
"""

FALLBACK_EMAIL_MOBILE_PROMPT = """
You are an intelligent resume data extraction engine.

The resume may contain icons, symbols, images, special characters, or non-standard formatting (such as 📞, ✉️, ☎, 📍, bullets, headers, or decorative fonts) instead of plain text for contact details.

IMPORTANT: The text has been cleaned to remove symbols and emojis. Look for email and phone patterns in the cleaned text.

Your task is to accurately extract the candidate's EMAIL ADDRESS and MOBILE PHONE NUMBER, even if:

- They appear near contact icons or symbols (which have been removed)
- They are split across lines
- They contain spaces, dots, brackets, or country codes
- They are embedded in headers, footers, or side sections
- They are written next to words like Contact, Phone, Mobile, Email, or icons
- The resume uses non-standard formatting

Extraction Rules:

1. Look for email patterns: text@domain.com format (case insensitive)
2. Look for phone patterns: 10-15 digits, may have +, -, spaces, parentheses
3. Normalize email into standard format (lowercase, no spaces)
4. Normalize mobile number to digits only (retain country code if present like +1)
5. Ignore location, fax, or other numbers
6. If multiple numbers exist, choose the most likely personal mobile number
7. If data is truly not present in the text, return null

Output Format (JSON only):

{"email": "<email_or_null>","mobile": "<mobile_or_null>"} 

Do not add explanations. Do not hallucinate values. Extract only from the given resume content. If email or mobile is not found, return null for that field.
"""


class EmailExtractor:
    """Service for extracting email addresses from resume text using OLLAMA LLM."""
    
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
            return {"email": None}
        
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
            if isinstance(parsed, dict) and "email" in parsed:
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
                    if isinstance(parsed, dict) and "email" in parsed:
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
        return {"email": None}
    
    def _extract_email_regex_fallback(self, text: str) -> Optional[str]:
        """
        Fast regex-based fallback for email extraction.
        Extracts the first valid email address found in the text.
        Uses multiple patterns and scans full text thoroughly.
        
        Args:
            text: The resume text
        
        Returns:
            Extracted email or None if not found
        """
        if not text:
            return None
        
        # Try cleaning text first to remove symbols that might interfere
        cleaned_text = remove_symbols_and_emojis(text)
        if cleaned_text:
            text = cleaned_text
        
        # Multiple email regex patterns to catch various formats
        # Order matters - more specific patterns first
        email_patterns = [
            # Email with label and mailto: Email: mailto:email@domain.com (HTML format)
            re.compile(r'(?:email|e-mail)\s*:\s*mailto\s*:?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b', re.IGNORECASE),
            # Email with mailto: prefix (standalone)
            re.compile(r'mailto\s*:?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b', re.IGNORECASE),
            # Email with label and colon (most common in headers): Email:email@domain.com
            re.compile(r'(?:email|e-mail)\s*:?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b', re.IGNORECASE),
            # Email in mixed text with pipes: |Email:email@domain.com| or |email@domain.com|
            re.compile(r'[|:]\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\s*[|]?', re.IGNORECASE),
            # Email with brackets: <email@domain.com>
            re.compile(r'<([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})>'),
            # Email in parentheses: (email@domain.com)
            re.compile(r'\(([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\)'),
            # Standard email pattern (word boundary)
            re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'),
            # Email with spaces (OCR errors): "email @ domain.com"
            re.compile(r'\b[a-zA-Z0-9._%+-]+\s*@\s*[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'),
            # Email at start of line (common in headers)
            re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', re.MULTILINE),
        ]
        
        # Try each pattern
        for pattern in email_patterns:
            matches = pattern.findall(text)
            if matches:
                # Handle tuple results (from groups)
                if isinstance(matches[0], tuple):
                    email_str = matches[0][0] if matches[0] else matches[0]
                else:
                    email_str = matches[0]
                
                # Clean up any spaces or special characters
                email_str = email_str.replace(' ', '').replace('<', '').replace('>', '').strip()
                
                # Normalize and validate
                email = normalize_email(email_str)
                if email:
                    logger.debug(f"Regex fallback extracted email: {email} (pattern: {pattern.pattern[:50]}...)")
                    return email
        
        # If no pattern matched, try a more aggressive search
        # Look for @ symbol and try to extract email around it
        # This handles cases where email is split by spaces or special characters
        at_positions = [i for i, char in enumerate(text) if char == '@']
        for pos in at_positions:
            # Extract more chars before and after @ to handle edge cases
            # Increased context to capture emails split by spaces and mailto: format
            start = max(0, pos - 100)
            end = min(len(text), pos + 100)
            snippet = text[start:end]
            
            # Try multiple patterns in the snippet, including emails with spaces and mailto:
            snippet_patterns = [
                r'mailto\s*:?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',  # mailto: format first
                r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # Standard (no spaces)
                r'[a-zA-Z0-9._%+-]+\s*@\s*[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # With spaces around @
                r'[a-zA-Z0-9._%+-]+\s+[a-zA-Z0-9._%+-]*\s*@\s*[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # With spaces in username
            ]
            
            for pattern in snippet_patterns:
                email_match = re.search(pattern, snippet, re.IGNORECASE)
                if email_match:
                    # Handle group matches (from mailto: pattern)
                    if email_match.groups():
                        email_str = email_match.group(1)
                    else:
                        email_str = email_match.group(0)
                    # Remove ALL spaces, newlines, and special formatting characters
                    email_str = email_str.replace(' ', '').replace('\n', '').replace('\r', '').replace('\t', '').strip()
                    # Remove any trailing punctuation that might have been captured
                    email_str = re.sub(r'[.,;:!?|]+$', '', email_str)
                    email = normalize_email(email_str)
                    if email:
                        logger.debug(f"Regex fallback extracted email from @ position: {email}")
                        return email
            
            # Special handling for emails split by spaces (e.g., "ramuponnaganti 8@gmail.com")
            # Look for text before @ that might be split
            before_at = text[max(0, pos - 50):pos]
            after_at = text[pos:min(len(text), pos + 50)]
            
            # Try to find username parts before @ (may be split by spaces)
            # Look for alphanumeric sequences before @
            username_parts = re.findall(r'[a-zA-Z0-9._%+-]+', before_at)
            if username_parts:
                # Take the last few parts (in case email is like "name 123 @gmail.com")
                # Combine them to form username
                username = ''.join(username_parts[-3:])  # Take up to 3 parts before @
                
                # Extract domain after @
                domain_match = re.search(r'@\s*([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', after_at)
                if domain_match:
                    domain = domain_match.group(1).replace(' ', '').strip()
                    email_str = f"{username}@{domain}"
                    # Remove any trailing punctuation
                    email_str = re.sub(r'[.,;:!?|]+$', '', email_str)
                    email = normalize_email(email_str)
                    if email and len(email) > 8:  # Basic validation (at least 8 chars like "a@b.co")
                        logger.debug(f"Regex fallback constructed email from split parts: {email}")
                        return email
        
        # Last resort: Look for any @ symbol and try to construct email
        # This handles cases where email might be split by spaces or have unusual formatting
        for pos in at_positions[:5]:  # Only check first 5 @ symbols to avoid false positives
            # Get context around @ (increased context to capture split emails)
            start = max(0, pos - 60)
            end = min(len(text), pos + 60)
            context = text[start:end]
            
            # Try to extract username and domain
            # Look for username before @ - may be split by spaces
            # First try standard pattern (no spaces)
            username_match = re.search(r'([a-zA-Z0-9._%+-]+)\s*@', context)
            if username_match:
                username = username_match.group(1).replace(' ', '').strip()
            else:
                # Try to find username parts that might be split by spaces
                # Look for alphanumeric sequences before @
                before_at_context = text[max(0, pos - 50):pos]
                username_parts = re.findall(r'[a-zA-Z0-9._%+-]+', before_at_context)
                if username_parts:
                    # Take the last few parts and combine them
                    username = ''.join(username_parts[-3:])  # Take up to 3 parts
                else:
                    continue  # Skip this @ if we can't find username
            
            # Look for domain after @
            domain_match = re.search(r'@\s*([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', context)
            if domain_match:
                domain = domain_match.group(1).replace(' ', '').strip()
                email_str = f"{username}@{domain}"
                # Remove any trailing punctuation
                email_str = re.sub(r'[.,;:!?|]+$', '', email_str)
                email = normalize_email(email_str)
                if email and len(email) > 8:  # Basic validation (at least 8 chars like "a@b.co")
                    logger.debug(f"Regex fallback constructed email from @ context: {email}")
                    return email
        
        return None
    
    async def _extract_with_fallback_prompt(self, resume_text: str, filename: str = "resume") -> Dict[str, Optional[str]]:
        """
        Fallback extraction method using specialized prompt for edge cases.
        Extracts both email and mobile together when regular extraction fails.
        Removes symbols and emojis before extraction to improve accuracy.
        
        Args:
            resume_text: The text content of the resume
            filename: Name of the resume file (for logging)
        
        Returns:
            Dictionary with "email" and "mobile" keys, or {"email": None, "mobile": None} if extraction fails
        """
        try:
            is_connected, available_model = await self._check_ollama_connection()
            if not is_connected:
                logger.debug(f"OLLAMA not accessible for fallback extraction: {filename}")
                return {"email": None, "mobile": None}
            
            model_to_use = self.model
            if available_model and "llama3.1" not in available_model.lower():
                model_to_use = available_model
            
            # Remove symbols and emojis before extraction
            cleaned_text = remove_symbols_and_emojis(resume_text)
            if not cleaned_text:
                cleaned_text = resume_text  # Fallback to original if cleaning removes everything
            
            logger.debug(
                f"Cleaned resume text for fallback extraction (removed symbols/emojis)",
                extra={"file_name": filename, "original_length": len(resume_text), "cleaned_length": len(cleaned_text)}
            )
            
            prompt = f"""{FALLBACK_EMAIL_MOBILE_PROMPT}

Resume content:
{cleaned_text[:10000]}

Output (JSON only, no other text, no explanations):"""
            
            logger.info(
                f"📤 CALLING OLLAMA API for fallback email/mobile extraction",
                extra={
                    "file_name": filename,
                    "model": model_to_use,
                    "ollama_host": self.ollama_host,
                }
            )
            
            result = None
            last_error = None
            
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
                except httpx.HTTPStatusError as e:
                    if e.response.status_code != 404:
                        raise
                    last_error = e
                    logger.debug("OLLAMA /api/generate returned 404, trying /api/chat endpoint")
                
                if result is None:
                    try:
                        response = await client.post(
                            f"{self.ollama_host}/api/chat",
                            json={
                                "model": model_to_use,
                                "messages": [
                                    {"role": "system", "content": "You are a fresh, isolated extraction agent. This is a new, independent task with no previous context."},
                                    {"role": "user", "content": prompt}
                                ],
                                "stream": False,
                                "options": {
                                    "temperature": 0.1,
                                    "top_p": 0.9,
                                    "num_predict": 500,
                                }
                            }
                        )
                        response.raise_for_status()
                        result = response.json()
                        if "message" in result and "content" in result["message"]:
                            result = {"response": result["message"]["content"]}
                        else:
                            raise ValueError("Unexpected response format from OLLAMA chat API")
                    except Exception as e2:
                        last_error = e2
                        logger.debug(f"OLLAMA /api/chat also failed: {e2}")
                
                if result is None:
                    logger.warning(f"All OLLAMA API endpoints failed for fallback extraction: {last_error}")
                    return {"email": None, "mobile": None}
            
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
            
            # Extract JSON from response
            cleaned_text = raw_output.strip()
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
                if isinstance(parsed, dict):
                    email = parsed.get("email")
                    mobile = parsed.get("mobile")
                    
                    # Normalize email
                    if email:
                        email = normalize_email(str(email).strip())
                    
                    logger.info(
                        f"✅ FALLBACK EXTRACTION completed for {filename}",
                        extra={
                            "file_name": filename,
                            "email": email,
                            "mobile": mobile,
                            "status": "success"
                        }
                    )
                    
                    return {"email": email, "mobile": mobile}
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON from fallback extraction response: {filename}")
            
            return {"email": None, "mobile": None}
            
        except Exception as e:
            logger.debug(f"Fallback extraction failed: {e}", extra={"file_name": filename, "error": str(e)})
            return {"email": None, "mobile": None}
    
    async def extract_email(self, resume_text: str, filename: str = "resume") -> Optional[str]:
        """
        Extract email address from resume text using regex fallback first, then OLLAMA LLM.
        Scans full text multiple times with different strategies.
        
        Args:
            resume_text: The text content of the resume
            filename: Name of the resume file (for logging)
        
        Returns:
            The extracted email address or None if not found
        """
        if not resume_text or len(resume_text.strip()) < 5:
            logger.warning(f"Resume text too short for email extraction: {filename}")
            return None
        
        # Step 1: Try header-specific extraction first (email is usually in header)
        # Focus on first 2000 characters where contact info typically appears
        # For HTML files, skip emails in forwarding sections
        try:
            header_text = resume_text[:2000] if len(resume_text) > 2000 else resume_text
            
            # For HTML files, filter out forwarding emails
            if filename.lower().endswith(('.html', '.htm')):
                # Skip emails that appear near forwarding keywords
                forwarding_keywords = ['forwarded by', 'to:', 'from:', 'resume link', 'comments:', 'i thought you might be interested']
                lines = header_text.split('\n')
                filtered_lines = []
                skip_section = False
                
                for line in lines:
                    line_lower = line.lower()
                    # Detect forwarding section
                    if any(keyword in line_lower for keyword in forwarding_keywords):
                        skip_section = True
                        continue
                    # Detect end of forwarding section
                    if any(marker in line_lower for marker in ['personal profile', 'name:', 'phone:', 'email:']):
                        skip_section = False
                    # Skip lines in forwarding section
                    if skip_section:
                        continue
                    filtered_lines.append(line)
                
                header_text = '\n'.join(filtered_lines)
            
            regex_email = self._extract_email_regex_fallback(header_text)
            if regex_email:
                # For HTML files, validate that email is not a forwarding email
                if filename.lower().endswith(('.html', '.htm')):
                    # Common forwarding email patterns to skip
                    forwarding_emails = ['rc@mavensoft.com', 'noreply@', 'donotreply@', 'careerbuilder.com', 'monster.com']
                    if any(fwd_email.lower() in regex_email.lower() for fwd_email in forwarding_emails):
                        logger.debug(f"Skipping forwarding email: {regex_email}")
                    else:
                        logger.info(
                            f"✅ EMAIL EXTRACTED via regex from header of {filename}",
                            extra={"file_name": filename, "email": regex_email, "method": "regex_header"}
                        )
                        return regex_email
                else:
                    logger.info(
                        f"✅ EMAIL EXTRACTED via regex from header of {filename}",
                        extra={"file_name": filename, "email": regex_email, "method": "regex_header"}
                    )
                    return regex_email
        except Exception as e:
            logger.debug(f"Header regex email extraction failed: {e}")
        
        # Step 2: Try fast regex extraction on full text
        # For HTML files, prioritize emails in "Personal Profile" or "Name:" sections
        try:
            text_to_search = resume_text
            
            # For HTML files, focus on candidate sections
            if filename.lower().endswith(('.html', '.htm')):
                # Extract emails from "Personal Profile" section first
                # Look for section starting with "Personal Profile" and ending before "Experience"
                personal_profile_match = re.search(r'(?i)Personal\s+Profile.*?(?=Experience|Education|Skills|Company|Work|$)', resume_text, re.DOTALL)
                if personal_profile_match:
                    profile_section = personal_profile_match.group(0)
                    logger.debug(f"Found Personal Profile section in {filename}, length: {len(profile_section)}")
                    regex_email = self._extract_email_regex_fallback(profile_section)
                    if regex_email:
                        # Validate it's not a forwarding email
                        forwarding_emails = ['rc@mavensoft.com', 'noreply@', 'donotreply@', 'careerbuilder.com', 'monster.com']
                        if not any(fwd_email.lower() in regex_email.lower() for fwd_email in forwarding_emails):
                            logger.info(
                                f"✅ EMAIL EXTRACTED via regex from Personal Profile section of {filename}",
                                extra={"file_name": filename, "email": regex_email, "method": "regex_personal_profile"}
                            )
                            return regex_email
                        else:
                            logger.debug(f"Skipped forwarding email in Personal Profile: {regex_email}")
                
                # Also try "Name:" section if Personal Profile didn't work
                name_section_match = re.search(r'(?i)Name\s*:.*?Email\s*:.*?(?=\n|Experience|Education|Skills|$)', resume_text, re.DOTALL)
                if name_section_match:
                    name_section = name_section_match.group(0)
                    logger.debug(f"Found Name section in {filename}, length: {len(name_section)}")
                    regex_email = self._extract_email_regex_fallback(name_section)
                    if regex_email:
                        forwarding_emails = ['rc@mavensoft.com', 'noreply@', 'donotreply@', 'careerbuilder.com', 'monster.com']
                        if not any(fwd_email.lower() in regex_email.lower() for fwd_email in forwarding_emails):
                            logger.info(
                                f"✅ EMAIL EXTRACTED via regex from Name section of {filename}",
                                extra={"file_name": filename, "email": regex_email, "method": "regex_name_section"}
                            )
                            return regex_email
            
            # Fallback to full text search
            regex_email = self._extract_email_regex_fallback(text_to_search)
            if regex_email:
                # For HTML files, validate that email is not a forwarding email
                if filename.lower().endswith(('.html', '.htm')):
                    forwarding_emails = ['rc@mavensoft.com', 'noreply@', 'donotreply@', 'careerbuilder.com', 'monster.com']
                    if any(fwd_email.lower() in regex_email.lower() for fwd_email in forwarding_emails):
                        logger.debug(f"Skipping forwarding email: {regex_email}")
                        # Try to find another email
                        all_emails = re.findall(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', resume_text)
                        for email in all_emails:
                            if not any(fwd_email.lower() in email.lower() for fwd_email in forwarding_emails):
                                normalized = normalize_email(email)
                                if normalized:
                                    logger.info(
                                        f"✅ EMAIL EXTRACTED via regex (skipped forwarding email) from {filename}",
                                        extra={"file_name": filename, "email": normalized, "method": "regex_skip_forwarding"}
                                    )
                                    return normalized
                    else:
                        logger.info(
                            f"✅ EMAIL EXTRACTED via regex from {filename}",
                            extra={"file_name": filename, "email": regex_email, "method": "regex"}
                        )
                        return regex_email
                else:
                    logger.info(
                        f"✅ EMAIL EXTRACTED via regex from {filename}",
                        extra={"file_name": filename, "email": regex_email, "method": "regex"}
                    )
                    return regex_email
        except Exception as e:
            logger.debug(f"Regex email extraction failed: {e}")
        
        # Step 3: Try scanning footer section (sometimes email is at bottom)
        try:
            if len(resume_text) > 1000:
                footer_text = resume_text[-1000:]
                regex_email = self._extract_email_regex_fallback(footer_text)
                if regex_email:
                    logger.info(
                        f"✅ EMAIL EXTRACTED via regex from footer section of {filename}",
                        extra={"file_name": filename, "email": regex_email, "method": "regex_footer"}
                    )
                    return regex_email
        except Exception as e:
            logger.debug(f"Footer regex email extraction failed: {e}")
        
        # Step 4: Try LLM extraction if regex didn't find anything
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
            
            prompt = f"""{EMAIL_PROMPT}

Input resume text:
{resume_text[:10000]}

Output (JSON only, no other text, no explanations):"""
            
            logger.info(
                f"📤 CALLING OLLAMA API for email extraction",
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
                    logger.info("✅ Successfully used /api/generate endpoint for email extraction")
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
                        logger.info("Successfully used /api/chat endpoint for email extraction")
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
            email = parsed_data.get("email")
            
            # Normalize email
            if email:
                email = normalize_email(str(email).strip())
            
            # If regular extraction found email, return it
            if email:
                logger.info(
                    f"✅ EMAIL EXTRACTED from {filename}",
                    extra={
                        "file_name": filename,
                        "email": email,
                        "status": "success"
                    }
                )
                return email
            
            # If regular extraction returned null, try fallback prompt
            logger.info(
                f"⚠️ Regular email extraction returned null for {filename}, trying fallback prompt",
                extra={"file_name": filename, "status": "trying_fallback"}
            )
            try:
                fallback_result = await self._extract_with_fallback_prompt(resume_text, filename)
                email = fallback_result.get("email")
                if email:
                    logger.info(
                        f"✅ EMAIL EXTRACTED via fallback prompt from {filename}",
                        extra={"file_name": filename, "email": email, "method": "fallback_prompt"}
                    )
                    return email
            except Exception as fallback_error:
                logger.debug(f"Fallback prompt extraction failed: {fallback_error}")
            
            logger.info(
                f"❌ EMAIL NOT FOUND in {filename}",
                extra={"file_name": filename, "status": "not_found"}
            )
            return None
            
        except httpx.TimeoutException:
            logger.warning(f"OLLAMA timeout for email extraction: {filename}, trying fallback prompt")
            # Try fallback prompt before giving up
            try:
                fallback_result = await self._extract_with_fallback_prompt(resume_text, filename)
                email = fallback_result.get("email")
                if email:
                    logger.info(
                        f"✅ EMAIL EXTRACTED via fallback prompt (after timeout) from {filename}",
                        extra={"file_name": filename, "email": email, "method": "fallback_prompt"}
                    )
                    return email
            except Exception as fallback_error:
                logger.debug(f"Fallback prompt extraction failed after timeout: {fallback_error}")
            return None
        except httpx.HTTPError as e:
            logger.warning(
                f"HTTP error calling OLLAMA for email extraction: {e}",
                extra={"file_name": filename, "error": str(e)}
            )
            # Try fallback prompt before giving up
            try:
                fallback_result = await self._extract_with_fallback_prompt(resume_text, filename)
                email = fallback_result.get("email")
                if email:
                    logger.info(
                        f"✅ EMAIL EXTRACTED via fallback prompt (after HTTP error) from {filename}",
                        extra={"file_name": filename, "email": email, "method": "fallback_prompt"}
                    )
                    return email
            except Exception as fallback_error:
                logger.debug(f"Fallback prompt extraction failed after HTTP error: {fallback_error}")
            return None
        except Exception as e:
            logger.warning(
                f"Error extracting email with LLM: {e}",
                extra={"file_name": filename, "error": str(e)}
            )
            # Try fallback prompt before giving up
            try:
                fallback_result = await self._extract_with_fallback_prompt(resume_text, filename)
                email = fallback_result.get("email")
                if email:
                    logger.info(
                        f"✅ EMAIL EXTRACTED via fallback prompt from {filename}",
                        extra={"file_name": filename, "email": email, "method": "fallback_prompt"}
                    )
                    return email
            except Exception as fallback_error:
                logger.debug(f"Fallback prompt extraction also failed: {fallback_error}")
            return None
        
        # All extraction methods have been tried, return None
        return None

