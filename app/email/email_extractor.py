"""Service for extracting email addresses from resumes using OLLAMA LLM."""
import json
import re
from typing import Dict, Optional
import httpx
from httpx import Timeout

from app.config import settings
from app.utils.logging import get_logger
from app.utils.cleaning import normalize_email

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
    
    def _clean_email_from_mixed_text(self, email_str: str) -> Optional[str]:
        """
        Clean email by removing phone numbers and text that got mixed in.
        Handles cases like: 'cybersecurityanalystcontact678-689-7630dongthientrung292k@gmail.com'
        Should extract: 'dongthientrung292k@gmail.com'
        
        Args:
            email_str: Potentially mixed email string
        
        Returns:
            Clean email or None if invalid
        """
        if not email_str or '@' not in email_str:
            return None
        
        # Split into username and domain
        parts = email_str.split('@')
        if len(parts) != 2:
            return None
        
        username = parts[0]
        domain = parts[1]
        
        # Look for phone number patterns in username (XXX-XXX-XXXX, (XXX)XXX-XXXX, etc.)
        # Remove everything before the actual email username
        phone_patterns = [
            r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',  # 678-689-7630
            r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}',  # (678)689-7630
            r'\d{10}',  # 6786897630
        ]
        
        for pattern in phone_patterns:
            # Find phone number in username
            phone_match = re.search(pattern, username)
            if phone_match:
                # Get position after phone number
                phone_end = phone_match.end()
                # Extract username part after phone number
                clean_username = username[phone_end:]
                # If we have a valid username after phone, use it
                if clean_username and len(clean_username) >= 3:
                    # Check if it starts with a letter (valid email username)
                    if clean_username[0].isalpha():
                        return f"{clean_username}@{domain}"
        
        # Check if username starts with phone-like pattern
        # Remove leading digits/dashes that look like phone numbers
        # Pattern: starts with digits, dashes, or parentheses
        leading_phone_match = re.match(r'^[\d\(\)\s\-\.]+([a-zA-Z][a-zA-Z0-9._%+-]*)$', username)
        if leading_phone_match:
            clean_username = leading_phone_match.group(1)
            if len(clean_username) >= 3:
                return f"{clean_username}@{domain}"
        
        # Check if username has text before valid email pattern
        # Look for pattern: text + phone + valid_email_username
        # Example: "contact678-689-7630dong" -> extract "dong"
        # Also handles: "cybersecurityanalystcontact678-689-7630dongthientrung292k"
        text_phone_email = re.search(r'[a-zA-Z]+\d{3}[-.\s]?\d{3}[-.\s]?\d{4}([a-zA-Z][a-zA-Z0-9._%+-]+)$', username)
        if text_phone_email:
            clean_username = text_phone_email.group(1)
            if len(clean_username) >= 3:
                return f"{clean_username}@{domain}"
        
        # More aggressive: Look for any phone pattern followed by valid email username
        # Handles: "cybersecurityanalystcontact678-689-7630dongthientrung292k"
        # Pattern: any text + phone pattern + email username (must start with letter)
        aggressive_clean = re.search(r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}([a-zA-Z][a-zA-Z0-9._%+-]{2,})$', username)
        if aggressive_clean:
            clean_username = aggressive_clean.group(1)
            if len(clean_username) >= 3:
                return f"{clean_username}@{domain}"
        
        # Even more aggressive: Find the longest valid email username pattern at the end
        # This handles cases where text and phone are mixed: "tact678-689-7630dongthientrung292k"
        # Look for email-like pattern at the end (starts with letter, has reasonable length)
        end_email_pattern = re.search(r'([a-zA-Z][a-zA-Z0-9._%+-]{4,})$', username)
        if end_email_pattern:
            potential_username = end_email_pattern.group(1)
            # Check if it's not mostly digits
            digits_ratio = sum(1 for c in potential_username if c.isdigit()) / len(potential_username)
            if digits_ratio < 0.5 and len(potential_username) >= 5:  # At least 5 chars, less than 50% digits
                return f"{potential_username}@{domain}"
        
        # If username is mostly digits or starts with digits, it's likely wrong
        if username and (username[0].isdigit() or sum(c.isdigit() for c in username) / len(username) > 0.6):
            return None
        
        # Return original if no cleaning needed
        return email_str
    
    def _extract_email_regex_fallback(self, text: str) -> Optional[str]:
        """
        Fast regex-based fallback for email extraction.
        Extracts the first valid email address found in the text.
        Uses multiple patterns and scans full text thoroughly.
        Now includes cleaning to remove phone numbers mixed with emails.
        
        Args:
            text: The resume text
        
        Returns:
            Extracted email or None if not found
        """
        if not text:
            return None
        
        # Multiple email regex patterns to catch various formats
        # Order matters: more specific patterns first
        # Prefer patterns that isolate emails (word boundaries, separators)
        email_patterns = [
            # Email with label and colon (most common in headers): Email:email@domain.com
            # Use non-greedy match and ensure we capture the FULL email
            # This pattern is most reliable as it has a clear label
            # Also handles: "Em ail: fkdavis1025@ gm ail.com" (spaces in "Email" and domain)
            re.compile(r'(?:e\s*m\s*a\s*i\s*l|e\s*-\s*m\s*a\s*i\s*l|m\s*a\s*i\s*l)\s*:?\s*([a-zA-Z][a-zA-Z0-9._%+\-\s]*[a-zA-Z0-9])\s*@\s*([a-zA-Z0-9][a-zA-Z0-9.\-\s]*[a-zA-Z0-9])\s*\.\s*([a-zA-Z]{2,})', re.IGNORECASE),
            # Standard email with label (no spaces in label): Email:email@domain.com
            re.compile(r'(?:email|e-mail|mail)\s*:?\s*([a-zA-Z][a-zA-Z0-9._%+-]*[a-zA-Z0-9]@[a-zA-Z0-9][a-zA-Z0-9.-]*[a-zA-Z0-9]\.[a-zA-Z]{2,})', re.IGNORECASE),
            # Email with concatenated label (no space/colon): Emailagenticsam@gmail.com
            # Handles cases where "Email" is directly concatenated with email address
            re.compile(r'(?:email|e-mail|mail)([a-zA-Z][a-zA-Z0-9._%+-]*[a-zA-Z0-9]@[a-zA-Z0-9][a-zA-Z0-9.-]*[a-zA-Z0-9]\.[a-zA-Z]{2,})', re.IGNORECASE),
            # Email isolated by word boundaries (best for clean extraction)
            # Matches: " dongthientrung292k@gmail.com " (isolated email)
            # Also handles: "wgdavis3rd@gmail .com" (space before .com)
            re.compile(r'\b([a-zA-Z][a-zA-Z0-9._%+-]{2,}@[a-zA-Z0-9][a-zA-Z0-9.\-\s]*[a-zA-Z0-9])\s*\.\s*([a-zA-Z]{2,})\b'),
            # Email with pipes separator: |email@domain.com| or |Email:email@domain.com|
            # Handles: "913-249-9434 | LinkedIn Profile | ramuponnaganti8@gmail.com"
            # IMPORTANT: Username must start with letter to avoid phone numbers like "243-3892"
            re.compile(r'\|?\s*([a-zA-Z][a-zA-Z0-9._%+\-\s]*[a-zA-Z0-9])\s*@\s*([a-zA-Z0-9][a-zA-Z0-9.-]*[a-zA-Z0-9]\.[a-zA-Z]{2,})\s*\|?', re.IGNORECASE),
            # Email with bullet point separator: • email@domain.com • or •Email:email@domain.com•
            # Handles: "Portland, OR 97229 • michael.sanganh.ho@gmail.com • (971) 282-1140"
            # IMPORTANT: Username must start with letter to avoid phone numbers
            re.compile(r'[•·]\s*([a-zA-Z][a-zA-Z0-9._%+\-\s]*[a-zA-Z0-9])\s*@\s*([a-zA-Z0-9][a-zA-Z0-9.-]*[a-zA-Z0-9]\.[a-zA-Z]{2,})\s*[•·]?', re.IGNORECASE),
            # Email in mixed text with pipes and label: |Email:email@domain.com|
            # When there's a label, we can be more flexible but still validate
            re.compile(r'\|?\s*(?:email|e-mail|mail)\s*:?\s*([a-zA-Z0-9][a-zA-Z0-9._%+\-\s]*[a-zA-Z0-9])\s*@\s*([a-zA-Z0-9][a-zA-Z0-9.-]*[a-zA-Z0-9]\.[a-zA-Z]{2,})\s*\|?', re.IGNORECASE),
            # Email with brackets: <email@domain.com>
            re.compile(r'<([a-zA-Z0-9][a-zA-Z0-9._%+-]*[a-zA-Z0-9]@[a-zA-Z0-9][a-zA-Z0-9.-]*[a-zA-Z0-9]\.[a-zA-Z]{2,})>'),
            # Email in parentheses: (email@domain.com)
            re.compile(r'\(([a-zA-Z0-9][a-zA-Z0-9._%+-]*[a-zA-Z0-9]@[a-zA-Z0-9][a-zA-Z0-9.-]*[a-zA-Z0-9]\.[a-zA-Z]{2,})\)'),
            # Email with mailto: prefix
            re.compile(r'mailto\s*:?\s*([a-zA-Z0-9][a-zA-Z0-9._%+-]*[a-zA-Z0-9]@[a-zA-Z0-9][a-zA-Z0-9.-]*[a-zA-Z0-9]\.[a-zA-Z]{2,})', re.IGNORECASE),
            # Email with spaces (OCR errors): "email @ domain.com" or "ramuponnaganti 8@gmail.com"
            # This handles cases where there's a space in the email username
            # IMPORTANT: Username must start with a letter to avoid phone numbers
            # Also handles: "wgdavis3rd@gmail .com" (space before .com)
            re.compile(r'\b([a-zA-Z][a-zA-Z0-9._%+\-\s]*[a-zA-Z0-9])\s*@\s*([a-zA-Z0-9][a-zA-Z0-9.\-\s]*[a-zA-Z0-9]\s*\.\s*[a-zA-Z]{2,})\b'),
            # Standard email pattern (word boundary) - most common, but check last
            # IMPORTANT: Username must start with a letter to avoid phone numbers
            re.compile(r'\b([a-zA-Z][a-zA-Z0-9._%+-]*[a-zA-Z0-9]@[a-zA-Z0-9][a-zA-Z0-9.-]*[a-zA-Z0-9]\.[a-zA-Z]{2,})\b'),
            # Email at start of line (common in headers)
            re.compile(r'^([a-zA-Z0-9][a-zA-Z0-9._%+-]*[a-zA-Z0-9]@[a-zA-Z0-9][a-zA-Z0-9.-]*[a-zA-Z0-9]\.[a-zA-Z]{2,})\b', re.MULTILINE),
            # Email with spaces between every character (OCR/spacing issues): "a a m i r . l a l a n i 7 8 6 @ g m a i l . c o m"
            # Pattern: letter/num, space, letter/num, space... @ letter/num, space... . letter, space, letter...
            # This is a special case where every character is separated by a space
            re.compile(r'\b((?:[a-zA-Z0-9]\s+){3,}[a-zA-Z0-9]\s*@\s*(?:[a-zA-Z0-9]\s+){2,}[a-zA-Z0-9]\s*\.\s*(?:[a-zA-Z]\s+){1,2}[a-zA-Z])\b', re.IGNORECASE),
        ]
        
        # Try each pattern and collect all valid matches, then pick the best one
        all_matches = []
        for pattern in email_patterns:
            # Use finditer to get all matches from this pattern
            for match in pattern.finditer(text):
                # Extract email from match
                # Handle patterns with multiple groups (e.g., username and domain separated)
                if len(match.groups()) == 3:
                    # Pattern has three groups (username, domain, TLD) - combine them
                    # Handles: "wgdavis3rd@gmail .com" -> "wgdavis3rd@gmail.com"
                    email_str = f"{match.group(1)}@{match.group(2)}.{match.group(3)}"
                elif len(match.groups()) == 2:
                    # Pattern has two groups (username and domain) - combine them
                    email_str = f"{match.group(1)}@{match.group(2)}"
                elif match.groups():
                    # Pattern has one capturing group - use it
                    email_str = match.group(1)
                else:
                    # No capturing groups - use the full match
                    email_str = match.group(0)
                
                # Clean up any spaces or special characters
                # Remove spaces from email (handles cases like "ramuponnaganti 8@gmail.com" or "fkdavis1025@ gm ail.com")
                # IMPORTANT: Remove all spaces to fix OCR errors like "fkdavis1025@ gm ail.com" -> "fkdavis1025@gmail.com"
                # Also handles: "shariffmaham@gmail.co m" -> "shariffmaham@gmail.com" (space before last letter)
                email_str = email_str.replace(' ', '').replace('<', '').replace('>', '').replace('|', '').replace('•', '').replace('·', '').strip()
                
                # Remove label prefixes if concatenated (e.g., "Emailagenticsam@gmail.com" -> "agenticsam@gmail.com")
                # Check if email starts with common labels
                label_prefixes = ['email', 'e-mail', 'mail', 'contact']
                email_lower = email_str.lower()
                for prefix in label_prefixes:
                    if email_lower.startswith(prefix) and len(email_str) > len(prefix):
                        # Check if the character after prefix is a valid email character (letter or number)
                        next_char = email_str[len(prefix):len(prefix)+1]
                        if next_char and (next_char.isalnum() or next_char in '._%+-'):
                            email_str = email_str[len(prefix):]
                            break
                
                # Additional validation: ensure we have a complete email with @ symbol
                if '@' not in email_str or email_str.count('@') != 1:
                    continue
                
                # Ensure email has both username and domain parts
                parts = email_str.split('@')
                if len(parts) != 2 or not parts[0] or not parts[1]:
                    continue
                
                # Clean email to remove phone numbers and mixed text FIRST
                # This handles cases like: "cybersecurityanalystcontact678-689-7630dongthientrung292k@gmail.com"
                cleaned_email = self._clean_email_from_mixed_text(email_str)
                if not cleaned_email:
                    continue
                
                # Update parts after cleaning
                parts = cleaned_email.split('@')
                if len(parts) != 2:
                    continue
                username = parts[0]
                
                # Additional validation: username should not be mostly digits (likely a phone number)
                digits_in_username = sum(1 for c in username if c.isdigit())
                if len(username) > 0 and digits_in_username / len(username) > 0.7 and len(username) >= 7:
                    # More than 70% digits and at least 7 chars - likely a phone number
                    continue
                
                # Normalize and validate
                email = normalize_email(cleaned_email)
                if email and len(email) > 5:  # Ensure minimum length (e.g., "a@b.co")
                    # Final check: ensure email doesn't start with phone-like pattern
                    if re.match(r'^\d{3}[-.\s]?\d{4}', email.split('@')[0]):
                        continue
                    
                    # Score this email match (prefer shorter, cleaner emails)
                    # Shorter emails are usually cleaner (less mixed text)
                    score = 1000 - len(email_str)  # Prefer shorter original strings
                    if 'email' in text[max(0, match.start()-20):match.start()].lower():
                        score += 100  # Bonus if preceded by "email" label
                    if match.start() < 1000:  # Bonus if in header area
                        score += 50
                    
                    all_matches.append((score, email, email_str))
        
        # Sort by score (highest first) and return the best match
        if all_matches:
            all_matches.sort(key=lambda x: x[0], reverse=True)
            best_email = all_matches[0][1]
            logger.debug(f"Regex fallback extracted email: {best_email} (from {len(all_matches)} candidates)")
            return best_email
        
        # Special handling for emails with spaces in label and domain (OCR errors)
        # Handles: "Em ail: fkdavis1025@ gm ail.com"
        # Look for pattern: "E m a i l" or "E m a i l:" followed by email with spaces
        spaced_email_label = re.compile(r'[eE]\s*[mM]\s*[aA]\s*[iI]\s*[lL]\s*:?\s*([a-zA-Z][a-zA-Z0-9._%+\-\s]*[a-zA-Z0-9])\s*@\s*([a-zA-Z0-9][a-zA-Z0-9.\-\s]*[a-zA-Z0-9])\s*\.\s*([a-zA-Z]{2,})', re.IGNORECASE)
        spaced_match = spaced_email_label.search(text)
        if spaced_match:
            email_str = f"{spaced_match.group(1)}@{spaced_match.group(2)}.{spaced_match.group(3)}"
            # Clean spaces
            email_str = email_str.replace(' ', '').strip()
            cleaned_email = self._clean_email_from_mixed_text(email_str)
            if cleaned_email:
                email = normalize_email(cleaned_email)
                if email and len(email) > 5:
                    logger.debug(f"Regex fallback extracted email with spaced label: {email}")
                    return email
        
        # If no pattern matched, try a more aggressive search
        # Look for @ symbol and try to extract email around it
        at_positions = [i for i, char in enumerate(text) if char == '@']
        for pos in at_positions:
            # Extract more chars before and after @ to handle edge cases
            start = max(0, pos - 60)
            end = min(len(text), pos + 60)
            snippet = text[start:end]
            
            # Try multiple patterns in the snippet
            snippet_patterns = [
                # Pattern for emails with spaces: "shariffmaham@gmail.co m" -> "shariffmaham@gmail.com"
                r'[a-zA-Z0-9._%+-]+\s*@\s*[a-zA-Z0-9.\-\s]*[a-zA-Z0-9]\s*\.\s*[a-zA-Z]{1,2}\s*[a-zA-Z]?',  # With spaces, handles "gmail.co m"
                r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # Standard
                r'[a-zA-Z0-9._%+-]+\s*@\s*[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # With spaces
            ]
            
            for pattern in snippet_patterns:
                email_match = re.search(pattern, snippet)
                if email_match:
                    email_str = email_match.group(0).replace(' ', '').replace('\n', '').replace('\r', '').strip()
                    # Remove any trailing punctuation that might have been captured
                    email_str = re.sub(r'[.,;:!?]+$', '', email_str)
                    
                    # Fix incomplete TLD (e.g., "gmail.co" should be "gmail.com" if we have more context)
                    # Check if domain ends with incomplete TLD
                    if '@' in email_str:
                        parts = email_str.split('@')
                        if len(parts) == 2 and '.' in parts[1]:
                            domain_parts = parts[1].split('.')
                            if len(domain_parts) >= 2 and len(domain_parts[-1]) < 2:
                                # TLD is too short, might be incomplete - check context for more
                                context_after = text[pos:pos+20] if pos+20 < len(text) else text[pos:]
                                tld_match = re.search(r'\.\s*([a-zA-Z]{2,})\b', context_after)
                                if tld_match:
                                    # Found complete TLD in context
                                    email_str = f"{parts[0]}@{'.'.join(domain_parts[:-1])}.{tld_match.group(1)}"
                    
                    # Validate: Check if email contains phone number pattern
                    parts = email_str.split('@')
                    if len(parts) == 2:
                        username = parts[0]
                        # Check if username starts with or contains phone pattern
                        if re.search(r'^\d{3}[-.\s]?\d{4}', username) or re.search(r'\d{3}[-.\s]?\d{4}', username):
                            # Try to extract just the email part after phone number
                            # Look for email pattern that doesn't start with digits
                            clean_email_match = re.search(r'[a-zA-Z][a-zA-Z0-9._%+-]*@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', snippet)
                            if clean_email_match:
                                email_str = clean_email_match.group(0).replace(' ', '').replace('\n', '').replace('\r', '').strip()
                                email_str = re.sub(r'[.,;:!?]+$', '', email_str)
                            else:
                                # Skip this match as it's likely a phone number
                                continue
                        
                        # Additional check: username shouldn't be mostly digits
                        digits_in_username = sum(1 for c in username if c.isdigit())
                        if len(username) > 0 and digits_in_username / len(username) > 0.7 and len(username) >= 7:
                            continue
                    
                    # Clean email to remove phone numbers and mixed text
                    cleaned_email = self._clean_email_from_mixed_text(email_str)
                    if not cleaned_email:
                        continue
                    
                    email = normalize_email(cleaned_email)
                    if email:
                        logger.debug(f"Regex fallback extracted email from @ position: {email}")
                        return email
        
        # Last resort: Look for any @ symbol and try to construct email
        # This handles cases where email might be split or have unusual formatting
        for pos in at_positions[:5]:  # Only check first 5 @ symbols to avoid false positives
            # Get context around @
            start = max(0, pos - 40)
            end = min(len(text), pos + 40)
            context = text[start:end]
            
            # Try to extract username and domain
            # Look for username before @ (alphanumeric, dots, underscores, hyphens)
            # IMPORTANT: Start with a letter, not a digit (to avoid phone numbers)
            username_match = re.search(r'([a-zA-Z][a-zA-Z0-9._%+-]*)\s*@', context)
            if username_match:
                username = username_match.group(1).replace(' ', '').strip()
                
                # Validate: username should not be mostly digits or contain phone patterns
                if re.search(r'\d{3}[-.\s]?\d{4}', username):
                    # Contains phone pattern, skip
                    continue
                
                digits_in_username = sum(1 for c in username if c.isdigit())
                if len(username) > 0 and digits_in_username / len(username) > 0.6:
                    # More than 60% digits - likely not an email
                    continue
                
                # Look for domain after @
                domain_match = re.search(r'@\s*([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', context)
                if domain_match:
                    domain = domain_match.group(1).replace(' ', '').strip()
                    email_str = f"{username}@{domain}"
                    # Clean email to remove phone numbers and mixed text
                    cleaned_email = self._clean_email_from_mixed_text(email_str)
                    if not cleaned_email:
                        continue
                    
                    email = normalize_email(cleaned_email)
                    if email and len(email) > 5:  # Basic validation
                        # Final validation: ensure it doesn't start with phone pattern
                        if not re.match(r'^\d{3}[-.\s]?\d{4}', email.split('@')[0]):
                            logger.debug(f"Regex fallback constructed email from @ context: {email}")
                            return email
        
        return None
    
    async def extract_email(self, resume_text: str, filename: str = "resume", aggressive: bool = False) -> Optional[str]:
        """
        Extract email address from resume text using regex fallback first, then OLLAMA LLM.
        Scans full text multiple times with different strategies.
        
        Args:
            resume_text: The resume text to extract from
            filename: Name of the resume file (for logging)
            aggressive: If True, use more aggressive extraction patterns (for retry attempts)
        
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
        try:
            header_text = resume_text[:2000] if len(resume_text) > 2000 else resume_text
            regex_email = self._extract_email_regex_fallback(header_text)
            if regex_email:
                logger.info(
                    f"✅ EMAIL EXTRACTED via regex from header of {filename}",
                    extra={"file_name": filename, "email": regex_email, "method": "regex_header"}
                )
                return regex_email
        except Exception as e:
            logger.debug(f"Header regex email extraction failed: {e}")
        
        # Step 2: Try fast regex extraction on full text
        try:
            regex_email = self._extract_email_regex_fallback(resume_text)
            if regex_email:
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
            
            logger.info(
                f"✅ EMAIL EXTRACTED from {filename}",
                extra={
                    "file_name": filename,
                    "email": email,
                    "status": "success" if email else "not_found"
                }
            )
            
            return email
            
        except httpx.TimeoutException:
            logger.warning(f"OLLAMA timeout for email extraction: {filename}, returning None")
            return None
        except httpx.HTTPError as e:
            logger.warning(
                f"HTTP error calling OLLAMA for email extraction: {e}",
                extra={"file_name": filename, "error": str(e)}
            )
            return None
        except Exception as e:
            logger.warning(
                f"Error extracting email with LLM: {e}",
                extra={"file_name": filename, "error": str(e)}
            )
            return None
        
        # If LLM extraction failed, return None (regex already tried)
        return None

