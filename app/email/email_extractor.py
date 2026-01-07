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
Extract ALL email addresses found in the profile text and identify the candidate's PRIMARY email address.

SELECTION RULES:
1. Look for email addresses in contact information sections.
2. Look for email addresses in header/footer sections.
3. Look for email addresses near phone numbers or addresses.
4. Extract all valid email addresses found in the resume.
5. The FIRST VALID PRIMARY email must be selected based on domain priority rules.
6.If DOMAIN PRIORITY mails are present, then select the first one as primary email.
7.If DOMAIN PRIORITY mail found then stop the extraction process and return the primary email.
DOMAIN PRIORITY RULES:
First Preference (Original / Personal Emails):
gmail.com,
outlook.com,
hotmail.com,
live.com,
yahoo.com,
icloud.com,
me.com,
zoho.com,
proton.me,
tutanota.com,
and other commonly used public or professional email providers.

Secondary Emails (NOT Original / Proxy / Job Portal Emails):
mail.dice.com,
dice.com,
linkedin.com,
indeedmail.com,
ziprecruiter.com,
glassdoor.com,
monster.com,
workday.com,
greenhouse.io,
lever.co

IMPORTANT DOMAIN HANDLING:
- If the extracted email domain belongs to a job portal or proxy service
  (e.g., mail.dice.com, linkedin.com, indeedmail.com),
  do NOT treat it as the candidate's primary email.
- Only consider emails from commonly used public or professional
  email providers as PRIMARY.
- If ONLY proxy or job portal emails are present,
  mark the primary email as "masked_email".

CONSTRAINTS:
- Extract ALL valid email addresses.
- Preserve emails exactly as written (normalization to lowercase is allowed).
- Email must be in valid format (user@domain.com).
- Return all extracted emails as a comma-separated string.
- Select ONE primary email based on domain priority rules.

ANTI-HALLUCINATION RULES:
- Never guess or infer an email address.
- Do not create email addresses from names or other information.
- If no explicit email is found, return null values.

OUTPUT FORMAT:
Return only valid JSON. No additional text. No explanations. No markdown formatting.

JSON SCHEMA:
{
  "primary_email": "string | masked_email | null",
  "all_emails": "comma_separated_string | null"
}

Example valid outputs:
{"primary_email": "john.doe@gmail.com", "all_emails": "john.doe@gmail.com,john.doe@linkedin.com"}
{"primary_email": "masked_email", "all_emails": "john.doe@mail.dice.com"}
{"primary_email": null, "all_emails": null}
"""

FALLBACK_EMAIL_MOBILE_PROMPT = """
You are an intelligent resume data extraction engine.

The resume may contain icons, symbols, images, special characters, or non-standard formatting
(such as 📞, ✉️, ☎, 📍, bullets, headers, or decorative fonts) instead of plain text for contact details.

IMPORTANT: The text has been cleaned to remove symbols and emojis.
Look for email and phone patterns in the cleaned text.

Your task is to accurately extract ALL EMAIL ADDRESSES and the candidate's MOBILE PHONE NUMBER.

EMAIL DOMAIN PRIORITY RULES:
First Preference (Original / Personal Emails):
gmail.com,
outlook.com,
hotmail.com,
live.com,
yahoo.com,
icloud.com,
me.com,
zoho.com,
proton.me,
tutanota.com,
and other commonly used public or professional email providers.

Secondary Emails (NOT Original / Proxy / Job Portal Emails):
mail.dice.com,
dice.com,
linkedin.com,
indeedmail.com,
ziprecruiter.com,
glassdoor.com,
monster.com,
workday.com,
greenhouse.io,
lever.co

IMPORTANT EMAIL HANDLING:
- Extract ALL valid email addresses found in the resume.
- Normalize emails (lowercase, remove spaces).
- Return ALL extracted emails as a comma-separated string.
- Select ONE primary email based on domain priority rules.
- If only proxy / job portal emails exist, set primary email as "masked_email".

PHONE EXTRACTION RULES:
1. Look for phone patterns: 10-15 digits, may include +, -, spaces, or parentheses.
2. Normalize mobile number to digits only (retain country code like +1 if present).
3. Ignore fax, office, or location numbers.
4. If multiple numbers exist, choose the most likely personal mobile number.

ANTI-HALLUCINATION RULES:
- Do not guess or infer data.
- Extract only from the provided resume content.
- If email or mobile is not found, return null.

OUTPUT FORMAT (JSON ONLY):
{
  "primary_email": "<email | masked_email | null>",
  "all_emails": "<comma_separated_emails | null>",
  "mobile": "<mobile_or_null>"
}

Do not add explanations. Do not hallucinate values.
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
    
    def _clean_and_fix_email(self, email_str: str) -> Optional[str]:
        """
        Clean and fix email addresses that may have extra text appended or formatting issues.
        
        Handles cases like:
        - "evansharenow@gmail.comemail" -> "evansharenow@gmail.com"
        - "user@domain.comemail" -> "user@domain.com"
        - Removes common words appended after TLD (email, com, net, org, etc.)
        - Validates email has reasonable username length
        
        Args:
            email_str: Raw email string that may have issues
            
        Returns:
            Cleaned email or None if invalid
        """
        if not email_str:
            return None
        
        # Remove spaces and normalize
        email_str = email_str.strip().lower()
        
        # Common words that might be appended after email domains
        appended_words = ['email', 'com', 'net', 'org', 'edu', 'gov', 'io', 'co', 'uk', 'us', 'ca', 'au', 'in', 'de', 'fr', 'jp', 'cn', 'info', 'biz', 'name', 'me', 'tv', 'cc', 'ws', 'mobi', 'asia', 'tel']
        
        # Quick check: if email is already valid, check if it has appended words
        # This prevents breaking valid emails like "60m-v6d-xhl@mail.dice.com" or "greg.harritt@gmail.com"
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if re.match(email_pattern, email_str):
            username_part, domain_part_final = email_str.split('@')
            # Require at least 2 characters for username to reject incomplete emails like "08@gmail.com"
            if len(username_part) >= 2 and len(domain_part_final) >= 4:
                # Check if domain ends with appended words (like "comemail" or "gmail.comemail")
                domain_lower = domain_part_final.lower()
                has_appended_words = False
                
                # Check if removing any appended word from the end leaves a valid domain
                # Only flag as needing cleaning if removing the word creates a valid domain
                # This ensures we don't break valid domains like "mail.dice.com" where "com" is the actual TLD
                for word in sorted(appended_words, key=len, reverse=True):
                    if domain_lower.endswith(word) and len(domain_part_final) > len(word):
                        test_domain = domain_part_final[:-len(word)]
                        # Make sure the test domain is still valid and ends with a proper TLD
                        if re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', test_domain):
                            # Get the TLD of the test domain
                            test_tld = test_domain.split('.')[-1].lower()
                            # Get the TLD of the original domain (last part after last dot)
                            original_tld = domain_part_final.split('.')[-1].lower()
                            
                            # Check if original TLD is longer than test TLD (meaning word was appended to TLD)
                            # Or if original ends with word repeated (like "comcom")
                            # Or if test domain is shorter (meaning word was appended)
                            tld_has_appended = len(original_tld) > len(test_tld) and original_tld.startswith(test_tld)
                            has_double_word = domain_lower.endswith(word.lower() + word.lower())
                            is_shorter = len(test_domain) < len(domain_part_final)
                            
                            if tld_has_appended or has_double_word or (is_shorter and test_tld != word.lower()):
                                # Found appended word, needs cleaning
                                has_appended_words = True
                                break
                
                # If no appended words found, email is clean and valid - return immediately
                if not has_appended_words:
                    return email_str.lower()
                # Otherwise, continue to cleaning logic below
        
        # Find the @ symbol position
        at_pos = email_str.find('@')
        if at_pos == -1 or at_pos == 0:
            return None  # No @ or @ at start (invalid)
        
        # Extract username and domain parts
        username = email_str[:at_pos]
        domain_part_raw = email_str[at_pos + 1:]
        
        # Validate username has at least 1 character
        if not username or len(username) == 0:
            return None
        
        # Reject emails with very short usernames that are likely incomplete (like "08@gmail.com")
        # Minimum username length should be at least 2 characters for valid emails
        # Exception: single character usernames are technically valid but rare, so we'll be conservative
        if len(username) < 2:
            # This is likely an incomplete extraction (e.g., "08" from "cherylbailey508")
            return None
        
        # Common TLDs (sorted by length descending to match longer ones first)
        common_tlds = ['info', 'mobi', 'asia', 'name', 'biz', 'tel', 'com', 'net', 'org', 'edu', 'gov', 'io', 'co', 'uk', 'us', 'ca', 'au', 'in', 'de', 'fr', 'jp', 'cn', 'me', 'tv', 'cc', 'ws']
        
        # appended_words is already defined above in the early validation check
        appended_words_pattern = r'(?:' + '|'.join(appended_words) + ')+'
        
        # Strategy: Directly extract valid domain.tld and remove all appended words
        # This handles cases like "gmail.comemail" -> "gmail.com"
        
        # Find the first valid domain.tld pattern
        domain_match = re.search(r'([a-zA-Z0-9][a-zA-Z0-9.-]*\.([a-zA-Z]{2,}))', domain_part_raw)
        if not domain_match:
            return None
        
        matched_domain = domain_match.group(1)  # e.g., "gmail.com" from "gmail.comemail"
        tld_part = domain_match.group(2).lower()  # e.g., "com" or "comemail"
        
        # Clean TLD: remove appended words from TLD itself
        cleaned_tld = tld_part
        for common_tld in sorted(common_tlds, key=len, reverse=True):
            if tld_part.startswith(common_tld):
                remaining = tld_part[len(common_tld):].lower()
                if remaining:
                    # Try to remove all appended words from remaining
                    temp_rem = remaining
                    while temp_rem:
                        found_word = False
                        for word in appended_words:
                            if temp_rem.startswith(word):
                                temp_rem = temp_rem[len(word):]
                                found_word = True
                                break
                        if not found_word:
                            break
                    if temp_rem == '':
                        cleaned_tld = common_tld
                        break
                else:
                    cleaned_tld = common_tld
                    break
        
        # Reconstruct domain
        if '.' in matched_domain:
            domain_base = '.'.join(matched_domain.split('.')[:-1])
            domain_part = f"{domain_base}.{cleaned_tld}"
        else:
            domain_part = cleaned_tld
        
        # Find what comes after the matched domain in the original string
        match_pos = domain_part_raw.find(matched_domain)
        if match_pos != -1:
            after_domain = domain_part_raw[match_pos + len(matched_domain):]
            if after_domain:
                # Remove appended words from after_domain
                after_lower = after_domain.lower()
                temp_after = after_lower
                while temp_after:
                    found_word = False
                    for word in appended_words:
                        if temp_after.startswith(word):
                            temp_after = temp_after[len(word):]
                            found_word = True
                            break
                    if not found_word:
                        break
                # If we removed everything, domain_part is correct
                # If there's still text, it's likely part of next word
        
        # Final direct cleanup: remove appended words from the END of domain_part
        # This is the most reliable method for "gmail.comemail" -> "gmail.com"
        domain_lower = domain_part.lower()
        max_iterations = 10  # Safety limit to prevent infinite loops
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            found_removal = False
            # Sort words by length (longest first) to match longer words first
            # Prioritize "email" as it's the most common appended word
            sorted_words = sorted(appended_words, key=lambda x: (x != 'email', -len(x)))
            for word in sorted_words:
                if domain_lower.endswith(word) and len(domain_part) > len(word):
                    test_domain = domain_part[:-len(word)]
                    # Validate the test domain is still valid
                    if re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', test_domain):
                        # Additional check: ensure we're not removing a valid TLD
                        # If the word is a TLD and the domain ends with it, check if it's actually appended
                        test_tld = test_domain.split('.')[-1].lower()
                        original_tld = domain_part.split('.')[-1].lower()
                        # If removing the word gives us a valid domain with a different TLD, it's appended
                        if test_tld != word.lower() or len(original_tld) > len(test_tld):
                            domain_part = test_domain
                            domain_lower = domain_part.lower()
                            found_removal = True
                            break
            if not found_removal:
                break
        
        # Validate final domain_part
        if not re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', domain_part):
            # Last resort: extract first valid domain.tld and use it
            simple_match = re.search(r'([a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,})', domain_part_raw)
            if simple_match:
                potential = simple_match.group(1)
                # Remove appended words from end (try multiple times)
                pot_lower = potential.lower()
                cleaned_potential = potential
                for _ in range(5):  # Try up to 5 times to remove multiple appended words
                    found_removal = False
                    for word in sorted(appended_words, key=len, reverse=True):
                        if pot_lower.endswith(word) and len(cleaned_potential) > len(word):
                            test = cleaned_potential[:-len(word)]
                            if re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', test):
                                cleaned_potential = test
                                pot_lower = cleaned_potential.lower()
                                found_removal = True
                                break
                    if not found_removal:
                        break
                
                if re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', cleaned_potential):
                    domain_part = cleaned_potential
                elif re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', potential):
                    domain_part = potential
                else:
                    return None
            else:
                return None
        
        # Reconstruct email
        cleaned_email = f"{username}@{domain_part}"
        
        # Final validation: ensure it's a valid email format
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if re.match(email_pattern, cleaned_email):
            # Additional validation: username should be at least 2 chars (reject incomplete like "08@gmail.com")
            # domain should be reasonable
            username_part, domain_part_final = cleaned_email.split('@')
            if len(username_part) >= 2 and len(domain_part_final) >= 4:  # At least 2 char username, "a.co" domain
                return cleaned_email.lower()
        
        # If cleaning resulted in invalid email, try one more simple approach:
        # Directly remove appended words from the entire original email string
        original_full_email = f"{username}@{domain_part_raw}"
        # Try removing appended words from the end of the entire email
        # Prioritize "email" as it's the most common appended word
        email_lower = original_full_email.lower()
        # Check for "email" first (most common case)
        if email_lower.endswith('email') and len(original_full_email) > len('email') + len(username) + 1:
            test_email = original_full_email[:-len('email')]
            if re.match(email_pattern, test_email):
                username_part, domain_part_final = test_email.split('@')
                if len(username_part) >= 2 and len(domain_part_final) >= 4:  # Require at least 2 char username
                    return test_email.lower()
        
        # Try other appended words
        for word in sorted([w for w in appended_words if w != 'email'], key=len, reverse=True):
            if email_lower.endswith(word) and len(original_full_email) > len(word) + len(username) + 1:  # +1 for @
                test_email = original_full_email[:-len(word)]
                if re.match(email_pattern, test_email):
                    username_part, domain_part_final = test_email.split('@')
                    if len(username_part) >= 2 and len(domain_part_final) >= 4:  # Require at least 2 char username
                        return test_email.lower()
        
        # Last resort: if original email was close to valid, try to return it
        # This ensures we don't lose valid emails due to over-aggressive cleaning
        # But still require minimum 2 char username to reject incomplete emails
        if re.match(email_pattern, original_full_email):
            username_part, domain_part_final = original_full_email.split('@')
            if len(username_part) >= 2 and len(domain_part_final) >= 4:  # Require at least 2 char username
                return original_full_email.lower()
        
        return None
    
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
        # Updated patterns to handle emails with extra text appended (like "email" after TLD)
        email_patterns = [
            # Email with label and mailto: Email: mailto:email@domain.com (HTML format)
            re.compile(r'(?:email|e-mail)\s*:\s*mailto\s*:?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:email|com|net|org|edu|gov|io|co|uk|us|ca|au|in|de|fr|jp|cn)*)\b', re.IGNORECASE),
            # Email with mailto: prefix (standalone)
            re.compile(r'mailto\s*:?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:email|com|net|org|edu|gov|io|co|uk|us|ca|au|in|de|fr|jp|cn)*)\b', re.IGNORECASE),
            # Email with label and colon (most common in headers): Email:email@domain.com
            re.compile(r'(?:email|e-mail)\s*:?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:email|com|net|org|edu|gov|io|co|uk|us|ca|au|in|de|fr|jp|cn)*)\b', re.IGNORECASE),
            # Email in mixed text with pipes: |Email:email@domain.com| or |email@domain.com|
            re.compile(r'[|:]\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:email|com|net|org|edu|gov|io|co|uk|us|ca|au|in|de|fr|jp|cn)*)\s*[|]?', re.IGNORECASE),
            # Email with brackets: <email@domain.com>
            re.compile(r'<([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:email|com|net|org|edu|gov|io|co|uk|us|ca|au|in|de|fr|jp|cn)*)>'),
            # Email in parentheses: (email@domain.com)
            re.compile(r'\(([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:email|com|net|org|edu|gov|io|co|uk|us|ca|au|in|de|fr|jp|cn)*)\)'),
            # Standard email pattern with potential extra text (word boundary)
            re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:email|com|net|org|edu|gov|io|co|uk|us|ca|au|in|de|fr|jp|cn)*\b'),
            # Email with spaces (OCR errors): "email @ domain.com"
            re.compile(r'\b[a-zA-Z0-9._%+-]+\s*@\s*[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:email|com|net|org|edu|gov|io|co|uk|us|ca|au|in|de|fr|jp|cn)*\b'),
            # Email at start of line (common in headers)
            re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:email|com|net|org|edu|gov|io|co|uk|us|ca|au|in|de|fr|jp|cn)*\b', re.MULTILINE),
            # Email on its own line or after whitespace (common in contact sections)
            re.compile(r'(?:^|\s)([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:email|com|net|org|edu|gov|io|co|uk|us|ca|au|in|de|fr|jp|cn)*)(?:\s|$)', re.MULTILINE),
            # Standard email pattern (word boundary) - fallback without extra text
            re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'),
        ]
        
        # Try each pattern and collect all matches, then prioritize longer/more complete emails
        all_matches = []
        for pattern in email_patterns:
            try:
                matches = pattern.findall(text)
                if matches:
                    for match in matches:
                        # Handle tuple results (from groups)
                        if isinstance(match, tuple):
                            # Get first non-empty group, or first element if all empty
                            email_str = next((m for m in match if m and len(str(m).strip()) > 0), match[0] if match else '')
                            if not email_str:
                                continue
                        else:
                            email_str = match
                        
                        if not email_str or len(email_str.strip()) == 0:
                            continue
                        
                        # Clean up any spaces or special characters
                        email_str = email_str.replace(' ', '').replace('<', '').replace('>', '').strip()
                        
                        # Skip if too short to be valid
                        if len(email_str) < 5:  # Minimum like "a@b.c"
                            continue
                        
                        # REJECT emails with very short numeric-only usernames (like "08")
                        # But be more lenient - only reject if it's clearly incomplete
                        if '@' in email_str:
                            username_part = email_str.split('@')[0]
                            # Only reject if username is:
                            # 1. Less than 2 chars, OR
                            # 2. Exactly 2 chars and all digits (like "08" from "508")
                            # But allow longer usernames even if they start with digits
                            if len(username_part) < 2:
                                continue  # Too short
                            elif len(username_part) == 2 and username_part.isdigit():
                                # This is likely incomplete (e.g., "08" from "cherylbailey508")
                                # But only skip if we haven't found better matches yet
                                # We'll collect it but prioritize longer ones later
                                pass  # Don't skip yet, let it be collected and filtered later
                        
                        # First try to clean and fix email (handles extra text appended)
                        cleaned_email = self._clean_and_fix_email(email_str)
                        if cleaned_email:
                            # Validate with normalize_email
                            email = normalize_email(cleaned_email)
                            username_len = len(email.split('@')[0])
                            # Require at least 2 chars
                            # For 2-digit usernames, we'll filter them out when prioritizing
                            if username_len >= 2:
                                # Store email with its length for prioritization
                                all_matches.append((email, len(email), email_str))
                        
                        # If cleaning didn't work, try normal normalization
                        if not cleaned_email:
                            email = normalize_email(email_str)
                            username_len = len(email.split('@')[0])
                            # Require at least 2 chars
                            if username_len >= 2:
                                all_matches.append((email, len(email), email_str))
            except Exception as e:
                logger.debug(f"Error processing pattern {pattern.pattern[:50]}: {e}")
                continue
        
        # If we found matches, prioritize longer emails (more complete) and return the best one
        if all_matches:
            # Filter out emails with very short usernames (likely incomplete)
            # Keep only emails with username length >= 2
            valid_matches = [(email, length, orig) for email, length, orig in all_matches 
                            if len(email.split('@')[0]) >= 2]
            
            if valid_matches:
                # Filter out 2-digit-only usernames (like "08") - these are likely incomplete
                # But keep them as fallback if no better matches exist
                best_matches = [(email, length, orig) for email, length, orig in valid_matches
                               if not (len(email.split('@')[0]) == 2 and email.split('@')[0].isdigit())]
                
                # If we have matches after filtering 2-digit usernames, use those
                if best_matches:
                    # Sort by username length first (longest username = more complete), then total length
                    best_matches.sort(key=lambda x: (len(x[0].split('@')[0]), x[1]), reverse=True)
                    best_email = best_matches[0][0]
                    logger.debug(f"Regex fallback extracted email: {best_email} (from {len(best_matches)} best matches out of {len(valid_matches)} valid)")
                    return best_email
                # If only 2-digit usernames found, but we have valid matches, check if they're reasonable
                elif valid_matches:
                    # Sort by total length and return longest if it's reasonable
                    valid_matches.sort(key=lambda x: x[1], reverse=True)
                    longest_email = valid_matches[0][0]
                    # Only return if email is reasonably long (at least 10 chars like "ab@cd.com")
                    if len(longest_email) >= 10:
                        logger.debug(f"Regex fallback extracted email (fallback, 2-digit username): {longest_email} (from {len(valid_matches)} matches)")
                        return longest_email
            else:
                # If no valid matches with >= 2 char username, but we have matches, 
                # check if any are actually valid (might be edge case)
                if all_matches:
                    # Sort by total length and return longest if it's reasonable
                    all_matches.sort(key=lambda x: x[1], reverse=True)
                    longest_email = all_matches[0][0]
                    # Only return if email is reasonably long (at least 10 chars like "ab@cd.com")
                    if len(longest_email) >= 10:
                        logger.debug(f"Regex fallback extracted email (fallback): {longest_email} (from {len(all_matches)} matches)")
                        return longest_email
            # If no valid matches, return None (don't return incomplete emails)
            logger.debug(f"No valid emails found (all {len(all_matches)} matches had short usernames or were too short)")
            return None
        
        # If no pattern matched, try a more aggressive search
        # Look for @ symbol and try to extract email around it
        # This handles cases where email is split by spaces or special characters
        at_positions = [i for i, char in enumerate(text) if char == '@']
        
        # Collect all potential emails from @ positions, then prioritize longer ones
        potential_emails = []
        for pos in at_positions:
            # Extract more chars before and after @ to handle edge cases
            # Increased context to capture emails split by spaces and mailto: format
            # Increased before context to capture longer usernames
            start = max(0, pos - 150)  # Increased from 100 to 150 to capture longer usernames
            end = min(len(text), pos + 100)
            snippet = text[start:end]
            
            # Try multiple patterns in the snippet, including emails with spaces and mailto:
            # Updated to handle emails with extra text appended
            snippet_patterns = [
                r'mailto\s*:?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:email|com|net|org|edu|gov|io|co|uk|us|ca|au|in|de|fr|jp|cn)*)',  # mailto: format first
                r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:email|com|net|org|edu|gov|io|co|uk|us|ca|au|in|de|fr|jp|cn)*',  # Standard (no spaces, with potential extra text)
                r'[a-zA-Z0-9._%+-]+\s*@\s*[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:email|com|net|org|edu|gov|io|co|uk|us|ca|au|in|de|fr|jp|cn)*',  # With spaces around @
                r'[a-zA-Z0-9._%+-]+\s+[a-zA-Z0-9._%+-]*\s*@\s*[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:email|com|net|org|edu|gov|io|co|uk|us|ca|au|in|de|fr|jp|cn)*',  # With spaces in username
                r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # Standard (no spaces, no extra text) - fallback
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
                    
                    # REJECT emails with very short usernames that are likely incomplete
                    # Like "08" from "cherylbailey 508@gmail.com" - these are partial matches
                    if '@' in email_str:
                        username_part = email_str.split('@')[0]
                        # Reject if username is:
                        # 1. Shorter than 2 chars (too short)
                        # 2. Only 2 chars and all numeric (like "08" - likely incomplete)
                        if len(username_part) < 2 or (len(username_part) == 2 and username_part.isdigit()):
                            continue  # Skip this match, look for better ones
                    
                        # Try to clean and fix email (handles extra text appended)
                        cleaned_email = self._clean_and_fix_email(email_str)
                        if cleaned_email:
                            email = normalize_email(cleaned_email)
                            username_len = len(email.split('@')[0])
                            # Require at least 2 chars - we'll filter 2-digit usernames when prioritizing
                            if username_len >= 2:
                                potential_emails.append((email, username_len, email_str))
                        
                        # If cleaning didn't work, try normal normalization
                        if not cleaned_email:
                            email = normalize_email(email_str)
                            username_len = len(email.split('@')[0])
                            # Require at least 2 chars - we'll filter 2-digit usernames when prioritizing
                            if username_len >= 2:
                                potential_emails.append((email, username_len, email_str))
        
        # Special handling for emails split by spaces (e.g., "cherylbailey 508@gmail.com" -> "cherylbailey508@gmail.com")
        # Check each @ position for split usernames - this is CRITICAL for catching split emails
        for pos in at_positions:
            # Increase context before @ to capture longer usernames that might be split
            before_at = text[max(0, pos - 100):pos]  # Increased from 50 to 100
            after_at = text[pos:min(len(text), pos + 50)]
            
            # Try to find username parts before @ (may be split by spaces)
            # Look for alphanumeric sequences before @ - capture more context
            username_parts = re.findall(r'[a-zA-Z0-9._%+-]+', before_at)
            if username_parts:
                # Try different combinations of username parts
                # Start with more parts and work down to find the longest valid username
                max_parts_to_try = min(5, len(username_parts))  # Try up to 5 parts
                for num_parts in range(max_parts_to_try, 0, -1):  # Start with most parts, work down
                    # Take the last N parts and combine them
                    username = ''.join(username_parts[-num_parts:])
                    
                    # Skip if username is too short (likely incomplete)
                    if len(username) < 2:
                        continue
                    
                    # Extract domain after @ (updated to handle potential extra text)
                    domain_match = re.search(r'@\s*([a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:email|com|net|org|edu|gov|io|co|uk|us|ca|au|in|de|fr|jp|cn)*)', after_at)
                    if domain_match:
                        domain = domain_match.group(1).replace(' ', '').strip()
                        email_str = f"{username}@{domain}"
                        # Remove any trailing punctuation
                        email_str = re.sub(r'[.,;:!?|]+$', '', email_str)
                        
                        # Try to clean and fix email (handles extra text appended)
                        cleaned_email = self._clean_and_fix_email(email_str)
                        if cleaned_email and len(cleaned_email.split('@')[0]) >= 2:
                            email = normalize_email(cleaned_email)
                            if email:
                                potential_emails.append((email, len(email.split('@')[0]), email_str))
                                # If we found a good email with longer username, prioritize it
                                # Don't break here - collect all possibilities and sort later
                        
                        # If cleaning didn't work, try normal normalization
                        if not cleaned_email:
                            email = normalize_email(email_str)
                            if email and len(email.split('@')[0]) >= 2:
                                potential_emails.append((email, len(email.split('@')[0]), email_str))
        
        # After checking all @ positions, prioritize longer usernames
        if potential_emails:
            # Filter out 2-digit-only usernames first, but keep them as fallback
            best_emails = [(email, username_len, orig) for email, username_len, orig in potential_emails
                          if not (username_len == 2 and email.split('@')[0].isdigit())]
            
            if best_emails:
                # Sort by username length (longest first), then total email length
                best_emails.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
                best_email = best_emails[0][0]
                logger.debug(f"Regex fallback extracted email from @ positions: {best_email} (from {len(best_emails)} best matches out of {len(potential_emails)} total)")
                return best_email
            else:
                # If only 2-digit usernames found, still return the longest one if reasonable
                potential_emails.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
                best_email = potential_emails[0][0]
                # Only return if email is reasonably long
                if len(best_email) >= 10:
                    logger.debug(f"Regex fallback extracted email from @ positions (2-digit fallback): {best_email} (from {len(potential_emails)} matches)")
                    return best_email
        
        # Final fallback: Try to find ANY email pattern in the text (very permissive)
        # This catches emails that might have been missed by other patterns
        final_fallback_pattern = re.compile(r'\b([a-zA-Z0-9][a-zA-Z0-9._%+-]*@[a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,})\b', re.IGNORECASE)
        final_matches = final_fallback_pattern.findall(text)
        if final_matches:
            # Collect all candidates and prioritize longer usernames
            final_candidates = []
            for email_candidate in final_matches:
                if isinstance(email_candidate, tuple):
                    email_candidate = email_candidate[0] if email_candidate else ''
                if not email_candidate:
                    continue
                
                email_candidate = email_candidate.strip()
                # Skip if too short
                if len(email_candidate) < 5:
                    continue
                
                # REJECT short numeric-only usernames
                if '@' in email_candidate:
                    username_part = email_candidate.split('@')[0]
                    if len(username_part) < 2 or (len(username_part) == 2 and username_part.isdigit()):
                        continue  # Skip short numeric usernames
                
                # Clean and validate
                cleaned = self._clean_and_fix_email(email_candidate)
                if cleaned:
                    username_len = len(cleaned.split('@')[0])
                    if username_len >= 2 and not (username_len == 2 and cleaned.split('@')[0].isdigit()):
                        normalized = normalize_email(cleaned)
                        if normalized:
                            final_candidates.append((normalized, username_len))
                
                # Try without cleaning
                if not cleaned:
                    normalized = normalize_email(email_candidate)
                    username_len = len(normalized.split('@')[0]) if normalized else 0
                    if normalized and username_len >= 2 and not (username_len == 2 and normalized.split('@')[0].isdigit()):
                        final_candidates.append((normalized, username_len))
            
            # If we found candidates, return the one with longest username
            if final_candidates:
                final_candidates.sort(key=lambda x: x[1], reverse=True)  # Sort by username length
                best_email = final_candidates[0][0]
                logger.debug(f"Regex fallback extracted email via final fallback: {best_email} (from {len(final_candidates)} candidates)")
                return best_email
        
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
            
            # Look for domain after @ (updated to handle potential extra text)
            domain_match = re.search(r'@\s*([a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:email|com|net|org|edu|gov|io|co|uk|us|ca|au|in|de|fr|jp|cn)*)', context)
            if domain_match:
                domain = domain_match.group(1).replace(' ', '').strip()
                email_str = f"{username}@{domain}"
                # Remove any trailing punctuation
                email_str = re.sub(r'[.,;:!?|]+$', '', email_str)
                
                # Try to clean and fix email (handles extra text appended)
                cleaned_email = self._clean_and_fix_email(email_str)
                if cleaned_email:
                    email = normalize_email(cleaned_email)
                    if email and len(email) > 8:  # Basic validation (at least 8 chars like "a@b.co")
                        logger.debug(f"Regex fallback constructed and cleaned email from @ context: {email}")
                        return email
                
                # If cleaning didn't work, try normal normalization
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
                    # Handle new format: primary_email, all_emails, mobile
                    primary_email = parsed.get("primary_email")
                    all_emails = parsed.get("all_emails")
                    mobile = parsed.get("mobile")
                    
                    # Use primary_email if available, otherwise fall back to all_emails (first one)
                    email = None
                    if primary_email and primary_email != "masked_email" and primary_email.lower() != "null":
                        email = primary_email
                    elif all_emails:
                        # Extract first email from comma-separated list
                        email_list = [e.strip() for e in str(all_emails).split(',') if e.strip()]
                        if email_list:
                            email = email_list[0]
                    
                    # Clean, fix, and normalize email
                    if email:
                        email_str = str(email).strip()
                        # First try to clean and fix email (handles extra text appended)
                        cleaned_email = self._clean_and_fix_email(email_str)
                        if cleaned_email:
                            email = normalize_email(cleaned_email)
                        else:
                            # If cleaning didn't work, try normal normalization
                            email = normalize_email(email_str)
                    
                    logger.info(
                        f"✅ FALLBACK EXTRACTION completed for {filename}",
                        extra={
                            "file_name": filename,
                            "email": email,
                            "primary_email": primary_email,
                            "all_emails": all_emails,
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
                                # Try to clean and fix email first
                                cleaned = self._clean_and_fix_email(email)
                                normalized = normalize_email(cleaned) if cleaned else normalize_email(email)
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
            # Handle new format: primary_email, all_emails
            primary_email = parsed_data.get("primary_email")
            all_emails = parsed_data.get("all_emails")
            # Fallback to old format for backward compatibility
            email = parsed_data.get("email")
            
            # Use primary_email if available, otherwise fall back to all_emails or old email format
            if not email:
                if primary_email and primary_email != "masked_email" and primary_email.lower() != "null":
                    email = primary_email
                elif all_emails:
                    # Extract first email from comma-separated list
                    email_list = [e.strip() for e in str(all_emails).split(',') if e.strip()]
                    if email_list:
                        email = email_list[0]
            
            # Clean, fix, and normalize email
            if email:
                email_str = str(email).strip()
                # First try to clean and fix email (handles extra text appended)
                cleaned_email = self._clean_and_fix_email(email_str)
                if cleaned_email:
                    email = normalize_email(cleaned_email)
                else:
                    # If cleaning didn't work, try normal normalization
                    email = normalize_email(email_str)
            
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

