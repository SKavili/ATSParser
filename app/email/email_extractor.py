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

# HIGH-PRIORITY personal domains for PRIMARY email selection
PREFERRED_PRIMARY_DOMAINS = {
    "gmail.com",
    "outlook.com",
    "hotmail.com",
    "live.com",
    "yahoo.com",
    "icloud.com",
    "me.com",
}

# PRIMARY email regex pattern - matches primary domains with .com requirement
PRIMARY_EMAIL_REGEX = re.compile(
    r'[a-zA-Z0-9._%+-]+@(?:gmail|outlook|hotmail|live|yahoo|icloud|me)\.com',
    re.IGNORECASE
)

# SECONDARY proxy domains for fallback extraction
SECONDARY_PROXY_DOMAINS = {
    "mail.dice.com",
    "dice.com",
    "linkedin.com",
    "indeedmail.com",
    "ziprecruiter.com",
    "glassdoor.com",
    "monster.com",
    "workday.com",
    "greenhouse.io",
    "lever.co"
}

# OLLAMA is accessed via HTTP API directly

EMAIL_PROMPT = """
IMPORTANT: This is a FRESH, ISOLATED extraction task.
Ignore any previous context or conversations.

ROLE:
You are an ATS resume parsing expert specializing in US IT staffing profiles.

CONTEXT:
Candidate profiles and resumes may be unstructured and inconsistently formatted.
Email refers ONLY to the candidate's contact email address explicitly written in the resume.

TASK:
Extract  one email addresse found in the resume text and identify the candidate's PRIMARY email address.

CRITICAL EXTRACTION REQUIREMENTS:
- Scan the ENTIRE resume text (header, footer, body, contact section, anywhere).
- Ensure  email must have  both '@' and '.com' . 
- Stop extraction if one email found.
- Preserve email exactly as written (lowercasing allowed).

PRIMARY EMAIL DOMAIN ALLOWLIST (STRICT):
ONLY the following domains are allowed to be selected as "primary_email":

- gmail.com
- outlook.com
- hotmail.com
- live.com
- yahoo.com
- icloud.com
- me.com

SELECTION RULES:
1.Scan the document that resume text provided. 
2.If any email exists whose domain exactly matches one of the allowed domains:
*Select the FIRST such email found.
*Set it as primary_email.
3.If emails exist but none belong to the allowed domain list:
*Set primary_email to "masked_email".
4.If no email exists at all:
Set primary_email to "masked_email".

IMPORTANT DOMAIN HANDLING:
- ONLY emails from the PRIMARY EMAIL DOMAIN ALLOWLIST can be selected as primary_email.
- Any email NOT from the allowlist MUST result in primary_email = "masked_email".

CONSTRAINTS:
- Email must be in valid format (user@domain.com).
- Do NOT infer or guess emails.
- Do NOT fabricate emails.
- If no explicit email exists, return null values.
- "all_emails" must include ALL extracted emails, comma-separated.

ANTI-HALLUCINATION RULES:
- Never create emails from names or usernames.
- Never assume a personal email exists.
- Output only what is explicitly present.

OUTPUT FORMAT:
Return ONLY valid JSON.
No explanations.
No markdown.
No extra text.

JSON SCHEMA:
{
  "primary_email": "string | masked_email | null",
  "all_emails": "comma_separated_string | null"
}

EXAMPLES:
{"primary_email": "john.doe@gmail.com", "charan.kumar@gmail.com","jane.smith@yahoo.com"}
{"primary_email": "jane.smith@yahoo.com","eric.martin@gmail.com}
{"primary_email": null}
"""



class EmailExtractor:
    """Service for extracting email addresses from resume text using OLLAMA LLM."""
    
    def __init__(self):
        self.ollama_host = settings.ollama_host
        self.model = "llama3.1"
        # HTML forwarding email patterns to skip (noreply/forwarding only)
        self.html_forwarding_patterns = [
            "noreply@",
            "donotreply@",
        ]
    
    def _is_forwarding_email(self, email: str, filename: str) -> bool:
        """Check if email is a forwarding/noreply email (for HTML files)."""
        if not filename.lower().endswith(('.html', '.htm')):
            return False
        email_lower = email.lower()
        return any(pattern.lower() in email_lower for pattern in self.html_forwarding_patterns)
    
    def extract_primary_email_with_context(
        self,
        resume_text: str,
        before: int = 400,
        after: int = 200
    ) -> Optional[str]:
        """
        Scan resume text top-to-bottom.
        STOP execution immediately when the FIRST primary email is found.
        Uses 400 chars before and 200 chars after the regex match.
        
        Args:
            resume_text: The text content of the resume
            before: Number of characters to include before match (default: 400)
            after: Number of characters to include after match (default: 200)
        
        Returns:
            First valid primary email found, or None if not found
        """
        if not resume_text:
            return None
        
        for match in PRIMARY_EMAIL_REGEX.finditer(resume_text):
            start, end = match.span()
            
            context = resume_text[
                max(0, start - before) : min(len(resume_text), end + after)
            ]
            
            cleaned_context = remove_symbols_and_emojis(context)
            candidate = self._clean_and_fix_email(match.group())
            
            if candidate:
                email = normalize_email(candidate)
                if email:
                    # Validate email contains both '@' and '.com'
                    if '@' in email and '.com' in email:
                        logger.info(
                            "Primary email found — stopping execution",
                            extra={"email": email, "position": start}
                        )
                        return email
        
        return None
    
    # def extract_secondary_proxy_email(
    #     self,
    #     resume_text: str,
    #     before: int = 400,
    #     after: int = 200
    # ) -> Optional[str]:
    #     """
    #     Executed ONLY if primary extraction fails.
    #     Extracts secondary/proxy emails from allowed proxy domains.
        
    #     Args:
    #         resume_text: The text content of the resume
    #         before: Number of characters to include before match (default: 400)
    #         after: Number of characters to include after match (default: 200)
        
    #     Returns:
    #         First valid secondary proxy email found, or None if not found
    #     """
    #     if not resume_text:
    #         return None
        
    #     proxy_pattern = re.compile(
    #         r'[a-zA-Z0-9._%+-]+@(?:' +
    #         '|'.join(map(re.escape, SECONDARY_PROXY_DOMAINS)) +
    #         r')',
    #         re.IGNORECASE
    #     )
        
    #     for match in proxy_pattern.finditer(resume_text):
    #         start, end = match.span()
            
    #         context = resume_text[
    #             max(0, start - before) : min(len(resume_text), end + after)
    #         ]
            
    #         cleaned_context = remove_symbols_and_emojis(context)
    #         candidate = self._clean_and_fix_email(match.group())
            
    #         if candidate:
    #             email = normalize_email(candidate)
    #             if email:
    #                 # Validate email contains both '@' and '.com'
    #                 if '@' in email and '.com' in email:
    #                     logger.info(
    #                         "Secondary proxy email extracted",
    #                         extra={"email": email}
    #                     )
    #                     return email
        
    #     return None

    def _select_first_valid_email(self, text: str) -> Optional[str]:
        """
        Find the FIRST valid primary domain email in document order (top-to-bottom).
        Stops immediately when the first valid email is found.
        
        Args:
            text: The text to search in
            
        Returns:
            First valid primary domain email found, or None if none found
        """
        if not text:
            return None
        
        # Use regex to find emails in order (finditer preserves order)
        email_pattern = re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', re.IGNORECASE)
        
        for match in email_pattern.finditer(text):
            email_str = match.group(0).strip()
            
            # Clean and normalize the email
            cleaned_email = self._clean_and_fix_email(email_str)
            if cleaned_email:
                normalized = normalize_email(cleaned_email)
            else:
                normalized = normalize_email(email_str)
            
            if not normalized:
                continue
            
            # Validate basic structure
            if "@" not in normalized:
                continue
            username, domain = normalized.split("@", 1)
            if len(username) < 2 or len(domain) < 4:
                continue
            
            # Check if it's a primary domain email
            domain_lower = domain.lower()
            username_lower = username.lower()
            
            # Check if domain is in PREFERRED_PRIMARY_DOMAINS
            is_preferred = domain_lower in PREFERRED_PRIMARY_DOMAINS
            
            # Reject noreply / forwarding patterns
            is_noreply = (
                username_lower.startswith("noreply")
                or username_lower.startswith("no-reply")
                or "noreply" in domain_lower
                or "no-reply" in domain_lower
            )
            
            # Return FIRST valid primary domain email found
            if is_preferred and not is_noreply:
                logger.debug(
                    f"_select_first_valid_email: Found first valid email '{normalized}' at position {match.start()}",
                    extra={"email": normalized, "position": match.start()}
                )
                return normalized
        
        return None
    
    # def _select_best_email_from_list(self, candidates: list[str]) -> Optional[str]:
    #     """
    #     Given a list of raw email strings, normalize them and select the BEST PRIMARY email.  
    #     Selection rules (STRICT PRIMARY-ONLY):
    #     1. ONLY accept emails from PREFERRED_PRIMARY_DOMAINS (gmail.com, outlook.com, etc.)
    #     2. Reject ALL other domains (only primary domains accepted)
    #     3. Within primary domains, prefer longer username part (more complete).
    #     4. As a final tie‑breaker, prefer longer full email string.
    #     5. If NO primary emails exist, return "masked_email".
    #     """
    #     if not candidates:
    #         return None

    #     cleaned_unique: dict[str, str] = {}
    #     rejected_reasons = []

    #     for raw in candidates:
    #         if not raw:
    #             continue
    #         email_str = str(raw).strip()

    #         # First try to clean and fix email (handles extra text appended)
    #         cleaned_email = self._clean_and_fix_email(email_str)
    #         if cleaned_email:
    #             normalized = normalize_email(cleaned_email)
    #         else:
    #             normalized = normalize_email(email_str)

    #         if not normalized:
    #             rejected_reasons.append(f"{email_str[:50]}: normalization failed")
    #             continue

    #         # Ensure basic validity and minimum username length
    #         if "@" not in normalized:
    #             rejected_reasons.append(f"{normalized[:50]}: no @ symbol")
    #             continue
    #         username, domain = normalized.split("@", 1)
    #         if len(username) < 2:
    #             rejected_reasons.append(f"{normalized[:50]}: username too short ({len(username)} chars)")
    #             continue
    #         if len(domain) < 4:
    #             rejected_reasons.append(f"{normalized[:50]}: domain too short ({len(domain)} chars)")
    #             continue

    #         cleaned_unique[normalized] = normalized

    #     if not cleaned_unique:
    #         logger.debug(
    #             f"_select_best_email_from_list: All {len(candidates)} candidates rejected",
    #             extra={"rejected_reasons": rejected_reasons[:5]}  # Log first 5 rejection reasons
    #         )
    #         return None

    #     # Filter to ONLY primary domain emails
    #     primary_emails = []
    #     for email in cleaned_unique.values():
    #         username, domain = email.split("@", 1)
    #         domain_lower = domain.lower()
    #         username_lower = username.lower()
            
    #         # Check if domain is in PREFERRED_PRIMARY_DOMAINS
    #         is_preferred = domain_lower in PREFERRED_PRIMARY_DOMAINS
            
    #         # Reject noreply / forwarding patterns even in primary domains
    #         is_noreply = (
    #             username_lower.startswith("noreply")
    #             or username_lower.startswith("no-reply")
    #             or "noreply" in domain_lower
    #             or "no-reply" in domain_lower
    #         )
            
    #         # ONLY accept preferred primary domains that are not noreply
    #         if is_preferred and not is_noreply:
    #             primary_emails.append(email)
        
    #     # If NO primary emails found, return "masked_email"
    #     if not primary_emails:
    #         logger.debug(
    #             "_select_best_email_from_list: No primary domain emails found, returning 'masked_email'",
    #             extra={
    #                 "total_candidates": len(candidates),
    #                 "example_emails": list(cleaned_unique.values())[:3],
    #             },
    #         )
    #         return "masked_email"
        
    #     # Select best primary email (prefer longer username, then longer email)
    #     def score(email: str) -> tuple[int, int]:
    #         """Return sorting score for an email (higher is better)."""
    #         username, _ = email.split("@", 1)
    #         username_len = len(username)
    #         total_len = len(email)
    #         return (username_len, total_len)
        
    #     best_email = max(primary_emails, key=score)
        
    #     # Log selection details for debugging
    #     all_scores = [(email, score(email)) for email in primary_emails]
    #     all_scores.sort(key=lambda x: x[1], reverse=True)

    #     logger.debug(
    #         f"_select_best_email_from_list: Selected '{best_email}' from {len(cleaned_unique)} valid emails",
    #         extra={
    #             "selected": best_email,
    #             "top_3_scores": all_scores[:3],  # Log top 3 for debugging
    #             "total_candidates": len(candidates)
    #         }
    #     )
    #     return best_email
    
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
    
    # # def _clean_and_fix_email(self, email_str: str) -> Optional[str]:
    # #     """
    # #     Clean and fix email addresses that may have extra text appended or formatting issues.
        
    # #     Handles cases like:
    # #     - "evansharenow@gmail.comemail" -> "evansharenow@gmail.com"
    # #     - "user@domain.comemail" -> "user@domain.com"
    # #     - Removes common words appended after TLD (email, com, net, org, etc.)
    # #     - Validates email has reasonable username length
        
    # #     Args:
    # #         email_str: Raw email string that may have issues
            
    # #     Returns:
    # #         Cleaned email or None if invalid
    # #     """
    # #     if not email_str:
    # #         return None
        
    # #     # Remove spaces and normalize
    # #     email_str = email_str.strip().lower()
        
    # #     # Common words that might be appended after email domains
    # #     appended_words = ['email', 'com', 'net', 'org', 'edu', 'gov', 'io', 'co', 'uk', 'us', 'ca', 'au', 'in', 'de', 'fr', 'jp', 'cn', 'info', 'biz', 'name', 'me', 'tv', 'cc', 'ws', 'mobi', 'asia', 'tel']
        
    # #     # Quick check: if email is already valid, check if it has appended words
    # #     # This prevents breaking valid emails like "john.doe@gmail.com" or "greg.harritt@gmail.com"
    # #     email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    # #     if re.match(email_pattern, email_str):
    # #         username_part, domain_part_final = email_str.split('@')
    # #         # Require at least 2 characters for username to reject incomplete emails like "08@gmail.com"
    # #         if len(username_part) >= 2 and len(domain_part_final) >= 4:
    # #             # Check if domain ends with appended words (like "comemail" or "gmail.comemail")
    # #             domain_lower = domain_part_final.lower()
    # #             has_appended_words = False
                
    # #             # Check if removing any appended word from the end leaves a valid domain
    # #             # Only flag as needing cleaning if removing the word creates a valid domain
    # #             # This ensures we don't break valid domains like "gmail.com" where "com" is the actual TLD
    # #             for word in sorted(appended_words, key=len, reverse=True):
    # #                 if domain_lower.endswith(word) and len(domain_part_final) > len(word):
    # #                     test_domain = domain_part_final[:-len(word)]
    # #                     # Make sure the test domain is still valid and ends with a proper TLD
    # #                     if re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', test_domain):
    # #                         # Get the TLD of the test domain
    # #                         test_tld = test_domain.split('.')[-1].lower()
    # #                         # Get the TLD of the original domain (last part after last dot)
    # #                         original_tld = domain_part_final.split('.')[-1].lower()
                            
    # #                         # Check if original TLD is longer than test TLD (meaning word was appended to TLD)
    # #                         # Or if original ends with word repeated (like "comcom")
    # #                         # Or if test domain is shorter (meaning word was appended)
    # #                         tld_has_appended = len(original_tld) > len(test_tld) and original_tld.startswith(test_tld)
    # #                         has_double_word = domain_lower.endswith(word.lower() + word.lower())
    # #                         is_shorter = len(test_domain) < len(domain_part_final)
                            
    # #                         if tld_has_appended or has_double_word or (is_shorter and test_tld != word.lower()):
    # #                             # Found appended word, needs cleaning
    # #                             has_appended_words = True
    # #                             break
                
    # #             # If no appended words found, email is clean and valid - return immediately
    # #             if not has_appended_words:
    # #                 return email_str.lower()
    # #             # Otherwise, continue to cleaning logic below
        
    # #     # Find the @ symbol position
    # #     at_pos = email_str.find('@')
    # #     if at_pos == -1 or at_pos == 0:
    # #         return None  # No @ or @ at start (invalid)
        
    # #     # Extract username and domain parts
    # #     username = email_str[:at_pos]
    # #     domain_part_raw = email_str[at_pos + 1:]
        
    # #     # Validate username has at least 1 character
    # #     if not username or len(username) == 0:
    # #         return None
        
    # #     # Reject emails with very short usernames that are likely incomplete (like "08@gmail.com")
    # #     # Minimum username length should be at least 2 characters for valid emails
    # #     # Exception: single character usernames are technically valid but rare, so we'll be conservative
    # #     if len(username) < 2:
    # #         # This is likely an incomplete extraction (e.g., "08" from "cherylbailey508")
    # #         return None
        
    # #     # Common TLDs (sorted by length descending to match longer ones first)
    # #     common_tlds = ['info', 'mobi', 'asia', 'name', 'biz', 'tel', 'com', 'net', 'org', 'edu', 'gov', 'io', 'co', 'uk', 'us', 'ca', 'au', 'in', 'de', 'fr', 'jp', 'cn', 'me', 'tv', 'cc', 'ws']
        
    # #     # appended_words is already defined above in the early validation check
    # #     appended_words_pattern = r'(?:' + '|'.join(appended_words) + ')+'
        
    # #     # Strategy: Directly extract valid domain.tld and remove all appended words
    # #     # This handles cases like "gmail.comemail" -> "gmail.com"
        
    # #     # Find the first valid domain.tld pattern
    # #     domain_match = re.search(r'([a-zA-Z0-9][a-zA-Z0-9.-]*\.([a-zA-Z]{2,}))', domain_part_raw)
    # #     if not domain_match:
    # #         return None
        
    # #     matched_domain = domain_match.group(1)  # e.g., "gmail.com" from "gmail.comemail"
    # #     tld_part = domain_match.group(2).lower()  # e.g., "com" or "comemail"
        
    # #     # Clean TLD: remove appended words from TLD itself
    # #     cleaned_tld = tld_part
    # #     for common_tld in sorted(common_tlds, key=len, reverse=True):
    # #         if tld_part.startswith(common_tld):
    # #             remaining = tld_part[len(common_tld):].lower()
    # #             if remaining:
    # #                 # Try to remove all appended words from remaining
    # #                 temp_rem = remaining
    # #                 while temp_rem:
    # #                     found_word = False
    # #                     for word in appended_words:
    # #                         if temp_rem.startswith(word):
    # #                             temp_rem = temp_rem[len(word):]
    # #                             found_word = True
    # #                             break
    # #                     if not found_word:
    # #                         break
    # #                 if temp_rem == '':
    # #                     cleaned_tld = common_tld
    # #                     break
    # #             else:
    # #                 cleaned_tld = common_tld
    # #                 break
        
    # #     # Reconstruct domain
    # #     if '.' in matched_domain:
    # #         domain_base = '.'.join(matched_domain.split('.')[:-1])
    # #         domain_part = f"{domain_base}.{cleaned_tld}"
    # #     else:
    # #         domain_part = cleaned_tld
        
    # #     # Find what comes after the matched domain in the original string
    # #     match_pos = domain_part_raw.find(matched_domain)
    # #     if match_pos != -1:
    # #         after_domain = domain_part_raw[match_pos + len(matched_domain):]
    # #         if after_domain:
    # #             # Remove appended words from after_domain
    # #             after_lower = after_domain.lower()
    # #             temp_after = after_lower
    # #             while temp_after:
    # #                 found_word = False
    # #                 for word in appended_words:
    # #                     if temp_after.startswith(word):
    # #                         temp_after = temp_after[len(word):]
    # #                         found_word = True
    # #                         break
    # #                 if not found_word:
    # #                     break
    # #             # If we removed everything, domain_part is correct
    # #             # If there's still text, it's likely part of next word
        
    # #     # Final direct cleanup: remove appended words from the END of domain_part
    # #     # This is the most reliable method for "gmail.comemail" -> "gmail.com"
    # #     domain_lower = domain_part.lower()
    # #     max_iterations = 10  # Safety limit to prevent infinite loops
    # #     iteration = 0
    # #     while iteration < max_iterations:
    # #         iteration += 1
    # #         found_removal = False
    # #         # Sort words by length (longest first) to match longer words first
    # #         # Prioritize "email" as it's the most common appended word
    # #         sorted_words = sorted(appended_words, key=lambda x: (x != 'email', -len(x)))
    # #         for word in sorted_words:
    # #             if domain_lower.endswith(word) and len(domain_part) > len(word):
    # #                 test_domain = domain_part[:-len(word)]
    # #                 # Validate the test domain is still valid
    # #                 if re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', test_domain):
    # #                     # Additional check: ensure we're not removing a valid TLD
    # #                     # If the word is a TLD and the domain ends with it, check if it's actually appended
    # #                     test_tld = test_domain.split('.')[-1].lower()
    # #                     original_tld = domain_part.split('.')[-1].lower()
    # #                     # If removing the word gives us a valid domain with a different TLD, it's appended
    # #                     if test_tld != word.lower() or len(original_tld) > len(test_tld):
    # #                         domain_part = test_domain
    # #                         domain_lower = domain_part.lower()
    # #                         found_removal = True
    # #                         break
    # #         if not found_removal:
    # #             break
        
    # #     # Validate final domain_part
    # #     if not re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', domain_part):
    # #         # Last resort: extract first valid domain.tld and use it
    # #         simple_match = re.search(r'([a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,})', domain_part_raw)
    # #         if simple_match:
    # #             potential = simple_match.group(1)
    # #             # Remove appended words from end (try multiple times)
    # #             pot_lower = potential.lower()
    # #             cleaned_potential = potential
    # #             for _ in range(5):  # Try up to 5 times to remove multiple appended words
    # #                 found_removal = False
    # #                 for word in sorted(appended_words, key=len, reverse=True):
    # #                     if pot_lower.endswith(word) and len(cleaned_potential) > len(word):
    # #                         test = cleaned_potential[:-len(word)]
    # #                         if re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', test):
    # #                             cleaned_potential = test
    # #                             pot_lower = cleaned_potential.lower()
    # #                             found_removal = True
    # #                             break
    # #                 if not found_removal:
    # #                     break
                
    # #             if re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', cleaned_potential):
    # #                 domain_part = cleaned_potential
    # #             elif re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', potential):
    # #                 domain_part = potential
    # #             else:
    # #                 return None
    # #         else:
    # #             return None
        
    # #     # Reconstruct email
    # #     cleaned_email = f"{username}@{domain_part}"
        
    # #     # Final validation: ensure it's a valid email format
    # #     email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    # #     if re.match(email_pattern, cleaned_email):
    # #         # Additional validation: username should be at least 2 chars (reject incomplete like "08@gmail.com")
    # #         # domain should be reasonable
    # #         username_part, domain_part_final = cleaned_email.split('@')
    # #         if len(username_part) >= 2 and len(domain_part_final) >= 4:  # At least 2 char username, "a.co" domain
    # #             return cleaned_email.lower()
        
    # #     # If cleaning resulted in invalid email, try one more simple approach:
    # #     # Directly remove appended words from the entire original email string
    # #     original_full_email = f"{username}@{domain_part_raw}"
    # #     # Try removing appended words from the end of the entire email
    # #     # Prioritize "email" as it's the most common appended word
    # #     email_lower = original_full_email.lower()
    # #     # Check for "email" first (most common case)
    # #     if email_lower.endswith('email') and len(original_full_email) > len('email') + len(username) + 1:
    # #         test_email = original_full_email[:-len('email')]
    # #         if re.match(email_pattern, test_email):
    # #             username_part, domain_part_final = test_email.split('@')
    # #             if len(username_part) >= 2 and len(domain_part_final) >= 4:  # Require at least 2 char username
    # #                 return test_email.lower()
        
    # #     # Try other appended words
    # #     for word in sorted([w for w in appended_words if w != 'email'], key=len, reverse=True):
    # #         if email_lower.endswith(word) and len(original_full_email) > len(word) + len(username) + 1:  # +1 for @
    # #             test_email = original_full_email[:-len(word)]
    # #             if re.match(email_pattern, test_email):
    # #                 username_part, domain_part_final = test_email.split('@')
    # #                 if len(username_part) >= 2 and len(domain_part_final) >= 4:  # Require at least 2 char username
    # #                     return test_email.lower()
        
    # #     # Last resort: if original email was close to valid, try to return it
    # #     # This ensures we don't lose valid emails due to over-aggressive cleaning
    # #     # But still require minimum 2 char username to reject incomplete emails
    # #     if re.match(email_pattern, original_full_email):
    # #         username_part, domain_part_final = original_full_email.split('@')
    # #         if len(username_part) >= 2 and len(domain_part_final) >= 4:  # Require at least 2 char username
    # #             return original_full_email.lower()
        
    # #     return None
    
    def _extract_json(self, text: str) -> Dict:
        """Extract JSON object from LLM response."""
        if not text:
            logger.warning("Empty response from LLM")
            return {"email": None}
        
        cleaned_text = text.strip()
        # Remove markdown code fences
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()
        
        # Try simple extraction first
        start_idx = cleaned_text.find('{')
        end_idx = cleaned_text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = cleaned_text[start_idx:end_idx + 1]
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict):
                    logger.debug(f"Successfully extracted JSON: {parsed}")
                    return parsed
            except json.JSONDecodeError:
                pass
        
        # Try balanced braces extraction
        start_idx = cleaned_text.find('{')
        if start_idx != -1:
            brace_count = 0
            for i in range(start_idx, len(cleaned_text)):
                if cleaned_text[i] == '{':
                    brace_count += 1
                elif cleaned_text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = cleaned_text[start_idx:i + 1]
                        try:
                            parsed = json.loads(json_str)
                            if isinstance(parsed, dict):
                                logger.debug(f"Successfully extracted JSON with balanced braces: {parsed}")
                                return parsed
                        except json.JSONDecodeError:
                            pass
                        break
        
        logger.error(
            "ERROR: Failed to parse JSON from LLM response", 
            extra={
                "response_preview": text[:500],
                "response_length": len(text),
                "cleaned_preview": cleaned_text[:500]
            }
        )
        return {"email": None}
    

    
    def _extract_all_emails_regex(self, text: str) -> set:
        """
        Extract ALL email addresses from text using comprehensive regex patterns.
        This is similar to _extract_email_regex_fallback() but collects ALL unique emails.
        
        Args:
            text: The text to extract emails from
        
        Returns:
            Set of all unique normalized email addresses found
        """
        if not text:
            return set()
        
        # Try cleaning text first to remove symbols that might interfere
        cleaned_text = remove_symbols_and_emojis(text)
        if cleaned_text:
            text = cleaned_text
        
        all_emails_found = set()  # Use set to avoid duplicates
        
        # Use the same comprehensive patterns as _extract_email_regex_fallback
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
        
        # Try each pattern and collect ALL matches
        for pattern in email_patterns:
            try:
                matches = pattern.findall(text)
                if matches:
                    for match in matches:
                        # Handle tuple results (from groups)
                        if isinstance(match, tuple):
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
                        # But ALLOW usernames with numbers if they contain letters (like "cherylbailey508" or "user123")
                        if '@' in email_str:
                            username_part = email_str.split('@')[0]
                            if len(username_part) < 2:
                                continue  # Too short
                            elif len(username_part) == 2 and username_part.isdigit():
                                # Only reject if it's EXACTLY 2 digits with no letters
                                # This catches "08" from "508" but allows "a1" or "1a"
                                continue  # Likely incomplete (e.g., "08" from "508")
                            # Allow usernames like "cherylbailey508", "greg.harriett", "user123" etc.
                        
                        # Try to clean and fix email (handles extra text appended)
                        cleaned_email = self._clean_and_fix_email(email_str)
                        if cleaned_email:
                            email = normalize_email(cleaned_email)
                            if email and '@' in email:
                                username, domain = email.split('@', 1)
                                if len(username) >= 2 and len(domain) >= 4:
                                    all_emails_found.add(email)
                        
                        # If cleaning didn't work, try normal normalization
                        if not cleaned_email:
                            email = normalize_email(email_str)
                            if email and '@' in email:
                                username, domain = email.split('@', 1)
                                if len(username) >= 2 and len(domain) >= 4:
                                    all_emails_found.add(email)
                        
                        # FINAL FALLBACK: If normalization failed but email looks valid, add it anyway
                        if not cleaned_email and not normalize_email(email_str):
                            # Check if email_str itself looks valid
                            if '@' in email_str and '.' in email_str:
                                username_check, domain_check = email_str.split('@', 1)
                                if len(username_check) >= 2 and len(domain_check) >= 4 and '.' in domain_check:
                                    # Add directly if it looks like a valid email
                                    all_emails_found.add(email_str.lower().strip())
            except Exception as e:
                logger.debug(f"Error processing pattern in _extract_all_emails_regex: {e}")
                continue
        
        # Also check @ positions for emails that might be split by spaces
        at_positions = [i for i, char in enumerate(text) if char == '@']
        for pos in at_positions[:20]:  # Check first 20 @ symbols
            try:
                # Get context around @
                start = max(0, pos - 150)
                end = min(len(text), pos + 100)
                snippet = text[start:end]
                
                # Try multiple patterns in the snippet
                snippet_patterns = [
                    r'mailto\s*:?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:email|com|net|org|edu|gov|io|co|uk|us|ca|au|in|de|fr|jp|cn)*)',
                    r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:email|com|net|org|edu|gov|io|co|uk|us|ca|au|in|de|fr|jp|cn)*',
                    r'[a-zA-Z0-9._%+-]+\s*@\s*[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:email|com|net|org|edu|gov|io|co|uk|us|ca|au|in|de|fr|jp|cn)*',
                    r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
                ]
                
                for pattern_str in snippet_patterns:
                    email_match = re.search(pattern_str, snippet, re.IGNORECASE)
                    if email_match:
                        if email_match.groups():
                            email_str = email_match.group(1)
                        else:
                            email_str = email_match.group(0)
                        
                        email_str = email_str.replace(' ', '').replace('\n', '').replace('\r', '').replace('\t', '').strip()
                        email_str = re.sub(r'[.,;:!?|]+$', '', email_str)
                        
                        if '@' in email_str:
                            username_part = email_str.split('@')[0]
                            # More lenient: only reject if username is too short OR is exactly 2 digits with no letters
                            # Allow usernames with numbers if they contain letters (like "cherylbailey508")
                            if len(username_part) < 2:
                                continue  # Too short
                            elif len(username_part) == 2 and username_part.isdigit():
                                # Only reject if it's EXACTLY 2 digits with no letters
                                continue
                            # Allow all other usernames (including those with numbers like "cherylbailey508")
                            
                            cleaned_email = self._clean_and_fix_email(email_str)
                            if cleaned_email:
                                email = normalize_email(cleaned_email)
                                if email and '@' in email:
                                    username, domain = email.split('@', 1)
                                    if len(username) >= 2 and len(domain) >= 4:
                                        all_emails_found.add(email)
                            
                            if not cleaned_email:
                                email = normalize_email(email_str)
                                if email and '@' in email:
                                    username, domain = email.split('@', 1)
                                    if len(username) >= 2 and len(domain) >= 4:
                                        all_emails_found.add(email)
                        break  # Found email at this position, move to next @
            except Exception as e:
                logger.debug(f"Error processing @ position in _extract_all_emails_regex: {e}")
                continue
        
        return all_emails_found
   
	
    async def extract_all_emails(self, resume_text: str, filename: str = "resume") -> str:
        """
        Extract ALL email addresses from resume text and return as comma-separated string.
        Uses comprehensive extraction logic to find ALL emails in the resume.
        This is the PRIMARY method for email extraction - it finds ALL emails.
        
        Args:
            resume_text: The text content of the resume
            filename: Name of the resume file (for logging)
        
        Returns:
            Comma-separated string of all emails found, or empty string if none found
        """
        if not resume_text or len(resume_text.strip()) < 5:
            logger.warning(f"Resume text too short for email extraction: {filename}")
            return ""
        
        all_emails_found = set()
        
        try:
            # Step 1: Use comprehensive regex extraction on full text (PRIMARY METHOD)
            regex_emails = self._extract_all_emails_regex(resume_text)
            all_emails_found.update(regex_emails)
            logger.debug(
                f"Step 1: Found {len(regex_emails)} emails in full text for {filename}",
                extra={"emails": list(regex_emails)[:5]}
            )
            
            # Step 2: Also try header section (first 3000 chars) - emails are often in header
            # Increased from 2000 to 3000 to catch more emails
            if len(resume_text) > 3000:
                header_text = resume_text[:3000]
                header_emails = self._extract_all_emails_regex(header_text)
                all_emails_found.update(header_emails)
                logger.debug(
                    f"Step 2: Found {len(header_emails)} emails in header for {filename}",
                    extra={"emails": list(header_emails)[:5]}
                )
            else:
                # If text is shorter, header is the same as full text, so skip
                pass
            
            # Step 3: Also try footer section (last 1500 chars) - sometimes emails are at bottom
            if len(resume_text) > 1500:
                footer_text = resume_text[-1500:]
                footer_emails = self._extract_all_emails_regex(footer_text)
                all_emails_found.update(footer_emails)
                logger.debug(
                    f"Step 3: Found {len(footer_emails)} emails in footer for {filename}",
                    extra={"emails": list(footer_emails)[:5]}
                )
            
            # Step 4: Try LLM extraction as additional source (non-blocking, optional)
            # The enhanced prompt explicitly instructs LLM to extract ALL emails
            try:
                # Use a shorter text for LLM (first 5000 chars) to avoid token limits
                llm_text = resume_text[:5000] if len(resume_text) > 5000 else resume_text
                prompt = f"{EMAIL_PROMPT}\n\nRESUME TEXT:\n{llm_text}\n\nExtract ALL email addresses found in the resume text above."
                
                # Try LLM extraction (but don't fail if it errors)
                async with httpx.AsyncClient(timeout=30.0) as client:
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
                                    "num_predict": 500,
                                }
                            }
                        )
                        response.raise_for_status()
                        result = response.json()
                        raw_output = result.get("response", "")
                        
                        # Extract JSON from LLM response using helper method
                        if raw_output:
                            parsed = self._extract_json(raw_output)
                            if isinstance(parsed, dict):
                                llm_all_emails = parsed.get("all_emails")
                                if llm_all_emails:
                                    # Parse comma-separated emails from LLM
                                    llm_email_list = [e.strip() for e in str(llm_all_emails).split(',') if e.strip()]
                                    for llm_email in llm_email_list:
                                        # Normalize and validate
                                        normalized = normalize_email(llm_email)
                                        if normalized and '@' in normalized:
                                            username, domain = normalized.split('@', 1)
                                            if len(username) >= 2 and len(domain) >= 4:
                                                all_emails_found.add(normalized)
                                    logger.debug(
                                        f"Step 4 (LLM): Found {len(llm_email_list)} emails from LLM for {filename}",
                                        extra={"emails": llm_email_list[:5]}
                                    )
                    except Exception as llm_error:
                        # Don't fail if LLM extraction fails - regex is primary method
                        logger.debug(f"LLM extraction failed (non-critical) for {filename}: {llm_error}")
            except Exception as e:
                logger.debug(f"LLM extraction step failed (non-critical) for {filename}: {e}")
            
            # Step 5: ABSOLUTE FINAL SAFETY CHECK - If still no emails, do one more ultra-simple scan
            if not all_emails_found:
                logger.warning(f"⚠️ No emails found after all steps for {filename}, running absolute final safety check")
                # Ultra-simple pattern - just find anything that looks like an email
                ultra_simple = re.findall(r'[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', resume_text, re.IGNORECASE)
                for email_candidate in ultra_simple:
                    email_candidate = email_candidate.strip().lower()
                    # Remove spaces
                    email_candidate = email_candidate.replace(' ', '').replace('\n', '').replace('\r', '').replace('\t', '')
                    # Remove trailing punctuation
                    email_candidate = re.sub(r'[.,;:!?|)\]]+$', '', email_candidate)
                    
                    if '@' in email_candidate:
                        username, domain = email_candidate.split('@', 1)
                        # Very lenient - just check basic structure
                        if len(username) >= 1 and len(domain) >= 3 and '.' in domain:
                            # Validate basic format - be very permissive
                            if '@' in email_candidate and '.' in domain:
                                all_emails_found.add(email_candidate)
                                logger.warning(f"⚠️ ABSOLUTE FINAL CHECK found email: {email_candidate}")
            
            # Convert to sorted list (for consistent ordering) then join
            emails_list = sorted(list(all_emails_found))
            all_emails_str = ','.join(emails_list) if emails_list else ""
            
            # CRITICAL: If we found emails in pre-step but lost them, restore them
            if not all_emails_str:
                # Do one final ultra-simple scan
                final_simple = re.findall(r'[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', resume_text, re.IGNORECASE)
                for email in final_simple:
                    email = email.strip().lower()
                    if '@' in email:
                        username, domain = email.split('@', 1)
                        if len(username) >= 2 and len(domain) >= 4 and '.' in domain:
                            emails_list.append(email)
                if emails_list:
                    emails_list = sorted(list(set(emails_list)))  # Remove duplicates
                    all_emails_str = ','.join(emails_list)
                    logger.warning(f"⚠️ Final simple scan recovered {len(emails_list)} email(s) for {filename}")
            
            if all_emails_str:
                logger.info(
                    f"📧 EXTRACTED ALL EMAILS from {filename}: {len(emails_list)} email(s)",
                    extra={
                        "file_name": filename,
                        "email_count": len(emails_list),
                        "emails": emails_list  # Log all emails for debugging
                    }
                )
            else:
                # Enhanced debugging: Check if @ symbols exist in text
                at_count = resume_text.count('@')
                logger.warning(
                    f"⚠️ NO EMAILS FOUND in {filename}",
                    extra={
                        "file_name": filename,
                        "text_length": len(resume_text),
                        "at_symbol_count": at_count,
                        "text_preview": resume_text[:1000],  # Log first 1000 chars for debugging
                        "text_contains_gmail": "gmail" in resume_text.lower(),
                        "text_contains_at": "@" in resume_text,
                    }
                )
                # If @ symbols exist but no emails found, log context around each @
                if at_count > 0:
                    at_positions = [i for i, char in enumerate(resume_text) if char == '@']
                    for i, pos in enumerate(at_positions[:5]):  # Check first 5 @ symbols
                        start = max(0, pos - 50)
                        end = min(len(resume_text), pos + 50)
                        context = resume_text[start:end]
                        logger.warning(
                            f"⚠️ Found @ symbol #{i+1} at position {pos} in {filename}",
                            extra={
                                "file_name": filename,
                                "position": pos,
                                "context": context,
                                "context_before": resume_text[max(0, pos-100):pos],
                                "context_after": resume_text[pos:min(len(resume_text), pos+100)]
                            }
                        )
            
            return all_emails_str
            
        except Exception as e:
            logger.error(
                f"ERROR in extract_all_emails for {filename}: {e}",
                extra={"file_name": filename, "error": str(e)},
                exc_info=True
            )
            # Return empty string on error, but log it
            return ""
    
    async def extract_email(self, resume_text: str, filename: str = "resume") -> Optional[str]:
        """
        Extract email address from resume text using PRIMARY-first approach, then fallback.
        Execution stops immediately when a PRIMARY email is found.
        
        Args:
            resume_text: The text content of the resume
            filename: Name of the resume file (for logging)
        
        Returns:
            The extracted email address or None if not found
        """
        if not resume_text or len(resume_text.strip()) < 5:
            logger.warning(f"Resume text too short for email extraction: {filename}")
            return None

        # ------------------------------------------------------------------
        # STEP 1: PRIMARY EMAIL EXTRACTION (HIGHEST PRIORITY)
        # ------------------------------------------------------------------
        # Extract primary email with context - STOPS IMMEDIATELY when found
        try:
            primary = self.extract_primary_email_with_context(resume_text)
            if primary:
                logger.info(
                    f"✅ PRIMARY EMAIL EXTRACTED from {filename}",
                    extra={
                        "file_name": filename,
                        "email": primary,
                        "method": "primary_email_with_context"
                    }
                )
                return primary
            else:
                logger.debug(
                    f"Step 1: No primary email found with context extraction",
                    extra={"file_name": filename}
                )
        except Exception as e:
            logger.warning(
                f"Primary email extraction failed: {e}",
                extra={"file_name": filename, "error": str(e)},
                exc_info=True
            )

        # ------------------------------------------------------------------
        # STEP 2: SECONDARY PROXY EMAIL EXTRACTION (FALLBACK)
        # ------------------------------------------------------------------
        # ONLY executed if primary extraction failed
        try:
            secondary = self.extract_secondary_proxy_email(resume_text)
            if secondary:
                logger.info(
                    f"✅ SECONDARY PROXY EMAIL EXTRACTED from {filename}",
                    extra={
                        "file_name": filename,
                        "email": secondary,
                        "method": "secondary_proxy_email"
                    }
                )
                return secondary
            else:
                logger.debug(
                    f"Step 2: No secondary proxy email found",
                    extra={"file_name": filename}
                )
        except Exception as e:
            logger.warning(
                f"Secondary proxy email extraction failed: {e}",
                extra={"file_name": filename, "error": str(e)},
                exc_info=True
            )

        # ------------------------------------------------------------------
        # STEP 3: EXISTING FALLBACK LOGIC (ONLY AFTER PRIMARY AND SECONDARY FAIL)
        # ------------------------------------------------------------------
        # Continue with existing regex fallback and/or LLM fallback logic
        # Scan the text sequentially and return the FIRST valid primary domain email found.
        # This ensures we stop immediately when we find the first email (top-to-bottom order).
        try:
            first_email = self._select_first_valid_email(resume_text)
            if first_email:
                logger.info(
                    f"✅ EMAIL EXTRACTED via simple full-text scan (first email found) from {filename}",
                    extra={
                        "file_name": filename, 
                        "email": first_email, 
                        "method": "simple_full_text_first_match"
                    },
                )
                return first_email
            else:
                logger.debug(
                    f"Step 3: No primary domain emails found in full text",
                    extra={"file_name": filename}
                )
        except Exception as e:
            logger.warning(
                f"Simple full-text email scan failed: {e}", 
                extra={"file_name": filename, "error": str(e)},
                exc_info=True
            )
        
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

            # Find the FIRST valid email in header (top-to-bottom order)
            first_header_email = self._select_first_valid_email(header_text)
            if first_header_email:
                if self._is_forwarding_email(first_header_email, filename):
                    logger.debug(f"Skipping forwarding email: {first_header_email}")
                else:
                    logger.info(
                        f"✅ EMAIL EXTRACTED from header of {filename} (first email found)",
                        extra={"file_name": filename, "email": first_header_email, "method": "header_first_match"}
                    )
                    return first_header_email

            # Fallback to existing regex-based header extraction if first match fails
            regex_email = self._extract_email_regex_fallback(header_text)
            if regex_email:
                if self._is_forwarding_email(regex_email, filename):
                    logger.debug(f"Skipping forwarding email: {regex_email}")
                else:
                    logger.info(
                        f"✅ EMAIL EXTRACTED via regex from header of {filename}",
                        extra={"file_name": filename, "email": regex_email, "method": "regex_header"}
                    )
                    return regex_email
        except Exception as e:
            logger.debug(f"Header email extraction failed: {e}")
        
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
                        if self._is_forwarding_email(regex_email, filename):
                            logger.debug(f"Skipped forwarding email in Personal Profile: {regex_email}")
                        else:
                            logger.info(
                                f"✅ EMAIL EXTRACTED via regex from Personal Profile section of {filename}",
                                extra={"file_name": filename, "email": regex_email, "method": "regex_personal_profile"}
                            )
                            return regex_email
                
                # Also try "Name:" section if Personal Profile didn't work
                name_section_match = re.search(r'(?i)Name\s*:.*?Email\s*:.*?(?=\n|Experience|Education|Skills|$)', resume_text, re.DOTALL)
                if name_section_match:
                    name_section = name_section_match.group(0)
                    logger.debug(f"Found Name section in {filename}, length: {len(name_section)}")
                    regex_email = self._extract_email_regex_fallback(name_section)
                    if regex_email:
                        if not self._is_forwarding_email(regex_email, filename):
                            logger.info(
                                f"✅ EMAIL EXTRACTED via regex from Name section of {filename}",
                                extra={"file_name": filename, "email": regex_email, "method": "regex_name_section"}
                            )
                            return regex_email
            
            # Fallback to full text search
            regex_email = self._extract_email_regex_fallback(text_to_search)
            if regex_email:
                if self._is_forwarding_email(regex_email, filename):
                    logger.debug(f"Skipping forwarding email: {regex_email}")
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
            
          
            if not email:
                if primary_email and isinstance(primary_email, str) and primary_email.lower() not in {"masked_email", "null"}:
                    email = primary_email
                elif all_emails:
                    email_list = [e.strip() for e in str(all_emails).split(',') if e and str(e).strip()]
                    if email_list:
                        email = self._select_best_email_from_list(email_list)
            
            # Clean, fix, and normalize email unless it is the special "masked_email" sentinel
            if email:
                if isinstance(email, str) and email == "masked_email":
                    # Do not attempt to normalize the sentinel value.
                    pass
                else:
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
                        "primary_email": primary_email,
                        "all_emails": all_emails,
                        "status": "success"
                    }
                )
                return email
           
            if not email and all_emails:
                email_list = [e.strip() for e in str(all_emails).split(',') if e.strip()]
                if email_list:
                    # Try each email in the list until we find a valid one
                    for email_candidate in email_list:
                        email_str = str(email_candidate).strip()
                        cleaned_email = self._clean_and_fix_email(email_str)
                        if cleaned_email:
                            email = normalize_email(cleaned_email)
                            if email:
                                logger.info(
                                    f"✅ EMAIL EXTRACTED from all_emails list for {filename}",
                                    extra={
                                        "file_name": filename,
                                        "email": email,
                                        "all_emails": all_emails,
                                        "status": "success"
                                    }
                                )
                                return email
            
            # If still no email, try fallback prompt
            logger.info(
                f"⚠️ Regular email extraction returned null for {filename}, trying fallback prompt",
                extra={"file_name": filename, "status": "trying_fallback"}
            )
        except Exception as e:
            logger.debug(f"LLM email extraction failed: {e}")
        
        return None
