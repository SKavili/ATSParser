"""Service for extracting years of experience from resumes using OLLAMA LLM."""
import json
import re
from typing import Dict, Optional, List, Tuple
from datetime import datetime
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

EXPERIENCE_PROMPT = """IMPORTANT: This is a FRESH, ISOLATED extraction task. Ignore any previous context.

ROLE:
You are an ATS resume parsing expert specializing in US IT staffing profiles.

CONTEXT:
Resumes may be unstructured and inconsistently formatted.
Experience refers to TOTAL PROFESSIONAL WORK EXPERIENCE in YEARS.

TASK:
Extract the candidate's total years of experience from the resume text.

SELECTION RULES (STRICT ORDER):

1. FIRST PRIORITY – EXPLICIT TOTAL EXPERIENCE:
   If the resume explicitly states total experience anywhere in summary, header, or profile:
   - Examples: "8 years experience", "10+ years", "over 15 years", "Total Experience: 12 years"
   - ALWAYS return this value AS-IS
   - Preserve "+" if present

2. SECOND PRIORITY – CALCULATED FROM WORK HISTORY (MOST IMPORTANT):
   If NO explicit total experience is found:

   CALCULATION METHOD (MANDATORY):
   - Identify ALL professional job date ranges
   - Determine:
     • Earliest job START date
     • Most recent job END date (or Present)
   - Calculate experience as:
     (Most Recent End Year) − (Earliest Start Year)

   IMPORTANT CALCULATION RULES:
   - DO NOT sum individual job durations
   - DO NOT use "(X years Y months)" text
   - Use ONLY date ranges
   - If end date is "Present", use CURRENT YEAR
   - Ignore education, certifications, internships,date of birth
   - Ignore overlapping jobs (use overall range only)
   - Round DOWN to nearest whole year

   Example:
   Dec 2017 – Present → 2025 − 2017 = 8 years

3. FORMATTING RULES:
   - Return ONLY whole years
   - Output format must be: "X years"
   - NEVER return months
   - NEVER return "+"
   - ALWAYS include the word "years"

ANTI-HALLUCINATION RULES:
- Do not guess or infer experience
- If dates are missing or ambiguous, return null
- Use ONLY resume content

OUTPUT FORMAT:
Return ONLY valid JSON. No explanation. No markdown.

JSON SCHEMA:
{
  "experience": "string | null"
}

VALID OUTPUT EXAMPLES:
{"experience": "8 years"}
{"experience": "15 years"}
{"experience": "25+ years"}
{"experience": null}"""



class ExperienceExtractor:
    """Service for extracting years of experience from resume text using OLLAMA LLM."""
    
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
    
    def _extract_dates_from_text(self, text: str) -> List[Tuple[datetime, str]]:
        """
        Extract all date patterns from text and return as datetime objects with context.
        
        Supports formats:
        - Month Year (e.g., "January 2020", "Jan 2020", "01/2020")
        - DD/MM/YY or DD/MM/YYYY (e.g., "15/01/20", "15/01/2020")
        - MM/DD/YY or MM/DD/YYYY (e.g., "01/15/20", "01/15/2020")
        - YYYY-MM-DD (e.g., "2020-01-15")
        - Year only (e.g., "2020")
        
        Returns:
            List of tuples (datetime, context_string) where context_string is surrounding text
        """
        dates = []
        month_names = {
            'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
            'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
            'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'sept': 9,
            'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
        }
        
        # Pattern 1: Month Year (e.g., "January 2020", "Jan 2020", "Jan. 2020")
        pattern1 = r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\.?\s+(\d{4})\b'
        for match in re.finditer(pattern1, text, re.IGNORECASE):
            month_str = match.group(1).lower()
            year = int(match.group(2))
            if month_str in month_names:
                try:
                    date_obj = datetime(year, month_names[month_str], 1)
                    context_start = max(0, match.start() - 100)
                    context_end = min(len(text), match.end() + 100)
                    context = text[context_start:context_end]
                    dates.append((date_obj, context))
                except ValueError:
                    continue
        
        # Pattern 2: DD/MM/YY or DD/MM/YYYY
        pattern2 = r'\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b'
        for match in re.finditer(pattern2, text):
            day = int(match.group(1))
            month = int(match.group(2))
            year_str = match.group(3)
            if len(year_str) == 2:
                year = 2000 + int(year_str) if int(year_str) < 50 else 1900 + int(year_str)
            else:
                year = int(year_str)
            
            # Heuristic: if day > 12, likely DD/MM format
            if day <= 31 and month <= 12:
                try:
                    if day > 12:  # Likely DD/MM format
                        date_obj = datetime(year, month, min(day, 28))
                    else:  # Could be MM/DD, but we'll try DD/MM first
                        date_obj = datetime(year, month, min(day, 28))
                    context_start = max(0, match.start() - 100)
                    context_end = min(len(text), match.end() + 100)
                    context = text[context_start:context_end]
                    dates.append((date_obj, context))
                except ValueError:
                    continue
        
        # Pattern 3: MM/DD/YY or MM/DD/YYYY (when day > 12, it's clearly MM/DD)
        pattern3 = r'\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b'
        for match in re.finditer(pattern3, text):
            first = int(match.group(1))
            second = int(match.group(2))
            year_str = match.group(3)
            if len(year_str) == 2:
                year = 2000 + int(year_str) if int(year_str) < 50 else 1900 + int(year_str)
            else:
                year = int(year_str)
            
            # If first <= 12 and second > 12, it's MM/DD format
            if first <= 12 and second > 12 and second <= 31:
                try:
                    date_obj = datetime(year, first, min(second, 28))
                    context_start = max(0, match.start() - 100)
                    context_end = min(len(text), match.end() + 100)
                    context = text[context_start:context_end]
                    dates.append((date_obj, context))
                except ValueError:
                    continue
        
        # Pattern 4: YYYY-MM-DD
        pattern4 = r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b'
        for match in re.finditer(pattern4, text):
            year = int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3))
            if 1 <= month <= 12 and 1 <= day <= 31:
                try:
                    date_obj = datetime(year, month, min(day, 28))
                    context_start = max(0, match.start() - 100)
                    context_end = min(len(text), match.end() + 100)
                    context = text[context_start:context_end]
                    dates.append((date_obj, context))
                except ValueError:
                    continue
        
        # Pattern 5: Year only (4 digits, reasonable range)
        pattern5 = r'\b(19[5-9]\d|20[0-3]\d)\b'
        for match in re.finditer(pattern5, text):
            year = int(match.group(1))
            if 1950 <= year <= datetime.now().year:
                try:
                    date_obj = datetime(year, 1, 1)
                    context_start = max(0, match.start() - 100)
                    context_end = min(len(text), match.end() + 100)
                    context = text[context_start:context_end]
                    dates.append((date_obj, context))
                except ValueError:
                    continue
        
        return dates
    
    def _is_education_date(self, context: str) -> bool:
        """
        Check if a date is related to education based on surrounding context.
        
        Args:
            context: Text surrounding the date (typically 200 chars)
        
        Returns:
            True if the date appears to be education-related, False otherwise
        """
        context_lower = context.lower()
        
        # Education-related keywords
        education_keywords = [
            'education', 'degree', 'bachelor', 'master', 'phd', 'doctorate',
            'graduation', 'graduated', 'graduate', 'university', 'college',
            'school', 'diploma', 'certificate', 'engineering', 'b.tech', 'm.tech',
            'b.e.', 'm.e.', 'b.sc', 'm.sc', 'b.a.', 'm.a.', 'passed', 'completed',
            'academic', 'qualification', 'studied', 'course', 'program'
        ]
        
        # Work-related keywords (if present, likely not education)
        work_keywords = [
            'experience', 'work', 'employment', 'job', 'position', 'role',
            'company', 'employer', 'organization', 'client', 'project',
            'responsibilities', 'achievements', 'skills', 'technologies',
            'professional', 'analyst', 'engineer', 'developer', 'manager',
            'intern', 'internship', 'solutions', 'academy', 'hyderabad',
            'remote', 'india', 'data analyst', 'quality assurance'
        ]
        
        # Check for education keywords
        has_education_keyword = any(keyword in context_lower for keyword in education_keywords)
        
        # Check for work keywords
        has_work_keyword = any(keyword in context_lower for keyword in work_keywords)
        
        # If it's in a section clearly marked as education (check this first)
        if any(section in context_lower for section in ['# education', 'education:', 'academic:', 'qualification:']):
            # But if it also has strong work indicators, it might be work-related
            if has_work_keyword and any(strong_work in context_lower for strong_work in ['professional experience', 'work experience', 'employment', 'job']):
                return False
            return True
        
        # If it has education keywords and no work keywords, it's likely education
        if has_education_keyword and not has_work_keyword:
            return True
        
        # If it has work keywords, it's likely work (even if it has some education keywords)
        if has_work_keyword:
            return False
        
        return False
    
    def _calculate_experience_from_dates(self, resume_text: str) -> Optional[str]:
        """
        Calculate years of experience by finding date range from work history.
        Excludes education-related dates.
        
        Returns:
            Experience string in format "X years" or None if cannot be calculated
        """
        if not resume_text:
            return None
        
        logger.info("📅 Starting date-based experience calculation")
        logger.debug(f"Resume text length: {len(resume_text)} characters")
        
        # Extract all dates with context (search full text, not limited)
        all_dates = self._extract_dates_from_text(resume_text)
        
        if not all_dates:
            logger.warning("❌ No dates found in resume text at all")
            # Try to see if there are any year-like numbers
            year_pattern = r'\b(20[0-2][0-9]|19[5-9][0-9])\b'
            years_found = re.findall(year_pattern, resume_text[:5000])
            if years_found:
                logger.debug(f"Found year-like numbers in text: {years_found[:10]}")
            return None
        
        logger.info(f"Found {len(all_dates)} total date occurrences in resume")
        
        # Filter out education-related dates, keeping context for work dates
        work_dates_with_context = []
        education_dates_count = 0
        for date_obj, context in all_dates:
            is_education = self._is_education_date(context)
            if not is_education:
                work_dates_with_context.append((date_obj, context))
                logger.debug(f"✅ Including work date: {date_obj.strftime('%Y-%m')} (context: {context[:100]}...)")
            else:
                education_dates_count += 1
                logger.debug(f"❌ Excluding education date: {date_obj.strftime('%Y-%m')} (context: {context[:100]}...)")
        
        logger.info(f"Date filtering: {len(work_dates_with_context)} work dates, {education_dates_count} education dates excluded")
        
        if not work_dates_with_context:
            logger.warning(f"❌ No work-related dates found after filtering. All {len(all_dates)} dates were filtered as education dates.")
            # Log why dates were filtered
            for date_obj, context in all_dates[:5]:  # Show first 5 filtered dates
                is_edu = self._is_education_date(context)
                logger.debug(f"Filtered date: {date_obj.strftime('%Y-%m')} - Is Education: {is_edu} - Context preview: {context[:100]}...")
            return None
        
        logger.info(f"✅ Found {len(work_dates_with_context)} work-related dates after filtering (excluded {len(all_dates) - len(work_dates_with_context)} education dates)")
        
        # Find oldest and most recent dates
        work_dates = [date_obj for date_obj, _ in work_dates_with_context]
        oldest_date = min(work_dates)
        most_recent_date = max(work_dates)
        
        # Check if "Present" or "Current" is mentioned near the most recent date
        # This helps identify ongoing positions
        current_date = datetime.now()
        most_recent_context = None
        for date_obj, context in work_dates_with_context:
            if date_obj == most_recent_date:
                most_recent_context = context.lower()
                break
        
        # If "Present" or "Current" is mentioned, treat most recent date as current
        if most_recent_context and any(keyword in most_recent_context for keyword in ['present', 'current', 'till date', 'till now']):
            most_recent_date = current_date
            logger.debug("Found 'Present' or 'Current' keyword, using current date as most recent")
        elif most_recent_date > current_date:
            # If most recent date is in the future, use current date
            most_recent_date = current_date
            logger.debug(f"Most recent date is in future, using current date")
        
        # Calculate years difference
        years_diff = (most_recent_date.year - oldest_date.year)
        if most_recent_date.month < oldest_date.month or (
            most_recent_date.month == oldest_date.month and most_recent_date.day < oldest_date.day
        ):
            years_diff -= 1
        
        # Add months for more accuracy (round to nearest year)
        months_diff = (most_recent_date.year - oldest_date.year) * 12 + (most_recent_date.month - oldest_date.month)
        if most_recent_date.day < oldest_date.day:
            months_diff -= 1
        
        # Round to nearest year
        years_diff = round(months_diff / 12)
        
        logger.debug(f"Date calculation: oldest={oldest_date.strftime('%Y-%m')}, most_recent={most_recent_date.strftime('%Y-%m')}, months_diff={months_diff}, years_diff={years_diff}")
        
        # Validate: should be between 0 and 50 years
        if years_diff < 0:
            logger.warning(f"Calculated negative years: {years_diff}, using 0")
            years_diff = 0
        elif years_diff > 50:
            logger.warning(f"Calculated unrealistic years: {years_diff}, capping at 50")
            years_diff = 50
        
        # Handle cases where experience is less than 1 year
        if years_diff == 0:
            # Check if we have at least 3 months of experience (be more lenient)
            if months_diff >= 3:
                experience_str = "1 year"  # Round up to 1 year if >= 3 months
                logger.info(f"✅ Date-based calculation: {experience_str} (from {oldest_date.strftime('%Y-%m')} to {most_recent_date.strftime('%Y-%m')}, {months_diff} months, rounded up)")
                return experience_str
            else:
                logger.warning(f"❌ Calculated {months_diff} months of experience from dates (less than 3 months, returning None)")
                logger.debug(f"Oldest date: {oldest_date.strftime('%Y-%m-%d')}, Most recent: {most_recent_date.strftime('%Y-%m-%d')}")
                return None
        
        experience_str = f"{years_diff} years"
        logger.info(f"✅ Date-based calculation: {experience_str} (from {oldest_date.strftime('%Y-%m')} to {most_recent_date.strftime('%Y-%m')}, {months_diff} months)")
        
        return experience_str
    
    def _extract_experience_fallback(self, resume_text: str) -> Optional[str]:
        """
        Fallback regex-based extraction if LLM fails.
        Looks for common experience patterns in the resume text.
        Also tries date-based calculation as a secondary method.
        """
        if not resume_text:
            logger.warning("Fallback extraction: resume_text is empty")
            return None
        
        logger.info(f"🔍 FALLBACK EXTRACTION: Starting regex-based experience extraction")
        logger.debug(f"Resume text length: {len(resume_text)} characters")
        
        # Common patterns for experience - capture full "X years" or "X+ years" format
        # Priority order: summary patterns first, then work history patterns
        # Patterns with "+" signs are prioritized
        summary_patterns = [
            # High priority: Summary/profile section patterns (these should be checked first)
            r'(\d+\+?\s*years?)\s+of\s+experience',  # "18+ years of experience"
            r'(\d+\+?\s*years?)\s+of\s+professional\s+experience',
            r'over\s+(\d+\+?\s*years?)',  # "over 25+ years" - HIGH PRIORITY
            r'more\s+than\s+(\d+\+?\s*years?)',
            r'(\d+\+?\s*years?)\s+experience',  # "18+ years experience"
            r'with\s+(\d+\+?\s*years?)',  # "with 18+ years"
            r'having\s+(\d+\+?\s*years?)',
            r'(\d+\+?\s*years?)\s+in\s+(?:the\s+)?(?:field|industry|profession)',
            r'(\d+\+?\s*years?)\s+professional',
        ]
        
        work_history_patterns = [
            # Lower priority: Work history section patterns
            r'(?:total\s+)?(?:work\s+)?experience[:\s]+(\d+\+?\s*years?)',  # "Total Work Experience: 18 years"
            r'experience[:\s]+(\d+\+?\s*years?)',
            r'(\d+\+?\s*years?)\s+work',
            r'(\d+\+?\s*years?)\s+in\s+',
        ]
        
        # Search in first 15000 characters (usually contains summary/profile sections)
        search_text = resume_text[:15000]
        summary_text = resume_text[:5000]  # First 5000 chars typically contain summary
        logger.debug(f"Searching in first {len(search_text)} characters")
        logger.debug(f"Summary section (first 500 chars): {summary_text[:500]}")
        
        all_matches = []  # Store all matches with priority info
        
        # First, check summary patterns in summary section (highest priority)
        for idx, pattern in enumerate(summary_patterns):
            try:
                matches = re.finditer(pattern, summary_text, re.IGNORECASE)
                for match in matches:
                    exp_str = match.group(1).strip()
                    has_plus = '+' in exp_str
                    position = match.start()
                    all_matches.append({
                        'text': exp_str,
                        'has_plus': has_plus,
                        'position': position,
                        'section': 'summary',
                        'pattern_idx': idx
                    })
                    logger.debug(f"Summary pattern {idx+1} '{pattern}': found '{exp_str}' at position {position}")
            except Exception as e:
                logger.warning(f"Error processing summary pattern {idx+1} '{pattern}': {e}")
                continue
        
        # Then, check all patterns in full search text
        all_patterns = summary_patterns + work_history_patterns
        for idx, pattern in enumerate(all_patterns):
            try:
                matches = re.finditer(pattern, search_text, re.IGNORECASE)
                for match in matches:
                    exp_str = match.group(1).strip()
                    has_plus = '+' in exp_str
                    position = match.start()
                    section = 'summary' if position < 5000 else 'work_history'
                    
                    # Skip if already found in summary section
                    if section == 'work_history':
                        # Check if we already have a better match from summary
                        if any(m['section'] == 'summary' and m['has_plus'] == has_plus for m in all_matches):
                            continue
                    
                    all_matches.append({
                        'text': exp_str,
                        'has_plus': has_plus,
                        'position': position,
                        'section': section,
                        'pattern_idx': idx
                    })
                    logger.debug(f"Pattern {idx+1} '{pattern}': found '{exp_str}' at position {position} in {section}")
            except Exception as e:
                logger.warning(f"Error processing pattern {idx+1} '{pattern}': {e}")
                continue
        
        # Prioritize matches: 1) Has "+" sign, 2) From summary section, 3) Earlier position
        if all_matches:
            # Sort: has_plus first, then summary section, then position
            all_matches.sort(key=lambda x: (
                not x['has_plus'],  # False (has +) comes before True (no +)
                x['section'] != 'summary',  # summary comes before work_history
                x['position']  # Earlier position comes first
            ))
            
            best_match = all_matches[0]
            exp_str = best_match['text']
            logger.debug(f"Selected best match: '{exp_str}' (has_plus={best_match['has_plus']}, section={best_match['section']}, position={best_match['position']})")
            
            # Ensure it has "years" or "year"
            if 'year' not in exp_str.lower():
                # Extract number and add "years"
                num_match = re.search(r'(\d+\+?)', exp_str)
                if num_match:
                    exp_str = f"{num_match.group(1)} years"
                else:
                    exp_str = f"{exp_str} years"
            
            # Normalize spacing and ensure proper format
            exp_str = re.sub(r'\s+', ' ', exp_str).strip()
            
            # Validate: should be between 0 and 50 years (reasonable range)
            num_match = re.search(r'(\d+)', exp_str)
            if num_match:
                years_num = int(num_match.group(1))
                if years_num > 50:
                    logger.debug(f"Skipping match '{exp_str}' - value {years_num} > 50 years")
                    # Try next match
                    if len(all_matches) > 1:
                        best_match = all_matches[1]
                        exp_str = best_match['text']
                        # Re-normalize
                        if 'year' not in exp_str.lower():
                            num_match = re.search(r'(\d+\+?)', exp_str)
                            if num_match:
                                exp_str = f"{num_match.group(1)} years"
                        exp_str = re.sub(r'\s+', ' ', exp_str).strip()
                        num_match = re.search(r'(\d+)', exp_str)
                        if num_match:
                            years_num = int(num_match.group(1))
                            if years_num > 50:
                                logger.debug(f"All matches exceed 50 years, skipping")
                                return None
                    else:
                        return None
            
            logger.info(f"✅ Fallback regex extracted experience: '{exp_str}' (priority: has_plus={best_match['has_plus']}, section={best_match['section']})")
            return exp_str
        
        # If no pattern matched, try a more aggressive search in the first 2000 chars
        logger.debug("No matches found with standard patterns, trying aggressive search in first 2000 chars")
        aggressive_text = resume_text[:2000].lower()
        
        # Look for any "X+ years" or "X years" near "experience" keyword
        aggressive_pattern = r'(\d+\+?\s*years?)'
        aggressive_matches = re.findall(aggressive_pattern, aggressive_text, re.IGNORECASE)
        
        if aggressive_matches:
            # Find the position of each match and check context
            for match in aggressive_matches[:5]:  # Check first 5 matches
                # Escape special regex characters in the match string
                escaped_match = re.escape(match)
                match_obj = re.search(escaped_match, aggressive_text, re.IGNORECASE)
                if match_obj:
                    start_pos = match_obj.start()
                    context_start = max(0, start_pos - 150)
                    context_end = min(len(aggressive_text), start_pos + len(match) + 150)
                    context = aggressive_text[context_start:context_end]
                    
                    # Must be near "experience" keyword
                    if 'experience' in context:
                        # Skip if it's clearly a skill (has dash or bullet before number)
                        if not re.search(r'[-•]\s*\d+\s*years?', context):
                            exp_str = match.strip()
                            if 'year' not in exp_str.lower():
                                num_match = re.search(r'(\d+\+?)', exp_str)
                                if num_match:
                                    exp_str = f"{num_match.group(1)} years"
                            
                            exp_str = re.sub(r'\s+', ' ', exp_str).strip()
                            
                            # Validate years
                            num_match = re.search(r'(\d+)', exp_str)
                            if num_match:
                                years_num = int(num_match.group(1))
                                if 1 <= years_num <= 50:
                                    logger.info(f"✅ Fallback aggressive search extracted experience: '{exp_str}'")
                                    return exp_str
        
        # If no pattern matched, try date-based calculation
        logger.debug("No matches found with standard patterns, trying date-based calculation")
        date_based_experience = self._calculate_experience_from_dates(resume_text)
        if date_based_experience:
            logger.info(f"✅ Fallback date-based calculation extracted experience: '{date_based_experience}'")
            return date_based_experience
        else:
            logger.debug("Date-based calculation also returned None - checking if dates were found")
            # Try to extract dates to see if any were found (for debugging)
            all_dates = self._extract_dates_from_text(resume_text[:10000])  # Check first 10k chars
            if all_dates:
                logger.debug(f"Found {len(all_dates)} dates in resume, but calculation returned None. This might indicate all dates were filtered as education dates.")
                # Log a sample of dates found
                for i, (date_obj, context) in enumerate(all_dates[:3]):  # Show first 3
                    is_edu = self._is_education_date(context)
                    logger.debug(f"Date {i+1}: {date_obj.strftime('%Y-%m')} - Education: {is_edu} - Context: {context[:80]}...")
            else:
                logger.debug("No dates found in resume text at all")
        
        logger.warning("❌ Fallback extraction: No experience pattern found in resume text")
        return None
    
    def _extract_json(self, text: str) -> Dict:
        """Extract JSON object from LLM response."""
        if not text:
            logger.warning("Empty response from LLM")
            return {"experience": None}
        
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
            if isinstance(parsed, dict) and "experience" in parsed:
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
                    if isinstance(parsed, dict) and "experience" in parsed:
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
        return {"experience": None}
    
    async def extract_experience(self, resume_text: str, filename: str = "resume") -> Optional[str]:
        """
        Extract years of experience from resume text using OLLAMA LLM.
        
        Args:
            resume_text: The text content of the resume
            filename: Name of the resume file (for logging)
        
        Returns:
            The extracted experience string or None if not found
        """
        model_to_use = self.model  # Initialize early for error handling
        try:
            is_connected, available_model = await self._check_ollama_connection()
            if not is_connected:
                # Try fallback before raising error
                logger.warning(f"OLLAMA not connected, trying fallback extraction for {filename}")
                experience = self._extract_experience_fallback(resume_text)
                if experience:
                    logger.info(f"✅ EXPERIENCE EXTRACTED via fallback (OLLAMA not connected) from {filename}: {experience}")
                    return experience
                raise RuntimeError(
                    f"OLLAMA is not accessible at {self.ollama_host}. "
                    "Please ensure OLLAMA is running. Start it with: ollama serve"
                )
            
            model_to_use = self.model
            if available_model and "llama3.1" not in available_model.lower():
                logger.warning(
                    f"llama3.1 not found, using available model: {available_model}",
                    extra={"available_model": available_model}
                )
                model_to_use = available_model
            
            # Increase text limit to capture more content, especially from summary sections
            text_to_send = resume_text[:20000]  # Increased from 10000 to capture more content
            
            prompt = f"""{EXPERIENCE_PROMPT}

Input resume text:
{text_to_send}

Output (JSON only, no other text, no explanations):"""
            
            logger.info(
                f"📤 CALLING OLLAMA API for experience extraction",
                extra={
                    "file_name": filename,
                    "model": model_to_use,
                    "ollama_host": self.ollama_host,
                }
            )
            
            result = None
            last_error = None
            
            async with httpx.AsyncClient(timeout=Timeout(1200.0)) as client:
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
                    logger.info("✅ Successfully used /api/generate endpoint for experience extraction")
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
                        logger.info("Successfully used /api/chat endpoint for experience extraction")
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
            experience = parsed_data.get("experience")
            
            if experience:
                experience = str(experience).strip()
                if not experience or experience.lower() in ["null", "none", ""]:
                    experience = None
            
            # Fallback to regex extraction if LLM returned null
            if not experience:
                logger.warning(
                    f"LLM returned null experience for {filename}, trying fallback regex extraction",
                    extra={"file_name": filename}
                )
                experience = self._extract_experience_fallback(resume_text)
                if experience:
                    logger.info(
                        f"✅ EXPERIENCE EXTRACTED via fallback from {filename}",
                        extra={
                            "file_name": filename,
                            "experience": experience,
                            "method": "regex_fallback"
                        }
                    )
                else:
                    # Try date-based calculation as last resort
                    logger.debug(f"Regex fallback failed, trying date-based calculation for {filename}")
                    date_based_exp = self._calculate_experience_from_dates(resume_text)
                    if date_based_exp:
                        experience = date_based_exp
                        logger.info(
                            f"✅ EXPERIENCE EXTRACTED via date-based calculation from {filename}",
                            extra={
                                "file_name": filename,
                                "experience": experience,
                                "method": "date_based_calculation"
                            }
                        )
            
            logger.info(
                f"✅ EXPERIENCE EXTRACTED from {filename}",
                extra={
                    "file_name": filename,
                    "experience": experience,
                    "status": "success" if experience else "not_found",
                    "method": "llm" if parsed_data.get("experience") else "fallback"
                }
            )
            
            return experience
            
        except httpx.HTTPError as e:
            error_details = {
                "error": str(e),
                "error_type": type(e).__name__,
                "ollama_host": self.ollama_host,
                "model": model_to_use,
            }
            logger.warning(
                f"HTTP error calling OLLAMA for experience extraction: {e}. Trying fallback regex extraction.",
                extra=error_details
            )
            # Try fallback extraction even when LLM fails
            experience = self._extract_experience_fallback(resume_text)
            if not experience:
                # Try date-based calculation as last resort
                experience = self._calculate_experience_from_dates(resume_text)
                if experience:
                    logger.info(
                        f"✅ EXPERIENCE EXTRACTED via date-based calculation after LLM failure from {filename}",
                        extra={
                            "file_name": filename,
                            "experience": experience,
                            "method": "date_based_after_llm_error"
                        }
                    )
                    return experience
            
            if experience:
                logger.info(
                    f"✅ EXPERIENCE EXTRACTED via fallback after LLM failure from {filename}",
                    extra={
                        "file_name": filename,
                        "experience": experience,
                        "method": "regex_fallback_after_llm_error"
                    }
                )
                return experience
            # If fallback also fails, raise the error
            logger.error(
                f"Both LLM and fallback extraction failed for {filename}",
                extra=error_details,
                exc_info=True
            )
            raise RuntimeError(f"Failed to extract experience with LLM and fallback: {e}")
        except Exception as e:
            logger.warning(
                f"Error extracting experience with LLM: {e}. Trying fallback regex extraction.",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "ollama_host": self.ollama_host,
                    "model": model_to_use,
                }
            )
            # Try fallback extraction even when other errors occur
            experience = self._extract_experience_fallback(resume_text)
            if not experience:
                # Try date-based calculation as last resort
                experience = self._calculate_experience_from_dates(resume_text)
                if experience:
                    logger.info(
                        f"✅ EXPERIENCE EXTRACTED via date-based calculation after error from {filename}",
                        extra={
                            "file_name": filename,
                            "experience": experience,
                            "method": "date_based_after_error"
                        }
                    )
                    return experience
            
            if experience:
                logger.info(
                    f"✅ EXPERIENCE EXTRACTED via fallback after error from {filename}",
                    extra={
                        "file_name": filename,
                        "experience": experience,
                        "method": "regex_fallback_after_error"
                    }
                )
                return experience
            # If fallback also fails, raise the original error
            logger.error(
                f"Both LLM and fallback extraction failed for {filename}",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "ollama_host": self.ollama_host,
                    "model": model_to_use,
                },
                exc_info=True
            )
            raise

