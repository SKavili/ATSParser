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

EXPERIENCE_PROMPT = """
IMPORTANT:
This is a FRESH, ISOLATED, SINGLE-TASK extraction.
Ignore ALL previous conversations, memory, instructions, or assumptions.

ROLE:
You are an ATS resume-parsing expert.

TASK:
Extract ONLY raw experience data from the resume.
DO NOT calculate total experience.
DO NOT infer missing data.

--------------------------------------------------
STEP 1: EXPLICIT EXPERIENCE CHECK (HIGHEST PRIORITY)

First, check the SUMMARY / PROFILE / ENTIRE RESUME
for an explicit total experience statement.

Examples:
• "5 years of experience"
• "5+ years total experience"
• "Over 17 years of professional experience"
• "3 and half years of experience" → Treat as 3 years (integer only, no decimals)

If found:
• Extract ONLY the NUMBER OF YEARS as an INTEGER (no decimals)
• Remove "+", "years", or text
• Example: "5+ years" → 5, "5.5 years" → 5, "5 and half years" → 5
• Return it in "summary_experience"
• DO NOT calculate from dates

--------------------------------------------------
STEP 2: DATE RANGE EXTRACTION (ONLY IF STEP 1 FAILS)

Extract employment date ranges ONLY from PAID WORK EXPERIENCE.

Include:
• Full-time
• Part-time
• Contract
• Freelance
• Internships ONLY if clearly employment

Exclude:
• Education
• Courses
• Certifications
• Academic projects
• Tool versions (e.g., Python 3.10)
• Company founding years

--------------------------------------------------
DATE HANDLING RULES:

Treat ALL of these as "Present":
Present, Current, Now, Till Date, Till Now, Still Date, Still, Ongoing, Working,
Working till Date, To Date, To Now, Until Present, Until Now, Until Date,
Up to Present, Up to Now, Up to Date, As of Now, As of Present, As of Date,
As of Today, Today, Continuing, Continue, Active, Currently, Currently Working,
Still Working, Still Employed, Still Active, Currently Employed, Currently Active

Accept ALL formats:
• Jan 2020 – Mar 2022
• Jan'22 – Aug'23
• 01/2020 – 08/2022
• 2021 – Present
• 2020 to 2023

--------------------------------------------------
CURRENT DATE CONTEXT:

• The current date will be provided below
• Dates in the CURRENT YEAR are VALID
• Only dates AFTER the provided current date are FUTURE dates
• Use the provided current year for two-digit year validation

--------------------------------------------------
TWO-DIGIT YEAR RULE:

• If (2000 + YY) ≤ CURRENT YEAR → use 2000s
• Otherwise → use 1900s
• Example: If CURRENT YEAR = 2026, then '25' → 2025, '26' → 2026

--------------------------------------------------
ANTI-HALLUCINATION RULES (STRICT):

• NEVER guess dates
• NEVER calculate totals
• NEVER infer experience
• If start date is missing → exclude
• If no valid data → return null

--------------------------------------------------
OUTPUT FORMAT (JSON ONLY):

{
  "summary_experience": number | null,
  "date_ranges": [
    {
      "start": "string",
      "end": "string",
      "company": "string | null",
      "title": "string | null"
    }
  ]
}

--------------------------------------------------
If no experience data exists:
Return:
{"summary_experience": null, "date_ranges": []}

"""



class ExperienceExtractor:
    """Service for extracting years of experience from resume text using OLLAMA LLM."""
    
    def __init__(self):
        self.ollama_host = settings.ollama_host
        self.model = "llama3.1"
        # Simple in-memory cache: {text_hash: (experience, timestamp)}
        self._cache = {}
        self._cache_ttl = 86400  # 24 hours in seconds
    
    def _clean_resume_text(self, resume_text: str) -> str:
        """
        Clean resume text by removing education, certification, and project blocks.
        This prevents false positives in experience extraction.
        
        Returns:
            Cleaned text containing only work-related sections
        """
        if not resume_text:
            return ""
        
        text = resume_text
        lines = text.split('\n')
        cleaned_lines = []
        skip_section = False
        current_section = ""
        
        # Section headers that indicate non-work content
        skip_keywords = [
            r'^#?\s*(education|academic|qualification|qualifications)',
            r'^#?\s*(certification|certifications|certificate|certificates)',
            r'^#?\s*(course|courses|training|trainings)',
            r'^#?\s*(project|projects)\s*$',  # Only if standalone (not "Project Manager")
            r'^#?\s*(award|awards|honor|honors)',
            r'^#?\s*(publication|publications|research)',
        ]
        
        # Work section indicators
        work_keywords = [
            r'^#?\s*(experience|work\s+experience|employment|professional\s+experience)',
            r'^#?\s*(career|career\s+history|work\s+history)',
        ]
        
        for i, line in enumerate(lines):
            line_lower = line.strip().lower()
            
            # Check if this line starts a section to skip
            should_skip = False
            for pattern in skip_keywords:
                if re.match(pattern, line_lower, re.IGNORECASE):
                    should_skip = True
                    skip_section = True
                    logger.debug(f"Removing section: {line.strip()[:50]}")
                    break
            
            # Check if this line starts a work section
            if not should_skip:
                for pattern in work_keywords:
                    if re.match(pattern, line_lower, re.IGNORECASE):
                        skip_section = False
                        break
            
            # Skip lines in non-work sections
            if skip_section:
                # Check if we've reached a new major section (usually starts with # or is blank followed by header)
                if i < len(lines) - 1:
                    next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
                    # If next line looks like a new section header, stop skipping
                    if next_line and (next_line.startswith('#') or len(next_line) < 50):
                        # Check if it's a work section
                        for pattern in work_keywords:
                            if re.match(pattern, next_line.lower(), re.IGNORECASE):
                                skip_section = False
                                break
                continue
            
            cleaned_lines.append(line)
        
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Remove ISO standards and version numbers (e.g., "ISO 9001:2015", "Python 3.10")
        cleaned_text = re.sub(r'\b(ISO|iso)\s+\d+:\d{4}\b', '', cleaned_text)
        cleaned_text = re.sub(r'\b[A-Za-z]+\s+\d+\.\d+\b(?=\s|$)', '', cleaned_text)  # Remove "Python 3.10" but keep "2020"
        
        # Remove standalone years that might be tool versions (but keep date ranges)
        # Use a simpler approach: find years and check context, then remove if not work-related
        # This avoids variable-width lookbehind issues
        year_pattern = r'\b(19[5-9]\d|20[0-3]\d)\b'
        
        def should_remove_year(match):
            """Check if a year should be removed (not part of work date range)."""
            start_pos = match.start()
            end_pos = match.end()
            
            # Get context around the match (100 chars before and after)
            context_start = max(0, start_pos - 100)
            context_end = min(len(cleaned_text), end_pos + 100)
            context = cleaned_text[context_start:context_end].lower()
            
            # Check if it's part of a date range (has separators like /, -, or month names nearby)
            # Pattern: MM/YYYY, YYYY-MM, or Month YYYY
            before_context = cleaned_text[max(0, start_pos - 20):start_pos]
            after_context = cleaned_text[end_pos:min(len(cleaned_text), end_pos + 20)]
            
            # If it has date separators nearby, it's likely a date - keep it
            if re.search(r'\d{1,2}[/-]', before_context) or re.search(r'[/-]\d{1,2}', after_context):
                return False  # Keep it - it's part of a date
            
            # If it's followed by date range indicators, keep it
            if re.search(r'\s*(?:–|-|to|present|current)\s*', after_context, re.IGNORECASE):
                return False  # Keep it - it's part of a date range
            
            # If it's preceded by month names, keep it
            month_pattern = r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\.?\s*$'
            if re.search(month_pattern, before_context, re.IGNORECASE):
                return False  # Keep it - it's part of a date
            
            # If context has work-related keywords, keep it
            work_keywords = ['company', 'worked', 'employed', 'joined', 'role', 'position', 'job', 'experience', 'developer', 'engineer', 'manager']
            if any(keyword in context for keyword in work_keywords):
                return False  # Keep it - it's work-related
            
            # Otherwise, it might be a tool version or unrelated year - remove it
            return True  # Remove it
        
        # Apply the filter: replace years that should be removed with empty string
        cleaned_text = re.sub(year_pattern, lambda m: '' if should_remove_year(m) else m.group(), cleaned_text)
        
        logger.debug(f"Text cleaning: {len(resume_text)} -> {len(cleaned_text)} characters")
        return cleaned_text
    
    def _extract_work_sections_only(self, resume_text: str) -> str:
        """
        Extract only work-related sections from resume text.
        Looks for employment sections with company names, roles, and dates.
        
        Returns:
            Text containing only work experience sections
        """
        if not resume_text:
            return ""
        
        lines = resume_text.split('\n')
        work_lines = []
        in_work_section = False
        work_section_headers = [
            r'experience', r'work\s+experience', r'employment', r'professional\s+experience',
            r'career', r'work\s+history', r'employment\s+history'
        ]
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            line_lower = line_stripped.lower()
            
            # Check if this is a work section header
            for pattern in work_section_headers:
                if re.match(rf'^#?\s*{pattern}', line_lower, re.IGNORECASE):
                    in_work_section = True
                    work_lines.append(line)
                    logger.debug(f"Found work section: {line_stripped[:50]}")
                    break
            else:
                # If we're in a work section, include the line
                if in_work_section:
                    # Stop if we hit another major section (education, certification, etc.)
                    if re.match(r'^#?\s*(education|certification|project|award)', line_lower, re.IGNORECASE):
                        in_work_section = False
                    else:
                        work_lines.append(line)
                # Also include lines that look like job entries (have company + date pattern)
                elif re.search(r'\b(company|corporation|inc\.|ltd\.|llc|technologies|solutions|systems)\b.*\d{4}', line_stripped, re.IGNORECASE):
                    work_lines.append(line)
        
        work_text = '\n'.join(work_lines)
        logger.debug(f"Work sections extraction: {len(resume_text)} -> {len(work_text)} characters")
        return work_text
    
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
    
    def _extract_work_date_ranges(self, text: str) -> List[Tuple[datetime, datetime, str]]:
        """
        Extract work date ranges (start-end pairs) from text.
        Only extracts dates that appear to be employment-related.
        
        Valid formats:
        - Jan 2020 – Mar 2023
        - Jan'22 - Jun'23 (apostrophe format with 2-digit year)
        - Jun'19-Aug21 (apostrophe and no-apostrophe mixed)
        - 2021 – Present
        - 02/2019 – 08/2022
        - January 2020 to March 2023
        
        Returns:
            List of tuples (start_date, end_date, context_string)
            end_date can be None if "Present" or "Current"
        """
        date_ranges = []
        current_date = datetime.now()
        current_year = current_date.year
        
        month_names = {
            'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
            'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
            'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'sept': 9,
            'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
        }
        
        def parse_two_digit_year(year_str: str) -> int:
            """Parse 2-digit year using two-digit year rule."""
            year = int(year_str)
            # If (year + 2000) <= current year → use 2000s
            # Otherwise → use 1900s
            if (year + 2000) <= current_year:
                return 2000 + year
            else:
                return 1900 + year
        
        # Pattern 0: Apostrophe format with 2-digit years (e.g., "Jan'22 - Jun'23" or "Jun'19-Aug21")
        # Handles: Jan'22, Jun'19, Aug21 (with or without apostrophe, with or without space)
        # Matches: Jan'22, Jan'22, Jan 22, Jan22, Aug21, etc.
        pattern0 = r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\.?\s*[\'\']?(\d{2,4})\s*[–\-]\s*(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|present|current)\.?\s*[\'\']?(\d{2,4})?'
        
        for match in re.finditer(pattern0, text, re.IGNORECASE):
            start_month_str = match.group(1).lower()
            start_year_str = match.group(2)
            end_month_str = match.group(3).lower()
            end_year_str = match.group(4)
            
            if start_month_str not in month_names:
                continue
            
            try:
                # Parse start year (handle 2-digit or 4-digit)
                if len(start_year_str) == 2:
                    start_year = parse_two_digit_year(start_year_str)
                else:
                    start_year = int(start_year_str)
                
                start_date = datetime(start_year, month_names[start_month_str], 1)
                
                # Parse end year
                if self._is_ongoing_keyword(end_month_str):
                    end_date = current_date
                elif end_month_str in month_names:
                    if not end_year_str:
                        continue
                    if len(end_year_str) == 2:
                        end_year = parse_two_digit_year(end_year_str)
                    else:
                        end_year = int(end_year_str)
                    end_date = datetime(end_year, month_names[end_month_str], 1)
                else:
                    continue
                
                # Validate: start date should be before end date
                if start_date > end_date:
                    continue
                
                context_start = max(0, match.start() - 150)
                context_end = min(len(text), match.end() + 150)
                context = text[context_start:context_end]
                
                # Filter out education dates first
                if self._is_education_date(context):
                    logger.debug(f"Skipping education date range: {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}")
                    continue
                
                # Only include if context suggests work (expanded keywords)
                work_keywords = [
                    'company', 'worked', 'employed', 'joined', 'role', 'position', 'job',
                    'developer', 'engineer', 'manager', 'analyst', 'software', 'programmer',
                    'consultant', 'specialist', 'coordinator', 'assistant', 'director',
                    'lead', 'senior', 'junior', 'intern', 'internship', 'professional',
                    'experience', 'employment', 'career', 'work history'
                ]
                if any(keyword in context.lower() for keyword in work_keywords):
                    date_ranges.append((start_date, end_date, context))
                    logger.debug(f"Extracted apostrophe date range: {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}")
            except (ValueError, TypeError) as e:
                logger.debug(f"Failed to parse apostrophe date range: {e}")
                continue
        
        # Pattern 1: Month Year – Month Year (e.g., "Jan 2020 – Mar 2023")
        pattern1 = r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\.?\s+(\d{4})\s*[–\-]\s*(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|present|current)\.?\s*(\d{4})?'
        
        for match in re.finditer(pattern1, text, re.IGNORECASE):
            start_month_str = match.group(1).lower()
            start_year = int(match.group(2))
            end_month_str = match.group(3).lower()
            end_year_str = match.group(4)
            
            if start_month_str in month_names:
                try:
                    start_date = datetime(start_year, month_names[start_month_str], 1)
                    
                    if self._is_ongoing_keyword(end_month_str) or not end_year_str:
                        end_date = current_date
                    elif end_month_str in month_names:
                        end_date = datetime(int(end_year_str), month_names[end_month_str], 1)
                    else:
                        continue
                    
                    # Validate: start date should be before end date
                    if start_date > end_date:
                        continue
                    
                    context_start = max(0, match.start() - 150)
                    context_end = min(len(text), match.end() + 150)
                    context = text[context_start:context_end]
                    
                    # Filter out education dates first
                    if self._is_education_date(context):
                        logger.debug(f"Skipping education date range: {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}")
                        continue
                    
                    # Only include if context suggests work (expanded keywords)
                    work_keywords = [
                        'company', 'worked', 'employed', 'joined', 'role', 'position', 'job',
                        'developer', 'engineer', 'manager', 'analyst', 'software', 'programmer',
                        'consultant', 'specialist', 'coordinator', 'assistant', 'director',
                        'lead', 'senior', 'junior', 'intern', 'internship', 'professional',
                        'experience', 'employment', 'career', 'work history'
                    ]
                    if any(keyword in context.lower() for keyword in work_keywords):
                        date_ranges.append((start_date, end_date, context))
                except (ValueError, TypeError):
                    continue
        
        # Pattern 2: Year – Year or Year – Present (e.g., "2021 – 2023" or "2021 – Present")
        pattern2 = r'\b(19[5-9]\d|20[0-3]\d)\s*[–\-]\s*(present|current|(19[5-9]\d|20[0-3]\d))\b'
        for match in re.finditer(pattern2, text, re.IGNORECASE):
            start_year = int(match.group(1))
            end_str = match.group(2).lower()
            
            try:
                start_date = datetime(start_year, 1, 1)
                
                if self._is_ongoing_keyword(end_str):
                    end_date = current_date
                else:
                    end_year = int(end_str)
                    end_date = datetime(end_year, 12, 31)
                
                if start_date > end_date:
                    continue
                
                context_start = max(0, match.start() - 150)
                context_end = min(len(text), match.end() + 150)
                context = text[context_start:context_end]
                
                # Filter out education dates first
                if self._is_education_date(context):
                    logger.debug(f"Skipping education date range: {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}")
                    continue
                
                # Only include if context suggests work (expanded keywords)
                work_keywords = [
                    'company', 'worked', 'employed', 'joined', 'role', 'position', 'job',
                    'developer', 'engineer', 'manager', 'analyst', 'software', 'programmer',
                    'consultant', 'specialist', 'coordinator', 'assistant', 'director',
                    'lead', 'senior', 'junior', 'intern', 'internship', 'professional',
                    'experience', 'employment', 'career', 'work history'
                ]
                if any(keyword in context.lower() for keyword in work_keywords):
                    date_ranges.append((start_date, end_date, context))
            except (ValueError, TypeError):
                continue
        
        # Pattern 3: MM/YYYY – MM/YYYY (e.g., "02/2019 – 08/2022")
        pattern3 = r'\b(\d{1,2})/(\d{4})\s*[–\-]\s*(\d{1,2})?/(\d{4}|present|current)\b'
        for match in re.finditer(pattern3, text, re.IGNORECASE):
            start_month = int(match.group(1))
            start_year = int(match.group(2))
            end_month_str = match.group(3)
            end_str = match.group(4).lower()
            
            if not (1 <= start_month <= 12):
                continue
            
            try:
                start_date = datetime(start_year, start_month, 1)
                
                if self._is_ongoing_keyword(end_str):
                    end_date = current_date
                else:
                    end_year = int(end_str)
                    end_month = int(end_month_str) if end_month_str else 12
                    if not (1 <= end_month <= 12):
                        end_month = 12
                    end_date = datetime(end_year, end_month, 28)  # Use 28 to avoid month-end issues
                
                if start_date > end_date:
                    continue
                
                context_start = max(0, match.start() - 150)
                context_end = min(len(text), match.end() + 150)
                context = text[context_start:context_end]
                
                # Filter out education dates first
                if self._is_education_date(context):
                    logger.debug(f"Skipping education date range: {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}")
                    continue
                
                # Only include if context suggests work (expanded keywords)
                work_keywords = [
                    'company', 'worked', 'employed', 'joined', 'role', 'position', 'job',
                    'developer', 'engineer', 'manager', 'analyst', 'software', 'programmer',
                    'consultant', 'specialist', 'coordinator', 'assistant', 'director',
                    'lead', 'senior', 'junior', 'intern', 'internship', 'professional',
                    'experience', 'employment', 'career', 'work history'
                ]
                if any(keyword in context.lower() for keyword in work_keywords):
                    date_ranges.append((start_date, end_date, context))
            except (ValueError, TypeError):
                continue
        
        logger.debug(f"Extracted {len(date_ranges)} work date ranges")
        return date_ranges
    
    def _extract_dates_from_text(self, text: str) -> List[Tuple[datetime, str]]:
        """
        Extract all date patterns from text and return as datetime objects with context.
        DEPRECATED: Use _extract_work_date_ranges for better accuracy.
        
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
        # Use larger context window to capture section headers
        context_window = 200  # Increased from 100 to capture section headers
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
                    context_start = max(0, match.start() - context_window)
                    context_end = min(len(text), match.end() + context_window)
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
                    context_start = max(0, match.start() - context_window)
                    context_end = min(len(text), match.end() + context_window)
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
                    context_start = max(0, match.start() - context_window)
                    context_end = min(len(text), match.end() + context_window)
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
                    context_start = max(0, match.start() - context_window)
                    context_end = min(len(text), match.end() + context_window)
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
                    context_start = max(0, match.start() - context_window)
                    context_end = min(len(text), match.end() + context_window)
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
            'academic', 'qualification', 'qualifications', 'studied', 'course', 'program',
            'gpa', 'cgpa', 'grade', 'semester', 'semesters'
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
        
        # Check for education section headers (case-insensitive, with or without colon/hash)
        # Patterns: "EDUCATION", "education", "# EDUCATION", "EDUCATION:", "education:", etc.
        education_section_patterns = [
            r'^#?\s*education\s*:?\s*$',  # Standalone "EDUCATION" or "# EDUCATION" or "EDUCATION:"
            r'^education\s*$',  # Just "education" as a line
            r'education\s*:',  # "education:" with colon
            r'#\s*education',  # "# education" with hash
        ]
        
        # Check if context contains section headers
        # First, check if EDUCATION section header exists
        edu_section_found = False
        for pattern in education_section_patterns:
            if re.search(pattern, context, re.IGNORECASE | re.MULTILINE):
                edu_section_found = True
                break
        
        # Check if EXPERIENCE section header exists
        exp_section_found = re.search(r'^#?\s*experience\s*:?\s*$|^experience\s*$|experience\s*:|#\s*experience', context, re.IGNORECASE | re.MULTILINE)
        
        # If EDUCATION section found, check proximity to education keywords
        if edu_section_found:
            # Strong education indicators that suggest the date is education-related
            strong_edu_indicators = ['gpa', 'cgpa', 'bachelor', 'bachelors', 'degree', 'college', 'university', 
                                    'engineering', 'b.tech', 'm.tech', 'b.e.', 'm.e.', 'b.sc', 'm.sc']
            
            # Check if strong education indicators are near the date (within 50 chars)
            context_lower = context.lower()
            for indicator in strong_edu_indicators:
                if indicator in context_lower:
                    # Check if EXPERIENCE section comes after this indicator
                    edu_ind_pos = context_lower.find(indicator)
                    if exp_section_found:
                        exp_ind_pos = context_lower.find('experience')
                        # If EXPERIENCE comes after education indicator, and date is before EXPERIENCE, it's education
                        if exp_ind_pos > edu_ind_pos:
                            # Date is likely in EDUCATION section
                            return True
                    else:
                        # No EXPERIENCE section, so it's education
                        return True
        
        # If EXPERIENCE section found but no EDUCATION section, it's work
        if exp_section_found and not edu_section_found:
            return False
        
        # If only EDUCATION section found (no EXPERIENCE), it's education
        if edu_section_found and not exp_section_found:
            return True
        
        # Check for education keywords
        has_education_keyword = any(keyword in context_lower for keyword in education_keywords)
        
        # Check for work keywords
        has_work_keyword = any(keyword in context_lower for keyword in work_keywords)
        
        # If it has education keywords and no work keywords, it's likely education
        if has_education_keyword and not has_work_keyword:
            return True
        
        # Special case: If context has both "engineering" (from degree) and work keywords,
        # check the position - if "engineering" is near "bachelor", "degree", "college", it's education
        if 'engineering' in context_lower:
            # Check if "engineering" is part of degree name (education) or job title (work)
            edu_indicators = ['bachelor', 'degree', 'college', 'university', 'b.tech', 'b.e.', 'm.tech', 'm.e.']
            work_indicators = ['engineer', 'developer', 'software engineer', 'mechanical engineer']
            
            has_edu_indicator = any(ind in context_lower for ind in edu_indicators)
            has_work_indicator = any(ind in context_lower for ind in work_indicators)
            
            # If it has education indicators but not work indicators, it's education
            if has_edu_indicator and not has_work_indicator:
                return True
        
        # If it has work keywords, it's likely work (even if it has some education keywords)
        if has_work_keyword:
            return False
        
        return False
    
    def _merge_overlapping_ranges(self, date_ranges: List[Tuple[datetime, datetime, str]]) -> List[Tuple[datetime, datetime]]:
        """
        Merge overlapping date ranges to avoid double-counting.
        
        Args:
            date_ranges: List of (start_date, end_date, context) tuples
            
        Returns:
            List of merged (start_date, end_date) tuples with no overlaps
        """
        if not date_ranges:
            return []
        
        # Sort by start date
        sorted_ranges = sorted([(start, end) for start, end, _ in date_ranges])
        merged = [sorted_ranges[0]]
        
        for current_start, current_end in sorted_ranges[1:]:
            last_start, last_end = merged[-1]
            
            # Check if current range overlaps with last merged range
            # Overlap if: current_start <= last_end
            if current_start <= last_end:
                # Merge: extend end date if current is later
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                # No overlap, add as new range
                merged.append((current_start, current_end))
        
        return merged
    
    def _calculate_experience_from_dates(self, resume_text: str) -> Optional[str]:
        """
        Calculate years of experience from work date ranges.
        Handles overlaps correctly by merging overlapping periods.
        
        Returns:
            Experience string in format "X years" or None if cannot be calculated
        """
        if not resume_text:
            return None
        
        logger.info("📅 Starting date-based experience calculation")
        logger.debug(f"Resume text length: {len(resume_text)} characters")
        
        # First, try to extract work date ranges (more accurate)
        date_ranges = self._extract_work_date_ranges(resume_text)
        
        if date_ranges:
            # Merge overlapping ranges
            merged_ranges = self._merge_overlapping_ranges(date_ranges)
            logger.debug(f"Extracted {len(date_ranges)} date ranges, merged to {len(merged_ranges)} non-overlapping ranges")
            
            if merged_ranges:
                # Calculate total months across all non-overlapping ranges
                total_months = 0
                for start_date, end_date in merged_ranges:
                    months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
                    if end_date.day < start_date.day:
                        months -= 1
                    total_months += max(0, months)  # Ensure non-negative
                    logger.debug(f"Range: {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')} = {months} months")
                
                # Convert to years using ATS-standard rounding
                # If remaining months >= 6, increment year by 1
                years_diff = total_months // 12
                remaining_months = total_months % 12
                
                if remaining_months >= 6:
                    years_diff += 1
                
                logger.debug(f"Total months: {total_months}, Remaining months: {remaining_months}, Years: {years_diff}")
                
                # Validate: should be between 0 and 50 years
                if years_diff < 0:
                    logger.warning(f"Calculated negative years: {years_diff}, using 0")
                    years_diff = 0
                elif years_diff > 50:
                    logger.warning(f"Calculated unrealistic years: {years_diff}, capping at 50")
                    years_diff = 50
                
                # Handle cases where experience is less than 1 year
                if years_diff == 0:
                    if total_months >= 3:
                        experience_str = "1 year"
                        logger.info(f"✅ Date-based calculation: {experience_str} ({total_months} months, rounded up)")
                        return experience_str
                    else:
                        logger.warning(f"❌ Calculated {total_months} months of experience (less than 3 months, returning None)")
                        return None
                
                experience_str = f"{years_diff} years"
                logger.info(f"✅ Date-based calculation: {experience_str} ({total_months} months total across {len(merged_ranges)} periods)")
                return experience_str
        
        # Fallback to old method if new method didn't find ranges
        logger.debug("Work date range extraction found no ranges, falling back to individual date extraction")
        all_dates = self._extract_dates_from_text(resume_text)
        
        if not all_dates:
            logger.warning("❌ No dates found in resume text at all")
            return None
        
        logger.info(f"Found {len(all_dates)} total date occurrences in resume")
        
        # Filter out education-related dates
        work_dates_with_context = []
        for date_obj, context in all_dates:
            is_education = self._is_education_date(context)
            if not is_education:
                work_dates_with_context.append((date_obj, context))
        
        if not work_dates_with_context:
            logger.warning(f"❌ No work-related dates found after filtering")
            return None
        
        # Find oldest and most recent dates (fallback method)
        work_dates = [date_obj for date_obj, _ in work_dates_with_context]
        oldest_date = min(work_dates)
        most_recent_date = max(work_dates)
        
        current_date = datetime.now()
        if most_recent_date > current_date:
            most_recent_date = current_date
        
        # Calculate months difference
        months_diff = (most_recent_date.year - oldest_date.year) * 12 + (most_recent_date.month - oldest_date.month)
        if most_recent_date.day < oldest_date.day:
            months_diff -= 1
        
        # Convert to years using ATS-standard rounding
        # If remaining months >= 6, increment year by 1
        years_diff = months_diff // 12
        remaining_months = months_diff % 12
        
        if remaining_months >= 6:
            years_diff += 1
        
        if years_diff < 0:
            years_diff = 0
        elif years_diff > 50:
            years_diff = 50
        
        if years_diff == 0:
            if months_diff >= 3:
                return "1 year"
            return None
        
        experience_str = f"{years_diff} years"
        logger.info(f"✅ Date-based calculation (fallback): {experience_str} (from {oldest_date.strftime('%Y-%m')} to {most_recent_date.strftime('%Y-%m')})")
        
        return experience_str
    
    def _extract_experience_fallback(self, resume_text: str) -> Optional[str]:
        """
        Fallback regex-based extraction if LLM fails.
        Looks for common experience patterns in the resume text.
        Uses cleaned text to avoid false positives.
        """
        if not resume_text:
            logger.warning("Fallback extraction: resume_text is empty")
            return None
        
        logger.info(f"🔍 FALLBACK EXTRACTION: Starting regex-based experience extraction")
        logger.debug(f"Resume text length: {len(resume_text)} characters")
        
        # Clean the text first to remove education/certification blocks
        cleaned_text = self._clean_resume_text(resume_text)
        
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
        search_text = cleaned_text[:15000]
        summary_text = cleaned_text[:5000]  # First 5000 chars typically contain summary
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
        aggressive_text = cleaned_text[:2000].lower()
        
        # First, check for "X YEAR Y month" format in aggressive search
        year_month_pattern = r'(\d+)\s+YEAR\s+(\d+)\s+months?'
        match = re.search(year_month_pattern, aggressive_text, re.IGNORECASE)
        if match:
            # Check if "experience" is in context (within 150 chars before/after)
            context_start = max(0, match.start() - 150)
            context_end = min(len(aggressive_text), match.end() + 150)
            context = aggressive_text[context_start:context_end]
            
            if 'experience' in context:
                years = int(match.group(1))
                months = int(match.group(2))
                
                # Apply rounding logic: if months >= 6, add 1 year
                if months >= 6:
                    years += 1
                    logger.debug(f"Aggressive search - Year+month format: {match.group(1)} year(s) + {months} month(s) → rounded to {years} years (months >= 6)")
                else:
                    logger.debug(f"Aggressive search - Year+month format: {match.group(1)} year(s) + {months} month(s) → {years} years (months < 6)")
                
                # Validate: should be between 1 and 50 years
                if 1 <= years <= 50:
                    exp_str = f"{years} years"
                    logger.info(f"✅ Aggressive search found year+month format: '{match.group(0)}' → '{exp_str}'")
                    return exp_str
        
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
        date_based_experience = self._calculate_experience_from_dates(cleaned_text)
        if date_based_experience:
            logger.info(f"✅ Fallback date-based calculation extracted experience: '{date_based_experience}'")
            return date_based_experience
        else:
            logger.debug("Date-based calculation also returned None - checking if dates were found")
            # Try to extract dates to see if any were found (for debugging)
            all_dates = self._extract_dates_from_text(cleaned_text[:10000])  # Check first 10k chars
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
            return {"summary_experience": None, "date_ranges": []}
        
        cleaned_text = text.strip()
        
        # Remove markdown code blocks
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()
        
        # Try to find JSON object - look for first { and matching }
        # This handles cases where LLM adds text before/after JSON
        start_idx = cleaned_text.find('{')
        if start_idx == -1:
            # No JSON found, try to extract date ranges from text response
            logger.warning("No JSON object found in LLM response, attempting to extract dates from text")
            return self._extract_dates_from_text_response(cleaned_text)
        
        # Find matching closing brace
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
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict):
                    result = {
                        "summary_experience": parsed.get("summary_experience"),
                        "date_ranges": parsed.get("date_ranges", [])
                    }
                    logger.debug(f"Successfully extracted JSON: {result}")
                    return result
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON string: {e}")
        
        # If JSON parsing failed, try to extract dates from text response
        logger.warning("JSON parsing failed, attempting to extract dates from text response")
        return self._extract_dates_from_text_response(cleaned_text)
    
    def _extract_dates_from_text_response(self, text: str) -> Dict:
        """
        Extract date ranges from LLM text response when JSON parsing fails.
        Looks for date patterns in the text and extracts them.
        """
        date_ranges = []
        seen_ranges = set()  # Avoid duplicates
        
        # Look for date patterns in the text
        # Pattern 1: "Jan 2020 – Mar 2022" or "Apr 2018 – Dec 2019"
        month_pattern = r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\.?\s+(\d{4})\s*[–\-]\s*(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|present|current)\.?\s*(\d{4})?'
        
        for match in re.finditer(month_pattern, text, re.IGNORECASE):
            start_str = f"{match.group(1)} {match.group(2)}"
            if match.group(3).lower() in ['present', 'current']:
                end_str = "Present"
            else:
                end_str = f"{match.group(3)} {match.group(4)}" if match.group(4) else match.group(3)
            
            range_key = (start_str.lower(), end_str.lower())
            if range_key not in seen_ranges:
                seen_ranges.add(range_key)
                date_ranges.append({
                    "start": start_str,
                    "end": end_str,
                    "company": None,
                    "title": None
                })
        
        # Pattern 2: "2020 – 2022" or "2018 – 2019"
        year_pattern = r'\b(19[5-9]\d|20[0-3]\d)\s*[–\-]\s*(19[5-9]\d|20[0-3]\d|present|current)\b'
        
        for match in re.finditer(year_pattern, text, re.IGNORECASE):
            start_str = match.group(1)
            end_str = match.group(2)
            
            range_key = (start_str.lower(), end_str.lower())
            if range_key not in seen_ranges:
                seen_ranges.add(range_key)
                date_ranges.append({
                    "start": start_str,
                    "end": end_str,
                    "company": None,
                    "title": None
                })
        
        if date_ranges:
            logger.info(f"Extracted {len(date_ranges)} date ranges from text response")
            return {"summary_experience": None, "date_ranges": date_ranges}
        
        logger.warning("Could not extract date ranges from text response")
        return {"summary_experience": None, "date_ranges": []}
    
    def _check_explicit_experience(self, resume_text: str) -> Optional[str]:
        """
        Check for explicit total experience statement in resume.
        This is STEP 3 of the pipeline.
        
        Returns:
            Experience string if found, None otherwise
        """
        if not resume_text:
            return None
        
        # Search in first 10000 characters (summary/profile sections)
        search_text = resume_text[:10000].lower()
        
        # 1) Handle common fractional phrases like "5 and half years" explicitly,
        #    but normalize them to an INTEGER number of years (no decimals).
        fractional_patterns = [
            r'(\d+)\s+and\s+half\s+years?\s+of\s+experience',
            r'(\d+)\s+and\s+half\s+years?\s+of\s+professional\s+experience',
            r'(\d+)\s+and\s+half\s+years?\s+experience\b',
            r'(\d+)\s+and\s+a\s+half\s+years?\s+of\s+experience',
            r'(\d+)\s+and\s+a\s+half\s+years?\s+of\s+professional\s+experience',
            r'(\d+)\s+and\s+a\s+half\s+years?\s+experience\b',
        ]
        
        for pattern in fractional_patterns:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                years_num = int(match.group(1))
                if 1 <= years_num <= 50:
                    exp_str = f"{years_num} years"
                    logger.info(f"✅ Found explicit fractional experience statement (normalized to integer): '{exp_str}'")
                    return exp_str
        
        # 1.5) Handle "X YEAR Y month" format (e.g., "1 YEAR 11 month" or "WORK EXPERIENCE ( 1 YEAR 11 month )")
        year_month_pattern = r'(\d+)\s+YEAR\s+(\d+)\s+months?'
        match = re.search(year_month_pattern, search_text, re.IGNORECASE)
        if match:
            years = int(match.group(1))
            months = int(match.group(2))
            
            # Apply rounding logic: if months >= 6, add 1 year
            if months >= 6:
                years += 1
                logger.debug(f"Year+month format: {match.group(1)} year(s) + {months} month(s) → rounded to {years} years (months >= 6)")
            else:
                logger.debug(f"Year+month format: {match.group(1)} year(s) + {months} month(s) → {years} years (months < 6)")
            
            # Validate: should be between 1 and 50 years
            if 1 <= years <= 50:
                exp_str = f"{years} years"
                logger.info(f"✅ Found year+month format: '{match.group(0)}' → '{exp_str}'")
                return exp_str
        
        # 2) Standard explicit experience statements
        explicit_patterns = [
            r'(\d+\+?\s*years?)\s+of\s+experience',
            r'(\d+\+?\s*years?)\s+of\s+professional\s+experience',
            r'total\s+experience[:\s]+(\d+\+?\s*years?)',
            r'over\s+(\d+\+?\s*years?)\s+of\s+experience',
            r'more\s+than\s+(\d+\+?\s*years?)\s+of\s+experience',
            r'(\d+\+?\s*years?)\s+experience\s+in',
        ]
        
        for pattern in explicit_patterns:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                exp_str = match.group(1).strip()
                
                # Normalize to contain "years" / "year" and ONLY the integer part (no decimals)
                num_match = re.search(r'(\d+)', exp_str)
                if not num_match:
                    continue
                years_num = int(num_match.group(1))
                
                # Validate: should be between 1 and 50 years
                if 1 <= years_num <= 50:
                    normalized = f"{years_num} years"
                    logger.info(f"✅ Found explicit experience statement: '{exp_str}' → normalized to '{normalized}'")
                    return normalized
        
        return None
    
    def _is_ongoing_keyword(self, text: str) -> bool:
        """
        Check if text represents ongoing employment (all variations).
        Handles: Present, Still Date, Till Date, etc.
        
        Args:
            text: String to check (e.g., "Present", "Still Date", "Till Date")
        
        Returns:
            True if text represents ongoing employment, False otherwise
        """
        if not text:
            return False
        
        # Normalize: lowercase, strip, remove punctuation
        normalized = text.lower().strip()
        normalized = re.sub(r'[.,;:!?]', '', normalized)
        normalized = re.sub(r'[\s\-_]+', ' ', normalized).strip()
        
        # Comprehensive list of ongoing keywords
        ongoing_keywords = {
            # Standard
            'present', 'current', 'now', 'today',
            # Till variations
            'till date', 'till now', 'till-date', 'till-now', 'tilldate', 'tillnow',
            'til date', 'til now', 'til-date', 'til-now', 'tildate', 'tilnow',
            # Still variations
            'still date', 'still now', 'still-date', 'still-now', 'stilldate', 'stillnow',
            'still', 'still working', 'still employed', 'still active',
            # To variations
            'to date', 'to now', 'to-date', 'to-now', 'todate', 'tonow',
            # Until variations
            'until present', 'until now', 'until date', 'until-present', 'until-now', 'until-date',
            'untilpresent', 'untilnow', 'untildate',
            # Up to variations
            'up to present', 'up to now', 'up to date', 'up-to-present', 'up-to-now', 'up-to-date',
            'uptopresent', 'uptonow', 'uptodate',
            # As of variations
            'as of now', 'as of present', 'as of date', 'as of today',
            'as-of-now', 'as-of-present', 'as-of-date', 'as-of-today',
            'asofnow', 'asofpresent', 'asofdate', 'asoftoday',
            # Ongoing variations
            'ongoing', 'on-going', 'on going',
            # Working variations
            'working', 'working till date', 'working till now', 'working still',
            'working-till-date', 'working-till-now', 'working-still',
            'workingtilldate', 'workingtillnow', 'workingstill',
            # Currently variations
            'currently', 'currently working', 'currently employed', 'currently active',
            'currently-working', 'currently-employed', 'currently-active',
            'currentlyworking', 'currentlyemployed', 'currentlyactive',
            # Continuing variations
            'continuing', 'continue', 'continues',
            # Active variations
            'active', 'still active',
        }
        
        if normalized in ongoing_keywords:
            return True
        
        # Pattern-based fallback for unknown variations
        patterns = [
            r'^still\s*(date|now|working|employed|active)?$',
            r'^til?l?\s*(date|now)$',
            r'^to\s*(date|now|present)$',
            r'^until\s*(date|now|present)$',
            r'^up\s*to\s*(date|now|present)$',
            r'^as\s*of\s*(date|now|present|today)$',
            r'^working\s*(till|til|still|date|now)?$',
            r'^currently\s*(working|employed|active)?$',
        ]
        
        for pattern in patterns:
            if re.match(pattern, normalized):
                return True
        
        return False
    
    def _parse_llm_date_string(self, date_str: str) -> Optional[datetime]:
        """
        Parse date string from LLM output to datetime object.
        Handles formats: "Jan 2020", "January 2020", "01/2020", "2020", "Present", etc.
        
        Args:
            date_str: Date string from LLM (e.g., "Jan 2020", "Present")
            
        Returns:
            datetime object or None if parsing fails
        """
        if not date_str or not isinstance(date_str, str):
            return None
        
        date_str = date_str.strip()
        current_date = datetime.now()
        current_year = current_date.year
        
        # Handle ongoing keywords (all variations)
        if self._is_ongoing_keyword(date_str):
            return current_date
        
        month_names = {
            'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
            'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
            'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'sept': 9,
            'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
        }
        
        # Pattern 1: Month Year (e.g., "Jan 2020", "January 2020")
        month_year_pattern = r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\.?\s+(\d{4})\b'
        match = re.search(month_year_pattern, date_str, re.IGNORECASE)
        if match:
            month_str = match.group(1).lower()
            year = int(match.group(2))
            if month_str in month_names:
                try:
                    return datetime(year, month_names[month_str], 1)
                except ValueError:
                    pass
        
        # Pattern 2: Apostrophe format (e.g., "Jan'22", "Jan'20")
        apostrophe_pattern = r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\.?\s*[\'\']?(\d{2,4})\b'
        match = re.search(apostrophe_pattern, date_str, re.IGNORECASE)
        if match:
            month_str = match.group(1).lower()
            year_str = match.group(2)
            if month_str in month_names:
                try:
                    if len(year_str) == 2:
                        # Two-digit year rule
                        year = int(year_str)
                        if (year + 2000) <= current_year:
                            year = 2000 + year
                        else:
                            year = 1900 + year
                    else:
                        year = int(year_str)
                    return datetime(year, month_names[month_str], 1)
                except (ValueError, TypeError):
                    pass
        
        # Pattern 3: MM/YYYY or YYYY/MM (e.g., "01/2020", "2020/01")
        numeric_pattern = r'\b(\d{1,2})/(\d{4})\b'
        match = re.search(numeric_pattern, date_str)
        if match:
            first = int(match.group(1))
            year = int(match.group(2))
            if 1 <= first <= 12:
                try:
                    return datetime(year, first, 1)
                except ValueError:
                    pass
        
        # Pattern 4: YYYY-MM (e.g., "2020-01")
        iso_pattern = r'\b(\d{4})-(\d{1,2})\b'
        match = re.search(iso_pattern, date_str)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            if 1 <= month <= 12:
                try:
                    return datetime(year, month, 1)
                except ValueError:
                    pass
        
        # Pattern 5: Year only (e.g., "2020")
        year_pattern = r'\b(19[5-9]\d|20[0-3]\d)\b'
        match = re.search(year_pattern, date_str)
        if match:
            year = int(match.group(1))
            if 1950 <= year <= current_year:
                try:
                    return datetime(year, 1, 1)  # Default to January
                except ValueError:
                    pass
        
        logger.debug(f"Failed to parse date string: {date_str}")
        return None
    
    def _parse_llm_date_ranges(self, llm_date_ranges: List[Dict]) -> List[Tuple[datetime, datetime]]:
        """
        Parse LLM date ranges from string format to datetime objects.
        
        Args:
            llm_date_ranges: List of dicts with "start", "end", "company", "title"
            
        Returns:
            List of (start_date, end_date) tuples
        """
        parsed_ranges = []
        current_date = datetime.now()
        
        for range_dict in llm_date_ranges:
            if not isinstance(range_dict, dict):
                continue
            
            start_str = range_dict.get("start")
            end_str = range_dict.get("end")
            
            if not start_str:
                continue
            
            start_date = self._parse_llm_date_string(start_str)
            if not start_date:
                logger.debug(f"Skipping invalid start date: {start_str}")
                continue
            
            # Parse end date (default to current date if ongoing keyword)
            if not end_str or self._is_ongoing_keyword(end_str):
                end_date = current_date
            else:
                end_date = self._parse_llm_date_string(end_str)
                if not end_date:
                    logger.debug(f"Using current date for invalid end date: {end_str}")
                    end_date = current_date
            
            # Validate: start should be before end
            if start_date > end_date:
                logger.debug(f"Skipping invalid range: {start_date} > {end_date}")
                continue
            
            # Validate: start should not be in future
            if start_date > current_date:
                logger.debug(f"Skipping future start date: {start_date}")
                continue
            
            parsed_ranges.append((start_date, end_date))
            logger.debug(f"Parsed LLM date range: {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}")
        
        return parsed_ranges
    
    def _calculate_experience_from_llm_dates(self, llm_date_ranges: List[Dict]) -> Optional[str]:
        """
        Calculate experience from LLM-extracted date ranges.
        
        Args:
            llm_date_ranges: List of date range dicts from LLM
            
        Returns:
            Experience string in format "X years" or None
        """
        if not llm_date_ranges:
            return None
        
        # Parse LLM date strings to datetime objects
        parsed_ranges = self._parse_llm_date_ranges(llm_date_ranges)
        if not parsed_ranges:
            logger.warning("Failed to parse any LLM date ranges")
            return None
        
        # Merge overlapping ranges
        date_ranges_with_context = [(start, end, "") for start, end in parsed_ranges]
        merged_ranges = self._merge_overlapping_ranges(date_ranges_with_context)
        
        if not merged_ranges:
            return None
        
        # Calculate total months
        total_months = 0
        for start_date, end_date in merged_ranges:
            months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
            if end_date.day < start_date.day:
                months -= 1
            total_months += max(0, months)
        
        # Convert to years using ATS-standard rounding
        # If remaining months >= 6, increment year by 1
        years_diff = total_months // 12
        remaining_months = total_months % 12
        
        if remaining_months >= 6:
            years_diff += 1
        
        # Validate
        if years_diff < 0:
            years_diff = 0
        elif years_diff > 50:
            years_diff = 50
        
        if years_diff == 0:
            if total_months >= 3:
                return "1 year"
            return None
        
        return f"{years_diff} years"
    
    def is_fresher(self, resume_text: str, date_ranges: list) -> bool:
        """
        Detect if the candidate is a fresher (no work experience).
        
        Args:
            resume_text: The resume text
            date_ranges: List of date ranges found (can be empty)
            
        Returns:
            True if fresher, False otherwise
        """
        if not resume_text:
            return False
        
        text = resume_text.lower()
        
        # Fresher keywords (using regex patterns to handle variations)
        fresher_patterns = [
            r'\brecent\s+(?:college\s+)?graduate\b',  # "recent graduate" or "recent college graduate"
            r'\bnew\s+graduate\b',
            r'\bfresh\s+graduate\b',
            r'\brecently\s+graduated\b',
            r'\bfresher\b',
            r'\bentry[\s-]?level\b',  # "entry level" or "entry-level"
            r'\blooking\s+for\s+first\s+opportunity\b',
            r'\bno\s+(?:work\s+)?experience\b',  # "no experience" or "no work experience"
            r'\bseeking\s+first\s+job\b',
            r'\bfirst\s+job\b',
            r'\bgraduate\s+seeking\b',
        ]
        
        # Check for fresher patterns
        for pattern in fresher_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                logger.info(f"Fresher detected: Found fresher keyword pattern: {pattern}")
                return True
        
        # If no date ranges found, check for education-only profile
        if not date_ranges:
            education_keywords = ["education", "degree", "b.tech", "bachelor", "master", "phd", "diploma"]
            work_keywords = ["experience", "worked", "employed", "job", "position", "role", "company"]
            
            has_education = any(keyword in text for keyword in education_keywords)
            has_work = any(keyword in text for keyword in work_keywords)
            
            if has_education and not has_work:
                logger.info("Fresher detected: Education found but no work experience")
                return True
        
        return False
    
    def _calculate_confidence_score(self, experience: Optional[str], source: str,
                                   explicit_found: bool, date_ranges_count: int,
                                   llm_summary: Optional[int], python_dates: Optional[str],
                                   llm_dates: Optional[str]) -> float:
        """
        Calculate confidence score (0.0 to 1.0) for extracted experience.
        
        Args:
            experience: Final experience string
            source: Extraction method used
            explicit_found: Whether explicit experience was found
            date_ranges_count: Number of date ranges found
            llm_summary: LLM summary experience value
            python_dates: Python date-based calculation
            llm_dates: LLM date-based calculation
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not experience:
            return 0.0
        
        confidence = 0.5  # Base confidence
        
        # Source-based confidence
        if source == "explicit_python":
            confidence = 0.95  # Very high - explicit statements are most reliable
        elif source == "explicit_llm":
            confidence = 0.90  # High - LLM found explicit statement
        elif source == "date_based_python":
            confidence = 0.85  # High - Python regex is deterministic
        elif source == "date_based_llm":
            confidence = 0.75  # Medium-high - LLM dates need parsing
        elif source == "fresher":
            confidence = 0.80  # High - Fresher detection is reliable
        elif source == "regex_fallback":
            confidence = 0.60  # Medium - Fallback regex
        
        # Agreement bonus: If multiple sources agree, increase confidence
        if explicit_found and llm_summary:
            # Extract years from both
            exp_match = re.search(r'(\d+)', experience)
            if exp_match:
                exp_years = int(exp_match.group(1))
                if abs(exp_years - llm_summary) <= 1:  # Within 1 year
                    confidence = min(1.0, confidence + 0.05)
        
        if python_dates and llm_dates:
            # Compare Python and LLM date-based calculations
            py_match = re.search(r'(\d+)', python_dates)
            llm_match = re.search(r'(\d+)', llm_dates)
            if py_match and llm_match:
                py_years = int(py_match.group(1))
                llm_years = int(llm_match.group(1))
                if abs(py_years - llm_years) <= 1:  # Within 1 year
                    confidence = min(1.0, confidence + 0.05)
        
        # Date range count bonus: More ranges = more reliable
        if date_ranges_count >= 3:
            confidence = min(1.0, confidence + 0.05)
        elif date_ranges_count >= 2:
            confidence = min(1.0, confidence + 0.03)
        
        return round(confidence, 2)
    
    async def extract_experience(self, resume_text: str, filename: str = "resume") -> Optional[str]:
        """
        Extract years of experience from resume text using recommended ATS pipeline.
        
        Pipeline:
        STEP 1: Explicit Experience (LLM + Python)
        STEP 2: Date Range Extraction (LLM) → Calculate in Python
        STEP 2B: Fresher Detection
        STEP 3: Regex Fallback
        
        Args:
            resume_text: The text content of the resume
            filename: Name of the resume file (for logging)
        
        Returns:
            The extracted experience string or None if not found
        """
        if not resume_text:
            logger.warning(f"Empty resume text for {filename}")
            return None

        logger.info(f"🔍 Starting experience extraction for {filename}")

        # STEP 0: Cache check
        import hashlib
        cache_key = hashlib.md5(resume_text[:5000].encode()).hexdigest()
        if cache_key in self._cache:
            cached_result, cached_time = self._cache[cache_key]
            if (datetime.now().timestamp() - cached_time) < self._cache_ttl:
                logger.info(f"✅ EXPERIENCE FROM CACHE for {filename}: {cached_result}")
                return cached_result
            del self._cache[cache_key]

        # STEP 1: Clean resume
        cleaned_text = self._clean_resume_text(resume_text)

        # STEP 2: Extract work sections
        work_text = self._extract_work_sections_only(cleaned_text)
        if not work_text.strip():
            work_text = cleaned_text[:15000]
        
        # Ensure work_text is not empty - use original resume_text as last resort
        if not work_text.strip():
            logger.warning(f"work_text is empty, using original resume_text for fallback")
            work_text = resume_text[:20000]

        # STEP 1: EXPLICIT EXPERIENCE (LLM + Python)
        # Initialize variables
        experience = None
        source = "unknown"
        python_explicit = None
        llm_experience = None
        llm_date_ranges = []
        llm_summary_years = None
        python_date = None
        
        # Check Python first (fast path)
        python_explicit = self._check_explicit_experience(resume_text)
        if python_explicit:
            logger.info(f"✅ Explicit experience found (Python fast path): {python_explicit}")
            experience = python_explicit
            source = "explicit_python"
        else:
            # Try LLM for explicit experience

            try:
                is_connected, available_model = await self._check_ollama_connection()
                if not is_connected:
                    logger.warning(f"OLLAMA not connected, skipping LLM for {filename}")
                    llm_experience = None
                else:
                    model_to_use = available_model or self.model
                    text_to_send = work_text[:20000]

                    current_date = datetime.now()
                    prompt = f"""{EXPERIENCE_PROMPT}

CURRENT DATE INFORMATION:
• Today's date: {current_date.strftime("%B %d, %Y")}
• Current year: {current_date.year}

Input resume text:
{text_to_send}

Output (JSON only):"""

                    async with httpx.AsyncClient(timeout=Timeout(1200.0)) as client:
                        response = await client.post(
                            f"{self.ollama_host}/api/generate",
                            json={
                                "model": model_to_use,
                                "prompt": prompt,
                                "stream": False,
                                "options": {
                                    "temperature": 0.1,
                                    "top_p": 0.9
                                }
                            }
                        )
                        response.raise_for_status()
                        result = response.json()

                    # Extract raw output
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

                    # Parse JSON
                    parsed_data = self._extract_json(raw_output)

                    # STEP 1A: Check explicit summary experience from LLM
                    summary_experience = parsed_data.get("summary_experience")
                    if summary_experience is not None:
                        try:
                            # Accept int/float directly, or extract the first integer from a string.
                            years: Optional[int] = None
                            
                            if isinstance(summary_experience, (int, float)):
                                years = int(summary_experience)
                            else:
                                # Extract leading integer from strings like "5", "5.5", "5 and half"
                                num_match = re.search(r'(\d+)', str(summary_experience))
                                if num_match:
                                    years = int(num_match.group(1))
                            
                            if years is not None and 1 <= years <= 50:
                                llm_experience = f"{years} years"
                                llm_summary_years = years
                                logger.info(f"✅ LLM explicit experience: {llm_experience}")
                            else:
                                logger.warning(f"Invalid or out-of-range summary_experience: {summary_experience}")
                        except Exception:
                            logger.warning(f"Failed to parse summary_experience: {summary_experience}", exc_info=True)

                    # STEP 2: Date ranges (only if no explicit experience)
                    if not llm_experience:
                        llm_date_ranges = parsed_data.get("date_ranges", [])
                        if isinstance(llm_date_ranges, list) and llm_date_ranges:
                            logger.info(f"LLM extracted {len(llm_date_ranges)} date ranges")
                            llm_experience = self._calculate_experience_from_llm_dates(llm_date_ranges)
                            if llm_experience:
                                logger.info(f"✅ LLM date-based experience: {llm_experience}")

            except Exception as e:
                logger.error(f"LLM extraction failed for {filename}: {e}", exc_info=True)
                llm_experience = None
                llm_date_ranges = []

            # STEP 1B: Use LLM explicit if found
            if llm_experience and llm_summary_years:
                experience = llm_experience
                source = "explicit_llm"
                confidence = 0.90
            # STEP 2: Use LLM date-based if found
            elif llm_experience:
                experience = llm_experience
                source = "date_based_llm"
                confidence = 0.75
            # STEP 2B: Fresher detection
            elif self.is_fresher(resume_text, llm_date_ranges):
                logger.info(f"✅ Fresher detected for {filename}")
                experience = "0 years"
                source = "fresher"
                confidence = 0.80
            # STEP 3: Python fallback
            else:
                # Ensure we have text to work with
                fallback_text = work_text if work_text and work_text.strip() else resume_text
                
                # Try Python date-based calculation
                python_date = self._calculate_experience_from_dates(fallback_text)
                if python_date:
                    experience = python_date
                    source = "date_based_python"
                    confidence = 0.85
                else:
                    # Final regex fallback
                    experience = self._extract_experience_fallback(fallback_text)
                    if experience:
                        source = "regex_fallback"
                        confidence = 0.60
                    else:
                        # Check fresher again (in case LLM didn't run)
                        if self.is_fresher(resume_text, []):
                            logger.info(f"✅ Fresher detected (fallback) for {filename}")
                            experience = "0 years"
                            source = "fresher"
                            confidence = 0.80

        # Calculate final confidence score
        final_confidence = self._calculate_confidence_score(
            experience=experience,
            source=source,
            explicit_found=(python_explicit is not None) or (llm_summary_years is not None),
            date_ranges_count=len(llm_date_ranges),
            llm_summary=llm_summary_years,
            python_dates=python_date,
            llm_dates=llm_experience if source == "date_based_llm" else None
        )

        logger.info(
            f"✅ EXPERIENCE RESULT for {filename}",
            extra={
                "file": filename,
                "experience": experience,
                "source": source,
                "confidence": final_confidence
            }
        )

        if experience:
            self._cache[cache_key] = (experience, datetime.now().timestamp())

        return experience
