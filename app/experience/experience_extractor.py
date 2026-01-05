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
You are an ATS resume-parsing expert specializing in accurate professional
work-experience extraction.

CURRENT DATE CONTEXT (CRITICAL):
• The current date will be provided below in "CURRENT DATE INFORMATION" section.
• Use that exact date for all date validations and calculations.
• The current year MUST be treated as valid for experience calculation.
• Dates equal to the current year are NOT future dates.
• Only dates AFTER the provided current date should be considered future dates.

TASK:
Extract the candidate’s TOTAL PROFESSIONAL WORK EXPERIENCE in YEARS.

DEFINITION:
Professional experience includes ONLY:
• Full-time paid employment
• Part-time paid employment
• Contract / freelance professional work
• Internships ONLY if explicitly stated as employment

DO NOT COUNT:
• Education or academic duration
• Certifications, courses, or training timelines
• Academic or personal projects
• Tool or standard versions (e.g., ISO 9001:2015, Python 3.10)
• Company founding years
• Research or teaching unless explicitly stated as employment

--------------------------------------------------
DATE FORMAT SUPPORT:
Accept ALL common resume date formats, including but not limited to:

TEXTUAL MONTH–YEAR:
• Jan 2022
• January 2022
• Jan, 2022
• January, 2022
• Jan-2022
• January-2022

APOSTROPHE YEAR FORMATS:
• Jan'22
• Feb’21
• Mar'19
• Dec’05

NUMERIC MONTH–YEAR:
• 01/2022
• 1/2022
• 01-2022
• 2022/01
• 2022-01

FULL DATE:
• 01 Jan 2022
• 01/01/2022
• 2022-01-01

YEAR-ONLY:
• 2020
• 2020 – 2022
• 2019 to 2023
(ONLY if employment context is present)

--------------------------------------------------
DATE RANGES:
• Jan 2020 – Mar 2022
• January 2020 to February 2023
• Jan'20 – Feb'23
• 01/2020 – 03/2022
• 2020-01 to 2022-03
• 2018 – Feb 2023
• [Previous Year] – Present (e.g., if current year is 2026, then "Jan 2025 – Present" is valid)
• [Current Year] – Present (e.g., if current year is 2026, then "Jan 2026 – Present" is valid)
• [Any Past Year] – Now
• [Any Past Year] – Till Date

SEPARATORS:
• "-", "–", "—", "to", "until", "till"

--------------------------------------------------
PRESENT / CURRENT INDICATORS (MANDATORY INTERPRETATION):
The following MUST be treated as TODAY’S DATE:
• Present
• Current
• Now
• Till Date
• Till Now
• Ongoing
• Working
• Working till date

--------------------------------------------------
TWO-DIGIT YEAR RULE:
• The current year will be provided in "CURRENT DATE INFORMATION" section below.
• If (year + 2000) ≤ current year (from provided date) → use 2000s
• Otherwise → use 1900s
• Example: If current year is 2026, then '25' → 2025, '26' → 2026, '27' → 2027

--------------------------------------------------
DATE VALIDATION RULES (VERY IMPORTANT):

• Start dates in the CURRENT YEAR are VALID
• Start dates earlier than today are VALID
• End date = Present / Now → VALID ongoing employment

FUTURE DATE HANDLING:
• Ignore ONLY date ranges where:
  – Start date is AFTER today
• DO NOT ignore ranges just because:
  – The year equals the current year
  – The end date is “Present / Now / Till Date”

--------------------------------------------------
DATE RANGE IDENTIFICATION:
A date is valid ONLY if it appears:
• Near a company name, OR
• Near a job title, OR
• Inside a work / employment section

Ignore dates near:
• Education keywords
• Certification keywords
• Projects without employment context

--------------------------------------------------
CALCULATION RULES:

1) EXPLICIT EXPERIENCE (HIGHEST PRIORITY)
If resume clearly states total experience
(e.g., “5 years experience”, “8+ years total experience”):
• Use numeric value only
• Remove any "+" sign

If multiple explicit values exist:
• Prefer summary/profile section
• Otherwise choose most specific

2) DATE-BASED EXPERIENCE
If explicit experience NOT present:
• Identify all valid employment date ranges
• Convert each range to months
• Merge overlapping periods
• Exclude gaps
• Sum total months
• Convert to years (months ÷ 12)
• Round DOWN to whole years

If explicit value contradicts dates by >3 years:
• Prefer date-based calculation

--------------------------------------------------
EDGE CASES:
• Start + End OR Start + Present required
• Single date without Present → return null
• Start after end → return null
• Less than 3 months total → return null
• 3–11 months → return "1 year"
• Experience > 50 years → return null

--------------------------------------------------
ANTI-HALLUCINATION RULES:
• NEVER guess
• NEVER infer
• NEVER assume missing dates
• Return null if confidence is low

--------------------------------------------------
OUTPUT REQUIREMENTS:
• Output ONLY valid JSON
• Whole numbers only
• Always include "years"
• No explanations, no markdown

JSON SCHEMA:
{
  "experience": "string | null"
}

Example output format:
{"experience": "<number> years"}
{"experience": null}

DO NOT use these exact values. Extract the actual experience from the resume.

"""



class ExperienceExtractor:
    """Service for extracting years of experience from resume text using OLLAMA LLM."""
    
    def __init__(self):
        self.ollama_host = settings.ollama_host
        self.model = "llama3.1"
    
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
                if end_month_str in ['present', 'current']:
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
                    
                    if end_month_str in ['present', 'current'] or not end_year_str:
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
                
                if end_str in ['present', 'current']:
                    end_date = current_date
                else:
                    end_year = int(end_str)
                    end_date = datetime(end_year, 12, 31)
                
                if start_date > end_date:
                    continue
                
                context_start = max(0, match.start() - 150)
                context_end = min(len(text), match.end() + 150)
                context = text[context_start:context_end]
                
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
                
                if end_str in ['present', 'current']:
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
                
                # Convert to years (round down as per requirements)
                years_diff = total_months // 12
                
                logger.debug(f"Total months: {total_months}, Years (rounded down): {years_diff}")
                
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
        
        # Round DOWN to nearest year (not round to nearest)
        years_diff = months_diff // 12
        
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
        
        # Patterns for explicit experience statements
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
                # Ensure it has "years" or "year"
                if 'year' not in exp_str.lower():
                    num_match = re.search(r'(\d+\+?)', exp_str)
                    if num_match:
                        exp_str = f"{num_match.group(1)} years"
                    else:
                        exp_str = f"{exp_str} years"
                
                # Validate: should be between 1 and 50 years
                num_match = re.search(r'(\d+)', exp_str)
                if num_match:
                    years_num = int(num_match.group(1))
                    if 1 <= years_num <= 50:
                        logger.info(f"✅ Found explicit experience statement: '{exp_str}'")
                        return exp_str
        
        return None
    
    async def extract_experience(self, resume_text: str, filename: str = "resume") -> Optional[str]:
        """
        Extract years of experience from resume text using production-grade pipeline.
        
        Pipeline:
        1. Clean resume text (remove education, certifications, projects)
        2. Extract work sections only
        3. Check for explicit total experience statement
        4. If not found, calculate from date ranges
        5. Send cleaned work text to LLM for validation
        6. Return result
        
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
        
        # STEP 1: Clean resume text
        cleaned_text = self._clean_resume_text(resume_text)
        
        # STEP 2: Extract work sections only
        work_text = self._extract_work_sections_only(cleaned_text)
        
        # If no work sections found, use cleaned text
        if not work_text.strip():
            work_text = cleaned_text[:15000]  # Fallback to cleaned text
            logger.debug("No work sections found, using cleaned text")
        
        # STEP 3: Check for explicit total experience statement
        explicit_experience = self._check_explicit_experience(resume_text)
        if explicit_experience:
            logger.info(f"✅ EXPERIENCE EXTRACTED (explicit) from {filename}: {explicit_experience}")
            return explicit_experience
        
        # STEP 4: Calculate experience from date ranges
        date_based_experience = self._calculate_experience_from_dates(work_text)
        if date_based_experience:
            logger.info(f"✅ EXPERIENCE EXTRACTED (date-based) from {filename}: {date_based_experience}")
            # Continue to LLM validation, but we have a good baseline
        
        # STEP 5: Send cleaned work text to LLM for validation/refinement
        model_to_use = self.model
        try:
            is_connected, available_model = await self._check_ollama_connection()
            if not is_connected:
                # Return date-based calculation if available
                if date_based_experience:
                    logger.warning(f"OLLAMA not connected, returning date-based calculation for {filename}")
                    return date_based_experience
                # Try fallback regex
                experience = self._extract_experience_fallback(work_text)
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
            
            # Send only cleaned work text to LLM (limit to 20000 chars)
            text_to_send = work_text[:20000]
            
            # Get current date information for prompt
            current_date = datetime.now()
            current_date_str = current_date.strftime("%B %d, %Y")  # e.g., "January 15, 2026"
            current_year = current_date.year  # e.g., 2026
            current_month_year = current_date.strftime("%B %Y")  # e.g., "January 2026"
            
            # Create dynamic prompt with current date information
            prompt = f"""{EXPERIENCE_PROMPT}

CURRENT DATE INFORMATION (USE THIS FOR ALL DATE VALIDATIONS):
• Today's date: {current_date_str}
• Current year: {current_year}
• Current month and year: {current_month_year}
• Any date in {current_year} or earlier is VALID
• Only dates AFTER {current_date_str} should be considered future dates
• Dates in {current_year} with "Present", "Now", or "Till Date" are VALID ongoing employment

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
            llm_experience = parsed_data.get("experience")
            
            if llm_experience:
                llm_experience = str(llm_experience).strip()
                if not llm_experience or llm_experience.lower() in ["null", "none", ""]:
                    llm_experience = None
            
            # STEP 6: Validate LLM output against date-based calculation
            # If LLM returned null or value differs significantly from date-based, use date-based
            if llm_experience:
                # Extract number from LLM result
                llm_num_match = re.search(r'(\d+)', llm_experience)
                if llm_num_match:
                    llm_years = int(llm_num_match.group(1))
                    
                    # If we have date-based calculation, compare
                    if date_based_experience:
                        date_num_match = re.search(r'(\d+)', date_based_experience)
                        if date_num_match:
                            date_years = int(date_num_match.group(1))
                            # If difference is more than 3 years, prefer date-based (more reliable)
                            if abs(llm_years - date_years) > 3:
                                logger.warning(
                                    f"LLM result ({llm_years} years) differs significantly from date-based ({date_years} years). Using date-based.",
                                    extra={"llm_experience": llm_experience, "date_experience": date_based_experience}
                                )
                                experience = date_based_experience
                            else:
                                # LLM result is reasonable, use it
                                experience = llm_experience
                        else:
                            experience = llm_experience
                    else:
                        # No date-based to compare, use LLM result
                        experience = llm_experience
                else:
                    # LLM returned invalid format, use date-based if available
                    experience = date_based_experience
            else:
                # LLM returned null, use date-based if available
                logger.warning(
                    f"LLM returned null experience for {filename}, using date-based calculation",
                    extra={"file_name": filename}
                )
                experience = date_based_experience
            
            # Final fallback: try regex extraction
            if not experience:
                logger.debug(f"Trying regex fallback for {filename}")
                experience = self._extract_experience_fallback(work_text)
                if experience:
                    logger.info(
                        f"✅ EXPERIENCE EXTRACTED via fallback from {filename}",
                        extra={
                            "file_name": filename,
                            "experience": experience,
                            "method": "regex_fallback"
                        }
                    )
            
            logger.info(
                f"✅ EXPERIENCE EXTRACTED from {filename}",
                extra={
                    "file_name": filename,
                    "experience": experience,
                    "status": "success" if experience else "not_found",
                    "method": "llm" if llm_experience else ("date_based" if date_based_experience else "fallback")
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

