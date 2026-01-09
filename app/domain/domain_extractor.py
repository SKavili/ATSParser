"""Service for extracting industry domain from resumes using OLLAMA LLM."""
import json
import re
from typing import Dict, Optional, List
from dataclasses import dataclass
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

DOMAIN_PROMPT = """
IMPORTANT:
This is a FRESH, ISOLATED, SINGLE-TASK extraction.
Ignore ALL previous conversations, memory, instructions, or assumptions.

ROLE:
You are an ATS resume parser specializing in conservative, evidence-based industry domain identification.

TASK:
Determine the industry domain of the candidate's MOST RECENT job role ONLY.
Return EXACTLY ONE domain OR null.

DEFINITION:
"Domain" = PRIMARY BUSINESS/INDUSTRY where candidate WORKED (company/industry).
NOT skills, tools, technologies, education, or academic projects.

EVIDENCE HIERARCHY (use in this order):
1. Employer organization/company
2. Paying client/customer
3. Commercial product/service
4. Business operations described
5. Industry-regulated terminology

CRITICAL RULES:
- Use ONLY business/industry context (company, client, product, operations).
- NEVER infer domain from skills, programming languages, tools, or technologies.
- NEVER guess or assume - require explicit evidence.
- NEVER default to IT without explicit IT company/business context.
- Job titles are unreliable unless clearly industry-specific.
- If domain is unclear or ambiguous → return null (prefer null over wrong classification).
- Same role text must produce same domain result (deterministic).

PLATFORM DOMAINS (return ONLY if work is clearly centered on that platform):
- "Salesforce" → ONLY if job title/role is "Salesforce Admin/Developer/Consultant/Architect"
- "AWS" → ONLY if job title/role is "AWS Solutions Architect/Engineer/Consultant" OR explicit AWS-focused role
- "SAP" → ONLY if job title/role is "SAP Consultant/Developer" OR primarily SAP implementation
- "Microsoft" → ONLY if job title/role is "Microsoft Consultant/Architect" OR primarily Microsoft stack work
- "Oracle" → ONLY if job title/role is "Oracle Consultant/Developer" OR primarily Oracle products
- "ServiceNow", "Workday", "Adobe", "Google Cloud", "Azure" → Same strict criteria

AWS ANTI-HALLUCINATION:
DO NOT return "AWS" if:
- Text only mentions "AWS" in skills/technologies used
- Text mentions "cloud" or "cloud services" without explicit AWS role
- Text mentions "EC2", "S3", "Lambda" without explicit AWS-focused job title
- Work is at a company that uses AWS but role is not AWS-specific

Return "AWS" ONLY if:
- Job title contains "AWS" (e.g., "AWS Solutions Architect", "AWS Engineer")
- Role description explicitly states PRIMARY work is AWS architecture/consulting

EXAMPLES:
- "Python Developer at Bank of America" → Banking (NOT IT, NOT Software)
- "Salesforce Developer at Bank of America" → Salesforce (platform-specific, NOT Banking)
- "Software Engineer using AWS at Bank" → Banking (NOT AWS - no explicit AWS role)
- "AWS Solutions Architect at TechCorp" → AWS (explicit AWS role)
- "Backend Engineer at TechCorp Solutions" → Information Technology (generic IT company)

IT vs SOFTWARE vs PLATFORM:
- "Information Technology" → Generic IT services company or IT department
- "Software & SaaS" → Software product company (builds/sells software products)
- "Salesforce/AWS/SAP/etc." → Specific platform domain (highest specificity, requires explicit platform role)

DOMAIN CONSISTENCY:
- Analyze the SAME role text consistently
- If domain is unclear or ambiguous → return null (do NOT flip between domains)
- If multiple domains are possible → return null (do NOT randomly pick one)
- Same role text should produce same domain result (deterministic)

OUTPUT:
Return ONLY valid JSON. No explanations. No extra text.

JSON SCHEMA:
{
  "domain": "string | null"
}
"""



class DomainExtractor:
    """Service for extracting industry domain from resume text using OLLAMA LLM."""
    
    # Domain precedence map: Higher priority domains come first
    # Used to resolve conflicts when multiple domains are detected
    DOMAIN_PRIORITY = [
        "Banking, Financial Services & Insurance (BFSI)",
        "Banking",
        "Insurance",
        "Capital Markets",
        "FinTech",
        "Finance",
        "Finance & Accounting",
        "Healthcare & Life Sciences",
        "Healthcare",
        "Pharmaceuticals & Clinical Research",
        "Retail & E-Commerce",
        "Retail",
        "E-Commerce",
        "Manufacturing & Production",
        "Manufacturing",
        "Supply Chain",
        "Operations & Supply Chain Management",
        "Logistics",
        "Logistics & Transportation",
        "Education, Training & Learning",
        "Education",
        "Government",
        "Public Sector",
        "Public Sector & Government Services",
        "Defense",
        "Energy, Utilities & Sustainability",
        "Energy",
        "Utilities",
        "Telecommunications",
        "Media & Entertainment",
        "Media, Advertising & Communications",
        "Gaming",
        "Real Estate & Facilities Management",
        "Real Estate",
        "Construction & Infrastructure",
        "Construction",
        "Hospitality",
        "Travel & Tourism",
        "Agriculture",
        "Agri-Business",
        "Legal, Risk & Corporate Governance",
        "Quality, Compliance & Audit",
        "Human Resources",
        "Sales & Marketing",
        "Customer Service & Customer Experience",
        "Administration & Office Management",
        "Non-Profit",
        "NGOs, Social Impact & CSR",
        "Transportation",
        "Automotive",
        "Aerospace",
        # IT domains at bottom - business domain must override IT
        "Information Technology",
        "Software & SaaS",
        "Cloud & Infrastructure",
        "Cybersecurity",
        "Data & Analytics",
        "Artificial Intelligence",
        # Specific Technology/Platform domains (highest specificity)
        "Salesforce",
        "AWS",
        "Microsoft",
        "Oracle",
        "SAP",
        "ServiceNow",
        "Workday",
        "Adobe",
        "Google Cloud",
        "Azure",
    ]
    
    # Employer domain mapping (deterministic - highest priority)
    # Maps known employer names to their domains
    EMPLOYER_DOMAIN_MAP = {
        # Healthcare
        "myeyedr": "Healthcare",
        "apollo": "Healthcare",
        "fortis": "Healthcare",
        "narayana": "Healthcare",
        "max healthcare": "Healthcare",
        "manipal": "Healthcare",
        "apollo hospitals": "Healthcare",
        "fortis healthcare": "Healthcare",
        "narayana health": "Healthcare",
        "mayo clinic": "Healthcare",
        "cleveland clinic": "Healthcare",
        "kaiser permanente": "Healthcare",
        "johns hopkins": "Healthcare",
        "mass general": "Healthcare",
        "vancouver clinic": "Healthcare",
        
        # Banking
        "bank of america": "Banking",
        "hdfc": "Banking",
        "icici": "Banking",
        "sbi": "Banking",
        "state bank": "Banking",
        "chase": "Banking",
        "wells fargo": "Banking",
        "citibank": "Banking",
        "jpmorgan": "Banking",
        "goldman sachs": "Banking",
        "morgan stanley": "Banking",
        
        # Retail
        "walmart": "Retail",
        "target": "Retail",
        "costco": "Retail",
        "home depot": "Retail",
        
        # E-Commerce
        "amazon": "E-Commerce",  # Note: Amazon retail vs AWS - context matters
        "ebay": "E-Commerce",
        "etsy": "E-Commerce",
        
        # Government/Defense
        "drdo": "Defense",
        "isro": "Aerospace",
        "barc": "Energy",
        "nasa": "Aerospace",
        "dod": "Defense",
        "department of defense": "Defense",
        
        # Legal
        "epiq systems": "Legal, Risk & Corporate Governance",
        
        # HR
        "hireright": "Human Resources",
    }
    
    # Healthcare keyword terms (deterministic override)
    HEALTHCARE_KEYWORDS = [
        "patient", "patients", "clinic", "clinics", "hospital", "hospitals",
        "optometry", "optometrist", "optometrists",
        "ehr", "emr", "electronic health record", "electronic medical record",
        "medical", "healthcare", "health care", "health system",
        "physician", "physicians", "nurse", "nurses", "doctor", "doctors",
        "clinical", "medicare", "medicaid", "hipaa",
        "pharmacy", "pharmaceutical", "diagnosis", "treatment", "therapy"
    ]
    
    # Banking keyword terms (deterministic override)
    BANKING_KEYWORDS = [
        "bank", "banking", "financial institution", "credit union",
        "mortgage", "lending", "loan", "deposit", "teller",
        "branch banking", "commercial bank", "retail banking",
        "investment bank", "banking services", "banking operations"
    ]
    
    # Retail keyword terms (deterministic override)
    RETAIL_KEYWORDS = [
        "retail", "retailer", "retail store", "retail chain",
        "store", "stores", "merchandising", "point of sale", "pos",
        "inventory management", "retail operations", "retail sales"
    ]
    
    def __init__(self):
        self.ollama_host = settings.ollama_host
        self.model = "llama3.1"
        self.MAX_ROLE_CHARS = 1800  # Character limit per role after isolation
    
    @dataclass
    class Role:
        """Represents a single job role with date information."""
        date_text: str
        start_year: Optional[int]
        end_year: Optional[int]
        is_current: bool
        text: str
        
        def get_score(self) -> int:
            """Get recency score: higher = more recent. Present/Current = highest score."""
            if self.is_current:
                return 999999  # Highest priority for current roles
            if self.end_year:
                return self.end_year * 100  # Year-based scoring
            if self.start_year:
                return self.start_year * 100  # Fallback to start year
            return 0  # Lowest priority if no dates
    
    def _extract_roles(self, resume_text: str) -> List['Role']:
        """
        Extract individual roles from resume text based on date range boundaries.
        Each distinct date range = one role.
        
        Args:
            resume_text: The full resume text
            
        Returns:
            List of Role objects, ordered by appearance in resume
        """
        if not resume_text:
            return []
        
        lines = resume_text.split('\n')
        roles = []
        current_role_lines = []
        current_date_text = ""
        current_start_year = None
        current_end_year = None
        current_is_current = False
        
        # Reuse existing date patterns from _extract_latest_experience
        # Enhanced to support em dash (—), en dash (–), "to", and various date formats
        date_patterns = [
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\.?\s+(\d{4})\b',
            r'\b(\d{1,2})/(\d{4})\b',  # MM/YYYY
            r'\b(\d{1,2})/(\d{4})\s*[-–—\s]+\s*(present|current|now|today|till\s+date|till\s+now|till-date|till-now|tilldate|tillnow|til\s+date|til\s+now|til-date|til-now|tildate|tilnow|still\s+date|still\s+now|still-date|still-now|stilldate|stillnow|still|still\s+working|still\s+employed|still\s+active|to\s+date|to\s+now|to-date|to-now|todate|tonow|until\s+present|until\s+now|until\s+date|until-present|until-now|until-date|untilpresent|untilnow|untildate|up\s+to\s+present|up\s+to\s+now|up\s+to\s+date|up-to-present|up-to-now|up-to-date|uptopresent|uptonow|uptodate|as\s+of\s+now|as\s+of\s+present|as\s+of\s+date|as\s+of\s+today|as-of-now|as-of-present|as-of-date|as-of-today|asofnow|asofpresent|asofdate|asoftoday|ongoing|on-going|on\s+going|working|continuing|continue|active|currently|currently\s+working|currently\s+employed|currently\s+active)\b',  # MM/YYYY – Present (supports -, –, —, "to")
            r'\b(\d{1,2})/(\d{4})\s+to\s+(\d{1,2})/(\d{4})\b',  # MM/YYYY to MM/YYYY
            r'\b(\d{4})[-\u2013\u2014\s]+\s*(\d{1,2})\b',  # YYYY-MM/YYYY–MM/YYYY—MM (hyphen, em dash, en dash)
            r'\b(\d{4})\s+to\s+(\d{4})\b',  # YYYY to YYYY
            r'\b(\d{4})\s*[-–—\s]+\s*(present|current|now|today|till\s+date|till\s+now|till-date|till-now|tilldate|tillnow|til\s+date|til\s+now|til-date|til-now|tildate|tilnow|still\s+date|still\s+now|still-date|still-now|stilldate|stillnow|still|still\s+working|still\s+employed|still\s+active|to\s+date|to\s+now|to-date|to-now|todate|tonow|until\s+present|until\s+now|until\s+date|until-present|until-now|until-date|untilpresent|untilnow|untildate|up\s+to\s+present|up\s+to\s+now|up\s+to\s+date|up-to-present|up-to-now|up-to-date|uptopresent|uptonow|uptodate|as\s+of\s+now|as\s+of\s+present|as\s+of\s+date|as\s+of\s+today|as-of-now|as-of-present|as-of-date|as-of-today|asofnow|asofpresent|asofdate|asoftoday|ongoing|on-going|on\s+going|working|continuing|continue|active|currently|currently\s+working|currently\s+employed|currently\s+active)\b',  # YYYY – Present (supports -, –, —, "to")
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\.?\s+(\d{4})\s+to\s+(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\.?\s+(\d{4})\b',  # Month YYYY to Month YYYY
            r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\'?(\d{2})\s*[-–—\s]+\s*(present|current|now|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|\d{2,4})\b',  # Jan'23 – Present or Jan'23 – Now (supports -, –, —, "to")
            r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\'?(\d{2})\s+to\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\'?(\d{2})\b',  # Jan'23 to Dec'23
            r'\b(19[5-9]\d|20[0-3]\d)\b',  # Year only
            # Comprehensive ongoing employment keywords
            r'\b(present|current|now|today|till\s+date|till\s+now|till-date|till-now|tilldate|tillnow|til\s+date|til\s+now|til-date|til-now|tildate|tilnow|still\s+date|still\s+now|still-date|still-now|stilldate|stillnow|still|still\s+working|still\s+employed|still\s+active|to\s+date|to\s+now|to-date|to-now|todate|tonow|until\s+present|until\s+now|until\s+date|until-present|until-now|until-date|untilpresent|untilnow|untildate|up\s+to\s+present|up\s+to\s+now|up\s+to\s+date|up-to-present|up-to-now|up-to-date|uptopresent|uptonow|uptodate|as\s+of\s+now|as\s+of\s+present|as\s+of\s+date|as\s+of\s+today|as-of-now|as-of-present|as-of-date|as-of-today|asofnow|asofpresent|asofdate|asoftoday|ongoing|on-going|on\s+going|working|continuing|continue|active|currently|currently\s+working|currently\s+employed|currently\s+active)\b',
        ]
        
        # Present/current keywords for is_current detection
        present_keywords = [
            "present", "current", "now", "today",
            "till date", "till now", "till-date", "till-now", "tilldate", "tillnow",
            "til date", "til now", "til-date", "til-now", "tildate", "tilnow",
            "still date", "still now", "still-date", "still-now", "stilldate", "stillnow",
            "still", "still working", "still employed", "still active",
            "to date", "to now", "to-date", "to-now", "todate", "tonow",
            "until present", "until now", "until date", "until-present", "until-now", "until-date",
            "untilpresent", "untilnow", "untildate",
            "up to present", "up to now", "up to date", "up-to-present", "up-to-now", "up-to-date",
            "uptopresent", "uptonow", "uptodate",
            "as of now", "as of present", "as of date", "as of today",
            "as-of-now", "as-of-present", "as-of-date", "as-of-today",
            "asofnow", "asofpresent", "asofdate", "asoftoday",
            "ongoing", "on-going", "on going",
            "working", "working till date", "working till now",
            "continuing", "continue",
            "active", "currently", "currently working", "currently employed", "currently active"
        ]
        
        def is_likely_header_or_contact_info(line: str) -> bool:
            """
            Check if a line is likely header/contact info (address, phone, email) rather than a date range.
            Returns True if line should be excluded from date detection.
            """
            line_lower = line.lower().strip()
            
            # Too short - likely header
            if len(line_lower) < 10:
                return True
            
            # Contains email pattern
            if '@' in line_lower or re.search(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', line_lower):
                return True
            
            # Contains phone number patterns
            phone_patterns = [
                r'\(\d{3}\)',  # (xxx)
                r'\d{3}[-.]\d{3}[-.]\d{4}',  # xxx-xxx-xxxx
                r'\+\d{1,3}[\s-]?\d{1,4}[\s-]?\d{1,4}[\s-]?\d{1,9}',  # +x xxx xxx xxxx
                r'phone|tel|mobile|cell',
            ]
            if any(re.search(pattern, line_lower) for pattern in phone_patterns):
                return True
            
            # Contains address indicators
            address_indicators = [
                r'\b(rd|road|st|street|ave|avenue|blvd|boulevard|dr|drive|ln|lane|ct|court|pl|place|way|cir|circle)\b',
                r'\b(apt|apartment|suite|unit|#)\s*\d+',
                r'\b(p\.o\.|po\s+box|post\s+office)',
                r'\b(zip|postal\s+code)',
                r'^\d+\s+[a-z]+\s+(rd|st|ave|blvd|dr|ln|ct|pl|way|cir)',  # "1509 Cedrus Rd"
            ]
            if any(re.search(pattern, line_lower) for pattern in address_indicators):
                return True
            
            # Contains URL patterns
            if re.search(r'https?://|www\.|linkedin\.com|github\.com', line_lower):
                return True
            
            # Contains only numbers and common non-date words (likely address/ID)
            if re.match(r'^[\d\s\-\(\)]+$', line_lower) and len(line_lower) < 20:
                return True
            
            return False
        
        def extract_years_from_line(line: str) -> tuple[Optional[int], Optional[int], bool]:
            """Extract start_year, end_year, and is_current from a line."""
            line_lower = line.lower()
            is_current = any(keyword in line_lower for keyword in present_keywords)
            
            # Extract years
            year_pattern = r'\b(20[0-3]\d|19[5-9]\d)\b'
            years = [int(y) for y in re.findall(year_pattern, line)]
            
            start_year = None
            end_year = None
            
            if years:
                if len(years) >= 2:
                    # Assume first year is start, last is end
                    start_year = min(years)
                    end_year = max(years)
                elif len(years) == 1:
                    # Single year - could be start or end
                    if is_current:
                        start_year = years[0]
                        end_year = None
                    else:
                        # Try to infer from context
                        end_year = years[0]
            
            return start_year, end_year, is_current
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                if current_role_lines:
                    current_role_lines.append(line)  # Keep blank lines within role
                continue
            
            line_lower = line_stripped.lower()
            
            # Skip header/contact info lines (addresses, phone numbers, emails)
            if is_likely_header_or_contact_info(line_stripped):
                continue
            
            # Check if this line contains a date range (role boundary)
            has_date_range = any(re.search(pattern, line_lower, re.IGNORECASE) for pattern in date_patterns)
            
            if has_date_range:
                # Save previous role if exists
                if current_role_lines:
                    role_text = '\n'.join(current_role_lines).strip()
                    if role_text:
                        roles.append(self.Role(
                            date_text=current_date_text,
                            start_year=current_start_year,
                            end_year=current_end_year,
                            is_current=current_is_current,
                            text=role_text
                        ))
                
                # Start new role
                current_date_text = line_stripped
                current_start_year, current_end_year, current_is_current = extract_years_from_line(line_stripped)
                current_role_lines = [line_stripped]
            else:
                # Add line to current role (if we're in a role context)
                if current_role_lines:
                    # We're in a role context (have seen a date range)
                    current_role_lines.append(line_stripped)
                # If no role context yet, skip (wait for first date range)
        
        # Save final role if exists
        if current_role_lines:
            role_text = '\n'.join(current_role_lines).strip()
            if role_text:
                roles.append(self.Role(
                    date_text=current_date_text,
                    start_year=current_start_year,
                    end_year=current_end_year,
                    is_current=current_is_current,
                    text=role_text
                ))
        
        logger.debug(
            f"Extracted {len(roles)} roles from resume",
            extra={"role_count": len(roles)}
        )
        
        return roles
    
    def _select_latest_role(self, roles: List['Role']) -> Optional['Role']:
        """
        Select the single most recent role from a list of roles.
        
        Scoring logic:
        1. is_current == True → highest priority
        2. Highest end_year
        3. If tie → first occurrence
        
        Args:
            roles: List of Role objects
            
        Returns:
            The most recent Role or None if list is empty
        """
        if not roles:
            return None
        
        if len(roles) == 1:
            return roles[0]
        
        # Score each role
        roles_with_scores = [(role, role.get_score()) for role in roles]
        
        # Sort by score (descending) - highest score = most recent
        roles_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get role with highest score
        latest_role = roles_with_scores[0][0]
        
        logger.info(
            f"Selected latest role (score: {roles_with_scores[0][1]}, is_current: {latest_role.is_current}, "
            f"end_year: {latest_role.end_year})",
            extra={
                "role_score": roles_with_scores[0][1],
                "is_current": latest_role.is_current,
                "end_year": latest_role.end_year,
                "total_roles": len(roles)
            }
        )
        
        return latest_role
    
    def _has_business_context(self, role_text: str) -> bool:
        """
        Check if role text has clear business/company context.
        This ensures we only call LLM when role has meaningful business indicators.
        
        Args:
            role_text: The role text to validate
            
        Returns:
            True if role has business context, False otherwise
        """
        if not role_text or len(role_text.strip()) < 20:
            return False
        
        text_lower = role_text.lower()
        
        # Company/business indicators (high confidence)
        company_indicators = [
            r'\b(company|corporation|corp|inc|ltd|llc|pvt|private|limited|enterprises|solutions|services|systems|technologies|tech|group|holdings)\b',
            r'\b(worked at|employed at|worked for|employed by|at [A-Z][a-z]+)\b',
            r'\b(client|customer|vendor|partner)\b',
        ]
        
        # Job title + company pattern
        job_title_patterns = [
            r'\b(engineer|developer|manager|director|analyst|consultant|specialist|architect|lead|senior|junior)\s+.*\b(at|for|with)\b',
            r'\b(software|senior|junior|principal|staff)\s+(engineer|developer|manager|analyst)\b',
        ]
        
        # Check for company indicators
        has_company = any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in company_indicators)
        
        # Check for job title + company pattern
        has_job_company_pattern = any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in job_title_patterns)
        
        # Check for explicit company names (capitalized words that look like company names)
        # Pattern: Job Title - Company Name or Company Name format
        company_name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(Pvt|Ltd|Inc|Corp|LLC|Limited|Corporation|Solutions|Services|Systems|Technologies|Group)\b'
        has_company_name = bool(re.search(company_name_pattern, role_text))
        
        # Also check for standalone capitalized company-like phrases
        # Pattern: "ABC Company" or "XYZ Solutions"
        standalone_company = re.search(r'\b([A-Z]{2,}(?:\s+[A-Z][a-z]+)?)\s+(Company|Corp|Inc|Ltd|Solutions|Services|Systems|Technologies)\b', role_text)
        has_standalone_company = bool(standalone_company)
        
        return has_company or has_job_company_pattern or has_company_name or has_standalone_company
    
    def _validate_role_isolation(self, role: 'Role', all_roles: List['Role'], resume_text: str) -> tuple[bool, str]:
        """
        Strictly validate that a role is properly isolated before sending to LLM.
        This prevents sending mixed/multiple roles to LLM.
        
        Args:
            role: The role to validate
            all_roles: All extracted roles (for context)
            resume_text: Full resume text (for validation)
            
        Returns:
            Tuple of (is_valid, reason)
        """
        if not role or not role.text:
            return False, "Role is empty"
        
        role_text = role.text.strip()
        
        # Check 1: Role text must be reasonable length
        if len(role_text) < 30:
            return False, f"Role text too short ({len(role_text)} chars)"
        
        if len(role_text) > self.MAX_ROLE_CHARS * 2:  # Allow some buffer
            return False, f"Role text suspiciously long ({len(role_text)} chars) - may contain multiple roles"
        
        # Check 2: Must have business context
        if not self._has_business_context(role_text):
            return False, "Role lacks clear business/company context"
        
        # Check 3: Check for multiple date ranges in role text (indicates multiple roles mixed)
        date_pattern = r'\b(19[5-9]\d|20[0-3]\d)\b'
        years_in_role = re.findall(date_pattern, role_text)
        if len(years_in_role) > 4:  # More than 2 date ranges (4 years) suggests multiple roles
            return False, f"Role text contains too many date references ({len(years_in_role)} years) - likely multiple roles"
        
        # Check 4: Check for multiple company indicators (suggests multiple roles)
        company_pattern = r'\b(company|corporation|corp|inc|ltd|llc|pvt|limited|solutions|services|systems|technologies)\b'
        company_matches = re.findall(company_pattern, role_text, re.IGNORECASE)
        if len(company_matches) > 3:  # Multiple company mentions suggest multiple roles
            return False, f"Role text contains too many company references ({len(company_matches)}) - likely multiple roles"
        
        # Check 5: Check for role separation keywords that suggest multiple roles
        separation_keywords = [
            r'\b(previous|prior|earlier|before|also worked|also|additionally)\s+(at|for|as|in)\b',
            r'\b(prior to|before joining|earlier role|previous position)\b',
        ]
        for pattern in separation_keywords:
            if re.search(pattern, role_text, re.IGNORECASE):
                return False, "Role text contains separation keywords - likely multiple roles mixed"
        
        # Check 6: If we have multiple roles, ensure selected role is clearly the latest
        if len(all_roles) > 1:
            # Verify this role has highest score
            role_scores = [(r, r.get_score()) for r in all_roles]
            role_scores.sort(key=lambda x: x[1], reverse=True)
            if role_scores[0][0] != role:
                return False, "Selected role is not the most recent according to scoring"
        
        # Check 7: Ensure role text doesn't contain education section markers
        education_markers = [
            r'\b(education|academic|qualification|degree|university|college|school)\s*:',
            r'\b(bachelor|master|phd|doctorate|graduated)\b',
        ]
        for pattern in education_markers:
            if re.search(pattern, role_text, re.IGNORECASE):
                # Check if it's in work context (e.g., "Education sector") vs academic section
                if not re.search(r'\b(worked|employed|role|position|job|experience|sector|industry)\b', role_text, re.IGNORECASE):
                    return False, "Role text contains education section markers without work context"
        
        # Check 8: Ensure role has a date (for proper isolation confidence)
        # If role has no date at all, it's harder to verify it's the latest
        if not role.date_text or not role.date_text.strip():
            # Allow if it's the only role, but be more strict if multiple roles exist
            if len(all_roles) > 1:
                return False, "Role has no date and multiple roles exist - cannot verify it's the latest"
            # If single role but no date, still require strong business context
            if not self._has_business_context(role_text):
                return False, "Single role with no date and weak business context - low confidence"
        
        return True, "Role is properly isolated"
    
    def _extract_employer_name(self, role_text: str) -> Optional[str]:
        """
        Extract employer/company name from role text.
        Returns normalized company name (lowercase, cleaned) or None.
        """
        if not role_text:
            return None
        
        text_lower = role_text.lower()
        
        # Pattern 1: "Job Title at Company Name" or "Job Title - Company Name"
        patterns = [
            r'\bat\s+([A-Z][a-zA-Z\s&]+(?:Inc|Ltd|LLC|Corp|Corporation|Limited|Pvt|Private|Solutions|Services|Systems|Technologies|Group|Holdings)?)',
            r'[-–—]\s*([A-Z][a-zA-Z\s&]+(?:Inc|Ltd|LLC|Corp|Corporation|Limited|Pvt|Private|Solutions|Services|Systems|Technologies|Group|Holdings)?)',
            r'\b([A-Z][a-zA-Z\s&]+(?:Inc|Ltd|LLC|Corp|Corporation|Limited|Pvt|Private))\b',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, role_text)
            if match:
                company = match.group(1).strip()
                # Clean up common suffixes
                company = re.sub(r'\s+(Inc|Ltd|LLC|Corp|Corporation|Limited|Pvt|Private|Solutions|Services|Systems|Technologies|Group|Holdings)$', '', company, flags=re.IGNORECASE)
                return company.lower().strip()
        
        # Pattern 2: Look for capitalized company-like phrases
        # "ABC Company" or "XYZ Solutions"
        company_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(Company|Corp|Inc|Ltd|Solutions|Services|Systems|Technologies|Group)\b'
        match = re.search(company_pattern, role_text)
        if match:
            company = match.group(1).strip()
            return company.lower().strip()
        
        return None
    
    def _check_employer_domain_map(self, role_text: str) -> Optional[str]:
        """
        STEP 1: Check employer domain mapping (deterministic).
        Returns domain if employer matches known companies, None otherwise.
        """
        if not role_text:
            return None
        
        employer_name = self._extract_employer_name(role_text)
        if not employer_name:
            return None
        
        # Direct match
        if employer_name in self.EMPLOYER_DOMAIN_MAP:
            domain = self.EMPLOYER_DOMAIN_MAP[employer_name]
            logger.info(
                f"✅ Deterministic: Employer domain map match - {employer_name} → {domain}",
                extra={"employer": employer_name, "domain": domain, "method": "employer_map"}
            )
            return domain
        
        # Partial match (employer name contains key)
        text_lower = role_text.lower()
        for employer_key, domain in self.EMPLOYER_DOMAIN_MAP.items():
            if employer_key in text_lower:
                logger.info(
                    f"✅ Deterministic: Employer domain map partial match - {employer_key} → {domain}",
                    extra={"employer_key": employer_key, "domain": domain, "method": "employer_map_partial"}
                )
                return domain
        
        return None
    
    def _check_healthcare_keywords(self, role_text: str) -> Optional[str]:
        """
        STEP 2: Check healthcare keyword override (deterministic).
        Returns "Healthcare" if 2+ healthcare terms found, None otherwise.
        """
        if not role_text:
            return None
        
        text_lower = role_text.lower()
        matches = sum(1 for keyword in self.HEALTHCARE_KEYWORDS if keyword in text_lower)
        
        if matches >= 2:
            logger.info(
                f"✅ Deterministic: Healthcare keyword override - {matches} healthcare terms found",
                extra={"matches": matches, "domain": "Healthcare", "method": "healthcare_keywords"}
            )
            return "Healthcare"
        
        return None
    
    def _check_banking_keywords(self, role_text: str) -> Optional[str]:
        """
        STEP 2: Check banking keyword override (deterministic).
        Returns "Banking" if 2+ banking terms found, None otherwise.
        """
        if not role_text:
            return None
        
        text_lower = role_text.lower()
        matches = sum(1 for keyword in self.BANKING_KEYWORDS if keyword in text_lower)
        
        if matches >= 2:
            logger.info(
                f"✅ Deterministic: Banking keyword override - {matches} banking terms found",
                extra={"matches": matches, "domain": "Banking", "method": "banking_keywords"}
            )
            return "Banking"
        
        return None
    
    def _check_retail_keywords(self, role_text: str) -> Optional[str]:
        """
        STEP 2: Check retail keyword override (deterministic).
        Returns "Retail" if 2+ retail terms found, None otherwise.
        """
        if not role_text:
            return None
        
        text_lower = role_text.lower()
        matches = sum(1 for keyword in self.RETAIL_KEYWORDS if keyword in text_lower)
        
        if matches >= 2:
            logger.info(
                f"✅ Deterministic: Retail keyword override - {matches} retail terms found",
                extra={"matches": matches, "domain": "Retail", "method": "retail_keywords"}
            )
            return "Retail"
        
        return None
    
    def _check_platform_domain_guard(self, role_text: str) -> Optional[str]:
        """
        STEP 3: Platform domain guard (deterministic).
        Returns platform domain ONLY if job title explicitly mentions the platform.
        Prevents hallucination from skills/technologies.
        """
        if not role_text:
            return None
        
        text_lower = role_text.lower()
        
        # AWS guard - STRICT
        aws_patterns = [
            r'\baws\s+(solutions\s+)?architect',
            r'\baws\s+(cloud\s+)?engineer',
            r'\baws\s+consultant',
            r'\baws\s+developer',
            r'\bamazon\s+web\s+services\s+(solutions\s+)?architect',
        ]
        for pattern in aws_patterns:
            if re.search(pattern, text_lower):
                logger.info(
                    f"✅ Deterministic: Platform domain guard - explicit AWS role detected",
                    extra={"domain": "AWS", "method": "platform_guard_aws"}
                )
                return "AWS"
        
        # Salesforce guard - STRICT
        salesforce_patterns = [
            r'\bsalesforce\s+(admin|administrator)',
            r'\bsalesforce\s+developer',
            r'\bsalesforce\s+consultant',
            r'\bsalesforce\s+architect',
        ]
        for pattern in salesforce_patterns:
            if re.search(pattern, text_lower):
                logger.info(
                    f"✅ Deterministic: Platform domain guard - explicit Salesforce role detected",
                    extra={"domain": "Salesforce", "method": "platform_guard_salesforce"}
                )
                return "Salesforce"
        
        # SAP guard - STRICT
        sap_patterns = [
            r'\bsap\s+consultant',
            r'\bsap\s+developer',
            r'\bsap\s+architect',
            r'\bsap\s+implementation',
        ]
        for pattern in sap_patterns:
            if re.search(pattern, text_lower):
                logger.info(
                    f"✅ Deterministic: Platform domain guard - explicit SAP role detected",
                    extra={"domain": "SAP", "method": "platform_guard_sap"}
                )
                return "SAP"
        
        # Oracle guard - STRICT
        oracle_patterns = [
            r'\boracle\s+consultant',
            r'\boracle\s+developer',
            r'\boracle\s+architect',
            r'\boracle\s+erp',
        ]
        for pattern in oracle_patterns:
            if re.search(pattern, text_lower):
                logger.info(
                    f"✅ Deterministic: Platform domain guard - explicit Oracle role detected",
                    extra={"domain": "Oracle", "method": "platform_guard_oracle"}
                )
                return "Oracle"
        
        # Microsoft guard - STRICT
        microsoft_patterns = [
            r'\bmicrosoft\s+consultant',
            r'\bmicrosoft\s+architect',
            r'\bmicrosoft\s+stack',
            r'\bmicrosoft\s+technologies',
        ]
        for pattern in microsoft_patterns:
            if re.search(pattern, text_lower):
                logger.info(
                    f"✅ Deterministic: Platform domain guard - explicit Microsoft role detected",
                    extra={"domain": "Microsoft", "method": "platform_guard_microsoft"}
                )
                return "Microsoft"
        
        # ServiceNow guard
        if re.search(r'\bservicenow\s+(admin|developer|consultant)', text_lower):
            logger.info(
                f"✅ Deterministic: Platform domain guard - explicit ServiceNow role detected",
                extra={"domain": "ServiceNow", "method": "platform_guard_servicenow"}
            )
            return "ServiceNow"
        
        # Workday guard
        if re.search(r'\bworkday\s+(consultant|developer|admin)', text_lower):
            logger.info(
                f"✅ Deterministic: Platform domain guard - explicit Workday role detected",
                extra={"domain": "Workday", "method": "platform_guard_workday"}
            )
            return "Workday"
        
        # Azure guard
        azure_patterns = [
            r'\bazure\s+(architect|engineer|consultant)',
            r'\bmicrosoft\s+azure\s+(architect|engineer)',
        ]
        for pattern in azure_patterns:
            if re.search(pattern, text_lower):
                logger.info(
                    f"✅ Deterministic: Platform domain guard - explicit Azure role detected",
                    extra={"domain": "Azure", "method": "platform_guard_azure"}
                )
                return "Azure"
        
        # Google Cloud guard
        gcp_patterns = [
            r'\bgoogle\s+cloud\s+(architect|engineer|consultant)',
            r'\bgcp\s+(architect|engineer)',
        ]
        for pattern in gcp_patterns:
            if re.search(pattern, text_lower):
                logger.info(
                    f"✅ Deterministic: Platform domain guard - explicit Google Cloud role detected",
                    extra={"domain": "Google Cloud", "method": "platform_guard_gcp"}
                )
                return "Google Cloud"
        
        return None
    
    def _extract_latest_role(self, resume_text: str) -> Optional['Role']:
        """
        Extract the single most recent role from resume text.
        This is the primary method for role-based domain extraction.
        
        Args:
            resume_text: The full resume text
            
        Returns:
            The most recent Role object, or None if no roles found
        """
        if not resume_text:
            return None
        
        # Extract all roles
        roles = self._extract_roles(resume_text)
        
        if not roles:
            # Fallback: try experience-based extraction
            logger.debug("No roles found with date ranges, falling back to experience-based extraction")
            return None
        
        # Select latest role
        latest_role = self._select_latest_role(roles)
        
        if latest_role:
            # Apply character limit after isolation
            if len(latest_role.text) > self.MAX_ROLE_CHARS:
                logger.debug(
                    f"Role text truncated from {len(latest_role.text)} to {self.MAX_ROLE_CHARS} characters",
                    extra={"original_length": len(latest_role.text)}
                )
                latest_role.text = latest_role.text[:self.MAX_ROLE_CHARS]
        
        return latest_role
    
    def _extract_latest_experience(self, resume_text: str) -> str:
        """
        Extract and prioritize the most recent work experience section.
        This ensures domain extraction focuses on the latest domain.
        
        Args:
            resume_text: The full resume text
            
        Returns:
            Latest experience section text (up to 3000 chars), or original text if not found
        """
        if not resume_text:
            return ""
        
        lines = resume_text.split('\n')
        experience_blocks = []
        current_block = []
        in_experience_section = False
        
        # Section headers that indicate experience/work
        experience_keywords = [
            r'^#?\s*(experience|work\s+experience|employment|professional\s+experience)',
            r'^#?\s*(career|career\s+history|work\s+history|employment\s+history)',
            r'^#?\s*(work|employment|professional)',
        ]
        
        # Date patterns to identify experience entries
        # Enhanced to support em dash (—), en dash (–), "to", and various date formats
        date_patterns = [
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\.?\s+(\d{4})\b',
            r'\b(\d{1,2})/(\d{4})\b',  # MM/YYYY
            r'\b(\d{1,2})/(\d{4})\s*[-–—\s]+\s*(present|current|now|today|till\s+date|till\s+now|till-date|till-now|tilldate|tillnow|til\s+date|til\s+now|til-date|til-now|tildate|tilnow|still\s+date|still\s+now|still-date|still-now|stilldate|stillnow|still|still\s+working|still\s+employed|still\s+active|to\s+date|to\s+now|to-date|to-now|todate|tonow|until\s+present|until\s+now|until\s+date|until-present|until-now|until-date|untilpresent|untilnow|untildate|up\s+to\s+present|up\s+to\s+now|up\s+to\s+date|up-to-present|up-to-now|up-to-date|uptopresent|uptonow|uptodate|as\s+of\s+now|as\s+of\s+present|as\s+of\s+date|as\s+of\s+today|as-of-now|as-of-present|as-of-date|as-of-today|asofnow|asofpresent|asofdate|asoftoday|ongoing|on-going|on\s+going|working|continuing|continue|active|currently|currently\s+working|currently\s+employed|currently\s+active)\b',  # MM/YYYY – Present (supports -, –, —, "to")
            r'\b(\d{1,2})/(\d{4})\s+to\s+(\d{1,2})/(\d{4})\b',  # MM/YYYY to MM/YYYY
            r'\b(\d{4})[-\u2013\u2014\s]+\s*(\d{1,2})\b',  # YYYY-MM/YYYY–MM/YYYY—MM (hyphen, em dash, en dash)
            r'\b(\d{4})\s+to\s+(\d{4})\b',  # YYYY to YYYY
            r'\b(\d{4})\s*[-–—\s]+\s*(present|current|now|today|till\s+date|till\s+now|till-date|till-now|tilldate|tillnow|til\s+date|til\s+now|til-date|til-now|tildate|tilnow|still\s+date|still\s+now|still-date|still-now|stilldate|stillnow|still|still\s+working|still\s+employed|still\s+active|to\s+date|to\s+now|to-date|to-now|todate|tonow|until\s+present|until\s+now|until\s+date|until-present|until-now|until-date|untilpresent|untilnow|untildate|up\s+to\s+present|up\s+to\s+now|up\s+to\s+date|up-to-present|up-to-now|up-to-date|uptopresent|uptonow|uptodate|as\s+of\s+now|as\s+of\s+present|as\s+of\s+date|as\s+of\s+today|as-of-now|as-of-present|as-of-date|as-of-today|asofnow|asofpresent|asofdate|asoftoday|ongoing|on-going|on\s+going|working|continuing|continue|active|currently|currently\s+working|currently\s+employed|currently\s+active)\b',  # YYYY – Present (supports -, –, —, "to")
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\.?\s+(\d{4})\s+to\s+(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\.?\s+(\d{4})\b',  # Month YYYY to Month YYYY
            r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\'?(\d{2})\s*[-–—\s]+\s*(present|current|now|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|\d{2,4})\b',  # Jan'23 – Present or Jan'23 – Now (supports -, –, —, "to")
            r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\'?(\d{2})\s+to\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\'?(\d{2})\b',  # Jan'23 to Dec'23
            r'\b(19[5-9]\d|20[0-3]\d)\b',  # Year only
            # Comprehensive ongoing employment keywords
            r'\b(present|current|now|today|till\s+date|till\s+now|till-date|till-now|tilldate|tillnow|til\s+date|til\s+now|til-date|til-now|tildate|tilnow|still\s+date|still\s+now|still-date|still-now|stilldate|stillnow|still|still\s+working|still\s+employed|still\s+active|to\s+date|to\s+now|to-date|to-now|todate|tonow|until\s+present|until\s+now|until\s+date|until-present|until-now|until-date|untilpresent|untilnow|untildate|up\s+to\s+present|up\s+to\s+now|up\s+to\s+date|up-to-present|up-to-now|up-to-date|uptopresent|uptonow|uptodate|as\s+of\s+now|as\s+of\s+present|as\s+of\s+date|as\s+of\s+today|as-of-now|as-of-present|as-of-date|as-of-today|asofnow|asofpresent|asofdate|asoftoday|ongoing|on-going|on\s+going|working|continuing|continue|active|currently|currently\s+working|currently\s+employed|currently\s+active)\b',
        ]
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            line_lower = line_stripped.lower()
            
            # Check if this line starts an experience section
            is_experience_header = False
            for pattern in experience_keywords:
                if re.match(pattern, line_lower, re.IGNORECASE):
                    is_experience_header = True
                    in_experience_section = True
                    # Save previous block if exists
                    if current_block:
                        experience_blocks.append('\n'.join(current_block))
                    current_block = [line]
                    break
            
            # If in experience section, collect lines
            if in_experience_section:
                current_block.append(line)
                
                # Check if line contains date (likely an experience entry)
                has_date = any(re.search(pattern, line_lower, re.IGNORECASE) for pattern in date_patterns)
                
                # If we hit a new major section (not experience), save current block
                if i < len(lines) - 1:
                    next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
                    # Check if next line is a new section header
                    is_new_section = any(
                        re.match(r'^#?\s*(education|academic|qualification|certification|skill|project)', 
                                next_line.lower(), re.IGNORECASE)
                    )
                    if is_new_section and current_block:
                        experience_blocks.append('\n'.join(current_block))
                        current_block = []
                        in_experience_section = False
            else:
                # Look for experience entries even without explicit section header
                has_date = any(re.search(pattern, line_lower, re.IGNORECASE) for pattern in date_patterns)
                has_job_indicators = any(keyword in line_lower for keyword in [
                    'company', 'corporation', 'inc', 'ltd', 'worked at', 'employed at',
                    'senior', 'junior', 'manager', 'developer', 'analyst', 'engineer'
                ])
                
                if has_date and has_job_indicators:
                    if not current_block:
                        current_block = []
                    current_block.append(line)
        
        # Add final block if exists
        if current_block:
            experience_blocks.append('\n'.join(current_block))
        
        # Rank experience blocks by date (most recent first)
        # Extract date score for each block
        from datetime import datetime
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        def get_date_score(block_text: str) -> int:
            """Get date score: higher = more recent. Present/Current = highest score."""
            text_lower = block_text.lower()
            
            # Check for present/current keywords (highest priority)
            # Comprehensive list matching experience extractor for consistency
            present_keywords = [
                # Standard
                "present", "current", "now", "today",
                # Till variations
                "till date", "till now", "till-date", "till-now", "tilldate", "tillnow",
                "til date", "til now", "til-date", "til-now", "tildate", "tilnow",
                # Still variations
                "still date", "still now", "still-date", "still-now", "stilldate", "stillnow",
                "still", "still working", "still employed", "still active",
                # To variations
                "to date", "to now", "to-date", "to-now", "todate", "tonow",
                # Until variations
                "until present", "until now", "until date", "until-present", "until-now", "until-date",
                "untilpresent", "untilnow", "untildate",
                # Up to variations
                "up to present", "up to now", "up to date", "up-to-present", "up-to-now", "up-to-date",
                "uptopresent", "uptonow", "uptodate",
                # As of variations
                "as of now", "as of present", "as of date", "as of today",
                "as-of-now", "as-of-present", "as-of-date", "as-of-today",
                "asofnow", "asofpresent", "asofdate", "asoftoday",
                # Ongoing variations
                "ongoing", "on-going", "on going",
                # Working variations
                "working", "working till date", "working till now",
                # Continuing variations
                "continuing", "continue",
                # Active variations
                "active", "currently", "currently working", "currently employed", "currently active"
            ]
            if any(keyword in text_lower for keyword in present_keywords):
                return 999999  # Highest score for present
            
            # Extract year from block
            year_pattern = r'\b(20[0-3]\d|19[5-9]\d)\b'
            years = re.findall(year_pattern, block_text)
            if years:
                # Use the highest year found (most recent)
                max_year = max(int(y) for y in years)
                # Score = year * 100 + month (if found)
                month_score = 0
                month_pattern = r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\.?\s+(\d{4})\b'
                month_matches = re.findall(month_pattern, text_lower, re.IGNORECASE)
                if month_matches:
                    month_names = {
                        'january': 1, 'jan': 1, 'february': 2, 'feb': 2,
                        'march': 3, 'mar': 3, 'april': 4, 'apr': 4,
                        'may': 5, 'june': 6, 'jun': 6,
                        'july': 7, 'jul': 7, 'august': 8, 'aug': 8,
                        'september': 9, 'sep': 9, 'sept': 9,
                        'october': 10, 'oct': 10, 'november': 11, 'nov': 11,
                        'december': 12, 'dec': 12
                    }
                    for month_str, year_str in month_matches:
                        if month_str.lower() in month_names and int(year_str) == max_year:
                            month_score = month_names[month_str.lower()]
                            break
                return max_year * 100 + month_score
            
            # If no date found, return 0 (lowest priority)
            return 0
        
        # Sort blocks by date score (descending - most recent first)
        if experience_blocks:
            experience_blocks_with_scores = [
                (block, get_date_score(block)) for block in experience_blocks
            ]
            experience_blocks_with_scores.sort(key=lambda x: x[1], reverse=True)
            latest_experience = experience_blocks_with_scores[0][0]
            
            # Note: No truncation here - truncation only belongs in role-based path
            # Note: No paragraph splitting - paragraph order is not a reliable recency signal
            logger.info(
                f"✅ Extracted latest experience section by date ({len(latest_experience)} chars)",
                extra={"experience_length": len(latest_experience), "date_score": experience_blocks_with_scores[0][1]}
            )
            return latest_experience
        
        # Fallback: return first 3000 chars of resume if no experience section found
        logger.debug("No explicit experience section found, using first 3000 chars of resume")
        return resume_text[:3000]
    
    def _filter_education_sections(self, resume_text: str) -> str:
        """
        Filter out education sections from resume text to prevent false domain detection.
        This ensures Education domain is only detected from work experience, not academic qualifications.
        
        Args:
            resume_text: The full resume text
            
        Returns:
            Text with education sections removed, containing only work-related content
        """
        if not resume_text:
            return ""
        
        lines = resume_text.split('\n')
        filtered_lines = []
        skip_section = False
        
        # Section headers that indicate education/academic content
        education_keywords = [
            r'^#?\s*(education|academic|qualification|qualifications)',
            r'^#?\s*(degree|degrees|bachelor|master|phd|doctorate)',
            r'^#?\s*(university|college|school)\s*$',  # Only if standalone header
        ]
        
        # Work section indicators - when we see these, stop skipping
        work_keywords = [
            r'^#?\s*(experience|work\s+experience|employment|professional\s+experience)',
            r'^#?\s*(career|career\s+history|work\s+history)',
            r'^#?\s*(project|projects)',  # Projects can indicate work
        ]
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            line_lower = line_stripped.lower()
            
            # Check if this line starts an education section
            is_education_header = False
            for pattern in education_keywords:
                if re.match(pattern, line_lower, re.IGNORECASE):
                    is_education_header = True
                    skip_section = True
                    logger.debug(f"Filtering education section: {line_stripped[:50]}")
                    break
            
            # Check if this line starts a work section (stops skipping)
            if not is_education_header:
                for pattern in work_keywords:
                    if re.match(pattern, line_lower, re.IGNORECASE):
                        skip_section = False
                        break
            
            # Skip lines in education sections
            if skip_section:
                # Check if we've reached a new major section
                if i < len(lines) - 1:
                    next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
                    # If next line looks like a work section header, stop skipping
                    if next_line:
                        for pattern in work_keywords:
                            if re.match(pattern, next_line.lower(), re.IGNORECASE):
                                skip_section = False
                                break
                continue
            
            filtered_lines.append(line)
        
        filtered_text = '\n'.join(filtered_lines)
        
        logger.debug(
            f"Education section filtering: {len(resume_text)} -> {len(filtered_text)} characters",
            extra={
                "original_length": len(resume_text),
                "filtered_length": len(filtered_text),
                "removed_chars": len(resume_text) - len(filtered_text)
            }
        )
        return filtered_text
    
    def _is_education_keyword_in_work_context(self, text: str, keyword: str) -> bool:
        """
        Check if an education-related keyword appears in work context (not education section).
        
        Args:
            text: The resume text to check
            keyword: The keyword to search for
            
        Returns:
            True if keyword appears in work context, False if in education section
        """
        if not text or not keyword:
            return False
        
        # Find all occurrences of the keyword
        text_lower = text.lower()
        keyword_lower = keyword.lower()
        
        # Get context around each occurrence (200 chars before and after)
        import re
        for match in re.finditer(re.escape(keyword_lower), text_lower):
            start = max(0, match.start() - 200)
            end = min(len(text), match.end() + 200)
            context = text_lower[start:end]
            
            # Check if context contains work indicators
            work_indicators = [
                'company', 'worked', 'employed', 'role', 'position', 'job', 'experience',
                'client', 'project', 'developed', 'managed', 'implemented', 'designed',
                'work experience', 'professional experience', 'employment'
            ]
            
            # Check if context contains education section indicators
            education_indicators = [
                'education:', 'academic:', 'qualification:', 'degree', 'bachelor', 'master',
                'phd', 'graduated', 'university', 'college', 'school'
            ]
            
            has_work_context = any(indicator in context for indicator in work_indicators)
            has_education_section = any(
                indicator in context and 
                ('education' in context[:context.find(indicator)] or 
                 'academic' in context[:context.find(indicator)] or
                 'qualification' in context[:context.find(indicator)])
                for indicator in education_indicators
            )
            
            # If it has work context and not clearly in education section, it's valid
            if has_work_context and not has_education_section:
                return True
        
        return False
    
    def _resolve_domain_precedence(self, domains: List[str]) -> str:
        """
        Resolve domain conflicts using precedence hierarchy.
        Returns the domain with highest priority.
        
        Args:
            domains: List of candidate domain strings
            
        Returns:
            Domain with highest priority
        """
        if not domains:
            return None
        
        if len(domains) == 1:
            return domains[0]
        
        # Find domain with highest priority (lowest index in DOMAIN_PRIORITY)
        best_domain = None
        best_priority = float('inf')
        
        for domain in domains:
            try:
                priority = self.DOMAIN_PRIORITY.index(domain)
                if priority < best_priority:
                    best_priority = priority
                    best_domain = domain
            except ValueError:
                # Domain not in priority list - assign lowest priority
                if best_priority == float('inf'):
                    best_domain = domain
        
        return best_domain if best_domain else domains[0]
    
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
    
    def _detect_domain_from_keywords(self, resume_text: str, filename: str = "resume") -> Optional[str]:
        """
        Keyword-based domain detection method.
        
        NOTE: This method is no longer used as a fallback when LLM returns null.
        LLM null is now respected as the authoritative decision (conservative behavior).
        
        This method may be used for validation or other purposes in the future.
        Uses strict scoring to ensure only business/industry domain terms are detected.
        Filters out education sections to prevent false Education domain detection.
        
        Args:
            resume_text: The resume text to analyze (should be single role text for recency)
            filename: Name of the file (for logging)
        
        Returns:
            Detected domain string or None if not found
        """
        if not resume_text:
            return None
        
        # Filter out education sections to prevent false positives
        filtered_text = self._filter_education_sections(resume_text)
        if not filtered_text or len(filtered_text.strip()) < 50:
            # If filtering removed too much, use original but be more careful with Education
            filtered_text = resume_text
        
        text_lower = filtered_text.lower()
        
        # Domain keyword mappings with weights (higher weight = more specific/important)
        # Keywords are categorized: high_weight (company names, specific terms), medium_weight (domain terms), low_weight (general terms)
        domain_keywords = {
            "Healthcare": {
                "high": [
                    "healthcare data", "healthcare analytics", "healthcare strategy", "healthcare practice",
                    "healthcare ecosystem", "healthcare it", "healthcare consulting", "healthcare services",
                    "epic", "cerner", "allscripts", "athenahealth", "meditech", "ehr", "emr", "emr system",
                    "population health", "value-based care", "vbc", "revenue cycle management", "rcm",
                    "mayo clinic", "cleveland clinic", "kaiser permanente", "johns hopkins", "mass general",
                    "vancouver clinic", "healthcare provider", "healthcare payer", "healthcare system"
                ],
                "medium": [
                    "healthcare", "health care", "hospital", "clinic", "medical center", "health system",
                    "clinical", "clinical data", "clinical analytics", "patient care", "patient data",
                    "medicare", "medicaid", "hipaa", "hl7", "fhir", "health information",
                    "life sciences", "biotech", "biotechnology",
                    # Removed "pharmaceutical", "pharma" - exclusive to "Pharmaceuticals & Clinical Research"
                    "health insurance", "health plan", "payer", "provider", "physician", "nurse"
                ],
                "low": [
                    "medical", "health", "wellness", "treatment", "diagnosis", "therapy"
                ]
            },
            "Banking": {
                "high": [
                    "bank of america", "chase", "wells fargo", "citibank", "jpmorgan", "goldman sachs",
                    "morgan stanley", "investment bank", "commercial bank", "retail banking",
                    "banking services", "banking operations", "banking technology"
                ],
                "medium": [
                    "bank", "banking", "financial institution", "credit union", "mortgage", "lending",
                    "loan", "deposit", "teller", "branch banking", "corporate banking"
                ],
                "low": []  # Removed abstract keywords: financial, finance, money, revenue, budget
            },
            "Finance": {
                "high": [
                    "capital markets", "investment management", "wealth management", "asset management",
                    "private equity", "venture capital", "hedge fund", "financial planning",
                    "financial services", "financial technology", "financial consulting"
                    # Removed "fintech" - exclusive to "FinTech" domain
                ],
                "medium": [
                    "finance", "financial", "accounting", "cpa", "audit", "tax", "treasury",
                    "financial analyst", "financial advisor", "financial reporting", "fp&a"
                ],
                "low": []  # Removed abstract keywords: accounting, budget, revenue
            },
            "FinTech": {
                "high": [
                    "fintech", "fintech company", "fintech platform", "fintech startup",
                    "digital banking", "mobile banking", "payment platform", "lending platform",
                    "cryptocurrency", "blockchain", "digital wallet", "payment gateway"
                ],
                "medium": [
                    "fintech", "financial technology", "digital finance", "payment solutions",
                    "lending technology", "banking technology", "payment processing"
                ],
                "low": []
            },
            "Insurance": {
                "high": [
                    "insurance company", "insurance carrier", "insurance agency", "insurance broker",
                    "life insurance", "health insurance", "property insurance", "casualty insurance",
                    "auto insurance", "home insurance", "insurance claims", "insurance underwriting"
                ],
                "medium": [
                    "insurance", "actuary", "underwriting", "claims", "policy", "premium",
                    "insurance services", "risk management", "actuarial"
                ],
                "low": [
                    "coverage", "policy"
                ]
            },
            "E-Commerce": {
                "high": [
                    "e-commerce", "ecommerce", "online retail", "online marketplace", "digital commerce",
                    "amazon", "ebay", "etsy", "shopify", "magento", "woocommerce", "online store"
                ],
                "medium": [
                    "online shopping", "digital retail", "e-commerce platform", "online sales",
                    "marketplace", "online business", "digital marketplace"
                ],
                "low": []  # Removed abstract keywords: online, digital, web, internet, ecommerce
            },
            "Retail": {
                "high": [
                    "retail chain", "retail store", "retail operations", "retail management",
                    "walmart", "target", "costco", "home depot", "retailer", "merchandising"
                ],
                "medium": [
                    "retail", "retailer", "store", "point of sale", "pos", "inventory management",
                    "brick and mortar", "retail sales", "store operations"
                ],
                "low": []  # Removed abstract keywords: sales, customer
            },
            "Manufacturing": {
                "high": [
                    "manufacturing", "production", "factory", "assembly line", "industrial manufacturing",
                    "automotive manufacturing", "aerospace manufacturing", "industrial automation",
                    "lean manufacturing", "six sigma", "quality control", "production management"
                ],
                "medium": [
                    "manufacturing", "production", "factory", "assembly", "industrial",
                    "manufacturing operations", "production planning", "manufacturing process"
                ],
                "low": [
                    "production", "industrial"
                ]
            },
            "Education": {
                "high": [
                    "school district", "educational institution", "edtech company", "education consulting",
                    "educational technology company", "lms platform", "e-learning platform", "education services"
                ],
                "medium": [
                    "education consulting", "educational technology", "curriculum development", "learning management system",
                    "worked at university", "worked at college", "education sector", "education industry"
                ],
                "low": []  # Removed generic terms - only use if clearly work-related
            },
            "Government": {
                "high": [
                    "federal government", "state government", "local government", "municipal government",
                    "government agency", "public sector", "government services", "public administration",
                    "civil service", "government contractor"
                ],
                "medium": [
                    "government", "public sector", "federal", "state", "municipal",
                    "government agency", "public administration", "civil service"
                ],
                "low": [
                    "public", "administration"
                ]
            },
            "Information Technology": {
                "high": [
                    "software company", "it company", "tech company", "saas company", "software as a service company",
                    "enterprise software", "software product", "it services company", "it consulting firm",
                    "cloud services company", "cybersecurity company", "data center company", "it infrastructure company"
                ],
                "medium": [
                    "information technology company", "software development company", "technology company",
                    "saas", "software as a service", "cloud computing company", "cybersecurity firm"
                ],
                "low": []  # Removed generic tech keywords - only use if IT company/product context found
            },
            "Software & SaaS": {
                "high": [
                    "software company", "saas company", "software as a service company", "software product company",
                    "enterprise software company", "saas platform", "software vendor"
                ],
                "medium": [
                    "saas", "software as a service", "software development company", "software product"
                ],
                "low": []
            },
            "Cloud & Infrastructure": {
                "high": [
                    "cloud services company", "cloud infrastructure company", "cloud provider", "aws", "azure", "gcp",
                    "cloud computing company", "infrastructure as a service", "iaas company"
                ],
                "medium": [
                    "cloud computing", "cloud services", "cloud infrastructure", "cloud platform"
                ],
                "low": []
            },
            "Cybersecurity": {
                "high": [
                    "cybersecurity company", "security software company", "security services company",
                    "cyber security firm", "information security company"
                ],
                "medium": [
                    "cybersecurity", "cyber security", "security company", "security services"
                ],
                "low": []
            },
            "Supply Chain": {
                "high": [
                    "supply chain management", "supply chain operations", "supply chain company",
                    "supply chain consulting", "supply chain services"
                ],
                "medium": [
                    "supply chain", "scm", "supply chain management"
                ],
                "low": []
            },
            "Defense": {
                "high": [
                    "defense contractor", "defense industry", "defense company", "defense sector",
                    "defense department", "department of defense", "dod contractor"
                ],
                "medium": [
                    "defense", "defence", "defense contractor", "defense industry"
                ],
                "low": []
            },
            "Public Sector": {
                "high": [
                    "public sector", "public sector services", "government services", "public administration",
                    "federal government", "state government", "municipal government"
                ],
                "medium": [
                    "public sector", "government services", "public administration"
                ],
                "low": []
            },
            "Finance & Accounting": {
                "high": [
                    "finance and accounting", "financial accounting", "accounting firm", "cpa firm",
                    "accounting services", "financial services company"
                ],
                "medium": [
                    "finance and accounting", "accounting", "financial accounting"
                ],
                "low": []
            },
            "Pharmaceuticals & Clinical Research": {
                "high": [
                    "pharmaceutical company", "pharma company", "pharmaceutical industry",
                    "clinical research", "clinical trials", "drug development", "pharmaceutical manufacturing",
                    "biopharmaceutical", "pharmaceutical research", "pharmaceutical sales"
                ],
                "medium": [
                    "pharmaceutical", "pharma", "pharmaceuticals", "clinical research",
                    "drug discovery", "pharmaceutical development"
                ],
                "low": []
            },
            "Banking, Financial Services & Insurance (BFSI)": {
                "high": [
                    "bfsi", "banking financial services insurance", "financial services and insurance",
                    "banking and financial services", "bfsi company", "bfsi sector"
                ],
                "medium": [
                    "bfsi", "banking financial services", "financial services insurance"
                ],
                "low": []
            },
            "Sales & Marketing": {
                "high": [
                    "sales and marketing", "marketing company", "sales company", "marketing agency",
                    "marketing services", "advertising agency"
                ],
                "medium": [
                    "sales and marketing", "marketing", "sales"
                ],
                "low": []
            },
            "Data & Analytics": {
                "high": [
                    "data analytics company", "analytics company", "data company", "data services company",
                    "analytics platform", "data platform", "business intelligence company", "bi company"
                ],
                "medium": [
                    "data analytics", "analytics company", "data company", "analytics platform"
                ],
                "low": []
            },
            "Artificial Intelligence": {
                "high": [
                    "ai company", "artificial intelligence company", "machine learning company", "ml company",
                    "ai platform", "ai product", "ai services company"
                ],
                "medium": [
                    "ai company", "artificial intelligence company", "machine learning company"
                ],
                "low": []
            },
            "Telecommunications": {
                "high": [
                    "telecommunications", "telecom", "wireless", "mobile network", "5g", "4g",
                    "verizon", "at&t", "t-mobile", "sprint", "network infrastructure",
                    "telecom services", "telecom operator", "network operator"
                ],
                "medium": [
                    "telecommunications", "telecom", "wireless", "mobile network", "network",
                    "telecom infrastructure", "network services"
                ],
                "low": []  # Removed abstract keywords: network, communication
            },
            "Energy": {
                "high": [
                    "power plant", "power station", "electric utility", "utility company",
                    "renewable energy", "solar energy", "wind energy", "oil and gas", "petroleum",
                    "energy sector", "energy company", "energy industry", "oil company", "gas company",
                    "solar company", "wind company", "renewable energy company",
                    "transmission", "distribution", "power grid", "electric grid", "energy grid"
                ],
                "medium": [
                    "utilities", "renewable", "solar", "wind", "oil", "gas", "petroleum",
                    "power generation", "energy generation", "electric power", "renewable power"
                ],
                "low": []  # Removed generic terms: "energy", "power" - too generic, match technical terms
            },
            "Logistics": {
                "high": [
                    "logistics", "supply chain", "warehouse", "distribution", "shipping",
                    "transportation", "freight", "logistics management", "supply chain management",
                    "logistics operations", "distribution center", "fulfillment center"
                ],
                "medium": [
                    "logistics", "supply chain", "warehouse", "distribution", "shipping",
                    "transportation", "freight", "logistics"
                ],
                "low": [
                    "shipping", "delivery"
                ]
            },
            "Real Estate": {
                "high": [
                    "real estate", "property", "realty", "real estate development",
                    "commercial real estate", "residential real estate", "property management",
                    "real estate broker", "real estate agent", "property development"
                ],
                "medium": [
                    "real estate", "property", "realty", "real estate", "property management",
                    "real estate services"
                ],
                "low": [
                    "property", "real estate"
                ]
            },
            "Media & Entertainment": {
                "high": [
                    "media", "entertainment", "broadcasting", "television", "film", "publishing",
                    "advertising", "marketing", "digital media", "content creation",
                    "media company", "entertainment industry", "broadcast media"
                ],
                "medium": [
                    "media", "entertainment", "broadcasting", "television", "film",
                    "publishing", "advertising", "marketing", "digital media"
                ],
                "low": []  # Removed abstract keywords: media, content, marketing
            },
            "Automotive": {
                "high": [
                    "automotive", "automobile", "car manufacturer", "auto industry",
                    "automotive manufacturing", "automotive engineering", "vehicle manufacturing"
                ],
                "medium": [
                    "automotive", "automobile", "auto", "vehicle", "car", "automotive industry"
                ],
                "low": [
                    "vehicle", "automotive"
                ]
            },
            "Aerospace": {
                "high": [
                    "aerospace", "aviation", "aircraft", "aerospace manufacturing",
                    "aerospace engineering", "defense contractor", "space", "satellite"
                ],
                "medium": [
                    "aerospace", "aviation", "aircraft", "aerospace", "aviation industry"
                ],
                "low": [
                    "aviation", "aircraft"
                ]
            },
            "Construction": {
                "high": [
                    "construction", "construction company", "construction management",
                    "general contractor", "construction project", "building construction"
                ],
                "medium": [
                    "construction", "contractor", "building", "construction management",
                    "construction industry"
                ],
                "low": [
                    "construction", "building"
                ]
            },
            "Hospitality": {
                "high": [
                    "hospitality", "hotel", "resort", "hospitality management",
                    "hotel management", "restaurant", "hospitality industry"
                ],
                "medium": [
                    "hospitality", "hotel", "resort", "restaurant", "hospitality services"
                ],
                "low": [
                    "hotel", "restaurant"
                ]
            },
            "Transportation": {
                "high": [
                    "transportation", "transit", "public transportation", "transportation services",
                    "transportation management", "fleet management", "transportation company"
                ],
                "medium": [
                    "transportation", "transit", "transport", "transportation services"
                ],
                "low": [
                    "transport", "transit"
                ]
            },
            "Salesforce": {
                "high": [
                    "salesforce admin", "salesforce developer", "salesforce consultant", "salesforce architect",
                    "salesforce platform", "salesforce crm", "salesforce.com", "salesforce org",
                    "apex", "visualforce", "lightning", "salesforce certification", "salesforce trailhead",
                    "salesforce administrator", "salesforce developer", "salesforce implementation",
                    "salesforce project", "salesforce work", "salesforce experience"
                ],
                "medium": [
                    "salesforce", "sfdc", "sales cloud", "service cloud", "marketing cloud",
                    "salesforce ecosystem", "salesforce environment"
                ],
                "low": []
            },
            "AWS": {
                "high": [
                    "aws solutions architect", "aws developer", "aws engineer", "aws consultant",
                    "aws cloud", "amazon web services", "aws platform", "aws services",
                    "aws certification", "aws infrastructure", "aws deployment",
                    "ec2", "s3", "lambda", "rds", "aws cloudformation", "aws cloudwatch"
                ],
                "medium": [
                    "aws", "amazon web services", "aws cloud", "aws platform"
                ],
                "low": []
            },
            "Microsoft": {
                "high": [
                    "microsoft technologies", "microsoft stack", "microsoft platform",
                    "microsoft azure", "microsoft dynamics", "office 365", "microsoft 365",
                    "sharepoint", "power platform", "power bi", "power apps", "power automate",
                    "microsoft consultant", "microsoft developer", "microsoft architect",
                    ".net", "asp.net", "microsoft certification"
                ],
                "medium": [
                    "microsoft", "microsoft technologies", "microsoft platform", "ms technologies"
                ],
                "low": []
            },
            "Oracle": {
                "high": [
                    "oracle erp", "oracle cloud", "oracle database", "oracle consultant",
                    "oracle developer", "oracle implementation", "oracle ebs", "oracle fusion",
                    "oracle financials", "oracle hcm", "oracle scm", "oracle platform",
                    "oracle certification", "oracle project"
                ],
                "medium": [
                    "oracle", "oracle technologies", "oracle platform", "oracle systems"
                ],
                "low": []
            },
            "SAP": {
                "high": [
                    "sap consultant", "sap developer", "sap implementation", "sap project",
                    "sap erp", "sap s4hana", "sap ecc", "sap fico", "sap mm", "sap sd",
                    "sap hcm", "sap abap", "sap basis", "sap platform", "sap certification",
                    "sap work", "sap experience"
                ],
                "medium": [
                    "sap", "sap erp", "sap system", "sap platform", "sap technologies"
                ],
                "low": []
            },
            "ServiceNow": {
                "high": [
                    "servicenow admin", "servicenow developer", "servicenow consultant",
                    "servicenow platform", "servicenow implementation", "servicenow project",
                    "servicenow itil", "servicenow certification", "servicenow instance",
                    "servicenow work", "servicenow experience"
                ],
                "medium": [
                    "servicenow", "service now", "servicenow platform", "servicenow system"
                ],
                "low": []
            },
            "Workday": {
                "high": [
                    "workday consultant", "workday developer", "workday implementation",
                    "workday hcm", "workday financials", "workday platform", "workday project",
                    "workday certification", "workday admin", "workday work", "workday experience"
                ],
                "medium": [
                    "workday", "workday platform", "workday system", "workday technologies"
                ],
                "low": []
            },
            "Adobe": {
                "high": [
                    "adobe marketing cloud", "adobe experience cloud", "adobe consultant",
                    "adobe developer", "adobe implementation", "adobe analytics", "adobe campaign",
                    "adobe target", "adobe aem", "adobe experience manager", "adobe platform",
                    "adobe certification", "adobe work", "adobe experience"
                ],
                "medium": [
                    "adobe marketing", "adobe experience", "adobe platform", "adobe technologies"
                ],
                "low": []
            },
            "Google Cloud": {
                "high": [
                    "google cloud platform", "gcp", "google cloud architect", "google cloud engineer",
                    "google cloud consultant", "google cloud certification", "gcp platform",
                    "google cloud services", "gcp deployment", "gcp infrastructure"
                ],
                "medium": [
                    "google cloud", "gcp", "google cloud platform", "google cloud services"
                ],
                "low": []
            },
            "Azure": {
                "high": [
                    "azure cloud", "microsoft azure", "azure architect", "azure engineer",
                    "azure consultant", "azure certification", "azure platform",
                    "azure services", "azure deployment", "azure infrastructure",
                    "azure devops", "azure functions", "azure sql"
                ],
                "medium": [
                    "azure", "microsoft azure", "azure cloud", "azure platform", "azure services"
                ],
                "low": []
            },
            "Legal, Risk & Corporate Governance": {
                "high": [
                    "legal services", "law firm", "legal consulting", "legal department",
                    "attorney", "lawyer", "legal counsel", "legal advisor", "litigation",
                    "class action", "legal case", "legal practice", "legal industry"
                ],
                "medium": [
                    "legal", "law", "legal services", "legal practice", "legal industry"
                ],
                "low": []
            },
            "Human Resources": {
                "high": [
                    "hr services", "human resources", "hr consulting", "hr department",
                    "background check", "talent acquisition", "hr platform", "hr technology",
                    "hr services company", "hr solutions"
                ],
                "medium": [
                    "human resources", "hr", "hr services", "hr consulting", "hr solutions"
                ],
                "low": []
            }
        }
        
        # Count keyword matches with weights
        domain_scores = {}
        for domain, keyword_groups in domain_keywords.items():
            score = 0
            high_matches = 0
            medium_matches = 0
            
            # High weight keywords (most specific)
            for keyword in keyword_groups.get("high", []):
                if keyword in text_lower:
                    score += 10  # High weight
                    high_matches += 1
            
            # Medium weight keywords
            for keyword in keyword_groups.get("medium", []):
                if keyword in text_lower:
                    score += 5  # Medium weight
                    medium_matches += 1
            
            # Low weight keywords (count even without other matches - more lenient)
            for keyword in keyword_groups.get("low", []):
                if keyword in text_lower:
                    score += 1  # Low weight
            
            # Fix #5: Only register domain if medium/high matches exist
            # This prevents single low-weight keyword from surviving until threshold check
            if score > 0 and (high_matches > 0 or medium_matches > 0):
                domain_scores[domain] = {
                    "score": score,
                    "high_matches": high_matches,
                    "medium_matches": medium_matches
                }
        
        # Platform-specific domains (highest priority when they have matches)
        platform_domains = ["Salesforce", "AWS", "Microsoft", "Oracle", "SAP", "ServiceNow", 
                           "Workday", "Adobe", "Google Cloud", "Azure"]
        
        # Return domain with highest score if score is significant
        if domain_scores:
            # First, check if any platform-specific domain has matches (prioritize these)
            # Check ALL domain_scores, not just those that passed threshold
            platform_matches = {
                domain: data for domain, data in domain_scores.items() 
                if domain in platform_domains and (data["high_matches"] > 0 or data["medium_matches"] > 0)
            }
            
            if platform_matches:
                # Use platform domain with highest score (even if below threshold)
                best_domain = max(platform_matches, key=lambda x: platform_matches[x]["score"])
                best_data = platform_matches[best_domain]
                best_score = best_data["score"]
                # For platform domains, lower threshold - just need any match
                if best_data["high_matches"] > 0 or best_data["medium_matches"] > 0:
                    logger.info(
                        f"✅ Platform-specific domain detected: {best_domain} (score: {best_score}, high: {best_data['high_matches']}, medium: {best_data['medium_matches']})",
                        extra={"file_name": filename, "domain": best_domain, "score": best_score}
                    )
                    return best_domain
                # If platform domain doesn't have matches, fall through to regular logic
            else:
                # No platform domain matches, use highest scoring domain
                best_domain = max(domain_scores, key=lambda x: domain_scores[x]["score"])
                best_data = domain_scores[best_domain]
                best_score = best_data["score"]
            
            # Special handling for Education domain - must have work context
            if best_domain == "Education":
                # For Education, require high-weight matches (work-related terms) or verify work context
                if best_data["high_matches"] == 0:
                    # No high-weight matches means no clear work context - skip Education
                    logger.debug(
                        f"Education domain detected but no work context found - skipping to avoid false positive",
                        extra={"file_name": filename, "score": best_score}
                    )
                    # Remove Education from scores and try next best domain
                    domain_scores.pop("Education", None)
                    if domain_scores:
                        best_domain = max(domain_scores, key=lambda x: domain_scores[x]["score"])
                        best_data = domain_scores[best_domain]
                        best_score = best_data["score"]
                    else:
                        return None
            
            # STRICT threshold: Require strong indicators (ATS-grade)
            # Require: score >= 10 AND at least 1 high-weight match OR 2+ medium-weight matches
            has_high_match = best_data["high_matches"] > 0
            has_medium_match = best_data["medium_matches"] >= 2
            has_strong_score = best_score >= 10
            
            # Accept only if we have strong indicators
            if has_strong_score and (has_high_match or has_medium_match):
                # Apply domain precedence: if multiple domains detected, choose highest priority
                candidate_domains = [
                    domain for domain, data in domain_scores.items()
                    if data["score"] >= 10 and (data["high_matches"] > 0 or data["medium_matches"] >= 2)
                ]
                
                if len(candidate_domains) > 1:
                    # Resolve conflict using domain precedence
                    best_domain = self._resolve_domain_precedence(candidate_domains)
                    logger.info(
                        f"✅ Domain precedence resolved: {best_domain} from {len(candidate_domains)} candidates",
                        extra={
                            "file_name": filename,
                            "candidates": candidate_domains,
                            "resolved": best_domain
                        }
                    )
                
                logger.info(
                    f"✅ Keyword-based domain detection: {best_domain} (score: {best_score}, "
                    f"high: {best_data['high_matches']}, medium: {best_data['medium_matches']})",
                    extra={
                        "file_name": filename,
                        "domain": best_domain,
                        "score": best_score,
                        "high_matches": best_data["high_matches"],
                        "medium_matches": best_data["medium_matches"]
                    }
                )
                return best_domain
            else:
                logger.debug(
                    f"Keyword detection found {best_domain} but threshold not met (score: {best_score}, "
                    f"high: {best_data['high_matches']}, medium: {best_data['medium_matches']})",
                    extra={
                        "file_name": filename, 
                        "domain": best_domain, 
                        "score": best_score,
                        "high_matches": best_data["high_matches"],
                        "medium_matches": best_data["medium_matches"]
                    }
                )
        
        return None
    
    def _infer_domain_from_job_titles(self, resume_text: str, filename: str = "resume") -> Optional[str]:
        """
        Conservative fallback: Only infer domain from clearly industry-specific job titles.
        Per new prompt rules: Never infer domain from job titles alone unless clearly industry-specific.
        This method is very conservative and only used as last resort.
        Filters out education sections to prevent false Education domain detection.
        
        Args:
            resume_text: The resume text to analyze
            filename: Name of the file (for logging)
        
        Returns:
            Inferred domain string or None if not found
        """
        if not resume_text:
            return None
        
        # Fix #2: Remove education filtering from fallback - rely on LLM prompt rules
        # Education filtering can remove valid EdTech experience
        text_lower = resume_text.lower()
        
        # Only use clearly industry-specific job titles that indicate business domain
        # NOT generic tech roles - those could be in any industry
        industry_specific_titles = {
            "Healthcare": [
                "healthcare director", "healthcare manager", "healthcare consultant", "healthcare analyst",
                "chief medical officer", "cmo", "chief nursing officer", "cno", "healthcare administrator",
                "hospital administrator", "clinic manager", "healthcare operations"
            ],
            "Banking": [
                "banker", "loan officer", "credit analyst", "mortgage officer", "branch manager",
                "bank manager", "commercial banker", "investment banker"
            ],
            "Finance": [
                "financial advisor", "financial planner", "wealth manager", "investment advisor",
                "asset manager", "portfolio manager", "financial consultant"
            ],
            "Insurance": [
                "insurance agent", "insurance broker", "actuary", "underwriter", "claims adjuster",
                "insurance sales", "insurance consultant"
            ],
            "Education": [
                # Only include titles that clearly indicate work in education industry (not academic roles)
                "education director", "education manager", "education consultant", "edtech manager",
                "lms administrator", "curriculum developer", "instructional designer", "education coordinator"
            ],
            "Government": [
                "government contractor", "federal employee", "state employee", "municipal employee",
                "civil servant", "public administrator"
            ],
            "Retail": [
                "store manager", "retail manager", "merchandiser", "retail operations manager",
                "buyer", "category manager"
            ],
            "Manufacturing": [
                "production manager", "manufacturing manager", "plant manager", "operations manager",
                "quality control manager", "supply chain manager"
            ]
        }
        
        # Check for industry-specific job titles only
        domain_matches = {}
        for domain, titles in industry_specific_titles.items():
            match_count = sum(1 for title in titles if title in text_lower)
            if match_count > 0:
                domain_matches[domain] = match_count
        
        # Return domain with most matches (only if clearly industry-specific)
        if domain_matches:
            best_domain = max(domain_matches, key=domain_matches.get)
            logger.info(
                f"✅ Domain inferred from industry-specific job title: {best_domain} (matches: {domain_matches[best_domain]})",
                extra={"file_name": filename, "domain": best_domain, "match_count": domain_matches[best_domain]}
            )
            return best_domain
        
        # Do NOT default to IT from generic tech keywords - per new prompt rules
        # IT domain should only be used if candidate worked at IT companies/products
        
        return None
    
    def _validate_llm_domain(self, llm_domain: str, role_text: str) -> tuple[bool, str]:
        """
        Validate LLM domain result to prevent hallucinations.
        
        Args:
            llm_domain: Domain returned by LLM
            role_text: The role text that was analyzed
            
        Returns:
            Tuple of (is_valid, reason)
        """
        if not llm_domain or not role_text:
            return False, "Empty domain or role text"
        
        domain_lower = llm_domain.lower()
        text_lower = role_text.lower()
        
        # Check 1: If domain matches employer map → ACCEPT (LLM is correct)
        employer_domain = self._check_employer_domain_map(role_text)
        if employer_domain and domain_lower == employer_domain.lower():
            return True, f"LLM domain matches employer map: {llm_domain}"
        
        # Check 2: If domain matches healthcare keywords → ACCEPT (LLM is correct)
        healthcare_domain = self._check_healthcare_keywords(role_text)
        if healthcare_domain and domain_lower == healthcare_domain.lower():
            return True, f"LLM domain matches healthcare keywords: {llm_domain}"
        
        # Check 3: If domain matches banking keywords → ACCEPT (LLM is correct)
        banking_domain = self._check_banking_keywords(role_text)
        if banking_domain and domain_lower == banking_domain.lower():
            return True, f"LLM domain matches banking keywords: {llm_domain}"
        
        # Check 4: If domain matches retail keywords → ACCEPT (LLM is correct)
        retail_domain = self._check_retail_keywords(role_text)
        if retail_domain and domain_lower == retail_domain.lower():
            return True, f"LLM domain matches retail keywords: {llm_domain}"
        
        # Check 5: Platform domain validation (CRITICAL - prevent hallucination)
        platform_domains = ["aws", "salesforce", "sap", "oracle", "microsoft", "servicenow", 
                           "workday", "adobe", "google cloud", "azure"]
        
        if domain_lower in platform_domains:
            # Check if explicit platform role exists
            platform_guard_domain = self._check_platform_domain_guard(role_text)
            if platform_guard_domain and domain_lower == platform_guard_domain.lower():
                # LLM returned platform domain AND explicit role exists → ACCEPT
                return True, f"LLM platform domain validated: {llm_domain} (explicit role found)"
            else:
                # LLM returned platform domain BUT no explicit role → REJECT (hallucination)
                return False, f"LLM platform domain REJECTED: {llm_domain} (no explicit platform role - likely hallucination)"
        
        # Check 6: If employer map suggests different domain → REJECT (LLM is wrong)
        if employer_domain and domain_lower != employer_domain.lower():
            return False, f"LLM domain REJECTED: {llm_domain} (employer map suggests: {employer_domain})"
        
        # Check 7: If healthcare keywords suggest Healthcare but LLM returned something else → REJECT
        if healthcare_domain and domain_lower != healthcare_domain.lower():
            return False, f"LLM domain REJECTED: {llm_domain} (healthcare keywords suggest: {healthcare_domain})"
        
        # Check 8: If banking keywords suggest Banking but LLM returned something else → REJECT
        if banking_domain and domain_lower != banking_domain.lower():
            return False, f"LLM domain REJECTED: {llm_domain} (banking keywords suggest: {banking_domain})"
        
        # If no clear validation rules match → ACCEPT (LLM may be handling edge case)
        return True, f"LLM domain ACCEPTED: {llm_domain} (no validation rules to contradict)"
    
    def _extract_json(self, text: str) -> Dict:
        """Extract JSON object from LLM response."""
        if not text:
            logger.warning("Empty response from LLM")
            return {"domain": None}
        
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
            if isinstance(parsed, dict) and "domain" in parsed:
                logger.debug(f"Successfully extracted JSON: {parsed}")
                # Ensure domain is properly handled (None, string, or null)
                if parsed.get("domain") is None:
                    parsed["domain"] = None
                elif isinstance(parsed.get("domain"), str):
                    parsed["domain"] = parsed["domain"].strip()
                    if not parsed["domain"] or parsed["domain"].lower() in ["null", "none", "nil"]:
                        parsed["domain"] = None
                return parsed
        except json.JSONDecodeError as e:
            logger.debug(f"First JSON parse attempt failed: {e}")
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
                    if isinstance(parsed, dict) and "domain" in parsed:
                        logger.debug(f"Successfully extracted JSON with balanced braces: {parsed}")
                        # Ensure domain is properly handled (None, string, or null)
                        if parsed.get("domain") is None:
                            parsed["domain"] = None
                        elif isinstance(parsed.get("domain"), str):
                            parsed["domain"] = parsed["domain"].strip()
                            if not parsed["domain"] or parsed["domain"].lower() in ["null", "none", "nil"]:
                                parsed["domain"] = None
                        return parsed
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON with balanced braces: {e}")
        
        logger.error(
            "ERROR: Failed to parse JSON from LLM response", 
            extra={
                "response_hash": hash(text[:1000]),
                "response_length": len(text),
                "cleaned_hash": hash(cleaned_text[:1000])
            }
        )
        return {"domain": None}
    
    async def extract_domain(self, resume_text: str, filename: str = "resume") -> Optional[str]:
        """
        Extract industry domain from resume text using OLLAMA LLM.
        
        Args:
            resume_text: The text content of the resume
            filename: Name of the resume file (for logging)
        
        Returns:
            The extracted domain string or None if not found
        """
        try:
            # Validate resume text
            if not resume_text or not resume_text.strip():
                logger.warning(
                    f"Empty or invalid resume text provided for domain extraction",
                    extra={"file_name": filename}
                )
                return None
            
            is_connected, available_model = await self._check_ollama_connection()
            if not is_connected:
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
            
            # Extract latest role (role-based approach - PRIMARY)
            roles = self._extract_roles(resume_text)
            latest_role = self._select_latest_role(roles) if roles else None
            
            # Handle no-date resume case: Fallback to experience-based extraction
            if not latest_role:
                logger.info(
                    "⚠️ No roles found with date ranges - falling back to experience-based extraction",
                    extra={
                        "file_name": filename,
                        "reason": "Cannot isolate single role without dates - using experience fallback"
                    }
                )
                # Fallback: Use experience-based extraction
                experience_text = self._extract_latest_experience(resume_text)
                if experience_text and experience_text.strip() and len(experience_text.strip()) > 50:
                    # Use experience text for domain extraction
                    text_to_analyze = experience_text[:self.MAX_ROLE_CHARS] if len(experience_text) > self.MAX_ROLE_CHARS else experience_text
                    logger.info(
                        f"✅ Using experience-based extraction fallback ({len(text_to_analyze)} chars)",
                        extra={
                            "file_name": filename,
                            "experience_length": len(experience_text),
                            "truncated_length": len(text_to_analyze),
                            "method": "experience_fallback"
                        }
                    )
                else:
                    # Final fallback: Use first part of resume text
                    text_to_analyze = resume_text[:self.MAX_ROLE_CHARS] if len(resume_text) > self.MAX_ROLE_CHARS else resume_text
                    logger.info(
                        f"⚠️ Experience extraction also failed - using resume text fallback ({len(text_to_analyze)} chars)",
                        extra={
                            "file_name": filename,
                            "resume_length": len(resume_text),
                            "method": "resume_text_fallback"
                        }
                    )
                
                # Skip role validation since we're using fallback
                if not text_to_analyze or not text_to_analyze.strip():
                    logger.warning(
                        f"No extractable text found for domain extraction",
                        extra={"file_name": filename}
                    )
                    return None
                
                # Use fallback text for LLM extraction (skip to LLM call section)
                role_text_hash = hash(text_to_analyze[:500])
                logger.info(
                    f"✅ Using fallback text for domain extraction (text: {len(text_to_analyze)} chars)",
                    extra={
                        "file_name": filename,
                        "text_length": len(text_to_analyze),
                        "text_hash": role_text_hash,
                        "text_preview": text_to_analyze[:150],
                        "method": "fallback_extraction"
                    }
                )
            else:
                # CRITICAL: Strict validation before calling LLM
                # If role cannot be properly isolated → return null (don't call LLM)
                is_valid, validation_reason = self._validate_role_isolation(latest_role, roles, resume_text)
                if not is_valid:
                    logger.info(
                        f"🔒 Role isolation validation FAILED - falling back to experience extraction",
                        extra={
                            "file_name": filename,
                            "reason": validation_reason,
                            "role_text_length": len(latest_role.text) if latest_role else 0,
                            "role_text_preview": latest_role.text[:200] if latest_role else None,
                        }
                    )
                    # Fallback to experience extraction
                    experience_text = self._extract_latest_experience(resume_text)
                    if experience_text and experience_text.strip() and len(experience_text.strip()) > 50:
                        text_to_analyze = experience_text[:self.MAX_ROLE_CHARS] if len(experience_text) > self.MAX_ROLE_CHARS else experience_text
                        logger.info(
                            f"✅ Using experience-based extraction fallback after validation failure ({len(text_to_analyze)} chars)",
                            extra={
                                "file_name": filename,
                                "method": "experience_fallback_after_validation"
                            }
                        )
                    else:
                        text_to_analyze = resume_text[:self.MAX_ROLE_CHARS] if len(resume_text) > self.MAX_ROLE_CHARS else resume_text
                        logger.info(
                            f"⚠️ Using resume text fallback after validation failure ({len(text_to_analyze)} chars)",
                            extra={
                                "file_name": filename,
                                "method": "resume_text_fallback_after_validation"
                            }
                        )
                    
                    if not text_to_analyze or not text_to_analyze.strip():
                        logger.warning(
                            f"No extractable text found for domain extraction",
                            extra={"file_name": filename}
                        )
                        return None
                    
                    role_text_hash = hash(text_to_analyze[:500])
                else:
                    # Role-based path: Use latest role text (already truncated to MAX_ROLE_CHARS in _extract_latest_role)
                    text_to_analyze = latest_role.text
                    
                    # Log role text hash for debugging (to detect if same role is being reused)
                    role_text_hash = hash(text_to_analyze[:500])  # Hash first 500 chars for comparison
                    
                    logger.info(
                        f"✅ Role isolation VALIDATED - using role-based extraction (role text: {len(text_to_analyze)} chars, "
                        f"is_current: {latest_role.is_current}, end_year: {latest_role.end_year})",
                        extra={
                            "file_name": filename,
                            "role_text_length": len(text_to_analyze),
                            "is_current": latest_role.is_current,
                            "end_year": latest_role.end_year,
                            "start_year": latest_role.start_year,
                            "validation_passed": True,
                            "role_text_hash": role_text_hash,
                            "role_text_preview": text_to_analyze[:150],  # First 150 chars for debugging
                            "total_roles_found": len(roles)
                        }
                    )
                    
                    if not text_to_analyze or not text_to_analyze.strip():
                        logger.warning(
                            f"No extractable text found for domain extraction",
                            extra={"file_name": filename}
                        )
                        return None
            
            # ============================================================
            # HYBRID APPROACH: LLM FIRST + VALIDATION
            # ============================================================
            logger.info(
                f"📤 Calling LLM first (with validation layer)",
                extra={"file_name": filename, "note": "LLM will be validated to prevent hallucinations"}
            )
            
            # LLM extraction (call first)
            prompt = f"""{DOMAIN_PROMPT}

IMPORTANT CONTEXT:
The text below represents ONE SINGLE, MOST RECENT JOB ROLE.
Do NOT infer domain from anything else.
If domain is unclear, return null (acceptable).

Input resume text (latest role):
{text_to_analyze}

Output (JSON only, no other text, no explanations):"""
            
            logger.info(
                f"📤 CALLING OLLAMA API for domain extraction",
                extra={
                    "file_name": filename,
                    "model": model_to_use,
                    "ollama_host": self.ollama_host,
                    "resume_text_length": len(resume_text),
                    "text_sent_length": len(text_to_analyze),
                }
            )
            
            result = None
            last_error = None
            
            # Fix #6: Reduce timeout to 120s (was 600s) to prevent thread starvation
            # Add retry logic for transient failures
            async with httpx.AsyncClient(timeout=Timeout(120.0)) as client:
                max_retries = 1
                for attempt in range(max_retries + 1):
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
                                }
                            }
                        )
                        response.raise_for_status()
                        result = response.json()
                        response_text = result.get("response", "") or result.get("text", "")
                        if not response_text and "message" in result:
                            response_text = result.get("message", {}).get("content", "")
                        result = {"response": response_text}
                        logger.info("✅ Successfully used /api/generate endpoint for domain extraction")
                        break  # Success, exit retry loop
                    except (httpx.TimeoutException, httpx.NetworkError) as e:
                        if attempt < max_retries:
                            logger.warning(f"OLLAMA request timeout/network error (attempt {attempt + 1}/{max_retries + 1}), retrying...")
                            last_error = e
                            continue
                        else:
                            last_error = e
                            logger.error(f"OLLAMA request failed after {max_retries + 1} attempts: {e}")
                            raise
                    except httpx.HTTPStatusError as e:
                        if e.response.status_code != 404:
                            raise
                        last_error = e
                        logger.warning("OLLAMA /api/generate returned 404, trying /api/chat endpoint")
                        break  # Try /api/chat instead
                
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
                                    "temperature": 0.0,
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
                        logger.info("Successfully used /api/chat endpoint for domain extraction")
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
            llm_domain = parsed_data.get("domain")
            
            # Handle both None and string "null" cases
            if llm_domain is not None:
                llm_domain = str(llm_domain).strip()
                if not llm_domain or llm_domain.lower() in ["null", "none", "nil", ""]:
                    llm_domain = None
            else:
                llm_domain = None
            
            # Log LLM response for debugging (regardless of value)
            logger.info(
                f"🔍 LLM Response for {filename}",
                extra={
                    "file_name": filename,
                    "llm_domain": llm_domain,
                    "parsed_data": parsed_data,
                    "raw_output_preview": raw_output[:200] if raw_output else None,
                    "raw_output_length": len(raw_output) if raw_output else 0
                }
            )
            
            # ============================================================
            # VALIDATE LLM RESULT (prevent hallucinations)
            # ============================================================
            domain = None
            if llm_domain:
                is_valid, validation_reason = self._validate_llm_domain(llm_domain, text_to_analyze)
                if is_valid:
                    # LLM domain is valid → ACCEPT
                    domain = llm_domain
                    logger.info(
                        f"✅ LLM domain VALIDATED and ACCEPTED: {domain}",
                        extra={
                            "file_name": filename,
                            "domain": domain,
                            "validation_reason": validation_reason,
                            "method": "llm_validated"
                        }
                    )
                else:
                    # LLM domain is invalid → REJECT (hallucination detected)
                    logger.warning(
                        f"⚠️ LLM domain REJECTED (hallucination detected): {llm_domain}",
                        extra={
                            "file_name": filename,
                            "rejected_domain": llm_domain,
                            "validation_reason": validation_reason,
                            "note": "Falling back to deterministic rules"
                        }
                    )
                    domain = None  # Reject LLM result
            else:
                # LLM returned null/None - log this for debugging
                logger.info(
                    f"ℹ️ LLM returned null/None for {filename}",
                    extra={
                        "file_name": filename,
                        "llm_domain": llm_domain,
                        "parsed_data": parsed_data,
                        "note": "LLM could not determine domain - using fallback methods"
                    }
                )
            
            # ============================================================
            # DETERMINISTIC RULES FALLBACK (if LLM failed or was rejected)
            # ============================================================
            if not domain:
                logger.info(
                    f"🔄 LLM failed or rejected - using deterministic rules as fallback",
                    extra={
                        "file_name": filename,
                        "llm_domain": llm_domain,
                        "note": "Deterministic rules will be used as fallback"
                    }
                )
                
                # STEP 1: Employer Domain Map (highest priority - deterministic)
                domain = self._check_employer_domain_map(text_to_analyze)
                if domain:
                    logger.info(
                        f"✅ Domain determined via employer map (fallback): {domain}",
                        extra={"file_name": filename, "domain": domain, "method": "deterministic_employer_map_fallback"}
                    )
                    return domain
                
                # STEP 2: Industry Keyword Override (deterministic)
                # Healthcare
                domain = self._check_healthcare_keywords(text_to_analyze)
                if domain:
                    logger.info(
                        f"✅ Domain determined via healthcare keywords (fallback): {domain}",
                        extra={"file_name": filename, "domain": domain, "method": "deterministic_healthcare_fallback"}
                    )
                    return domain
                
                # Banking
                domain = self._check_banking_keywords(text_to_analyze)
                if domain:
                    logger.info(
                        f"✅ Domain determined via banking keywords (fallback): {domain}",
                        extra={"file_name": filename, "domain": domain, "method": "deterministic_banking_fallback"}
                    )
                    return domain
                
                # Retail
                domain = self._check_retail_keywords(text_to_analyze)
                if domain:
                    logger.info(
                        f"✅ Domain determined via retail keywords (fallback): {domain}",
                        extra={"file_name": filename, "domain": domain, "method": "deterministic_retail_fallback"}
                    )
                    return domain
                
                # STEP 3: Platform Domain Guard (deterministic - prevents hallucination)
                domain = self._check_platform_domain_guard(text_to_analyze)
                if domain:
                    logger.info(
                        f"✅ Domain determined via platform guard (fallback): {domain}",
                        extra={"file_name": filename, "domain": domain, "method": "deterministic_platform_guard_fallback"}
                    )
                    return domain
                
                # ============================================================
                # FINAL SAFETY NET: Keyword Fallback (only if all methods fail)
                # ============================================================
                logger.info(
                    f"ℹ️ All deterministic rules exhausted - attempting keyword fallback as final safety net",
                    extra={
                        "file_name": filename,
                        "note": "LLM and deterministic rules exhausted, trying keyword fallback"
                    }
                )
                # Try keyword fallback if we have text to analyze (removed latest_role requirement)
                if text_to_analyze:
                    keyword_domain = self._detect_domain_from_keywords(text_to_analyze, filename)
                    if keyword_domain:
                        logger.info(
                            f"✅ Keyword fallback detected domain: {keyword_domain} for {filename}",
                            extra={
                                "file_name": filename,
                                "detected_domain": keyword_domain,
                                "method": "keyword_fallback_final_safety_net"
                            }
                        )
                        return keyword_domain
                    else:
                        logger.info(
                            f"ℹ️ Keyword fallback found no domain for {filename}",
                            extra={
                                "file_name": filename,
                                "text_analyzed_length": len(text_to_analyze) if text_to_analyze else 0,
                                "note": "All methods exhausted - returning null"
                            }
                        )
                else:
                    logger.warning(
                        f"⚠️ No text available for keyword fallback: {filename}",
                        extra={
                            "file_name": filename,
                            "has_text_to_analyze": bool(text_to_analyze),
                            "has_latest_role": bool(latest_role)
                        }
                    )
                # If no text or keyword fallback fails, return null
                return None
            
            # Log the raw response for debugging (enhanced for troubleshooting)
            # Only log LLM details if we used role-based extraction
            if latest_role:
                logger.info(
                    f"🔍 DOMAIN EXTRACTION DEBUG for {filename}",
                    extra={
                        "file_name": filename,
                        "raw_output_hash": hash(raw_output[:1000]) if raw_output else None,
                        "raw_output_length": len(raw_output) if raw_output else 0,
                        "parsed_data": parsed_data,
                        "llm_domain": llm_domain,
                        "final_domain": domain,
                        "resume_text_length": len(resume_text),
                        "text_sent_length": len(text_to_analyze),
                        "resume_text_hash": hash(resume_text[:1000]) if resume_text else None,
                        "extraction_method": "hybrid_llm_first_with_validation"
                    }
                )
                
                # If domain is null, log details (null is acceptable for ATS-grade systems)
                if not domain:
                    logger.info(
                        f"ℹ️ DOMAIN EXTRACTION RETURNED NULL for {filename} (acceptable - unclear domain)",
                        extra={
                            "file_name": filename,
                            "llm_domain": llm_domain,
                            "resume_text_length": len(resume_text),
                            "text_analyzed_length": len(text_to_analyze),
                            "text_sent_length": len(text_to_analyze),
                            "raw_output_hash": hash(raw_output[:2000]) if raw_output else None,
                            "parsed_data": parsed_data,
                            "text_analyzed_hash": hash(text_to_analyze[:1000]) if text_to_analyze else None,
                            "extraction_method": "hybrid_llm_first_with_validation",
                            "note": "Null is acceptable when domain is unclear (ATS-grade behavior)"
                        }
                    )
            
            logger.info(
                f"✅ DOMAIN EXTRACTED from {filename}",
                extra={
                    "file_name": filename,
                    "domain": domain,
                    "llm_domain": llm_domain,
                    "status": "success" if domain else "not_found"
                }
            )
            
            return domain
            
        except httpx.HTTPError as e:
            error_details = {
                "error": str(e),
                "error_type": type(e).__name__,
                "ollama_host": self.ollama_host,
                "model": model_to_use,
            }
            logger.error(
                f"HTTP error calling OLLAMA for domain extraction: {e}",
                extra=error_details,
                exc_info=True
            )
            raise RuntimeError(f"Failed to extract domain with LLM: {e}")
        except Exception as e:
            logger.error(
                f"Error extracting domain: {e}",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "ollama_host": self.ollama_host,
                    "model": model_to_use,
                },
                exc_info=True
            )
            raise

