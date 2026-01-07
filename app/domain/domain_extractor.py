"""Service for extracting industry domain from resumes using OLLAMA LLM."""
import json
import re
from typing import Dict, Optional, List
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
You are responsible for accurate resume domain classification.

ROLE:
You are an ATS resume-parsing expert specializing in industry and business domain identification.
You MUST behave conservatively, responsibly, and professionally.

DEFINITION:
"Domain" means the PRIMARY BUSINESS / INDUSTRY DOMAIN in which the candidate has worked.
It is NOT the candidate's skills, technologies, tools, job role, or academic background.

TASK:
Analyze the resume text and extract ONE primary industry domain.

WHERE TO LOOK (IN PRIORITY ORDER):
1. Company names
2. Client names
3. Project descriptions
4. Business context of work
5. Industry-specific terminology tied to business operations

DO NOT USE:
- Skills alone
- Programming languages
- Tools or platforms alone
- Job titles alone (unless clearly industry-specific)
- Academic qualifications or education history

CRITICAL EDUCATION DOMAIN DISAMBIGUATION (MANDATORY):
The "Education" domain MUST be extracted ONLY if the candidate has WORKED in the education industry.

HARD EXCLUSION RULE:
- NEVER infer "Education" as a domain from:
  - Degree names (e.g., B.Ed, M.Ed, MA Education, Instructional Technology)
  - University, college, or school names
  - Teaching credentials or certifications
  - Academic programs, majors, minors, or coursework
  - Sections titled "Education", "Academic Background", "Qualifications", or similar
  - Continuous degree listings with institutions and years

If "Education" appears ONLY in academic qualifications, then the domain is not Education, might be something else.

VALID INCLUSION RULE (ALL CONDITIONS REQUIRED):
Extract "Education" ONLY IF:
1. Education-related terms appear in WORK EXPERIENCE, PROJECTS, or CLIENT CONTEXT
2. The resume explicitly indicates professional work such as:
   - Education domain projects
   - EdTech products or platforms
   - eLearning or LMS development/implementation
   - Work for schools, universities, or academic institutions as clients
   - Student information systems, curriculum platforms, or online learning systems
3. The education reference is clearly tied to a job role, project, client, or product

If professional education-domain evidence is missing or ambiguous, return null.

DOMAIN SELECTION RULES:
1. If multiple domains appear, select the MOST RECENT or MOST DOMINANT one.
2. If IT skills are used INSIDE a non-IT industry (e.g., Banking, Healthcare), return the BUSINESS DOMAIN — NOT IT.
3. Use STANDARDIZED domain names from the allowed list below.
4. Do NOT invent, assume, or guess domains.
5. If no reasonable inference is possible, return null.

IMPORTANT DISTINCTIONS:
- "Information Technology", "Software & SaaS", "Cloud", "Cybersecurity", "AI", "Data & Analytics"
  → Use ONLY if the candidate worked on IT PRODUCTS or IT COMPANIES.
  → DO NOT use these if they are merely skills applied within another industry.

ANTI-HALLUCINATION RULES:
- Never infer domain from skills alone.
- Never infer domain from education or certifications.
- Never assume domain based on job title alone.
- Never combine multiple domains.
- Never return explanations or extra text.

ALLOWED SAMPLE DOMAIN LIST (STANDARDIZED OUTPUT VALUES):

Healthcare  
Healthcare & Life Sciences  
Pharmaceuticals & Clinical Research  

Banking  
Finance  
Finance & Accounting  
Banking, Financial Services & Insurance (BFSI)  
Insurance  
Capital Markets  
FinTech  

Retail  
E-Commerce  
Retail & E-Commerce  

Manufacturing  
Manufacturing & Production  
Supply Chain  
Operations & Supply Chain Management  
Logistics  
Logistics & Transportation  
Procurement & Vendor Management  

Education  
Education, Training & Learning  

Government  
Public Sector  
Public Sector & Government Services  
Defense  

Energy  
Utilities  
Energy, Utilities & Sustainability  

Telecommunications  
Media & Entertainment  
Media, Advertising & Communications  
Gaming  

Real Estate  
Real Estate & Facilities Management  

Construction  
Construction & Infrastructure  

Hospitality  
Travel & Tourism  

Agriculture  
Agri-Business  

Legal, Risk & Corporate Governance  
Quality, Compliance & Audit  

Human Resources  
Sales & Marketing  
Customer Service & Customer Experience  
Administration & Office Management  

Non-Profit  
NGOs, Social Impact & CSR  

Software & SaaS  
Cloud & Infrastructure  
Cybersecurity  
Information Technology  

Artificial Intelligence  
→ ONLY if candidate worked on AI products or AI companies (NOT skills)

Data & Analytics  
→ ONLY if candidate worked on data/analytics companies or platforms (NOT skills)

OUTPUT REQUIREMENTS:
- Output ONLY valid JSON
- No explanations
- No markdown
- No comments
- Exactly ONE domain or null

JSON SCHEMA:
{
  "domain": "string | null"
}

VALID OUTPUT EXAMPLES:
{"domain": "Healthcare"}
{"domain": "Banking"}
{"domain": "Finance"}
{"domain": "Banking, Financial Services & Insurance (BFSI)"}
{"domain": "Insurance"}
{"domain": "Capital Markets"}
{"domain": "FinTech"}
{"domain": "Retail"}
{"domain": "E-Commerce"}
{"domain": "Manufacturing"}
{"domain": "Supply Chain"}
{"domain": "Logistics"}
{"domain": "Transportation"}
{"domain": "Education"}
{"domain": "Government"}
{"domain": "Public Sector"}
{"domain": "Defense"}
{"domain": "Energy"}
{"domain": "Utilities"}
{"domain": "Telecommunications"}
{"domain": "Media & Entertainment"}
{"domain": "Gaming"}
{"domain": "Real Estate"}
{"domain": "Construction"}
{"domain": "Hospitality"}
{"domain": "Travel & Tourism"}
{"domain": "Automotive"}
{"domain": "Aerospace"}
{"domain": "Electronics & Semiconductors"}
{"domain": "Mining"}
{"domain": "Metals"}
{"domain": "Agriculture"}
{"domain": "AgriTech"}
{"domain": "Non-Profit"}
{"domain": "Software & SaaS"}
{"domain": "Cloud & Infrastructure"}
{"domain": "Cybersecurity"}
{"domain": "Information Technology"}
{"domain": "Artificial Intelligence"}
{"domain": "Data & Analytics"}
{"domain": null}
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
    ]
    
    def __init__(self):
        self.ollama_host = settings.ollama_host
        self.model = "llama3.1"
    
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
        date_patterns = [
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\.?\s+(\d{4})\b',
            r'\b(\d{1,2})/(\d{4})\b',  # MM/YYYY
            r'\b(\d{4})-(\d{1,2})\b',  # YYYY-MM
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
            
            # Fix #1: Split multi-role blocks - prefer last role paragraph within same company
            # This prevents old domain language from contaminating latest domain
            # Split on double newlines (paragraph breaks) and take first paragraph
            # This typically contains the most recent role
            if '\n\n' in latest_experience:
                paragraphs = latest_experience.split('\n\n')
                # Take first paragraph (usually most recent role)
                latest_experience = paragraphs[0]
                logger.debug(
                    f"Multi-role block detected, using first paragraph (most recent role)",
                    extra={"original_paragraphs": len(paragraphs)}
                )
            
            # Limit to 3000 characters for LLM context
            if len(latest_experience) > 3000:
                latest_experience = latest_experience[:3000]
                logger.debug(
                    f"Latest experience truncated to 3000 characters",
                    extra={"original_length": len(experience_blocks_with_scores[0][0])}
                )
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
        Fallback method to detect domain from keywords when LLM returns null.
        This is a safety net to catch obvious domain indicators that LLM might miss.
        Uses strict scoring to ensure only resume-related domains are detected.
        Filters out education sections to prevent false Education domain detection.
        
        Args:
            resume_text: The resume text to analyze
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
                    "energy", "power", "utilities", "electric utility", "renewable energy",
                    "solar energy", "wind energy", "oil and gas", "petroleum", "energy sector",
                    "power generation", "energy management", "utility company"
                ],
                "medium": [
                    "energy", "power", "utilities", "electric", "renewable", "solar",
                    "wind", "oil", "gas", "petroleum", "energy"
                ],
                "low": []  # Removed abstract keywords: power, energy
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
        
        # Return domain with highest score if score is significant
        if domain_scores:
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
            
            # Extract latest experience section (experience-first approach)
            latest_experience = self._extract_latest_experience(resume_text)
            
            # Use latest experience (up to 3000 characters) for LLM context
            # This ensures we prioritize the most recent domain
            text_to_send = latest_experience[:3000]
            if len(latest_experience) > 3000:
                logger.debug(
                    f"Latest experience truncated from {len(latest_experience)} to 3000 characters for domain extraction",
                    extra={"file_name": filename, "original_length": len(latest_experience)}
                )
            
            prompt = f"""{DOMAIN_PROMPT}

IMPORTANT CONTEXT:
- The text below contains the MOST RECENT work experience section
- Prioritize the domain from this latest experience
- If domain is unclear, return null (acceptable)

Input resume text (latest experience):
{text_to_send}

Output (JSON only, no other text, no explanations):"""
            
            logger.info(
                f"📤 CALLING OLLAMA API for domain extraction",
                extra={
                    "file_name": filename,
                    "model": model_to_use,
                    "ollama_host": self.ollama_host,
                    "resume_text_length": len(resume_text),
                    "text_sent_length": len(text_to_send),
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
            domain = parsed_data.get("domain")
            
            # Handle both None and string "null" cases
            if domain is not None:
                domain = str(domain).strip()
                if not domain or domain.lower() in ["null", "none", "nil", ""]:
                    domain = None
            else:
                domain = None
            
            # FALLBACK LAYER 1: Strict keyword-based domain detection (ATS-grade)
            # Only use latest experience for fallback to maintain "latest domain" priority
            if not domain and latest_experience:
                domain = self._detect_domain_from_keywords(latest_experience, filename)
                if domain:
                    logger.info(
                        f"✅ FALLBACK LAYER 1: Domain detected from keywords (latest experience): {domain} for {filename}",
                        extra={"file_name": filename, "detected_domain": domain, "method": "strict_keyword_fallback"}
                    )
            
            # FALLBACK LAYER 2: Conservative job title inference (only if Layer 1 fails)
            # Only use latest experience to maintain "latest domain" priority
            if not domain and latest_experience:
                domain = self._infer_domain_from_job_titles(latest_experience, filename)
                if domain:
                    logger.info(
                        f"✅ FALLBACK LAYER 2: Domain inferred from job titles (latest experience): {domain} for {filename}",
                        extra={"file_name": filename, "detected_domain": domain, "method": "job_title_inference"}
                    )
            
            # Log the raw response for debugging (enhanced for troubleshooting)
            logger.info(
                f"🔍 DOMAIN EXTRACTION DEBUG for {filename}",
                extra={
                    "file_name": filename,
                    "raw_output_hash": hash(raw_output[:1000]) if raw_output else None,
                    "raw_output_length": len(raw_output) if raw_output else 0,
                    "parsed_data": parsed_data,
                    "extracted_domain": domain,
                    "resume_text_length": len(resume_text),
                    "text_sent_length": len(text_to_send),
                    "resume_text_hash": hash(resume_text[:1000]) if resume_text else None
                }
            )
            
            # If domain is null, log details (null is acceptable for ATS-grade systems)
            if not domain:
                logger.info(
                    f"ℹ️ DOMAIN EXTRACTION RETURNED NULL for {filename} (acceptable - unclear domain)",
                    extra={
                        "file_name": filename,
                        "resume_text_length": len(resume_text),
                        "latest_experience_length": len(latest_experience),
                        "text_sent_length": len(text_to_send),
                        "raw_output_hash": hash(raw_output[:2000]) if raw_output else None,
                        "parsed_data": parsed_data,
                        "latest_experience_hash": hash(latest_experience[:1000]) if latest_experience else None,
                        "note": "Null is acceptable when domain is unclear (ATS-grade behavior)"
                    }
                )
            
            logger.info(
                f"✅ DOMAIN EXTRACTED from {filename}",
                extra={
                    "file_name": filename,
                    "domain": domain,
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

