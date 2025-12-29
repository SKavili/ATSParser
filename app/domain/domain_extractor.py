"""Service for extracting industry domain from resumes using OLLAMA LLM."""
import json
import re
from typing import Dict, Optional
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
IMPORTANT: This is a FRESH, ISOLATED extraction task. Ignore any previous context or conversations.

ROLE:
You are an ATS resume parsing expert specializing in US IT staffing profiles.

CONTEXT:
Candidate profiles and resumes may be unstructured and inconsistently formatted.
Domain refers to the primary industry domain or business domain the candidate has worked in (e.g., Healthcare, Finance, E-commerce, Banking, Insurance, Retail, etc.).

TASK:
Extract the candidate's primary industry domain from the profile text.

SELECTION RULES:
1. Look for domain information in work experience sections, company names, project descriptions, and client names.
2. Identify the industry domain from:
   - Company names (e.g., "Bank of America" → Banking, "Mayo Clinic" → Healthcare)
   - Project descriptions mentioning industry (e.g., "healthcare system", "financial application", "e-commerce platform")
   - Client names or client industries mentioned
   - Industry-specific keywords in job descriptions
3. If multiple domains are mentioned, select the most prominent or recent one.
4. Common domains: Healthcare, Finance, Banking, Insurance, E-commerce, Retail, Manufacturing, Education, Government, etc.
5. Look for industry indicators even if not explicitly stated (e.g., "worked on hospital management system" → Healthcare).

CONSTRAINTS:
- Extract only one primary domain.
- Preserve the domain name exactly as written or use standard industry terminology.
- Return as a string (e.g., "Healthcare", "Finance", "E-commerce").
- Be thorough in searching for domain clues throughout the resume.

ANTI-HALLUCINATION RULES:
- If absolutely no domain information can be inferred from company names, projects, or work descriptions, return null.
- You may infer domain from project descriptions and company context, but not from skills alone.
- Prefer returning a domain if there are reasonable indicators, even if not explicitly stated.

OUTPUT FORMAT:
Return only valid JSON. No additional text. No explanations. No markdown formatting.

JSON SCHEMA:
{
  "domain": "string | null"
}

Example valid outputs:
{"domain": "Healthcare"}
{"domain": "Defence"}
{"domain": "Energy"}
{"domain": "Education"}
{"domain": "Utility"}
{"domain": "Finance"}
{"domain": "Banking"}
{"domain": "Insurance"}
{"domain": "Capital Markets"}
{"domain": "FinTech"}
{"domain": "Public Sector"}
{"domain": "Government"}
{"domain": "Smart Cities"}
{"domain": "Law Enforcement"}
{"domain": "Judiciary"}
{"domain": "Regulatory & Compliance"}
{"domain": "Information Technology"}
{"domain": "Software & SaaS"}
{"domain": "Artificial Intelligence"}
{"domain": "Cybersecurity"}
{"domain": "Cloud & Infrastructure"}
{"domain": "Telecommunications"}
{"domain": "Manufacturing"}
{"domain": "Industrial Automation"}
{"domain": "Automotive"}
{"domain": "Aerospace"}
{"domain": "Electronics & Semiconductors"}
{"domain": "Logistics"}
{"domain": "Transportation"}
{"domain": "Supply Chain"}
{"domain": "Warehousing"}
{"domain": "Procurement"}
{"domain": "Human Resources"}
{"domain": "Finance & Accounting"}
{"domain": "Sales"}
{"domain": "Marketing"}
{"domain": "Customer Support"}
{"domain": "Retail"}
{"domain": "E-Commerce"}
{"domain": "Hospitality"}
{"domain": "Travel & Tourism"}
{"domain": "Media & Entertainment"}
{"domain": "Gaming"}
{"domain": "Real Estate"}
{"domain": "Construction"}
{"domain": "Facilities Management"}
{"domain": "Mining"}
{"domain": "Metals"}
{"domain": "Agriculture"}
{"domain": "AgriTech"}
{"domain": "Environmental Sustainability"}
{"domain": "ESG"}
{"domain": "ClimateTech"}
{"domain": "Non-Profit"}
{"domain": "Social Impact"}
{"domain": "Sports"}
{"domain": "Fitness"}
{"domain": null}
"""


class DomainExtractor:
    """Service for extracting industry domain from resume text using OLLAMA LLM."""
    
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
    
    def _detect_domain_from_keywords(self, resume_text: str, filename: str = "resume") -> Optional[str]:
        """
        Fallback method to detect domain from keywords when LLM returns null.
        This is a safety net to catch obvious domain indicators that LLM might miss.
        Uses strict scoring to ensure only resume-related domains are detected.
        
        Args:
            resume_text: The resume text to analyze
            filename: Name of the file (for logging)
        
        Returns:
            Detected domain string or None if not found
        """
        if not resume_text:
            return None
        
        text_lower = resume_text.lower()
        
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
                    "life sciences", "pharmaceutical", "pharma", "biotech", "biotechnology",
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
                "low": [
                    "financial", "finance", "money", "revenue", "budget"
                ]
            },
            "Finance": {
                "high": [
                    "capital markets", "investment management", "wealth management", "asset management",
                    "private equity", "venture capital", "hedge fund", "financial planning",
                    "financial services", "fintech", "financial technology", "financial consulting"
                ],
                "medium": [
                    "finance", "financial", "accounting", "cpa", "audit", "tax", "treasury",
                    "financial analyst", "financial advisor", "financial reporting", "fp&a"
                ],
                "low": [
                    "accounting", "budget", "revenue"
                ]
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
                "low": [
                    "online", "digital", "web", "internet", "ecommerce"
                ]
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
                "low": [
                    "sales", "customer"
                ]
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
                    "university", "college", "school district", "educational institution",
                    "higher education", "k-12", "academic", "curriculum development", "educational technology"
                ],
                "medium": [
                    "education", "school", "academic", "curriculum", "student", "teacher",
                    "professor", "educational", "learning management", "lms", "edtech"
                ],
                "low": [
                    "learning", "teaching", "training"
                ]
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
                    "information technology", "it services", "it consulting", "software development",
                    "saas", "software as a service", "cloud computing", "cybersecurity", "data center",
                    "it infrastructure", "enterprise software", "software company"
                ],
                "medium": [
                    "it", "software", "technology", "tech", "developer", "programming",
                    "application development", "system administration", "network administration"
                ],
                "low": [
                    "technology", "tech", "software"
                ]
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
                "low": [
                    "network", "communication"
                ]
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
                "low": [
                    "power", "energy"
                ]
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
                "low": [
                    "media", "content", "marketing"
                ]
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
            
            if score > 0:
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
            
            # Very lenient threshold: Accept if we have any matches (to avoid null results)
            # Priority: high matches > medium matches > any matches
            has_high_match = best_data["high_matches"] > 0
            has_medium_match = best_data["medium_matches"] > 0
            has_any_match = best_score > 0
            
            # Very lenient: Accept if we have ANY matches at all (score > 0)
            # This ensures we catch domains even with minimal keyword matches
            # Priority: high > medium > low, but accept any if found
            if best_score > 0:
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
                    f"Keyword detection found {best_domain} but score/confidence too low (score: {best_score})",
                    extra={"file_name": filename, "domain": best_domain, "score": best_score}
                )
        
        # Additional fallback: Try to infer from job titles and roles
        inferred_domain = self._infer_domain_from_job_titles(resume_text, filename)
        if inferred_domain:
            return inferred_domain
        
        # Final attempt: Even if no strong matches, return the domain with highest score if any score > 0
        # This ensures we extract domain for ALL resumes, even with weak indicators
        if domain_scores:
            best_domain = max(domain_scores, key=lambda x: domain_scores[x]["score"])
            best_data = domain_scores[best_domain]
            best_score = best_data["score"]
            
            # If we have ANY score, return it (ultra lenient to avoid null)
            if best_score > 0:
                logger.info(
                    f"✅ Ultra lenient domain detection: {best_domain} (score: {best_score}) - accepting minimal match to avoid null",
                    extra={
                        "file_name": filename,
                        "domain": best_domain,
                        "score": best_score,
                        "high_matches": best_data["high_matches"],
                        "medium_matches": best_data["medium_matches"],
                        "method": "ultra_lenient_fallback"
                    }
                )
                return best_domain
        
        # EXTRA SAFETY: If we have ANY domain with score > 0, return it (even if already checked above)
        # This is a double-check to ensure we never miss a domain
        if domain_scores:
            for domain_name, domain_info in domain_scores.items():
                if domain_info["score"] > 0:
                    logger.info(
                        f"✅ Safety net: Returning {domain_name} (score: {domain_info['score']}) to avoid null",
                        extra={"file_name": filename, "domain": domain_name, "score": domain_info["score"]}
                    )
                    return domain_name
        
        return None
    
    def _detect_domain_with_minimal_threshold(self, resume_text: str, filename: str = "resume") -> Optional[str]:
        """
        Ultimate fallback with minimal threshold - accepts domain with just 1 keyword match.
        This is the last resort to avoid null results.
        
        Args:
            resume_text: The resume text to analyze
            filename: Name of the file (for logging)
        
        Returns:
            Detected domain string or None if not found
        """
        if not resume_text:
            return None
        
        text_lower = resume_text.lower()
        
        # Comprehensive keyword list - check for any domain indicator with more keywords
        domain_indicators = {
            "Healthcare": [
                "healthcare", "health care", "hospital", "clinic", "medical", "epic", "cerner", "ehr", "emr", 
                "clinical", "patient", "physician", "nurse", "healthcare data", "healthcare analytics",
                "population health", "value-based care", "revenue cycle", "health insurance", "medicare", "medicaid"
            ],
            "Information Technology": [
                "software", "developer", "programmer", "engineer", "it", "information technology", "technology", 
                "tech", "coding", "programming", "application", "system", "software development", "it services",
                "cloud computing", "cybersecurity", "data center", "saas", "software as a service"
            ],
            "Finance": [
                "finance", "financial", "accounting", "cpa", "audit", "tax", "investment", "wealth",
                "financial services", "capital markets", "asset management", "financial planning", "fintech"
            ],
            "Banking": [
                "bank", "banking", "loan", "mortgage", "credit", "financial institution", "commercial bank",
                "retail banking", "investment bank", "banking services"
            ],
            "Education": [
                "education", "university", "college", "school", "teacher", "professor", "academic",
                "educational", "curriculum", "higher education", "k-12"
            ],
            "Government": [
                "government", "federal", "state", "public sector", "government agency", "public administration",
                "civil service", "municipal"
            ],
            "Retail": [
                "retail", "store", "merchandising", "retailer", "retail operations", "retail management",
                "point of sale", "inventory management"
            ],
            "Manufacturing": [
                "manufacturing", "production", "factory", "industrial", "manufacturing operations",
                "production planning", "quality control", "assembly line"
            ],
            "Insurance": [
                "insurance", "actuary", "underwriting", "claims", "insurance company", "insurance carrier",
                "life insurance", "health insurance", "property insurance"
            ],
            "E-Commerce": [
                "e-commerce", "ecommerce", "online retail", "online marketplace", "digital commerce",
                "online shopping", "e-commerce platform"
            ],
            "Telecommunications": [
                "telecom", "telecommunications", "wireless", "mobile network", "telecom services",
                "network infrastructure", "5g", "4g"
            ],
            "Energy": [
                "energy", "power", "utilities", "renewable", "renewable energy", "solar", "wind",
                "energy sector", "power generation", "utility company"
            ],
            "Logistics": [
                "logistics", "supply chain", "warehouse", "shipping", "transportation", "freight",
                "logistics management", "distribution", "supply chain management"
            ],
            "Real Estate": [
                "real estate", "property", "realty", "real estate development", "commercial real estate",
                "residential real estate", "property management"
            ],
            "Media & Entertainment": [
                "media", "entertainment", "broadcasting", "television", "film", "publishing",
                "advertising", "marketing", "digital media", "content creation"
            ],
            "Automotive": [
                "automotive", "automobile", "vehicle", "car manufacturer", "auto industry",
                "automotive manufacturing", "automotive engineering"
            ],
            "Aerospace": [
                "aerospace", "aviation", "aircraft", "aerospace manufacturing", "aerospace engineering",
                "defense contractor", "space", "satellite"
            ],
            "Construction": [
                "construction", "contractor", "building", "construction company", "construction management",
                "general contractor", "construction project"
            ],
            "Hospitality": [
                "hospitality", "hotel", "resort", "hospitality management", "hotel management",
                "restaurant", "hospitality industry"
            ],
            "Transportation": [
                "transportation", "transit", "transport", "transportation services", "transportation management",
                "fleet management", "public transportation"
            ]
        }
        
        # Count matches for each domain (even single match counts)
        domain_matches = {}
        for domain, keywords in domain_indicators.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > 0:
                domain_matches[domain] = matches
        
        # Return domain with most matches (even if just 1)
        if domain_matches:
            best_domain = max(domain_matches, key=domain_matches.get)
            best_count = domain_matches[best_domain]
            
            logger.info(
                f"✅ Minimal threshold detection: {best_domain} (matches: {best_count})",
                extra={"file_name": filename, "domain": best_domain, "match_count": best_count}
            )
            return best_domain
        
        # If still nothing, check for "Developer" role and default to IT
        if "developer" in text_lower:
            logger.info(
                f"✅ Minimal threshold: Defaulting to Information Technology (Developer role)",
                extra={"file_name": filename, "domain": "Information Technology", "reason": "developer_role"}
            )
            return "Information Technology"
        
        return None
    
    def _infer_domain_from_job_titles(self, resume_text: str, filename: str = "resume") -> Optional[str]:
        """
        Additional fallback: Infer domain from job titles, roles, and common patterns.
        This is a last-resort method to avoid returning null.
        
        Args:
            resume_text: The resume text to analyze
            filename: Name of the file (for logging)
        
        Returns:
            Inferred domain string or None if not found
        """
        if not resume_text:
            return None
        
        text_lower = resume_text.lower()
        
        # Job title patterns that indicate specific domains
        title_patterns = {
            "Healthcare": [
                "healthcare", "health care", "medical", "clinical", "nurse", "physician", "doctor",
                "healthcare data", "healthcare analytics", "healthcare it", "healthcare consultant",
                "cdo", "chief data officer", "healthcare director", "healthcare manager"
            ],
            "Information Technology": [
                "software engineer", "developer", "programmer", "it", "information technology",
                "tech lead", "technical", "software", "application", "system", "network",
                "data engineer", "data scientist", "data analyst", "cloud", "devops", "sre",
                "java developer", "python developer", ".net developer", "web developer",
                "full stack developer", "frontend developer", "backend developer", "mobile developer"
            ],
            "Finance": [
                "financial", "finance", "accountant", "cpa", "auditor", "treasury", "financial analyst",
                "investment", "wealth", "asset management", "financial advisor"
            ],
            "Banking": [
                "banker", "banking", "loan officer", "credit analyst", "mortgage", "bank"
            ],
            "Education": [
                "teacher", "professor", "educator", "academic", "curriculum", "education",
                "university", "college", "school"
            ],
            "Government": [
                "government", "federal", "state", "municipal", "public sector", "civil service"
            ],
            "Retail": [
                "retail", "store manager", "merchandiser", "retail sales", "store operations"
            ],
            "Manufacturing": [
                "manufacturing", "production", "factory", "industrial", "quality control"
            ]
        }
        
        # Check for job title patterns (more lenient - check all patterns)
        domain_matches = {}
        for domain, patterns in title_patterns.items():
            match_count = sum(1 for pattern in patterns if pattern in text_lower)
            if match_count > 0:
                domain_matches[domain] = match_count
        
        # Return domain with most matches
        if domain_matches:
            best_domain = max(domain_matches, key=domain_matches.get)
            logger.info(
                f"✅ Domain inferred from job title/role: {best_domain} (matches: {domain_matches[best_domain]})",
                extra={"file_name": filename, "domain": best_domain, "match_count": domain_matches[best_domain]}
            )
            return best_domain
        
        # Last resort: Check if it's clearly a tech/IT resume
        tech_keywords = ["software", "developer", "programmer", "engineer", "it", "technology", 
                        "data engineer", "data scientist", "cloud", "devops", "api", "database",
                        "java", "python", "javascript", "sql", "aws", "azure", "gcp", "coding",
                        "programming", "application", "system", "technical", "tech"]
        tech_count = sum(1 for keyword in tech_keywords if keyword in text_lower)
        
        # More lenient: Accept if 1+ tech keywords found (especially "developer")
        if tech_count >= 1 or "developer" in text_lower:
            logger.info(
                f"✅ Domain inferred as Information Technology from tech keywords (count: {tech_count})",
                extra={"file_name": filename, "domain": "Information Technology", "tech_keyword_count": tech_count}
            )
            return "Information Technology"
        
        return None
    
    def _infer_domain_from_general_patterns(self, resume_text: str, filename: str = "resume") -> Optional[str]:
        """
        Final fallback: Infer domain from general patterns, skills, and common industry indicators.
        This is the last resort to avoid returning null.
        
        Args:
            resume_text: The resume text to analyze
            filename: Name of the file (for logging)
        
        Returns:
            Inferred domain string or None if not found
        """
        if not resume_text:
            return None
        
        text_lower = resume_text.lower()
        
        # Check for very common domain indicators (even if not in main keywords)
        # These are patterns that strongly suggest a domain
        
        # Healthcare indicators (more lenient - accept with 1+ indicator)
        healthcare_indicators = sum(1 for indicator in [
            "patient", "clinical", "medical", "health", "hospital", "clinic",
            "epic", "cerner", "ehr", "emr", "hipaa", "medicare", "medicaid",
            "healthcare", "health care", "physician", "nurse", "healthcare data",
            "healthcare analytics", "population health", "value-based care"
        ] if indicator in text_lower)
        
        if healthcare_indicators >= 1:  # More lenient: accept with just 1 indicator
            logger.info(
                f"✅ Domain inferred as Healthcare from general patterns (indicators: {healthcare_indicators})",
                extra={"file_name": filename, "domain": "Healthcare", "indicator_count": healthcare_indicators}
            )
            return "Healthcare"
        
        # IT/Tech indicators (more lenient - accept with 1+ indicator)
        tech_indicators = sum(1 for indicator in [
            "software", "developer", "programmer", "engineer", "it", "technology",
            "java", "python", "javascript", "sql", "aws", "azure", "gcp", "cloud",
            "api", "database", "devops", "sre", "data engineer", "data scientist",
            "coding", "programming", "application", "system", "technical", "tech"
        ] if indicator in text_lower)
        
        if tech_indicators >= 1:  # More lenient: accept with just 1 indicator
            logger.info(
                f"✅ Domain inferred as Information Technology from general patterns (indicators: {tech_indicators})",
                extra={"file_name": filename, "domain": "Information Technology", "indicator_count": tech_indicators}
            )
            return "Information Technology"
        
        # Finance indicators (more lenient - accept with 1+ indicator)
        finance_indicators = sum(1 for indicator in [
            "financial", "finance", "accounting", "cpa", "audit", "tax", "treasury",
            "investment", "wealth", "asset management", "capital markets", "financial analyst"
        ] if indicator in text_lower)
        
        if finance_indicators >= 1:  # More lenient: accept with just 1 indicator
            logger.info(
                f"✅ Domain inferred as Finance from general patterns (indicators: {finance_indicators})",
                extra={"file_name": filename, "domain": "Finance", "indicator_count": finance_indicators}
            )
            return "Finance"
        
        # Banking indicators
        if any(indicator in text_lower for indicator in [
            "bank", "banking", "loan", "mortgage", "credit", "deposit", "teller"
        ]):
            logger.info(
                f"✅ Domain inferred as Banking from general patterns",
                extra={"file_name": filename, "domain": "Banking"}
            )
            return "Banking"
        
        # Education indicators
        if any(indicator in text_lower for indicator in [
            "university", "college", "school", "teacher", "professor", "academic", "curriculum"
        ]):
            logger.info(
                f"✅ Domain inferred as Education from general patterns",
                extra={"file_name": filename, "domain": "Education"}
            )
            return "Education"
        
        # Final comprehensive check - try all domains one more time with minimal threshold
        # This ensures we catch domains even with very weak indicators
        all_domain_keywords = {
            "Healthcare": ["healthcare", "health care", "hospital", "clinic", "medical", "clinical", "patient", "epic", "cerner"],
            "Information Technology": ["software", "developer", "programmer", "engineer", "it", "technology", "tech", "coding"],
            "Finance": ["finance", "financial", "accounting", "cpa", "audit", "tax", "investment"],
            "Banking": ["bank", "banking", "loan", "mortgage", "credit"],
            "Education": ["education", "university", "college", "school", "teacher", "professor"],
            "Government": ["government", "federal", "state", "public sector"],
            "Retail": ["retail", "store", "merchandising"],
            "Manufacturing": ["manufacturing", "production", "factory"],
            "Insurance": ["insurance", "actuary", "underwriting"],
            "E-Commerce": ["e-commerce", "ecommerce", "online retail"],
            "Telecommunications": ["telecom", "telecommunications", "wireless"],
            "Energy": ["energy", "power", "utilities"],
            "Logistics": ["logistics", "supply chain", "warehouse"],
            "Real Estate": ["real estate", "property", "realty"],
            "Media & Entertainment": ["media", "entertainment", "broadcasting"],
            "Automotive": ["automotive", "automobile", "vehicle"],
            "Aerospace": ["aerospace", "aviation", "aircraft"],
            "Construction": ["construction", "contractor", "building"],
            "Hospitality": ["hospitality", "hotel", "resort"],
            "Transportation": ["transportation", "transit", "transport"]
        }
        
        # Check each domain - return first match
        for domain, keywords in all_domain_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                logger.info(
                    f"✅ Domain inferred from comprehensive check: {domain}",
                    extra={"file_name": filename, "domain": domain}
                )
                return domain
        
        # Absolute final fallback: If resume has any content, default to IT (most common for tech resumes)
        if len(resume_text.strip()) > 100:
            if any(keyword in text_lower for keyword in ["developer", "software", "engineer", "programmer", "it", "tech", "technical", "system", "application"]):
                logger.info(
                    f"✅ Domain defaulted to Information Technology (final fallback - tech content found)",
                    extra={"file_name": filename, "domain": "Information Technology", "method": "final_fallback"}
                )
                return "Information Technology"
        
        return None
    
    def _comprehensive_domain_detection(self, resume_text: str, filename: str = "resume") -> Optional[str]:
        """
        Comprehensive domain detection - absolute last resort to avoid null.
        Checks for ANY domain indicator with minimal threshold.
        
        Args:
            resume_text: The resume text to analyze
            filename: Name of the file (for logging)
        
        Returns:
            Detected domain string or None if not found
        """
        if not resume_text or len(resume_text.strip()) < 50:
            return None
        
        text_lower = resume_text.lower()
        
        # Comprehensive domain keyword list - single keyword match is enough
        domain_checks = {
            "Healthcare": ["healthcare", "health care", "hospital", "clinic", "medical", "clinical", "patient", "epic", "cerner", "ehr", "emr", "physician", "nurse", "healthcare data", "healthcare analytics"],
            "Information Technology": ["software", "developer", "programmer", "engineer", "it", "technology", "tech", "coding", "programming", "application", "system", "technical", "data engineer", "data scientist", "cloud", "api", "database"],
            "Finance": ["finance", "financial", "accounting", "cpa", "audit", "tax", "investment", "wealth", "asset management", "capital markets"],
            "Banking": ["bank", "banking", "loan", "mortgage", "credit", "teller", "deposit"],
            "Education": ["education", "university", "college", "school", "teacher", "professor", "academic", "curriculum"],
            "Government": ["government", "federal", "state", "municipal", "public sector", "civil service"],
            "Retail": ["retail", "store", "merchandising", "retailer"],
            "Manufacturing": ["manufacturing", "production", "factory", "industrial"],
            "Insurance": ["insurance", "actuary", "underwriting", "claims"],
            "E-Commerce": ["e-commerce", "ecommerce", "online retail", "marketplace"],
            "Telecommunications": ["telecom", "telecommunications", "wireless", "mobile network"],
            "Energy": ["energy", "power", "utilities", "renewable"],
            "Logistics": ["logistics", "supply chain", "warehouse", "shipping"],
            "Real Estate": ["real estate", "property", "realty"],
            "Media & Entertainment": ["media", "entertainment", "broadcasting", "television"],
            "Automotive": ["automotive", "automobile", "vehicle"],
            "Aerospace": ["aerospace", "aviation", "aircraft"],
            "Construction": ["construction", "contractor", "building"],
            "Hospitality": ["hospitality", "hotel", "resort"],
            "Transportation": ["transportation", "transit", "transport"]
        }
        
        # Check each domain - return first match found
        for domain, keywords in domain_checks.items():
            if any(keyword in text_lower for keyword in keywords):
                logger.info(
                    f"✅ Comprehensive detection: {domain}",
                    extra={"file_name": filename, "domain": domain, "method": "comprehensive"}
                )
                return domain
        
        # If still nothing and resume has substantial content, default to IT (most common)
        if len(resume_text.strip()) > 200:
            # Check for any tech-related content
            if any(keyword in text_lower for keyword in ["developer", "software", "engineer", "programmer", "it", "tech", "technical", "system", "application"]):
                logger.info(
                    f"✅ Comprehensive detection: Defaulting to Information Technology (tech content found)",
                    extra={"file_name": filename, "domain": "Information Technology", "method": "comprehensive_default"}
                )
                return "Information Technology"
        
        return None
    
    def _aggressive_domain_detection(self, resume_text: str, filename: str = "resume") -> Optional[str]:
        """
        Most aggressive domain detection - scans for ANY domain indicator.
        This is the final safety net to avoid null results.
        
        Args:
            resume_text: The resume text to analyze
            filename: Name of the file (for logging)
        
        Returns:
            Detected domain string or None if not found
        """
        if not resume_text or len(resume_text.strip()) < 30:
            return None
        
        text_lower = resume_text.lower()
        
        # Ultra-aggressive keyword list - single keyword match is enough
        aggressive_keywords = {
            "Healthcare": ["health", "medical", "hospital", "clinic", "patient", "clinical", "epic", "cerner", "ehr", "emr", "physician", "nurse", "medicare", "medicaid"],
            "Information Technology": ["software", "developer", "programmer", "engineer", "it", "technology", "tech", "coding", "programming", "application", "system", "technical", "java", "python", "sql", "aws", "azure", "cloud", "api", "database", "devops"],
            "Finance": ["finance", "financial", "accounting", "cpa", "audit", "tax", "investment", "wealth", "asset", "capital"],
            "Banking": ["bank", "banking", "loan", "mortgage", "credit", "teller", "deposit"],
            "Education": ["education", "university", "college", "school", "teacher", "professor", "academic", "curriculum"],
            "Government": ["government", "federal", "state", "municipal", "public sector"],
            "Retail": ["retail", "store", "merchandising", "retailer"],
            "Manufacturing": ["manufacturing", "production", "factory", "industrial"],
            "Insurance": ["insurance", "actuary", "underwriting", "claims", "policy"],
            "E-Commerce": ["e-commerce", "ecommerce", "online retail", "marketplace"],
            "Telecommunications": ["telecom", "telecommunications", "wireless", "mobile network"],
            "Energy": ["energy", "power", "utilities", "renewable"],
            "Logistics": ["logistics", "supply chain", "warehouse", "shipping"],
            "Real Estate": ["real estate", "property", "realty"],
            "Media & Entertainment": ["media", "entertainment", "broadcasting", "television"],
            "Automotive": ["automotive", "automobile", "vehicle"],
            "Aerospace": ["aerospace", "aviation", "aircraft"],
            "Construction": ["construction", "contractor", "building"],
            "Hospitality": ["hospitality", "hotel", "resort"],
            "Transportation": ["transportation", "transit", "transport"]
        }
        
        # Check each domain - return FIRST match found (most aggressive)
        for domain, keywords in aggressive_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    logger.info(
                        f"✅ Aggressive detection: {domain} (keyword: {keyword})",
                        extra={"file_name": filename, "domain": domain, "keyword": keyword}
                    )
                    return domain
        
        # If still nothing, check for common tech patterns (most common domain)
        if any(word in text_lower for word in ["developer", "software", "engineer", "programmer", "it", "tech", "technical", "system", "application", "code", "coding"]):
            logger.info(
                f"✅ Aggressive detection: Defaulting to Information Technology (tech keywords found)",
                extra={"file_name": filename, "domain": "Information Technology", "method": "tech_default"}
            )
            return "Information Technology"
        
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
                "response_preview": text[:500],
                "response_length": len(text),
                "cleaned_preview": cleaned_text[:500]
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
            
            # Use up to 10000 characters, but log if truncated
            text_to_send = resume_text[:10000]
            if len(resume_text) > 10000:
                logger.debug(
                    f"Resume text truncated from {len(resume_text)} to 10000 characters for domain extraction",
                    extra={"file_name": filename, "original_length": len(resume_text)}
                )
            
            prompt = f"""{DOMAIN_PROMPT}

Input resume text:
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
            
            async with httpx.AsyncClient(timeout=Timeout(600.0)) as client:
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
                    logger.info("✅ Successfully used /api/generate endpoint for domain extraction")
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
            
            # FALLBACK LAYER 1: If LLM returned null, try keyword-based domain detection
            if not domain and resume_text:
                domain = self._detect_domain_from_keywords(resume_text, filename)
                if domain:
                    logger.info(
                        f"✅ FALLBACK LAYER 1: Domain detected from keywords: {domain} for {filename}",
                        extra={"file_name": filename, "detected_domain": domain, "method": "keyword_fallback"}
                    )
            
            # FALLBACK LAYER 2: If still null, try inferring from job titles
            if not domain and resume_text:
                domain = self._infer_domain_from_job_titles(resume_text, filename)
                if domain:
                    logger.info(
                        f"✅ FALLBACK LAYER 2: Domain inferred from job titles: {domain} for {filename}",
                        extra={"file_name": filename, "detected_domain": domain, "method": "job_title_inference"}
                    )
            
            # FALLBACK LAYER 3: Final attempt - try to infer from general patterns and skills
            if not domain and resume_text:
                domain = self._infer_domain_from_general_patterns(resume_text, filename)
                if domain:
                    logger.info(
                        f"✅ FALLBACK LAYER 3: Domain inferred from general patterns: {domain} for {filename}",
                        extra={"file_name": filename, "detected_domain": domain, "method": "general_pattern_inference"}
                    )
            
            # FALLBACK LAYER 4: Ultimate fallback - if still null and resume has content, try one more time with even more lenient rules
            if not domain and resume_text and len(resume_text.strip()) > 100:
                # Try keyword detection one more time with minimal threshold
                domain = self._detect_domain_with_minimal_threshold(resume_text, filename)
                if domain:
                    logger.info(
                        f"✅ FALLBACK LAYER 4: Domain detected with minimal threshold: {domain} for {filename}",
                        extra={"file_name": filename, "detected_domain": domain, "method": "minimal_threshold_fallback"}
                    )
            
            # FALLBACK LAYER 5: Absolute last resort - comprehensive pattern matching for ALL domains
            if not domain and resume_text and len(resume_text.strip()) > 50:
                domain = self._comprehensive_domain_detection(resume_text, filename)
                if domain:
                    logger.info(
                        f"✅ FALLBACK LAYER 5: Domain detected via comprehensive detection: {domain} for {filename}",
                        extra={"file_name": filename, "detected_domain": domain, "method": "comprehensive_detection"}
                    )
            
            # FALLBACK LAYER 6: Final safety net - if resume has ANY content, try one more aggressive keyword scan
            if not domain and resume_text and len(resume_text.strip()) > 30:
                domain = self._aggressive_domain_detection(resume_text, filename)
                if domain:
                    logger.info(
                        f"✅ FALLBACK LAYER 6: Domain detected via aggressive detection: {domain} for {filename}",
                        extra={"file_name": filename, "detected_domain": domain, "method": "aggressive_detection"}
                    )
            
            # Log the raw response for debugging (enhanced for troubleshooting)
            logger.info(
                f"🔍 DOMAIN EXTRACTION DEBUG for {filename}",
                extra={
                    "file_name": filename,
                    "raw_output_preview": raw_output[:1000] if raw_output else "",
                    "raw_output_length": len(raw_output) if raw_output else 0,
                    "parsed_data": parsed_data,
                    "extracted_domain": domain,
                    "resume_text_length": len(resume_text),
                    "text_sent_length": len(text_to_send),
                    "resume_text_preview": resume_text[:500] if resume_text else ""
                }
            )
            
            # If domain is null, log more details for debugging
            if not domain:
                logger.warning(
                    f"⚠️ DOMAIN EXTRACTION RETURNED NULL for {filename}",
                    extra={
                        "file_name": filename,
                        "resume_text_length": len(resume_text),
                        "text_sent_length": len(text_to_send),
                        "raw_output": raw_output[:2000] if raw_output else "",
                        "parsed_data": parsed_data,
                        "resume_text_preview": resume_text[:1000] if resume_text else "",
                        "resume_text_contains_company": "company" in resume_text.lower() or "corporation" in resume_text.lower() or "inc" in resume_text.lower() if resume_text else False,
                        "resume_text_contains_experience": "experience" in resume_text.lower() or "worked" in resume_text.lower() or "project" in resume_text.lower() if resume_text else False
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

