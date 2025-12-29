"""Service for extracting master category (IT/NON_IT) from resumes using OLLAMA LLM."""
import json
import re
from typing import Optional
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

MASTERCATEGORY_PROMPT = """IMPORTANT: This is a FRESH, ISOLATED classification task.
Ignore all prior context, memory, or previous conversations.

ROLE:
You are an Enterprise ATS Domain Classification Gateway.

Your sole responsibility is to determine whether a candidate profile
belongs to the IT domain or the NON-IT domain.

CONTEXT:
- Resume content may be unstructured, partial, or inconsistently formatted.
- Decisions must be made using ONLY the provided resume text.
- Do NOT infer intent, career aspirations, or future roles.
- Do NOT normalize, reinterpret, or guess missing information.

INPUT SCOPE:
- You are provided with the first 1000 characters of resume text.

MASTER DOMAIN DEFINITIONS:

IT DOMAIN includes (but is not limited to):
- Full Stack Development (Java, Python, .NET)
- Programming & Scripting
- Databases & Data Technologies
- Cloud Platforms (Azure, AWS)
- DevOps & Platform Engineering
- Artificial Intelligence & Machine Learning
- Generative AI & Large Language Models
- Data Science
- Data Analysis & Business Intelligence
- Networking & Security
- Software Tools & Platforms
- Methodologies & Practices (Agile, DevOps, SDLC)
- Web & Mobile Development
- Microsoft Dynamics & Power Platform
- SAP Ecosystem
- Salesforce Ecosystem
- ERP Systems
- IT Business Analysis
- IT Project / Program Management

NON-IT DOMAIN includes (but is not limited to):
- Business & Management
- Finance & Accounting
- Banking, Financial Services & Insurance (BFSI)
- Sales & Marketing
- Human Resources (HR)
- Operations & Supply Chain Management
- Procurement & Vendor Management
- Manufacturing & Production
- Quality, Compliance & Audit
- Project Management (Non-IT)
- Strategy & Consulting
- Entrepreneurship & Startups
- Education, Training & Learning
- Healthcare & Life Sciences
- Pharmaceuticals & Clinical Research
- Retail & E-Commerce (Non-Tech)
- Logistics & Transportation
- Real Estate & Facilities Management
- Construction & Infrastructure
- Energy, Utilities & Sustainability
- Agriculture & Agri-Business
- Hospitality, Travel & Tourism
- Media, Advertising & Communications
- Legal, Risk & Corporate Governance
- Public Sector & Government Services
- NGOs, Social Impact & CSR
- Customer Service & Customer Experience
- Administration & Office Management
- Product Management (Business / Functional)
- Data, Analytics & Decision Sciences (Non-Technical)

TASK:
Determine whether the resume belongs to IT or NON-IT domain.

CLASSIFICATION RULES (STRICT):


1. Explicit IT Technical Indicators:
   - Programming languages, frameworks, databases, cloud platforms,
     DevOps tools, AI/ML, ERP technical platforms, or software systems.

2. IT Job Titles or Roles:
   - Developer, Engineer, Architect, Data Scientist, Data Engineer,
     DevOps Engineer, Cloud Engineer, QA / Automation,
     Business Analyst (IT), IT Project / Program Manager,
     AI / ML / GenAI roles.

3. IT Work Descriptions:
   - Designing, developing, coding, configuring, deploying,
     integrating, automating, optimizing, debugging, maintaining
     software systems, infrastructure, platforms, or applications.

IMPORTANT EXCLUSIONS:
- Ignore generic management, coordination, sales, HR, finance,
  operations, teaching, consulting, or customer service content
  unless explicitly tied to IT systems or platforms.
- Do NOT treat tools like Excel, PowerPoint, or basic reporting
  as IT indicators unless linked to technical platforms or systems.

DECISION LOGIC (HARD STOP):
- If ANY IT indicator is detected:
  - STOP further analysis immediately
  - Classify as IT

- If NO IT indicator is detected:
  - Classify as NON_IT

OUTPUT RULES (ABSOLUTE):
- Output exactly ONE line
- No explanations, no reasoning, no metadata

ALLOWED OUTPUTS ONLY:
- NAVIGATE_TO_IT_SKILLS_EXTRACTION
- NAVIGATE_TO_NON_IT_SKILLS_EXTRACTION"""


class MasterCategoryExtractor:
    """Service for extracting master category (IT/NON_IT) from resume text using OLLAMA LLM."""
    
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
    
    def _parse_mastercategory(self, text: str) -> str:
        """
        Parse mastercategory from LLM response.
        
        Args:
            text: Raw response from LLM
            
        Returns:
            "IT" or "NON_IT" or "NON_IT" as default
        """
        if not text:
            logger.warning("Empty response from LLM for mastercategory classification")
            return "NON_IT"  # Default to NON_IT
        
        cleaned_text = text.strip().upper()
        
        # Look for the navigation commands
        if "NAVIGATE_TO_IT_SKILLS_EXTRACTION" in cleaned_text:
            return "IT"
        elif "NAVIGATE_TO_NON_IT_SKILLS_EXTRACTION" in cleaned_text:
            return "NON_IT"
        
        # Fallback: Look for IT or NON_IT keywords
        if "IT" in cleaned_text and "NON" not in cleaned_text:
            return "IT"
        
        # Default to NON_IT if unclear
        return "NON_IT"
    
    async def extract_mastercategory(self, resume_text: str, filename: str = "resume") -> str:
        """
        Extract master category (IT or NON_IT) from resume text using OLLAMA LLM.
        
        Args:
            resume_text: The text content of the resume
            filename: Name of the resume file (for logging)
        
        Returns:
            "IT" or "NON_IT"
        """
        try:
            is_connected, available_model = await self._check_ollama_connection()
            if not is_connected:
                logger.warning(
                    "OLLAMA not accessible for mastercategory classification, defaulting to NON_IT",
                    extra={"file_name": filename}
                )
                return "NON_IT"
            
            model_to_use = self.model
            if available_model and "llama3.1" not in available_model.lower():
                model_to_use = available_model
            
            # Use first 1000 characters as per prompt
            text_to_send = resume_text[:1000]
            prompt = f"""{MASTERCATEGORY_PROMPT}

Input resume text:
{text_to_send}

Output (one line only, no explanations):"""
            
            logger.info(
                "[MASTERCATEGORY] Classifying master category",
                extra={
                    "file_name": filename,
                    "model": model_to_use,
                    "resume_text_length": len(resume_text)
                }
            )
            
            result = None
            async with httpx.AsyncClient(timeout=Timeout(300.0)) as client:
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
                    if e.response.status_code == 404:
                        # Try /api/chat endpoint
                        response = await client.post(
                            f"{self.ollama_host}/api/chat",
                            json={
                                "model": model_to_use,
                                "messages": [
                                    {"role": "system", "content": "You are a fresh, isolated classification agent. This is a new, independent task with no previous context."},
                                    {"role": "user", "content": prompt}
                                ],
                                "stream": False,
                                "options": {
                                    "temperature": 0.1,
                                    "top_p": 0.9,
                                    "num_predict": 50,  # Short response for classification
                                }
                            }
                        )
                        response.raise_for_status()
                        result = response.json()
                        if "message" in result and "content" in result["message"]:
                            result = {"response": result["message"]["content"]}
                    else:
                        raise
            
            raw_output = ""
            if isinstance(result, dict):
                raw_output = str(result.get("response", "") or result.get("text", "") or result.get("content", ""))
            else:
                raw_output = str(result)
            
            mastercategory = self._parse_mastercategory(raw_output)
            
            logger.info(
                "[MASTERCATEGORY] Master category classified",
                extra={
                    "file_name": filename,
                    "mastercategory": mastercategory,
                    "raw_response": raw_output[:200]
                }
            )
            
            return mastercategory
            
        except Exception as e:
            logger.error(
                f"Master category classification failed, defaulting to NON_IT: {e}",
                extra={"file_name": filename, "error": str(e)}
            )
            return "NON_IT"

