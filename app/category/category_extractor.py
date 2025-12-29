"""Service for extracting category from resumes using OLLAMA LLM based on mastercategory."""
import json
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

IT_CATEGORY_PROMPT = """IMPORTANT: This is a FRESH, ISOLATED extraction task.
Ignore all prior context, memory, or previous conversations.

ROLE:
You are an Enterprise ATS IT Skills Categorization Engine.

OBJECTIVE:
Assess the candidate profile content and determine the MOST APPROPRIATE IT CATEGORY
that best represents the candidate's primary technical specialization.

CONTEXT:
- The profile has ALREADY been classified as IT.
- Resume content may be unstructured, partial, or inconsistently formatted.
- The category selection MUST be based on a reasoned assessment of the profile content.
- Use ONLY the information explicitly present in the resume text.
- Do NOT infer future intent, aspirational roles, or missing details.
- Do NOT normalize, merge, or invent new categories.

INPUT SCOPE:
- You are provided with the first 1000 characters of resume text.

SAMPLE IT CATEGORIES:

1. Full Stack Development (Java)
2. Full Stack Development (Python)
3. Full Stack Development (.NET)
4. Programming & Scripting
5. Databases & Data Technologies
6. Cloud Platforms (Azure)
7. Cloud Platforms (AWS)
8. DevOps & Platform Engineering
9. Artificial Intelligence & Machine Learning
10. Generative AI & Large Language Models
11. Data Science
12. Data Analysis & Business Intelligence
13. Networking & Security
14. Software Tools & Platforms
15. Methodologies & Practices (Agile, DevOps, SDLC)
16. Web & Mobile Development
17. Microsoft Dynamics & Power Platform
18. SAP Ecosystem
19. Salesforce Ecosystem
20. ERP Systems
21. IT Business Analysis
22. IT Project / Program Management

ASSESSMENT & SELECTION RULES (STRICT):

1. Evaluate the profile holistically within the allowed input scope and
   determine which IT category MOST ACCURATELY reflects the candidate's
   core technical work and repeated themes.

2. If a clearly identifiable technology stack or platform dominates,
   select the category that best aligns with that stack.

3. If AI/ML, GenAI, LLMs, NLP, embeddings, agents, or model workflows are present,
   select the MOST SPECIFIC applicable AI-related category.

4. If multiple IT areas appear, prioritize:
   - Frequency of mention
   - Depth of responsibility
   - Centrality to the role description

5. If the profile is technical but broadly defined,
   select "Programming & Scripting".

CONSTRAINTS:
- Select ONLY ONE category.
- The selected category MUST be chosen from the allowed list.
- Do NOT output reasoning, explanation, or additional metadata.

OUTPUT FORMAT (ABSOLUTE):
- Output exactly ONE line.
- Output ONLY the category name."""

NON_IT_CATEGORY_PROMPT = """IMPORTANT: This is a FRESH, ISOLATED extraction task.
Ignore all prior context, memory, or previous conversations.

ROLE:
You are an Enterprise ATS Non-IT Domain Categorization Engine.

OBJECTIVE:
Assess the candidate profile content and determine the MOST APPROPRIATE NON-IT CATEGORY
that best represents the candidate's primary functional or industry focus.

CONTEXT:
- The profile has ALREADY been classified as NON-IT.
- Resume content may be unstructured, partial, or inconsistently formatted.
- The category selection MUST be based on a reasoned assessment of the profile content.
- Use ONLY the information explicitly stated in the resume text.
- Do NOT reinterpret business roles as technical roles.
- Do NOT infer hidden IT ownership or technical depth.

INPUT SCOPE:
- You are provided with the first 1000 characters of resume text.
 
SAMPLE NON-IT CATEGORIES:

1. Business & Management
2. Finance & Accounting
3. Banking, Financial Services & Insurance (BFSI)
4. Sales & Marketing
5. Human Resources (HR)
6. Operations & Supply Chain Management
7. Procurement & Vendor Management
8. Manufacturing & Production
9. Quality, Compliance & Audit
10. Project Management (Non-IT)
11. Strategy & Consulting
12. Entrepreneurship & Startups
13. Education, Training & Learning
14. Healthcare & Life Sciences
15. Pharmaceuticals & Clinical Research
16. Retail & E-Commerce (Non-Tech)
17. Logistics & Transportation
18. Real Estate & Facilities Management
19. Construction & Infrastructure
20. Energy, Utilities & Sustainability
21. Agriculture & Agri-Business
22. Hospitality, Travel & Tourism
23. Media, Advertising & Communications
24. Legal, Risk & Corporate Governance
25. Public Sector & Government Services
26. NGOs, Social Impact & CSR
27. Customer Service & Customer Experience
28. Administration & Office Management
29. Product Management (Business / Functional)
30. Data, Analytics & Decision Sciences (Non-Technical)

ASSESSMENT & SELECTION RULES (STRICT):

1. Assess the profile holistically and identify the PRIMARY functional or
   industry area explicitly demonstrated through roles, responsibilities,
   and repeated terminology.

2. If industry-specific experience dominates (e.g., BFSI, Healthcare),
   select the most relevant industry category over generic management.

3. If multiple non-IT functions appear,
   select the category that is MOST CENTRAL and REPEATED.

4. If the role focuses on coordination, delivery, governance, or oversight
   without technical system ownership,
   select the closest applicable non-IT category.

CONSTRAINTS:
- Select ONLY ONE category.
- The selected category MUST be chosen from the allowed list.
- Do NOT output reasoning, explanation, or additional metadata.

OUTPUT FORMAT (ABSOLUTE):
- Output exactly ONE line.
- Output ONLY the category name."""


class CategoryExtractor:
    """Service for extracting category from resume text using OLLAMA LLM based on mastercategory."""
    
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
    
    def _parse_category(self, text: str) -> Optional[str]:
        """
        Parse category from LLM response.
        
        Args:
            text: Raw response from LLM
            
        Returns:
            Category string or None if not found
        """
        if not text:
            logger.warning("Empty response from LLM for category classification")
            return None
        
        # Clean the text - take first line only, strip whitespace
        lines = text.strip().split('\n')
        category = lines[0].strip() if lines else ""
        
        # Remove any markdown formatting or quotes
        category = category.strip('"').strip("'").strip('`').strip()
        
        # Remove leading/trailing punctuation
        category = category.strip('.,;:')
        
        if not category:
            return None
        
        return category
    
    async def extract_category(
        self, 
        resume_text: str, 
        mastercategory: str,
        filename: str = "resume"
    ) -> Optional[str]:
        """
        Extract category from resume text based on mastercategory.
        
        Args:
            resume_text: The text content of the resume
            mastercategory: "IT" or "NON_IT"
            filename: Name of the resume file (for logging)
        
        Returns:
            Category string or None if not found
        """
        try:
            # Validate mastercategory
            if mastercategory not in ["IT", "NON_IT"]:
                logger.warning(
                    f"Invalid mastercategory '{mastercategory}', defaulting to NON_IT",
                    extra={"file_name": filename, "mastercategory": mastercategory}
                )
                mastercategory = "NON_IT"
            
            # Select appropriate prompt
            if mastercategory == "IT":
                prompt_template = IT_CATEGORY_PROMPT
            else:
                prompt_template = NON_IT_CATEGORY_PROMPT
            
            is_connected, available_model = await self._check_ollama_connection()
            if not is_connected:
                logger.warning(
                    f"OLLAMA not accessible for category classification, returning None",
                    extra={"file_name": filename, "mastercategory": mastercategory}
                )
                return None
            
            model_to_use = self.model
            if available_model and "llama3.1" not in available_model.lower():
                model_to_use = available_model
            
            # Use first 1000 characters as per prompt
            text_to_send = resume_text[:1000]
            prompt = f"""{prompt_template}

Input resume text:
{text_to_send}

Output (one line only, category name only, no explanations):"""
            
            logger.info(
                "[CATEGORY] Classifying category",
                extra={
                    "file_name": filename,
                    "mastercategory": mastercategory,
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
                                    {"role": "system", "content": "You are a fresh, isolated categorization agent. This is a new, independent task with no previous context."},
                                    {"role": "user", "content": prompt}
                                ],
                                "stream": False,
                                "options": {
                                    "temperature": 0.1,
                                    "top_p": 0.9,
                                    "num_predict": 100,  # Short response for category name
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
            
            category = self._parse_category(raw_output)
            
            logger.info(
                "[CATEGORY] Category classified",
                extra={
                    "file_name": filename,
                    "mastercategory": mastercategory,
                    "category": category,
                    "raw_response": raw_output[:200]
                }
            )
            
            return category
            
        except Exception as e:
            logger.error(
                f"Category classification failed: {e}",
                extra={"file_name": filename, "mastercategory": mastercategory, "error": str(e)}
            )
            return None

