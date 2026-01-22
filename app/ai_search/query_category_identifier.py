"""Service for identifying category from search queries using OLLAMA LLM."""
import json
import re
from typing import Optional, Tuple
import httpx
from httpx import Timeout

from app.config import settings
from app.utils.logging import get_logger

# Timeout settings for category identification
MASTERCATEGORY_TIMEOUT = 600.0  # 600 seconds for mastercategory identification
CATEGORY_TIMEOUT = 600.0  # 600 seconds for category identification

logger = get_logger(__name__)

# Try to import OLLAMA Python client
try:
    import ollama
    OLLAMA_CLIENT_AVAILABLE = True
except ImportError:
    OLLAMA_CLIENT_AVAILABLE = False
    logger.warning("OLLAMA Python client not available, using HTTP API directly")

QUERY_MASTERCATEGORY_PROMPT = """IMPORTANT: This is a FRESH, ISOLATED classification task.
Ignore all prior context, memory, or previous conversations.

ROLE:
You are an Enterprise ATS Domain Classification Gateway for search queries.

TASK:
Determine whether a search query belongs to the IT domain or the NON-IT domain.

CONTEXT:
- Search queries are typically short (job titles, skills, or brief descriptions).
- Examples: "QA Automation Engineer", "Data Analyst", "Business Manager", "Java Developer"
- Decisions must be made using ONLY the provided query text.
- Do NOT infer intent beyond what is explicitly stated.

IT DOMAIN indicators:
- Job titles: Developer, Engineer, Architect, Data Scientist, DevOps Engineer, QA/Automation Engineer, etc.
- IT abbreviations (ALWAYS IT): "QA" (Quality Assurance), "SDET" (Software Development Engineer in Test), "SDE" (Software Development Engineer), "QA Engineer", "Test Engineer", "Automation Engineer"
- Technologies: Programming languages, frameworks, databases, cloud platforms, AI/ML, etc.
- Examples: "Java Developer", "Python Engineer", "Data Analyst", "Cloud Architect", "QA Automation Engineer", "QA", "SDET"

NON-IT DOMAIN indicators:
- Job titles: Manager, Analyst (business), Consultant, Sales, HR, Finance, Operations, etc.
- Functions: Business, Finance, Sales, Marketing, HR, Operations, etc.
- NON-IT Quality roles: "Quality Manager", "Quality Compliance", "Quality Audit", "Compliance Officer" (when NOT related to software testing)
- Examples: "Business Analyst", "Sales Manager", "HR Executive", "Finance Analyst", "Quality Compliance & Audit"

CRITICAL DISTINCTIONS:
- "QA" alone → IT (Quality Assurance in software/testing context)
- "QA Engineer", "QA Automation", "Test QA" → IT
- "Quality Compliance", "Quality Audit", "Compliance & Audit" → NON-IT (regulatory/business compliance, NOT software testing)
- "Data Analyst" → Usually IT (technical data analysis) unless explicitly "Business Data Analyst"
- "Business Analyst" → NON-IT unless explicitly "IT Business Analyst"

CLASSIFICATION RULES:
1. If query contains IT technical terms (Developer, Engineer, Programmer, Architect, DevOps, etc.) → IT
2. If query contains IT abbreviations ("QA", "SDET", "SDE", "QA Engineer", "Test Engineer") → IT
3. If query contains programming languages, frameworks, or technical platforms → IT
4. If query is ambiguous but contains "Analyst" or "Manager" → Consider context
   - "Data Analyst" → IT (technical data analysis)
   - "Business Analyst" → NON-IT unless explicitly "IT Business Analyst"
   - "Quality Manager" → NON-IT (business quality management, NOT software QA)
5. If query contains "Quality Compliance", "Quality Audit", "Compliance & Audit" → NON-IT
6. If no clear IT indicators → NON_IT

OUTPUT RULES (ABSOLUTE):
- Output exactly ONE line
- No explanations, no reasoning

ALLOWED OUTPUTS ONLY:
- IT
- NON_IT"""

QUERY_CATEGORY_PROMPT_IT = """IMPORTANT: This is a FRESH, ISOLATED classification task.
Ignore all prior context, memory, or previous conversations.

ROLE:
You are an Enterprise ATS IT Category Classifier for search queries.

TASK:
Identify the MOST APPROPRIATE IT CATEGORY for the search query.

CONTEXT:
- The query has ALREADY been classified as IT.
- Search queries are typically short (job titles or brief descriptions).
- You may receive the job role/designation and required skills separately for better accuracy.
- Use BOTH role and skills together to identify the most specific category.
- Select the category that BEST matches the query intent.

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

SELECTION RULES:
1. If role AND skills are provided, use BOTH together for most accurate category:
   - Example: Role="QA tester" + Skills=["Selenium"] → "Full Stack Development (Selenium)" or "Programming & Scripting"
   - Example: Role="Developer" + Skills=["Java", "Spring"] → "Full Stack Development (Java)"
   - Example: Role="Data Engineer" + Skills=["Python", "Spark"] → "Data Science" or "Data Analysis & Business Intelligence"
2. Match job title to category (e.g., "QA Automation Engineer" → "Programming & Scripting" or "Web & Mobile Development")
3. Match technologies mentioned to category (e.g., ".NET Developer" → "Full Stack Development (.NET)")
4. Match domain to category (e.g., "Data Analyst" → "Data Analysis & Business Intelligence")
5. If multiple categories fit, select the MOST SPECIFIC one based on role + skills combination.
6. If only role is provided (no skills), use role-based matching.
7. If unclear, select the most general applicable category.

OUTPUT RULES (ABSOLUTE):
- Output exactly ONE line
- Output ONLY the category name from the list above
- No explanations, no reasoning"""

QUERY_CATEGORY_PROMPT_NON_IT = """IMPORTANT: This is a FRESH, ISOLATED classification task.
Ignore all prior context, memory, or previous conversations.

ROLE:
You are an Enterprise ATS Non-IT Category Classifier for search queries.

TASK:
Identify the MOST APPROPRIATE NON-IT CATEGORY for the search query.

CONTEXT:
- The query has ALREADY been classified as NON-IT.
- Search queries are typically short (job titles or brief descriptions).
- You may receive the job role/designation and required skills separately for better accuracy.
- Use BOTH role and skills together to identify the most specific category.
- Select the category that BEST matches the query intent.

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

SELECTION RULES:
1. If role AND skills are provided, use BOTH together for most accurate category:
   - Example: Role="Business Analyst" + Skills=["SAP"] → "Business & Management" or "ERP Systems"
   - Example: Role="Project Manager" + Skills=["Agile", "Scrum"] → "Project Management (Non-IT)"
2. Match job title to category (e.g., "Business Analyst" → "Business & Management")
3. Match function to category (e.g., "HR Manager" → "Human Resources (HR)")
4. Match industry to category (e.g., "Healthcare Analyst" → "Healthcare & Life Sciences")
5. If multiple categories fit, select the MOST SPECIFIC one based on role + skills combination.
6. If only role is provided (no skills), use role-based matching.
7. If unclear, select the most general applicable category.

OUTPUT RULES (ABSOLUTE):
- Output exactly ONE line
- Output ONLY the category name from the list above
- No explanations, no reasoning"""


class QueryCategoryIdentifier:
    """Service for identifying mastercategory and category from search queries."""
    
    def __init__(self):
        self.ollama_host = settings.ollama_host
        self.model = "llama3.1"
    
    async def _check_ollama_connection(self) -> Tuple[bool, Optional[str]]:
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
    
    def _parse_mastercategory(self, text: str) -> Optional[str]:
        """Parse mastercategory from LLM response."""
        if not text:
            return None
        
        cleaned_text = text.strip().upper()
        
        if "IT" in cleaned_text and "NON_IT" not in cleaned_text:
            return "IT"
        elif "NON_IT" in cleaned_text or "NON-IT" in cleaned_text:
            return "NON_IT"
        
        # Check for IT indicators
        it_indicators = ["NAVIGATE_TO_IT", "DEVELOPER", "ENGINEER", "PROGRAMMER", "ARCHITECT"]
        if any(indicator in cleaned_text for indicator in it_indicators):
            return "IT"
        
        return None
    
    def _parse_category(self, text: str) -> Optional[str]:
        """Parse category from LLM response."""
        if not text:
            return None
        
        # Clean and extract category name
        cleaned_text = text.strip()
        
        # Remove common prefixes/suffixes
        cleaned_text = re.sub(r'^(category|output|result):\s*', '', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = cleaned_text.strip()
        
        # If it's a valid category name (contains letters and possibly numbers/parentheses), return it
        if cleaned_text and len(cleaned_text) > 2:
            return cleaned_text
        
        return None
    
    async def identify_mastercategory(self, query: str) -> Optional[str]:
        """
        Identify mastercategory (IT/NON_IT) from search query.
        
        Args:
            query: Search query string
            
        Returns:
            "IT", "NON_IT", or None if identification fails
        """
        try:
            is_connected, available_model = await self._check_ollama_connection()
            if not is_connected:
                logger.warning("OLLAMA not accessible for mastercategory identification")
                return None
            
            model_to_use = self.model
            if available_model and "llama3.1" not in available_model.lower():
                model_to_use = available_model
            
            prompt = f"""{QUERY_MASTERCATEGORY_PROMPT}

Input Query: {query}

Output (one line only, IT or NON_IT):"""
            
            # Try using OLLAMA Python client first
            result = None
            if OLLAMA_CLIENT_AVAILABLE:
                try:
                    import asyncio
                    loop = asyncio.get_event_loop()
                    
                    def _generate():
                        client = ollama.Client(
                            host=self.ollama_host.replace("http://", "").replace("https://", "")
                        )
                        response = client.generate(
                            model=model_to_use,
                            prompt=prompt,
                            options={
                                "temperature": 0.1,
                                "top_p": 0.9,
                            }
                        )
                        return {"response": response.get("response", "")}
                    
                    result = await loop.run_in_executor(None, _generate)
                except Exception as e:
                    logger.warning(f"OLLAMA Python client failed: {e}")
                    result = None
            
            # Fallback to HTTP API
            if result is None:
                async with httpx.AsyncClient(timeout=Timeout(MASTERCATEGORY_TIMEOUT)) as client:
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
                    except httpx.HTTPStatusError as e:
                        if e.response.status_code == 404:
                            # Try /api/chat endpoint
                            try:
                                response = await client.post(
                                    f"{self.ollama_host}/api/chat",
                                    json={
                                        "model": model_to_use,
                                        "messages": [
                                            {"role": "system", "content": QUERY_MASTERCATEGORY_PROMPT},
                                            {"role": "user", "content": query}
                                        ],
                                        "stream": False,
                                        "options": {
                                            "temperature": 0.1,
                                            "top_p": 0.9,
                                        }
                                    }
                                )
                                response.raise_for_status()
                                result = response.json()
                                if "message" in result and "content" in result["message"]:
                                    result = {"response": result["message"]["content"]}
                            except Exception as e2:
                                logger.warning(f"OLLAMA API failed: {e2}")
                                return None
                        else:
                            logger.warning(f"OLLAMA API failed: {e}")
                            return None
            
            if result is None:
                return None
            
            raw_output = result.get("response", "")
            mastercategory = self._parse_mastercategory(raw_output)
            
            if mastercategory:
                logger.info(
                    f"Identified mastercategory from query: {mastercategory}",
                    extra={"query": query, "mastercategory": mastercategory}
                )
            
            return mastercategory
            
        except Exception as e:
            logger.warning(
                f"Failed to identify mastercategory from query: {e}",
                extra={"query": query, "error": str(e)}
            )
            return None
    
    async def identify_category(
        self, 
        query: str, 
        mastercategory: str,
        role: Optional[str] = None,
        skills: Optional[list] = None
    ) -> Optional[str]:
        """
        Identify specific category from search query based on mastercategory.
        Uses role and skills if provided for more accurate category identification.
        
        Args:
            query: Search query string (original query for context)
            mastercategory: "IT" or "NON_IT"
            role: Optional job role/designation extracted from query
            skills: Optional list of skills extracted from query
            
        Returns:
            Category string or None if identification fails
        """
        try:
            # Validate mastercategory
            if mastercategory not in ["IT", "NON_IT"]:
                logger.warning(f"Invalid mastercategory '{mastercategory}' for category identification")
                return None
            
            is_connected, available_model = await self._check_ollama_connection()
            if not is_connected:
                logger.warning("OLLAMA not accessible for category identification")
                return None
            
            model_to_use = self.model
            if available_model and "llama3.1" not in available_model.lower():
                model_to_use = available_model
            
            # Select appropriate prompt
            if mastercategory == "IT":
                prompt_template = QUERY_CATEGORY_PROMPT_IT
            else:
                prompt_template = QUERY_CATEGORY_PROMPT_NON_IT
            
            # Build enhanced input with role and skills if available
            input_parts = [f"Input Query: {query}"]
            
            if role:
                input_parts.append(f"Job Role/Designation: {role}")
            
            if skills and len(skills) > 0:
                # Combine all skills (from must_have_all and must_have_one_of_groups)
                all_skills = []
                for skill in skills:
                    if isinstance(skill, list):
                        all_skills.extend(skill)
                    else:
                        all_skills.append(skill)
                if all_skills:
                    skills_str = ", ".join([str(s) for s in all_skills if s])
                    input_parts.append(f"Required Skills: {skills_str}")
            
            input_text = "\n".join(input_parts)
            
            prompt = f"""{prompt_template}

{input_text}

Output (one line only, category name only):"""
            
            # Try using OLLAMA Python client first
            result = None
            if OLLAMA_CLIENT_AVAILABLE:
                try:
                    import asyncio
                    loop = asyncio.get_event_loop()
                    
                    def _generate():
                        client = ollama.Client(
                            host=self.ollama_host.replace("http://", "").replace("https://", "")
                        )
                        response = client.generate(
                            model=model_to_use,
                            prompt=prompt,
                            options={
                                "temperature": 0.1,
                                "top_p": 0.9,
                            }
                        )
                        return {"response": response.get("response", "")}
                    
                    result = await loop.run_in_executor(None, _generate)
                except Exception as e:
                    logger.warning(f"OLLAMA Python client failed: {e}")
                    result = None
            
            # Fallback to HTTP API
            if result is None:
                async with httpx.AsyncClient(timeout=Timeout(MASTERCATEGORY_TIMEOUT)) as client:
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
                    except httpx.HTTPStatusError as e:
                        if e.response.status_code == 404:
                            # Try /api/chat endpoint
                            try:
                                # Build user message with role and skills if available
                                user_content = query
                                if role:
                                    user_content += f"\nJob Role/Designation: {role}"
                                if skills and len(skills) > 0:
                                    all_skills = []
                                    for skill in skills:
                                        if isinstance(skill, list):
                                            all_skills.extend(skill)
                                        else:
                                            all_skills.append(skill)
                                    if all_skills:
                                        skills_str = ", ".join([str(s) for s in all_skills if s])
                                        user_content += f"\nRequired Skills: {skills_str}"
                                
                                response = await client.post(
                                    f"{self.ollama_host}/api/chat",
                                    json={
                                        "model": model_to_use,
                                        "messages": [
                                            {"role": "system", "content": prompt_template},
                                            {"role": "user", "content": user_content}
                                        ],
                                        "stream": False,
                                        "options": {
                                            "temperature": 0.1,
                                            "top_p": 0.9,
                                        }
                                    }
                                )
                                response.raise_for_status()
                                result = response.json()
                                if "message" in result and "content" in result["message"]:
                                    result = {"response": result["message"]["content"]}
                            except Exception as e2:
                                logger.warning(f"OLLAMA API failed: {e2}")
                                return None
                        else:
                            logger.warning(f"OLLAMA API failed: {e}")
                            return None
            
            if result is None:
                return None
            
            raw_output = result.get("response", "")
            category = self._parse_category(raw_output)
            
            if category:
                logger.info(
                    f"Identified category from query: {category}",
                    extra={
                        "query": query, 
                        "mastercategory": mastercategory, 
                        "category": category,
                        "role": role,
                        "skills": skills
                    }
                )
            
            return category
            
        except Exception as e:
            logger.warning(
                f"Failed to identify category from query: {e}",
                extra={"query": query, "mastercategory": mastercategory, "error": str(e)}
            )
            return None
    
    async def identify_category_from_query(
        self, 
        query: str,
        role: Optional[str] = None,
        skills: Optional[list] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Identify both mastercategory and category from search query.
        Uses role and skills if provided for more accurate category identification.
        
        Args:
            query: Search query string
            role: Optional job role/designation extracted from parsed query
            skills: Optional list of skills extracted from parsed query (can include must_have_all and must_have_one_of_groups)
            
        Returns:
            Tuple of (mastercategory, category) or (None, None) if identification fails
        """
        try:
            # First identify mastercategory
            mastercategory = await self.identify_mastercategory(query)
            
            if not mastercategory:
                logger.info("Could not identify mastercategory from query, skipping category identification")
                return None, None
            
            # Then identify specific category (with role and skills for better accuracy)
            category = await self.identify_category(query, mastercategory, role=role, skills=skills)
            
            return mastercategory, category
            
        except Exception as e:
            logger.warning(
                f"Failed to identify category from query: {e}",
                extra={"query": query, "role": role, "skills": skills, "error": str(e)}
            )
            return None, None

