"""Service for extracting skills from resumes using OLLAMA LLM."""
import json
import re
from typing import Dict, List, Optional
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

# ============================================================================
# GATEWAY PROMPT - Profile Classification
# ============================================================================
GATEWAY_PROMPT = """
IMPORTANT: This is a FRESH, ISOLATED classification task. Ignore any previous context or conversations.

ROLE:
You are an ATS resume classification engine.

TASK:
Classify the candidate profile into ONE of the following profile types:
- IT
- NON_IT

If NON_IT, also identify the PRIMARY functional domain.

CLASSIFICATION RULES:
1. Use explicit evidence from the profile text.
2. Do NOT infer based on job titles alone.
3. If technical skills like programming, cloud, databases, DevOps, software engineering, or IT infrastructure are DOMINANT → IT
4. Otherwise → NON_IT

DOMAIN IDENTIFICATION (only for NON_IT):
If NON_IT, identify the primary domain from:
Healthcare, Defence, Energy, Education, Utility, Finance, Banking, Insurance, Capital Markets, FinTech,
Public Sector, Government, Smart Cities, Law Enforcement, Judiciary, Regulatory & Compliance,
Information Technology, Software & SaaS, Artificial Intelligence, Cybersecurity, Cloud & Infrastructure,
Telecommunications, Manufacturing, Industrial Automation, Automotive, Aerospace, Electronics & Semiconductors,
Logistics, Transportation, Supply Chain, Warehousing, Procurement, Human Resources, Finance & Accounting,
Sales, Marketing, Customer Support, Retail, E-Commerce, Hospitality, Travel & Tourism, Media & Entertainment,
Gaming, Real Estate, Construction, Facilities Management, Mining, Metals, Agriculture, AgriTech,
Environmental Sustainability, ESG, ClimateTech, Non-Profit, Social Impact, Sports, Fitness

ANTI-HALLUCINATION RULES:
- Use ONLY explicit evidence from the profile.
- Never guess or infer.
- If uncertain, default to NON_IT with domain as null.

OUTPUT FORMAT:
Return only valid JSON. No additional text. No explanations. No markdown formatting.

JSON SCHEMA:
{
  "profile_type": "IT" | "NON_IT",
  "domain": "string | null"
}

Example valid outputs:
{"profile_type": "IT", "domain": null}
{"profile_type": "NON_IT", "domain": "Healthcare"}
{"profile_type": "NON_IT", "domain": "Real Estate"}
{"profile_type": "NON_IT", "domain": "Insurance"}
{"profile_type": "NON_IT", "domain": null}
"""

# ============================================================================
# IT SKILLS PROMPT
# ============================================================================
IT_SKILLS_PROMPT = """
IMPORTANT: This is a FRESH, ISOLATED extraction task. Ignore any previous context or conversations.

ROLE:
You are an ATS resume parsing expert specializing in IT and technical professional profiles.

CONTEXT:
Candidate profiles and resumes may be unstructured and inconsistently formatted.
Skills refer ONLY to practical, applied, and demonstrable professional capabilities,
technical knowledge areas, tools, techniques, methodologies, or certifications
that a candidate can actively use or perform in IT contexts.

TASK:
Extract ALL IT and technical skills from the profile text.
Focus on skills explicitly mentioned in the profile summary, designation, career objective,
work experience, or anywhere in the profile.
Include and must cover all skills from ALL IT skills categories and return them as a **single combined list**.


Partial list of IT skills categories:
1 Full Stack Development (Java):
   - Backend: Java, Core Java, Java EE, Spring, Spring Boot, Hibernate, Microservices, REST APIs
   - Build & Persistence: Maven, Gradle, JPA, JDBC
   - Frontend: HTML, CSS, JavaScript, React, Angular

2. Full Stack Development (Python):
   - Backend: Python, Django, Flask, FastAPI, Microservices, REST APIs
   - Databases: PostgreSQL, MySQL, MongoDB
   - Frontend: HTML, CSS, JavaScript, React

3. Full Stack Development (.NET):
   - Backend: C#, .NET, .NET Core, ASP.NET, ASP.NET MVC, Web API
   - ORM & Language Features: Entity Framework, LINQ
   - Cloud & UI: Azure, Blazor
   - Frontend: HTML, CSS, JavaScript
  
4- Business Analysis: Requirements Gathering, Stakeholder Management, BRD, FRD, Use Case Modeling, Process Mapping, Gap Analysis, Data Analysis
   - Documentation & Modeling: UML, BPMN, User Stories, Acceptance Criteria

5. Project & Program Management (IT):
   - Project Management: Project Management, Program Management, Agile Project Management, Risk Management, Resource Management, Change Management
   - Tools & Frameworks: Scrum, Sprint Planning, JIRA, MS Project, PMO Processes

1. Programming & Scripting:
   - Languages: Python, Java, C#, C++, JavaScript, TypeScript, Go, Ruby, PHP, R, Scala, Kotlin, Swift
   - Scripting/Automation: Bash, PowerShell, Perl

3. Databases & Data Technologies:
   - RDBMS: MySQL, PostgreSQL, Oracle, SQL Server
   - NoSQL: MongoDB, Cassandra, Redis, DynamoDB
   - Big Data/Analytics: Hadoop, Spark, Kafka, Hive, Presto
   - BI/Visualization: Tableau, Power BI, Looker, Qlik

4.  Azure/Microsoft Azure Cloud:
   - Azure Fundamentals (Entry Level): Azure Basics, Azure Portal, Azure Resource Groups, Azure Virtual Machines, Azure Storage Accounts, Azure Blob Storage, Azure Virtual Network, Azure Load Balancer
   - Azure Compute & Networking: Azure VM Scale Sets, Azure App Service, Azure Functions, Azure Kubernetes Service (AKS), Azure VPN Gateway, Azure Application Gateway, Azure DNS
   - Azure Data & Integration: Azure SQL Database, Azure Cosmos DB, Azure Data Factory, Azure Synapse Analytics, Azure Service Bus, Azure Event Grid
   - Azure Security & Identity: Azure Active Directory, Azure RBAC, Azure Key Vault, Azure Security Center, Azure Defender, Azure MFA
   - Azure DevOps & Automation: Azure DevOps, Azure Pipelines, ARM Templates, Bicep, Azure Automation, CI/CD Pipelines
   - Azure Monitoring & Management: Azure Monitor, Log Analytics, Application Insights, Azure Cost Management
   - Azure Administration & Architecture: Azure Backup, Azure Site Recovery, Azure Governance, Azure Policy, Azure Blueprints, High Availability, Disaster Recovery

5 AWS/Amazon Web Services (AWS) Cloud:
   - AWS Fundamentals (Entry Level): AWS Basics, AWS Management Console, IAM Basics, EC2 Basics, S3 Basics, VPC Basics, AWS Regions and Availability Zones
   - AWS Compute & Networking: Amazon EC2, Auto Scaling, Elastic Load Balancer, Amazon ECS, Amazon EKS, AWS Lambda, Amazon VPC, Route 53
   - AWS Storage & Databases: Amazon S3, S3 Glacier, Amazon EBS, Amazon EFS, Amazon RDS, DynamoDB, Amazon Aurora
   - AWS Security & Identity: AWS IAM, Security Groups, Network ACLs, AWS KMS, AWS Shield, AWS WAF, AWS Secrets Manager
   - AWS DevOps & Automation: AWS CloudFormation, AWS CDK, CodePipeline, CodeBuild, CodeDeploy, CI/CD Pipelines
   - AWS Monitoring & Management: Amazon CloudWatch, AWS CloudTrail, AWS Config, AWS Cost Explorer
   - AWS Administration & Architecture: Backup and Recovery, Fault Tolerance, High Availability, Disaster Recovery, Well-Architected Framework, Multi-Account Strategy

8. DevOps & Platform Engineering:
   - DevOps Fundamentals (Entry Level): DevOps Basics, CI/CD Basics, Version Control, Git, Linux Basics, Shell Scripting, YAML
   - CI/CD & Build Tools: Jenkins, GitHub Actions, GitLab CI, Azure DevOps Pipelines, Bitbucket Pipelines
   - Containerization & Orchestration: Docker, Docker Compose, Kubernetes, Helm, Kubernetes Networking, Kubernetes Security
   - Infrastructure as Code (IaC): Terraform, Ansible, CloudFormation, ARM Templates, Bicep
   - Configuration & Automation: Ansible Playbooks, Chef, Puppet, SaltStack
   - Cloud DevOps Practices: Blue-Green Deployment, Canary Deployment, Rolling Deployment, Auto Scaling
   - Monitoring & Logging: Prometheus, Grafana, ELK Stack, Fluentd, Loki, Datadog, New Relic
   - Security & DevSecOps: DevSecOps, Secrets Management, HashiCorp Vault, SAST, DAST, Container Security, OWASP
   - Reliability & Operations: Site Reliability Engineering (SRE), Incident Management, Root Cause Analysis, High Availability, Disaster Recovery
   - DevOps Administration & Architecture: Platform Engineering, CI/CD Architecture, GitOps, ArgoCD, FluxCD, Scalability, Performance Optimization

9. Artificial Intelligence & Machine Learning:
   - AI/ML Fundamentals (Entry Level): Machine Learning Basics, Supervised Learning, Unsupervised Learning, Feature Engineering, Model Evaluation, Data Preprocessing
   - Machine Learning Algorithms: Linear Regression, Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, XGBoost, Support Vector Machines, K-Means
   - Deep Learning: Neural Networks, Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), LSTM, Transformers
   - ML Frameworks & Libraries: Scikit-learn, TensorFlow, PyTorch, Keras
   - Model Training & Optimization: Hyperparameter Tuning, Cross Validation, Regularization, Model Deployment
   - MLOps & Production: ML Pipelines, Model Versioning, Model Monitoring, MLflow, Kubeflow, Model Serving

10. Generative AI & Large Language Models:
   - Generative AI Fundamentals: Generative AI, Large Language Models (LLMs), Prompt Engineering, In-Context Learning
   - LLM Frameworks & APIs: OpenAI API, Azure OpenAI, Hugging Face Transformers, LangChain, LlamaIndex
   - Text & Language Models: GPT, BERT, T5, LLaMA
   - Image & Multimodal Models: Stable Diffusion, DALL·E, Vision Transformers, Multimodal Models
   - Vector Databases & RAG: Embeddings, Vector Search, FAISS, Pinecone, Weaviate, ChromaDB, Retrieval-Augmented Generation (RAG)
   - Fine-Tuning & Optimization: LoRA, PEFT, Instruction Tuning, Model Quantization
   - GenAI Deployment & Governance: LLM Deployment, Model Monitoring, Prompt Evaluation, AI Safety, Responsible AI

11. Data Science:
   - Data Science Fundamentals (Entry Level): Data Analysis, Exploratory Data Analysis (EDA), Statistics, Probability, Data Cleaning
   - Programming & Libraries: Python, NumPy, Pandas, SciPy, Matplotlib, Seaborn
   - Advanced Analytics & Modeling: Predictive Modeling, Time Series Analysis, Forecasting, Anomaly Detection
   - Big Data & Distributed Computing: Spark, PySpark, Hadoop, Hive
   - Data Visualization: Tableau, Power BI, Plotly
   - Data Science Workflows: Feature Engineering, Model Validation, Experimentation, A/B Testing
   - Data Science Deployment: Model Deployment, API Integration, Data Pipelines

12. Data Analysis & Business Intelligence:
   - Data Analyst Fundamentals (Entry Level): Data Analysis, Data Interpretation, Business Metrics, KPI Tracking
   - Querying & Databases: SQL, Advanced SQL, Joins, Subqueries, Window Functions
   - Data Visualization & Reporting: Power BI, Tableau, Excel Dashboards, Data Storytelling
   - Spreadsheet & Tools: MS Excel, Pivot Tables, Power Query, VBA
   - BI & Reporting Platforms: Looker, Qlik, SSRS
   - Data Governance & Quality: Data Validation, Data Quality Checks, Master Data Management
   - Advanced Data Analysis: Trend Analysis, Cohort Analysis, Root Cause Analysis

6. Networking & Security:
   - Networking: TCP/IP, DNS, DHCP, VPN, Firewalls
   - Security: Penetration Testing, Ethical Hacking, OWASP, CIS Controls, SIEM tools

7. Software Tools & Platforms:
   - Version Control: Git, SVN, Mercurial
   - IDEs & Editors: VS Code, PyCharm, Eclipse, IntelliJ, NetBeans
   - Project Management: Jira, Trello, Confluence

8. Methodologies & Practices:
   - Agile, Scrum, Kanban, DevOps, Test-Driven Development, Continuous Integration/Delivery
   - Software Development Life Cycle (SDLC), ITIL, Six Sigma (if IT-related)

2. Web & Mobile Development:
   - Frontend: HTML, CSS, JavaScript, React, Angular, Vue.js
   - Backend: Node.js, Django, Spring Boot, Flask, Express.js
   - Mobile: Android, iOS, React Native, Flutter, Swift, Kotlin


13. Microsoft Dynamics & Power Platform:
   - Dynamics 365: Microsoft Dynamics 365, Dynamics Business Central, NAV, AL Development, C/AL
   - Power Platform: Power BI, Power Apps, Power Automate, Power Virtual Agents, Dataverse, DAX

14. SAP Ecosystem:
   - SAP Core: SAP, SAP S/4HANA, SAP ECC, SAP HANA
   - SAP Modules: SAP FICO, SAP MM, SAP SD, SAP CRM, SAP BW
   - SAP Technical: SAP ABAP, SAP Basis

15. Salesforce Ecosystem:
   - Salesforce Core: Salesforce, Salesforce CRM, Salesforce Administration
   - Salesforce Development: Apex, Visualforce, Lightning Web Components (LWC)
   - Salesforce Clouds: Sales Cloud, Service Cloud
   - Salesforce Integration: Salesforce Integration


12. ERP Systems:
   - ERP Core: ERP Implementation, ERP Configuration, ERP Integration, ERP Migration, ERP Support
   - ERP Functional Areas: Finance Modules, Supply Chain Modules, Manufacturing Modules



9. Certifications:
   - Include only IT certifications explicitly mentioned (e.g., AWS Certified Solutions Architect, PMP, CCNA, MCSE)




CONSTRAINTS:
- Extract ONLY IT and technical skills.
- Preserve skill names exactly as written (case-sensitive).
- Remove duplicates.
- Limit to a maximum of 50 skills.

ANTI-HALLUCINATION RULES:
- Extract skills ONLY if they are explicitly mentioned in the resume.
- Never guess, infer, or assume skills.
- Do NOT convert topics, events, or titles into skills.
- Do NOT derive skills from job titles or organization names alone.

OUTPUT FORMAT:
Return only valid JSON.
No additional text.
No explanations.
No markdown formatting.

JSON SCHEMA:
{
  "skills": ["skill1", "skill2", "skill3", ...]
}

VALID OUTPUT EXAMPLES:
{"skills": ["Python", "Java", "Spring Boot", "AWS", "Docker", "React", "Agile", "Git", "MySQL"]}
{"skills": ["Azure", "C#", ".NET Core", "SQL Server", "Kubernetes", "CI/CD"]}
"""

# ============================================================================
# NON-IT SKILLS PROMPT (Domain-Aware)
# ============================================================================
def get_non_it_skills_prompt(domain: Optional[str] = None) -> str:
    """
    Generate domain-aware Non-IT skills extraction prompt.
    
    Args:
        domain: The detected domain for the profile (e.g., "Healthcare", "Real Estate")
    
    Returns:
        Prompt string for Non-IT skills extraction
    """
    domain_context = ""
    if domain:
        domain_context = f"""
DOMAIN CONTEXT:
The candidate profile has been classified as: {domain}

Extract ONLY skills relevant to the {domain} domain.
Focus on domain-specific skills, functional skills, tools, certifications, and methodologies
that are relevant to {domain} professionals.

Examples of {domain} domain skills:
"""
        # Add domain-specific examples
        domain_examples = {
            "Healthcare": "Hospital Operations, NABH Compliance, Medical Records Management, Patient Care, HIPAA",
            "Real Estate": "Property Leasing, RERA Compliance, Facility Management, Property Valuation, Real Estate Law",
            "Insurance": "Underwriting, Claims Processing, Policy Administration, Actuarial Analysis, Insurance Regulations",
            "Hospitality": "Front Office Operations, Guest Relations, Revenue Management, Event Planning, Hotel Management",
            "Sales": "Lead Generation, Account Management, Negotiation, CRM, Sales Forecasting",
            "HR": "Recruitment, Payroll, Performance Management, Talent Acquisition, Employee Relations",
            "Finance": "Financial Analysis, Accounting, Budgeting, Financial Reporting, Auditing",
            "Education": "Curriculum Development, Student Assessment, Educational Administration, Teaching Methodologies",
            "Retail": "Store Operations, Inventory Management, Customer Service, Retail Management, Visual Merchandising",
            "Logistics": "Supply Chain Management, Warehouse Operations, Transportation Planning, Inventory Control",
            "Manufacturing": "Production Planning, Quality Control, Lean Manufacturing, Six Sigma, Operations Management"
        }
        
        if domain in domain_examples:
            domain_context += f"- {domain_examples[domain]}"
        else:
            domain_context += f"- Domain-specific skills relevant to {domain}"
    else:
        domain_context = """
Extract professional skills relevant to the candidate's domain and functional area.
Focus on business skills, functional skills, domain knowledge, tools (non-technical), 
methodologies, and certifications relevant to non-IT professionals.
"""

    return f"""
IMPORTANT: This is a FRESH, ISOLATED extraction task. Ignore any previous context or conversations.

ROLE:
You are an ATS resume parsing expert specializing in NON-IT professional profiles.

CONTEXT:
Candidate profiles and resumes may be unstructured and inconsistently formatted.
Skills refer ONLY to practical, applied, and demonstrable professional capabilities,
domain knowledge areas, tools (non-technical), techniques, methodologies, or certifications
that a candidate can actively use or perform in business or functional contexts.
{domain_context}

TASK:
Extract ONLY non-IT professional skills from the profile text.
Focus on skills explicitly mentioned in the profile summary, designation, career objective,
work experience, or anywhere in the profile.

SKILL CATEGORIES TO EXTRACT:
1. Domain Skills: Industry-specific knowledge and capabilities
2. Functional Skills: Business function capabilities (e.g., Sales, Marketing, HR, Operations)
3. Tools (Non-Technical): Business software, CRM, ERP modules, MS Office, etc.
4. Methodologies: Business methodologies, frameworks, process improvement
5. Certifications: Professional certifications (non-IT)
6. Regulations & Compliance: Industry-specific regulations and compliance knowledge
7. Soft Skills: Leadership, communication, negotiation (if explicitly mentioned)

CONSTRAINTS:
- Extract ONLY non-IT professional skills.
- Do NOT extract programming languages, cloud platforms, or IT technical skills.
- Preserve skill names exactly as written (case-sensitive).
- Remove duplicates.
- Limit to a maximum of 50 skills.

ANTI-HALLUCINATION RULES:
- Extract skills ONLY if they are explicitly mentioned in the resume.
- Never guess, infer, or assume skills.
- Do NOT convert topics, events, or titles into skills.
- Do NOT derive skills from job titles or organization names alone.
- Do NOT include generic skills like "Communication" or "Teamwork" unless explicitly listed.

OUTPUT FORMAT:
Return only valid JSON.
No additional text.
No explanations.
No markdown formatting.

JSON SCHEMA:
{{
  "skills": ["skill1", "skill2", "skill3", ...]
}}

VALID OUTPUT EXAMPLES:
{{"skills": ["Property Leasing", "RERA Compliance", "Real Estate Law", "MS Excel", "CRM"]}}
{{"skills": ["Underwriting", "Claims Processing", "Policy Administration", "Insurance Regulations"]}}
{{"skills": ["Lead Generation", "Account Management", "Sales Forecasting", "Negotiation", "CRM"]}}
"""




class SkillsExtractor:
    """Service for extracting skills from resume text using OLLAMA LLM."""
    
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
    
    async def _classify_profile(self, resume_text: str, filename: str = "resume") -> tuple[str, Optional[str]]:
        """
        Classify profile as IT or NON_IT using Gateway Prompt.
        
        Args:
            resume_text: The text content of the resume
            filename: Name of the resume file (for logging)
        
        Returns:
            Tuple of (profile_type, domain) where profile_type is "IT" or "NON_IT"
        """
        try:
            is_connected, available_model = await self._check_ollama_connection()
            if not is_connected:
                logger.warning(
                    "OLLAMA not accessible for classification, defaulting to IT",
                    extra={"file_name": filename}
                )
                return "IT", None
            
            model_to_use = self.model
            if available_model and "llama3.1" not in available_model.lower():
                model_to_use = available_model
            
            text_to_send = resume_text[:5000]  # Use shorter text for classification
            prompt = f"""{GATEWAY_PROMPT}

Input resume text:
{text_to_send}

Output (JSON only, no other text, no explanations):"""
            
            logger.info(
                "[GATEWAY] Classifying profile type",
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
                                    "num_predict": 200,
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
            
            # Extract JSON with different schema (profile_type and domain)
            parsed_data = self._extract_classification_json(raw_output)
            
            profile_type = parsed_data.get("profile_type", "IT")
            domain = parsed_data.get("domain")
            
            # Validate profile_type
            if profile_type not in ["IT", "NON_IT"]:
                logger.warning(
                    f"Invalid profile_type '{profile_type}', defaulting to IT",
                    extra={"file_name": filename, "parsed_data": parsed_data}
                )
                profile_type = "IT"
            
            logger.info(
                "[GATEWAY] Profile classified",
                extra={
                    "file_name": filename,
                    "profile_type": profile_type,
                    "domain": domain
                }
            )
            
            return profile_type, domain
            
        except Exception as e:
            logger.error(
                f"Profile classification failed, defaulting to IT: {e}",
                extra={"file_name": filename, "error": str(e)}
            )
            return "IT", None
    
    def _extract_classification_json(self, text: str) -> Dict:
        """Extract JSON object from classification response (profile_type, domain)."""
        if not text:
            return {"profile_type": "IT", "domain": None}
        
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
            if isinstance(parsed, dict) and "profile_type" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass
        
        # Try balanced braces
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
                    if isinstance(parsed, dict) and "profile_type" in parsed:
                        return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        
        return {"profile_type": "IT", "domain": None}
    
    def _extract_json(self, text: str) -> Dict:
        """Extract JSON object from LLM response."""
        if not text:
            logger.warning(
                "Empty response from LLM during JSON extraction",
                extra={
                    "failure_reason": "empty_text_in_extract_json"
                }
            )
            return {"skills": []}
        
        # Clean the text - remove markdown code blocks if present
        cleaned_text = text.strip()
        had_markdown = False
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
            had_markdown = True
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
            had_markdown = True
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
            had_markdown = True
        cleaned_text = cleaned_text.strip()
        
        # Find the first { and last }
        start_idx = cleaned_text.find('{')
        end_idx = cleaned_text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            cleaned_text = cleaned_text[start_idx:end_idx + 1]
        elif start_idx == -1:
            logger.error(
                "JSON extraction failed: No opening brace '{' found in response",
                extra={
                    "response_preview": text[:500],
                    "response_length": len(text),
                    "cleaned_preview": cleaned_text[:500],
                    "had_markdown": had_markdown,
                    "failure_reason": "no_opening_brace"
                }
            )
            return {"skills": []}
        elif end_idx == -1:
            logger.error(
                "JSON extraction failed: No closing brace '}' found in response",
                extra={
                    "response_preview": text[:500],
                    "response_length": len(text),
                    "cleaned_preview": cleaned_text[:500],
                    "had_markdown": had_markdown,
                    "failure_reason": "no_closing_brace"
                }
            )
            return {"skills": []}
        
        # Try parsing the cleaned text
        try:
            parsed = json.loads(cleaned_text)
            if isinstance(parsed, dict) and "skills" in parsed:
                logger.debug(f"Successfully extracted JSON: {parsed}")
                return parsed
            elif isinstance(parsed, dict):
                logger.error(
                    "JSON extraction failed: Parsed dict missing 'skills' key",
                    extra={
                        "parsed_keys": list(parsed.keys()),
                        "parsed_data": str(parsed)[:500],
                        "response_preview": text[:500],
                        "failure_reason": "missing_skills_key_in_parsed_dict"
                    }
                )
                return {"skills": []}
            else:
                logger.error(
                    "JSON extraction failed: Parsed result is not a dict",
                    extra={
                        "parsed_type": type(parsed).__name__,
                        "parsed_value": str(parsed)[:500],
                        "response_preview": text[:500],
                        "failure_reason": "parsed_result_not_dict"
                    }
                )
                return {"skills": []}
        except json.JSONDecodeError as e:
            logger.warning(
                f"JSON parsing failed on first attempt: {e}",
                extra={
                    "json_error": str(e),
                    "json_error_position": getattr(e, 'pos', None),
                    "cleaned_text_preview": cleaned_text[:500],
                    "failure_reason": "json_decode_error_first_attempt"
                }
            )
        
        # Try to find JSON with balanced braces
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
                    if isinstance(parsed, dict) and "skills" in parsed:
                        logger.debug(f"Successfully extracted JSON with balanced braces: {parsed}")
                        return parsed
                    else:
                        logger.error(
                            "JSON extraction with balanced braces failed: Missing 'skills' key or not a dict",
                            extra={
                                "parsed_type": type(parsed).__name__ if parsed else None,
                                "parsed_keys": list(parsed.keys()) if isinstance(parsed, dict) else None,
                                "json_str_preview": json_str[:500],
                                "failure_reason": "balanced_braces_missing_skills"
                            }
                        )
                        return {"skills": []}
                else:
                    logger.error(
                        "JSON extraction failed: Unbalanced braces in response",
                        extra={
                            "brace_count": brace_count,
                            "response_preview": text[:500],
                            "cleaned_preview": cleaned_text[:500],
                            "failure_reason": "unbalanced_braces"
                        }
                    )
                    return {"skills": []}
            else:
                logger.error(
                    "JSON extraction failed: No opening brace found for balanced brace search",
                    extra={
                        "response_preview": text[:500],
                        "cleaned_preview": cleaned_text[:500],
                        "failure_reason": "no_opening_brace_balanced_search"
                    }
                )
                return {"skills": []}
        except json.JSONDecodeError as e:
            logger.error(
                f"JSON parsing failed with balanced braces: {e}",
                extra={
                    "json_error": str(e),
                    "json_error_position": getattr(e, 'pos', None),
                    "response_preview": text[:500],
                    "cleaned_preview": cleaned_text[:500],
                    "failure_reason": "json_decode_error_balanced_braces"
                }
            )
            return {"skills": []}
        except (ValueError, Exception) as e:
            logger.error(
                f"Unexpected error during balanced brace JSON extraction: {e}",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "response_preview": text[:500],
                    "cleaned_preview": cleaned_text[:500],
                    "failure_reason": "unexpected_error_balanced_braces"
                }
            )
            return {"skills": []}
        
        logger.error(
            "JSON extraction failed: All parsing attempts exhausted", 
            extra={
                "response_preview": text[:500],
                "response_length": len(text),
                "cleaned_preview": cleaned_text[:500],
                "had_markdown": had_markdown,
                "failure_reason": "all_parsing_attempts_failed"
            }
        )
        return {"skills": []}
    
    def _validate_and_clean_skills(self, skills: List, profile_type: str, domain: Optional[str] = None) -> List[str]:
        """
        Validate and clean extracted skills based on profile type.
        
        Args:
            skills: Raw skills list from LLM
            profile_type: "IT" or "NON_IT"
            domain: Domain name for NON_IT profiles
        
        Returns:
            Cleaned and validated list of skills
        """
        if not skills or not isinstance(skills, list):
            return []
        
        original_count = len(skills)
        
        # Step 1: Convert to strings and strip whitespace
        cleaned_skills = []
        for skill in skills:
            if skill is None:
                continue
            skill_str = str(skill).strip()
            if not skill_str or skill_str.lower() in ["null", "none", ""]:
                continue
            
            # Step 2: Split concatenated skills (handle cases like "Python, Java, AWS")
            if "," in skill_str:
                split_skills = [s.strip() for s in skill_str.split(",")]
                cleaned_skills.extend([s for s in split_skills if s])
            else:
                cleaned_skills.append(skill_str)
        
        # Step 3: Remove duplicates while preserving order
        seen = set()
        unique_skills = []
        for skill in cleaned_skills:
            # Normalize for comparison (case-insensitive)
            skill_lower = skill.lower()
            if skill_lower not in seen and len(skill) > 0:
                seen.add(skill_lower)
                unique_skills.append(skill)
        
        # Step 4: Basic validation - remove obviously invalid entries
        valid_skills = []
        for skill in unique_skills:
            # Skip if too short or too long (likely errors)
            if len(skill) < 2 or len(skill) > 100:
                continue
            # Skip if it's just punctuation or numbers
            if not any(c.isalpha() for c in skill):
                continue
            valid_skills.append(skill)
        
        # Step 5: Limit to 50 skills
        valid_skills = valid_skills[:50]
        
        after_cleaning_count = len(valid_skills)
        
        if original_count > 0 and after_cleaning_count == 0:
            logger.warning(
                "All skills were filtered out during validation",
                extra={
                    "original_count": original_count,
                    "profile_type": profile_type,
                    "domain": domain
                }
            )
        
        return valid_skills
    
    async def _extract_skills_with_prompt(self, resume_text: str, prompt: str, filename: str, model_to_use: str) -> List[str]:
        """
        Extract skills using a specific prompt.
        
        Args:
            resume_text: The text content of the resume
            prompt: The prompt to use for extraction
            filename: Name of the resume file
            model_to_use: Model name to use
        
        Returns:
            List of extracted skills
        """
        text_to_send = resume_text[:10000]
        full_prompt = f"""{prompt}

Input resume text:
{text_to_send}

Output (JSON only, no other text, no explanations):"""
        
        result = None
        last_error = None
        
        async with httpx.AsyncClient(timeout=Timeout(3600.0)) as client:
            try:
                response = await client.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": model_to_use,
                        "prompt": full_prompt,
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
                logger.debug("Successfully used /api/generate endpoint for skills extraction")
            except httpx.HTTPStatusError as e:
                if e.response.status_code != 404:
                    raise
                last_error = e
            
            if result is None:
                try:
                    response = await client.post(
                        f"{self.ollama_host}/api/chat",
                        json={
                            "model": model_to_use,
                            "messages": [
                                {"role": "system", "content": "You are a fresh, isolated extraction agent. This is a new, independent task with no previous context. Ignore any previous conversations."},
                                {"role": "user", "content": full_prompt}
                            ],
                            "stream": False,
                            "options": {
                                "temperature": 0.1,
                                "top_p": 0.9,
                                "num_predict": 2000,
                            }
                        }
                    )
                    response.raise_for_status()
                    result = response.json()
                    if "message" in result and "content" in result["message"]:
                        result = {"response": result["message"]["content"]}
                    else:
                        raise ValueError("Unexpected response format from OLLAMA chat API")
                    logger.debug("Successfully used /api/chat endpoint for skills extraction")
                except Exception as e2:
                    last_error = e2
                    raise RuntimeError(f"All OLLAMA API endpoints failed. Last error: {last_error}")
        
        raw_output = ""
        if isinstance(result, dict):
            raw_output = str(result.get("response", "") or result.get("text", "") or result.get("content", ""))
            if "message" in result and isinstance(result.get("message"), dict):
                raw_output = str(result["message"].get("content", ""))
        else:
            raw_output = str(result)
        
        if not raw_output or not raw_output.strip():
            logger.warning("Empty response from LLM for skills extraction", extra={"file_name": filename})
            return []
        
        parsed_data = self._extract_json(raw_output)
        skills = parsed_data.get("skills", [])
        
        if not isinstance(skills, list):
            logger.error("Skills is not a list", extra={"file_name": filename, "skills_type": type(skills).__name__})
            return []
        
        return skills
    
    async def extract_skills(self, resume_text: str, filename: str = "resume") -> List[str]:
        """
        Extract skills from resume text using OLLAMA LLM.
        
        Args:
            resume_text: The text content of the resume
            filename: Name of the resume file (for logging)
        
        Returns:
            List of extracted skills
        """
        try:
            # Log input validation
            if not resume_text or not resume_text.strip():
                logger.error(
                    f"❌ SKILLS EXTRACTION FAILED: Empty or null resume text",
                    extra={
                        "file_name": filename,
                        "resume_text_length": len(resume_text) if resume_text else 0,
                        "resume_text_is_none": resume_text is None,
                        "resume_text_is_empty": resume_text == "" if resume_text else True,
                        "failure_reason": "empty_resume_text"
                    }
                )
                return []
            
            if len(resume_text.strip()) < 10:
                logger.warning(
                    f"⚠️ SKILLS EXTRACTION WARNING: Resume text is very short",
                    extra={
                        "file_name": filename,
                        "resume_text_length": len(resume_text),
                        "resume_text_preview": resume_text[:200],
                        "failure_reason": "resume_text_too_short"
                    }
                )
            
            is_connected, available_model = await self._check_ollama_connection()
            if not is_connected:
                logger.error(
                    f"❌ SKILLS EXTRACTION FAILED: OLLAMA connection check failed",
                    extra={
                        "file_name": filename,
                        "ollama_host": self.ollama_host,
                        "failure_reason": "ollama_not_accessible",
                        "resume_text_length": len(resume_text)
                    }
                )
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
            
            # ============================================================================
            # STEP 1: GATEWAY - Classify Profile Type
            # ============================================================================
            logger.info(
                "[ROUTING] Step 1: Classifying profile type",
                extra={"file_name": filename}
            )
            
            profile_type, domain = await self._classify_profile(resume_text, filename)
            
            logger.info(
                "[ROUTING] Profile classified",
                extra={
                    "file_name": filename,
                    "profile_type": profile_type,
                    "domain": domain
                }
            )
            
            # ============================================================================
            # STEP 2: ROUTE to Appropriate Skills Extraction Prompt
            # ============================================================================
            if profile_type == "IT":
                logger.info(
                    "[ROUTING] Routing to IT skills extraction",
                    extra={"file_name": filename}
                )
                prompt_to_use = IT_SKILLS_PROMPT
            else:
                logger.info(
                    "[ROUTING] Routing to Non-IT skills extraction",
                    extra={
                        "file_name": filename,
                        "domain": domain
                    }
                )
                prompt_to_use = get_non_it_skills_prompt(domain)
            
            # ============================================================================
            # STEP 3: EXTRACT SKILLS using appropriate prompt
            # ============================================================================
            raw_skills = await self._extract_skills_with_prompt(
                resume_text=resume_text,
                prompt=prompt_to_use,
                filename=filename,
                model_to_use=model_to_use
            )
            
            # ============================================================================
            # STEP 4: VALIDATE and CLEAN skills
            # ============================================================================
            validated_skills = self._validate_and_clean_skills(
                skills=raw_skills,
                profile_type=profile_type,
                domain=domain
            )
            
            # ============================================================================
            # STEP 5: LOG RESULTS
            # ============================================================================
            if not validated_skills or len(validated_skills) == 0:
                logger.warning(
                    "[SKILLS] No skills extracted after validation",
                    extra={
                        "file_name": filename,
                        "profile_type": profile_type,
                        "domain": domain,
                        "raw_skills_count": len(raw_skills) if raw_skills else 0
                    }
                )
            else:
                logger.info(
                    "[SKILLS] Skills extracted successfully",
                    extra={
                        "file_name": filename,
                        "profile_type": profile_type,
                        "domain": domain,
                        "skills_count": len(validated_skills),
                        "skills_preview": validated_skills[:10]  # Log first 10
                    }
                )
            
            return validated_skills
            
        except httpx.HTTPError as e:
            error_details = {
                "file_name": filename,
                "error": str(e),
                "error_type": type(e).__name__,
                "ollama_host": self.ollama_host,
                "model": getattr(self, "model", "unknown"),
                "resume_text_length": len(resume_text) if resume_text else 0,
                "failure_reason": "http_error"
            }
            logger.error(
                f"❌ SKILLS EXTRACTION FAILED: HTTP error calling OLLAMA for skills extraction: {e}",
                extra=error_details,
                exc_info=True
            )
            raise RuntimeError(f"Failed to extract skills with LLM: {e}")
        except Exception as e:
            logger.error(
                f"❌ SKILLS EXTRACTION FAILED: Unexpected error extracting skills: {e}",
                extra={
                    "file_name": filename,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "ollama_host": self.ollama_host,
                    "model": getattr(self, "model", "unknown"),
                    "resume_text_length": len(resume_text) if resume_text else 0,
                    "failure_reason": "unexpected_exception"
                },
                exc_info=True
            )
            raise

