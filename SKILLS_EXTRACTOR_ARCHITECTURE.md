# Skills Extractor - Method Call Hierarchy & Flow

```
================================================================================
                    SKILLS EXTRACTION ARCHITECTURE
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                    EXTERNAL CALLER (e.g., SkillsService)                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ Call: extract_skills()
                                   │ Input: resume_text: str, filename: str
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SkillsExtractor.extract_skills()                         │
│  [MAIN ENTRY POINT - ROUTING ORCHESTRATOR]                                  │
│                                                                              │
│  Input:                                                                      │
│    - resume_text: str                                                        │
│    - filename: str = "resume"                                                │
│                                                                              │
│  Output:                                                                     │
│    - List[str] (validated skills list)                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
        ▼                          ▼                          ▼
   [VALIDATE]              [CHECK OLLAMA]              [GET MODEL]
   Input Validation        Connection Check            Model Selection
        │                          │                          │
        │                          ▼                          │
        │              ┌───────────────────────────┐         │
        │              │ _check_ollama_connection()│         │
        │              │                           │         │
        │              │ Input: None               │         │
        │              │ Output:                   │         │
        │              │   - tuple[bool,           │         │
        │              │        Optional[str]]     │         │
        │              │   - (is_connected,        │         │
        │              │      available_model)     │         │
        │              └───────────────────────────┘         │
        │                          │                          │
        └──────────────────────────┴──────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │  STEP 1: GATEWAY CLASSIFY    │
                    └──────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│            SkillsExtractor._classify_profile()                              │
│  [GATEWAY - Profile Classification]                                         │
│                                                                              │
│  Input:                                                                      │
│    - resume_text: str                                                        │
│    - filename: str                                                           │
│                                                                              │
│  Process:                                                                    │
│    1. Call _check_ollama_connection()                                       │
│    2. Prepare GATEWAY_PROMPT + resume_text[:5000]                           │
│    3. Call OLLAMA API (/api/generate or /api/chat)                         │
│    4. Parse JSON response                                                    │
│                                                                              │
│  Output:                                                                     │
│    - tuple[str, Optional[str]]                                               │
│      (profile_type, domain)                                                  │
│      profile_type: "IT" | "NON_IT"                                          │
│      domain: "Healthcare" | "Real Estate" | ... | None                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                ┌──────────────────┴──────────────────┐
                │                                     │
                ▼                                     ▼
    ┌───────────────────────┐           ┌─────────────────────────────┐
    │ _extract_classification_json()    │  OLLAMA API CALL            │
    │                                    │  - /api/generate            │
    │ Input:                             │  - /api/chat (fallback)    │
    │   - text: str (raw LLM response)   │                            │
    │                                    │ Input:                      │
    │ Output:                            │   - prompt: str             │
    │   - Dict {                         │   - model: str              │
    │       "profile_type": str,         │   - options: dict           │
    │       "domain": str | None         │                            │
    │   }                                 │ Output:                     │
    └───────────────────────┘             │   - JSON response          │
                                          └─────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │  STEP 2: ROUTING DECISION    │
                    └──────────────────────────────┘
                                   │
                ┌──────────────────┴──────────────────┐
                │                                     │
        profile_type == "IT"              profile_type == "NON_IT"
                │                                     │
                ▼                                     ▼
    ┌───────────────────────┐           ┌─────────────────────────────┐
    │  IT_SKILLS_PROMPT     │           │ get_non_it_skills_prompt()  │
    │  (Static prompt)      │           │                             │
    │                       │           │ Input:                      │
    │  Contains IT skill    │           │   - domain: Optional[str]   │
    │  categories:          │           │                             │
    │  - Programming        │           │ Output:                     │
    │  - Cloud (AWS/Azure)  │           │   - str (domain-aware       │
    │  - DevOps             │           │      prompt)                │
    │  - AI/ML              │           │                             │
    │  - Databases          │           │ Process:                    │
    │  - etc.               │           │   - If domain provided:     │
    │                       │           │     add domain-specific     │
    │                       │           │     examples                │
    └───────────────────────┘           └─────────────────────────────┘
                │                                     │
                └──────────────────┬──────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │  STEP 3: EXTRACT SKILLS      │
                    └──────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│        SkillsExtractor._extract_skills_with_prompt()                        │
│  [Generic Skills Extraction Method]                                         │
│                                                                              │
│  Input:                                                                      │
│    - resume_text: str                                                        │
│    - prompt: str (IT_SKILLS_PROMPT or Non-IT prompt)                        │
│    - filename: str                                                           │
│    - model_to_use: str                                                       │
│                                                                              │
│  Process:                                                                    │
│    1. Prepare full prompt: prompt + resume_text[:10000]                     │
│    2. Call OLLAMA API (/api/generate or /api/chat)                         │
│    3. Extract raw output                                                    │
│    4. Parse JSON response                                                   │
│                                                                              │
│  Output:                                                                     │
│    - List[str] (raw skills list from LLM)                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                ┌──────────────────┴──────────────────┐
                │                                     │
                ▼                                     ▼
    ┌───────────────────────┐           ┌─────────────────────────────┐
    │ _extract_json()       │           │  OLLAMA API CALL            │
    │                       │           │  - /api/generate            │
    │ Input:                │           │  - /api/chat (fallback)    │
    │   - text: str         │           │                            │
    │                       │           │ Input:                      │
    │ Output:               │           │   - prompt: str             │
    │   - Dict {            │           │   - model: str              │
    │       "skills": [     │           │   - options: dict           │
    │         "skill1",     │           │                            │
    │         "skill2",     │           │ Output:                     │
    │         ...           │           │   - JSON response           │
    │       ]               │           │     {"skills": [...]}       │
    │   }                   │           │                            │
    │                       │           │ Process:                    │
    │ Process:              │           │   - Remove markdown         │
    │   - Remove markdown   │           │   - Extract JSON with       │
    │   - Find JSON braces  │           │     balanced braces         │
    │   - Parse JSON        │           │   - Parse JSON              │
    │   - Handle errors     │           │                            │
    └───────────────────────┘           └─────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │  STEP 4: VALIDATE & CLEAN    │
                    └──────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│        SkillsExtractor._validate_and_clean_skills()                         │
│  [Validation & Cleanup Layer]                                               │
│                                                                              │
│  Input:                                                                      │
│    - skills: List (raw skills from LLM)                                     │
│    - profile_type: str ("IT" | "NON_IT")                                    │
│    - domain: Optional[str]                                                   │
│                                                                              │
│  Process:                                                                    │
│    1. Convert to strings and strip whitespace                               │
│    2. Split concatenated skills (handle "Python, Java, AWS")               │
│    3. Remove duplicates (case-insensitive)                                  │
│    4. Validate skill format:                                                │
│       - Length: 2-100 characters                                            │
│       - Must contain alphabetic characters                                  │
│    5. Limit to 50 skills max                                                │
│                                                                              │
│  Output:                                                                     │
│    - List[str] (cleaned and validated skills, max 50)                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │  STEP 5: LOG & RETURN        │
                    └──────────────────────────────┘
                                   │
                                   ▼
                          ┌─────────────────┐
                          │  Return Skills  │
                          │  List[str]      │
                          └─────────────────┘


================================================================================
                         DATA FLOW SUMMARY
================================================================================

1. INPUT
   └─> resume_text: str, filename: str

2. GATEWAY CLASSIFICATION
   └─> resume_text[:5000] → GATEWAY_PROMPT → OLLAMA
       └─> Output: {"profile_type": "IT"|"NON_IT", "domain": str|null}

3. ROUTING
   ├─> IT: IT_SKILLS_PROMPT
   └─> NON_IT: get_non_it_skills_prompt(domain)

4. SKILLS EXTRACTION
   └─> resume_text[:10000] + prompt → OLLAMA
       └─> Output: {"skills": ["skill1", "skill2", ...]}

5. VALIDATION
   └─> Raw skills → Clean → Validate → Deduplicate → Limit(50)
       └─> Output: List[str] (max 50 skills)

6. OUTPUT
   └─> List[str] (validated, cleaned skills)


================================================================================
                         PROMPT STRUCTURE
================================================================================

GATEWAY_PROMPT
├─> Purpose: Classify profile type (IT vs NON_IT)
├─> Input: resume_text[:5000]
├─> Output Schema: {"profile_type": "IT"|"NON_IT", "domain": str|null}
└─> Models: Classification only (short, cheap)

IT_SKILLS_PROMPT
├─> Purpose: Extract IT technical skills
├─> Input: resume_text[:10000]
├─> Output Schema: {"skills": ["skill1", "skill2", ...]}
├─> Categories:
│   ├─> Programming & Scripting
│   ├─> Cloud Platforms (AWS, Azure)
│   ├─> DevOps & Platform Engineering
│   ├─> AI/ML & Generative AI
│   ├─> Data Science & BI
│   ├─> Databases & Data Technologies
│   ├─> Web & Mobile Development
│   ├─> ERP Systems (SAP, Salesforce, Dynamics)
│   └─> Certifications (IT)
└─> Constraints: Only IT skills, max 50

NON_IT_SKILLS_PROMPT (domain-aware)
├─> Purpose: Extract non-IT professional skills
├─> Input: resume_text[:10000], domain context
├─> Output Schema: {"skills": ["skill1", "skill2", ...]}
├─> Categories:
│   ├─> Domain Skills (domain-specific)
│   ├─> Functional Skills (Sales, HR, Operations)
│   ├─> Tools (Non-Technical: CRM, MS Office, etc.)
│   ├─> Methodologies (Business frameworks)
│   ├─> Certifications (Non-IT)
│   ├─> Regulations & Compliance
│   └─> Soft Skills (if explicitly mentioned)
└─> Constraints: No IT skills, domain-focused, max 50


================================================================================
                         ERROR HANDLING
================================================================================

Classification Errors:
├─> OLLAMA not accessible → Default to ("IT", None)
├─> JSON parse failure → Default to ("IT", None)
└─> Invalid profile_type → Default to "IT"

Extraction Errors:
├─> OLLAMA not accessible → Raise RuntimeError
├─> JSON parse failure → Return []
├─> Empty response → Return []
└─> Invalid skills format → Filtered out in validation

Validation Errors:
├─> Empty skills list → Return []
├─> Invalid skill format → Filtered out
└─> Too many skills → Truncated to 50


================================================================================
                         EXTERNAL DEPENDENCIES
================================================================================

OLLAMA API:
├─> Endpoints:
│   ├─> GET /api/tags (check connection)
│   ├─> POST /api/generate (primary)
│   └─> POST /api/chat (fallback)
├─> Model: llama3.1 (or available model)
├─> Timeouts:
│   ├─> Connection check: 5 seconds
│   ├─> Classification: 300 seconds
│   └─> Skills extraction: 3600 seconds
└─> Options:
    ├─> temperature: 0.1
    ├─> top_p: 0.9
    └─> num_predict: 200-2000 (depending on task)


================================================================================
                         HELPER FUNCTIONS
================================================================================

Module-level:
├─> get_non_it_skills_prompt(domain: Optional[str]) -> str
│   └─> Generates domain-aware Non-IT prompt with examples

Class Methods:
├─> _check_ollama_connection() -> tuple[bool, Optional[str]]
│   └─> Checks OLLAMA availability and finds model
│
├─> _classify_profile(resume_text, filename) -> tuple[str, Optional[str]]
│   └─> Uses GATEWAY_PROMPT to classify profile
│
├─> _extract_classification_json(text) -> Dict
│   └─> Parses classification JSON: {profile_type, domain}
│
├─> _extract_json(text) -> Dict
│   └─> Parses skills JSON: {skills: [...]}
│
├─> _extract_skills_with_prompt(resume_text, prompt, filename, model) -> List[str]
│   └─> Generic method to extract skills using any prompt
│
└─> _validate_and_clean_skills(skills, profile_type, domain) -> List[str]
    └─> Validates, cleans, deduplicates, and limits skills


================================================================================
                         PERFORMANCE CONSIDERATIONS
================================================================================

1. Text Truncation:
   ├─> Classification: First 5,000 chars (faster, cheaper)
   └─> Skills extraction: First 10,000 chars (more context)

2. API Calls:
   ├─> Classification: 1 call (lightweight)
   └─> Skills extraction: 1 call (heavier)

3. Total API Calls per Resume: 2
   ├─> Gateway classification: ~1-5 seconds
   └─> Skills extraction: ~5-60 seconds (depending on complexity)

4. Validation:
   └─> In-memory processing (fast, no API calls)


