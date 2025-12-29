# Profile Processing Flow Review

## Overview
This document reviews the sequential flow of profile processing, focusing on mastercategory, category, and skills extraction.

---

## Current Execution Order

### STEP 1: Database Record Creation
**Location:** `app/controllers/resume_controller.py` (lines 250-289)

**Actions:**
- Check for existing record by filename
- If exists: Update existing record (candidatename, jobrole, status)
- If not exists: Create new record with `mastercategory=None`, `category=None`
- Set status to `STATUS_PROCESSING`

**Output:**
- `resume_metadata` object with `id` and `status="processing"`

---

### STEP 2: Mastercategory Classification (IT vs NON_IT)
**Location:** `app/controllers/resume_controller.py` (lines 311-333)

**Flow:**
```
Resume Text (first 1000 chars)
    ↓
MasterCategoryExtractor.extract_mastercategory()
    ↓
MASTERCATEGORY_PROMPT (Navigation-based output)
    ↓
Parse: "NAVIGATE_TO_IT_SKILLS_EXTRACTION" → "IT"
       "NAVIGATE_TO_NON_IT_SKILLS_EXTRACTION" → "NON_IT"
    ↓
MasterCategoryService.extract_and_save_mastercategory()
    ↓
UPDATE resume_metadata SET mastercategory = "IT" | "NON_IT"
```

**Key Points:**
- ✅ Runs immediately after record creation
- ✅ Uses first 1000 characters of resume text
- ✅ Saves result to database before proceeding
- ✅ On error: Defaults to "NON_IT" and continues processing
- ✅ Logging: Comprehensive logging at each step

**Error Handling:**
- If OLLAMA fails → Defaults to "NON_IT"
- If parsing fails → Defaults to "NON_IT"
- If DB update fails → Logs error but continues

**Files:**
- `app/mastercategory/mastercategory_extractor.py`
- `app/mastercategory/mastercategory_service.py`

---

### STEP 3: Category Classification
**Location:** `app/controllers/resume_controller.py` (lines 335-380)

**Flow:**
```
REFRESH resume_metadata (get latest mastercategory from DB)
    ↓
Check: IF mastercategory IS NOT NULL
    ↓
IF mastercategory == "IT":
    CategoryExtractor.extract_category(resume_text, mastercategory="IT")
    ↓
    IT_CATEGORY_PROMPT (22 IT categories)
    ↓
    Parse category name from LLM response
    ↓
ELSE IF mastercategory == "NON_IT":
    CategoryExtractor.extract_category(resume_text, mastercategory="NON_IT")
    ↓
    NON_IT_CATEGORY_PROMPT (30 Non-IT categories)
    ↓
    Parse category name from LLM response
    ↓
CategoryService.extract_and_save_category()
    ↓
UPDATE resume_metadata SET category = <extracted_category>
```

**Key Points:**
- ✅ **Refresh happens** - Gets latest mastercategory from DB (line 339)
- ✅ **Conditional execution** - Only runs if mastercategory exists
- ✅ Uses first 1000 characters of resume text
- ✅ Prompt selection is dynamic based on mastercategory
- ✅ Saves result to database before proceeding
- ✅ On error: Saves NULL and continues processing

**Error Handling:**
- If mastercategory is NULL → Skips category extraction (logs warning)
- If OLLAMA fails → Saves NULL and continues
- If parsing fails → Saves NULL and continues
- If DB update fails → Logs error but continues

**Files:**
- `app/category/category_extractor.py`
- `app/category/category_service.py`

---

### STEP 4-N: Other Module Extractions
**Location:** `app/controllers/resume_controller.py` (lines 407-481)

**Modules Extracted (in order):**
1. Designation
2. Name
3. Email
4. Mobile
5. Experience
6. Domain
7. Education

**Key Points:**
- ✅ Runs after mastercategory and category extraction
- ✅ Each module is independent
- ✅ If one fails, others continue
- ✅ No dependencies on mastercategory/category

---

### STEP 8: Skills Extraction
**Location:** `app/controllers/resume_controller.py` (lines 483-494)

**Flow:**
```
REFRESH resume_metadata (get latest mastercategory & category from DB)
    ↓
SkillsService.extract_and_save_skills()
    ↓
Get resume_metadata from DB
    ↓
Read: mastercategory, category
    ↓
IF mastercategory AND category exist:
    ├─ Try: PromptRepository.get_by_category(mastercategory, category)
    │   └─ Found? Use database prompt
    │   └─ Not found? Try generic prompt
    │
    └─ Try: PromptRepository.get_by_mastercategory(mastercategory)
        └─ Found? Use generic database prompt
        └─ Not found? Fallback to gateway routing
    
ELSE:
    └─ Fallback to gateway routing
        ├─ Gateway decision (IT vs NON_IT)
        └─ Use hardcoded SKILLS_PROMPT or NON_IT_SKILLS_PROMPT
    ↓
SkillsExtractor.extract_skills(resume_text, custom_prompt=...)
    ↓
Extract skills using selected prompt
    ↓
Validate, clean, deduplicate, limit to 50 skills
    ↓
UPDATE resume_metadata SET skillset = <comma-separated-skills>
```

**Key Points:**
- ✅ **Refresh happens** - Gets latest mastercategory & category from DB (line 485)
- ✅ **Prompt priority:**
  1. Category-specific prompt from database (mastercategory + category)
  2. Generic mastercategory prompt from database (mastercategory only)
  3. Fallback to gateway routing with hardcoded prompts
- ✅ Uses full resume text (first 10,000 characters for LLM)
- ✅ Saves result to database
- ✅ On error: Saves NULL and continues processing

**Error Handling:**
- If mastercategory/category missing → Falls back to gateway routing
- If prompt not found in DB → Falls back to gateway routing
- If OLLAMA fails → Raises exception (logged, continues)
- If parsing fails → Returns empty list, saves NULL

**Files:**
- `app/skills/skills_extractor.py`
- `app/skills/skills_service.py`
- `app/repositories/prompt_repo.py`

---

## Data Dependencies

```
resume_metadata.id (STEP 1)
    ↓
mastercategory (STEP 2) ──┐
    ↓                     │
category (STEP 3) ────────┼──→ Skills Extraction (STEP 8)
    ↓                     │
Other fields...          │
```

**Critical Path:**
1. Record must exist (STEP 1) before mastercategory extraction
2. Mastercategory must exist (STEP 2) for category extraction to run
3. Both mastercategory and category should exist (STEP 3) for optimal skills extraction

---

## Verification Checklist

### ✅ Mastercategory Extraction
- [x] Runs immediately after record creation
- [x] Saves result to database
- [x] Uses proper prompt (MASTERCATEGORY_PROMPT)
- [x] Parses navigation commands correctly
- [x] Error handling defaults to "NON_IT"
- [x] Logging is comprehensive

### ✅ Category Extraction
- [x] Refreshes resume_metadata before reading mastercategory
- [x] Only runs if mastercategory exists
- [x] Uses correct prompt based on mastercategory (IT vs NON_IT)
- [x] Saves result to database
- [x] Error handling saves NULL
- [x] Logging is comprehensive

### ✅ Skills Extraction
- [x] Refreshes resume_metadata before reading mastercategory/category
- [x] Fetches prompt from database based on category
- [x] Falls back gracefully if prompt not found
- [x] Uses full resume text for extraction
- [x] Validates and cleans extracted skills
- [x] Saves result to database
- [x] Error handling saves NULL
- [x] Logging is comprehensive

---

## Potential Issues & Recommendations

### ⚠️ Issue 1: Refresh Timing
**Status:** ✅ FIXED
- Category extraction refreshes before reading mastercategory (line 339)
- Skills extraction refreshes before reading mastercategory/category (line 485)

### ⚠️ Issue 2: Skills Extraction Happens After Other Modules
**Current Behavior:**
- Skills extraction is step 8, after designation, name, email, etc.
- This is fine since it doesn't depend on those fields

**Recommendation:** ✅ Current order is correct
- Mastercategory and category must be extracted first
- Other modules can run in parallel or sequentially
- Skills extraction needs mastercategory/category, so it's correctly positioned

### ⚠️ Issue 3: Gateway Routing Still Present in Skills Extractor
**Status:** ✅ BY DESIGN
- Gateway routing is a fallback when:
  1. mastercategory/category are NULL
  2. No prompt found in database
- This ensures backward compatibility

**Recommendation:** ✅ Keep as is - Provides robust fallback

### ✅ Issue 4: Error Propagation
**Status:** ✅ GOOD
- Each step continues even if previous step fails
- Errors are logged but don't stop processing
- Database always updated (even with NULL/default values)

---

## Execution Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Create/Update Database Record                       │
│ - Insert record or update existing                          │
│ - Set status = "processing"                                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Mastercategory Classification                       │
│ - Extract mastercategory (IT/NON_IT)                        │
│ - Save to resume_metadata.mastercategory                    │
│ - Uses: MASTERCATEGORY_PROMPT                               │
│ - Input: resume_text[:1000]                                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ REFRESH: Get latest mastercategory from DB                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Category Classification                             │
│ - IF mastercategory EXISTS:                                 │
│   - Extract category based on mastercategory                │
│   - Save to resume_metadata.category                        │
│   - Uses: IT_CATEGORY_PROMPT or NON_IT_CATEGORY_PROMPT      │
│   - Input: resume_text[:1000]                               │
│ - ELSE: Skip (log warning)                                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4-7: Other Module Extractions                          │
│ - Designation, Name, Email, Mobile, Experience, Domain,     │
│   Education                                                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ REFRESH: Get latest mastercategory & category from DB       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 8: Skills Extraction                                   │
│ - Read mastercategory & category from DB                    │
│ - Try to fetch prompt from prompts table:                   │
│   1. By mastercategory + category                           │
│   2. By mastercategory only (generic)                       │
│   3. Fallback: Gateway routing + hardcoded prompts          │
│ - Extract skills using selected prompt                      │
│ - Validate, clean, deduplicate, limit to 50                 │
│ - Save to resume_metadata.skillset                          │
│ - Input: resume_text[:10000]                                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ FINAL: Update status = "completed"                          │
│ Generate embeddings, store in vector DB                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Summary

### ✅ Strengths
1. **Correct Order**: Mastercategory → Category → Skills (with proper dependencies)
2. **Robust Error Handling**: Each step continues even if previous fails
3. **Database Consistency**: All steps save to DB with proper error handling
4. **Refresh Points**: Critical refreshes before reading mastercategory/category
5. **Fallback Logic**: Skills extraction has robust fallback mechanism
6. **Comprehensive Logging**: Each step logs start, progress, and completion

### 📝 Recommendations
1. ✅ Current implementation is correct and well-structured
2. ✅ All dependencies are properly handled
3. ✅ Error handling is comprehensive
4. ✅ Logging provides good visibility

### 🎯 Conclusion
The extraction flow is **correctly implemented** and follows the intended sequential order:
1. Mastercategory extraction (STEP 2)
2. Category extraction (STEP 3) - depends on mastercategory
3. Skills extraction (STEP 8) - depends on mastercategory + category

All steps properly refresh database state and handle errors gracefully.

