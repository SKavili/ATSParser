# Experience Extractor Architecture

## Overview

The `ExperienceExtractor` class is responsible for extracting years of professional work experience from resume text using a **3-tier extraction strategy**:

1. **Primary**: LLM-based extraction (OLLAMA)
2. **Secondary**: Regex pattern matching
3. **Tertiary**: Date-based calculation from work history

---

## Class Structure

```python
class ExperienceExtractor:
    - ollama_host: str
    - model: str (default: "llama3.1")
    
    Methods:
    - __init__()
    - _check_ollama_connection()
    - _extract_dates_from_text()
    - _is_education_date()
    - _calculate_experience_from_dates()
    - _extract_experience_fallback()
    - _extract_json()
    - extract_experience() [MAIN ENTRY POINT]
```

---

## Extraction Flow (3-Tier Strategy)

```
┌─────────────────────────────────────────────────────────────┐
│                    extract_experience()                      │
│                    [Main Entry Point]                        │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │  Check OLLAMA Connection       │
        └───────────────┬───────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                                │
        ▼                                ▼
┌───────────────┐              ┌──────────────────┐
│ TIER 1: LLM   │              │ OLLAMA Not        │
│ Extraction    │              │ Available         │
│ (Primary)     │              │ → Skip to Tier 2  │
└───────┬───────┘              └──────────────────┘
        │
        ├─→ Call OLLAMA API (/api/generate or /api/chat)
        ├─→ Parse JSON response
        └─→ Extract experience value
                        │
                        ▼
        ┌───────────────────────────────┐
        │  Experience Found?             │
        └───────┬───────────────┬───────┘
                │ YES           │ NO
                ▼               ▼
        ┌───────────┐   ┌──────────────────────┐
        │ RETURN    │   │ TIER 2: Regex        │
        │ Result    │   │ Fallback             │
        └───────────┘   └──────┬───────────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
        ┌──────────────┐            ┌──────────────────┐
        │ Summary      │            │ Work History      │
        │ Patterns     │            │ Patterns          │
        │ (Priority)   │            │ (Lower Priority)  │
        └──────┬───────┘            └────────┬─────────┘
               │                              │
               └──────────┬───────────────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │ Prioritize Matches: │
                │ 1. Has "+" sign     │
                │ 2. Summary section   │
                │ 3. Earlier position │
                └──────────┬──────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Experience Found?    │
                └──────┬───────────┬───┘
                       │ YES       │ NO
                       ▼           ▼
                ┌──────────┐  ┌──────────────────────┐
                │ RETURN   │  │ TIER 3: Date-Based   │
                │ Result   │  │ Calculation          │
                └──────────┘  └──────┬───────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                   │
                    ▼                                   ▼
        ┌──────────────────────┐          ┌──────────────────────┐
        │ Extract All Dates   │          │ Filter Education     │
        │ from Text           │          │ Dates                │
        └──────────┬──────────┘          └──────────┬───────────┘
                   │                                 │
                   └──────────────┬──────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │ Find Oldest & Most      │
                    │ Recent Work Dates       │
                    └──────────┬──────────────┘
                               │
                               ▼
                    ┌─────────────────────────┐
                    │ Calculate Years          │
                    │ Difference               │
                    └──────────┬───────────────┘
                               │
                               ▼
                    ┌─────────────────────────┐
                    │ RETURN "X years"        │
                    │ or None                 │
                    └─────────────────────────┘
```

---

## Method Details

### 1. `extract_experience(resume_text, filename)` - Main Entry Point

**Purpose**: Orchestrates the 3-tier extraction strategy

**Flow**:
1. Check OLLAMA connection
2. If connected → Call LLM (Tier 1)
3. If LLM returns null → Try regex fallback (Tier 2)
4. If regex fails → Try date-based calculation (Tier 3)
5. Return result or None

**Returns**: `Optional[str]` - Experience string like "5 years", "10+ years", or None

---

### 2. `_check_ollama_connection()` - Connection Checker

**Purpose**: Verify OLLAMA service is accessible

**Process**:
- Makes GET request to `/api/tags`
- Checks for available models
- Prefers "llama3.1" model
- Returns `(is_connected: bool, available_model: Optional[str])`

---

### 3. `_extract_dates_from_text(text)` - Date Extractor

**Purpose**: Extract all date patterns from resume text with context

**Supported Date Formats**:
1. **Month Year**: "January 2020", "Jan 2020", "Aug 2022"
2. **DD/MM/YY or DD/MM/YYYY**: "15/01/20", "15/01/2020"
3. **MM/DD/YY or MM/DD/YYYY**: "01/15/20", "01/15/2020"
4. **YYYY-MM-DD**: "2020-01-15"
5. **Year Only**: "2020" (1950-2030 range)

**Returns**: `List[Tuple[datetime, str]]` - List of (date_object, context_string) tuples

**Context**: Captures 200 characters (100 before + 100 after) around each date for filtering

---

### 4. `_is_education_date(context)` - Education Filter

**Purpose**: Determine if a date is education-related based on surrounding context

**Logic**:
```
IF date is in "# EDUCATION" section header:
    IF has strong work keywords (professional experience, work experience):
        RETURN False (it's work-related)
    ELSE:
        RETURN True (it's education)
ELSE IF has education keywords AND no work keywords:
    RETURN True (it's education)
ELSE IF has work keywords:
    RETURN False (it's work-related)
ELSE:
    RETURN False (default to work-related)
```

**Education Keywords**:
- education, degree, bachelor, master, phd, graduation, university, college, school, diploma, certificate, etc.

**Work Keywords**:
- experience, work, employment, job, position, role, company, employer, professional, analyst, engineer, intern, internship, etc.

---

### 5. `_calculate_experience_from_dates(resume_text)` - Date-Based Calculator

**Purpose**: Calculate experience by finding date range from work history

**Process**:
1. Extract all dates from resume text
2. Filter out education-related dates
3. Find oldest and most recent work dates
4. Check for "Present" or "Current" keywords
5. Calculate months difference
6. Round to nearest year
7. Validate (0-50 years range)
8. Handle edge cases (< 3 months returns None, 3-11 months rounds to "1 year")

**Returns**: `Optional[str]` - Experience string or None

**Edge Cases**:
- If most recent date > current date → use current date
- If "Present" mentioned → use current date
- If < 3 months → return None
- If 3-11 months → return "1 year" (rounded up)
- If > 50 years → cap at 50 years

---

### 6. `_extract_experience_fallback(resume_text)` - Regex Fallback

**Purpose**: Extract experience using regex patterns when LLM fails

**Strategy**:
1. **Summary Patterns** (High Priority):
   - "X+ years of experience"
   - "over X+ years"
   - "with X+ years"
   - Searches first 5000 characters (summary section)

2. **Work History Patterns** (Lower Priority):
   - "Total Work Experience: X years"
   - "Work Experience: X years"
   - Searches first 15000 characters

3. **Aggressive Search** (Last Resort):
   - Find any "X years" near "experience" keyword
   - Searches first 2000 characters

4. **Date-Based Calculation** (Final Fallback):
   - If no patterns match, try date-based calculation

**Prioritization Logic**:
```python
Sort matches by:
1. Has "+" sign (True comes before False)
2. Section (summary comes before work_history)
3. Position (earlier comes first)
```

**Returns**: `Optional[str]` - Best matching experience string or None

---

### 7. `_extract_json(text)` - JSON Parser

**Purpose**: Extract JSON object from LLM response (handles markdown, extra text)

**Process**:
1. Remove markdown code blocks (```json, ```)
2. Find first `{` and last `}`
3. Try parsing JSON
4. If fails, try balanced braces parsing
5. Validate "experience" key exists

**Returns**: `Dict` - Parsed JSON with "experience" key

---

## Priority System

### Experience Format Priority (Highest to Lowest):
1. **"X+ years"** format (e.g., "25+ years") - Explicit minimum
2. **"over X+ years"** format
3. **"X years"** in summary section
4. **"X years"** in work history section
5. **Calculated from dates** (last resort)

### Section Priority:
1. **Summary/Profile sections** (first 5000 chars)
2. **Work history sections** (after 5000 chars)

---

## Date Extraction Patterns

### Pattern 1: Month Year
```regex
\b(january|february|...|dec)\.?\s+(\d{4})\b
```
**Examples**: "August 2022", "Dec 2021", "Jan. 2020"

### Pattern 2: DD/MM/YY or DD/MM/YYYY
```regex
\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b
```
**Examples**: "15/01/20", "15/01/2020"
**Heuristic**: If day > 12, likely DD/MM format

### Pattern 3: MM/DD/YY or MM/DD/YYYY
```regex
\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b
```
**Examples**: "01/15/20", "01/15/2020"
**Heuristic**: If first <= 12 and second > 12, it's MM/DD format

### Pattern 4: YYYY-MM-DD
```regex
\b(\d{4})-(\d{1,2})-(\d{1,2})\b
```
**Examples**: "2020-01-15"

### Pattern 5: Year Only
```regex
\b(19[5-9]\d|20[0-3]\d)\b
```
**Examples**: "2020", "1995"
**Range**: 1950 to current year

---

## Education Date Filtering Logic

### Decision Tree:
```
Is date in "# EDUCATION" section?
├─ YES → Has work keywords (professional experience, work experience)?
│   ├─ YES → Return False (work-related)
│   └─ NO → Return True (education)
│
└─ NO → Has education keywords?
    ├─ YES → Has work keywords?
    │   ├─ YES → Return False (work-related)
    │   └─ NO → Return True (education)
    │
    └─ NO → Has work keywords?
        ├─ YES → Return False (work-related)
        └─ NO → Return False (default to work-related)
```

---

## Error Handling

### LLM Errors:
- **Connection Failed**: Falls back to regex extraction
- **HTTP Error**: Falls back to regex extraction
- **JSON Parse Error**: Falls back to regex extraction
- **Timeout**: Falls back to regex extraction

### Fallback Chain:
```
LLM Error → Regex Fallback → Date Calculation → Return None
```

### All Methods Have Try-Except:
- Graceful degradation
- Detailed error logging
- Multiple fallback strategies

---

## Logging Strategy

### Log Levels:
- **INFO**: Successful extractions, method used
- **WARNING**: Fallbacks, filtered dates, edge cases
- **DEBUG**: Detailed matching, date filtering, calculations
- **ERROR**: Critical failures, JSON parse errors

### Key Log Messages:
- `📅 Starting date-based experience calculation`
- `✅ Date-based calculation: X years`
- `✅ EXPERIENCE EXTRACTED via [method]`
- `❌ No work-related dates found after filtering`
- `Found X total date occurrences in resume`

---

## Configuration

### Settings:
- `ollama_host`: OLLAMA service URL (default: from settings)
- `model`: LLM model name (default: "llama3.1")
- `text_limit`: Max resume text to send to LLM (20000 chars)

### Timeouts:
- Connection check: 5 seconds
- LLM API call: 1200 seconds (20 minutes)

---

## Example Flow

### Scenario: Resume with "25+ years" in summary and "28 years" in work history

```
1. LLM Extraction:
   - Finds "25+ years" in summary
   - Returns {"experience": "25+ years"}
   - ✅ SUCCESS: Returns "25+ years"

2. If LLM fails → Regex Fallback:
   - Finds "25+ years" in summary (has_plus=True, section=summary)
   - Finds "28 years" in work history (has_plus=False, section=work_history)
   - Prioritizes: "25+ years" (has "+" and in summary)
   - ✅ SUCCESS: Returns "25+ years"

3. If Regex fails → Date Calculation:
   - Extracts all dates
   - Filters education dates
   - Finds work dates: Dec 2021 to Aug 2023
   - Calculates: ~20 months = 2 years
   - ✅ SUCCESS: Returns "2 years"
```

---

## Key Features

1. **Multi-Format Support**: Handles various date formats and experience statements
2. **Priority System**: Always prefers explicit "+" format over calculated values
3. **Education Filtering**: Intelligently excludes graduation dates
4. **Robust Fallbacks**: 3-tier strategy ensures maximum extraction success
5. **Context-Aware**: Uses surrounding text to determine date relevance
6. **Comprehensive Logging**: Detailed logs for debugging
7. **Error Resilience**: Graceful handling of all error scenarios

---

## Dependencies

- `httpx`: Async HTTP client for OLLAMA API
- `json`: JSON parsing
- `re`: Regex pattern matching
- `datetime`: Date calculations
- `typing`: Type hints
- `app.config.settings`: Configuration
- `app.utils.logging`: Logging utilities

---

## Future Enhancements

1. **Parallel Processing**: Run LLM and regex simultaneously
2. **Caching**: Cache LLM responses for similar resumes
3. **Confidence Scores**: Return confidence level with extraction
4. **More Date Formats**: Support additional international date formats
5. **ML-Based Filtering**: Use ML to improve education date filtering

