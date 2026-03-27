"""Query parser for AI search using OLLAMA or OpenAI LLM (via LLM_MODEL)."""
import json
import re
from typing import Dict, List, Optional
import httpx
from httpx import Timeout

from app.config import settings
from app.utils.logging import get_logger
from app.utils.nlp_utils import fix_common_query_typos, is_likely_name_query
from app.utils.cleaning import normalize_skill_list
from app.services.llm_service import use_openai, generate as openai_generate
# QueryCategoryIdentifier removed - category is now provided explicitly in payload

logger = get_logger(__name__)

# Try to import OLLAMA Python client
try:
    import ollama
    OLLAMA_CLIENT_AVAILABLE = True
except ImportError:
    OLLAMA_CLIENT_AVAILABLE = False
    logger.warning("OLLAMA Python client not available, using HTTP API directly")

AI_SEARCH_PROMPT = """
You are an ATS query parser for recruiters. Your job is to convert the natural language or boolean query into structured JSON. Be conservative about intent, but you MUST fix obvious spelling mistakes in tech/role terms so search and embeddings match real resumes.

0. Spelling, OCR, and keyboard errors (apply FIRST — overrides "literal" for typos only)
   - Recruiters often mistype skills and titles. When a token is clearly a misspelling of a common IT skill, language, framework, or job-title word, **normalize it to standard spelling** in designation, must_have_all, must_have_one_of_groups, and text_for_embedding.
   - Examples of normalization (not exhaustive): developor/devaloper → developer; pythan/pytaan/pythoon → python; enginer → engineer; java script/javascript → javascript (if clearly one term); reactjs → react; anguler/angular; djangoo → django; sqll/sql → sql; kubernates → kubernetes.
   - Rules: (1) Only fix spelling when the intended word is unambiguous (same meaning as the correct term). (2) Do NOT change one technology into another (e.g. never turn "python" into "java"). (3) Do NOT invent skills or roles not implied by the query. (4) For person names or company names, do NOT "correct" spelling unless it is clearly a known tech term mispasted into a name field.
   - If the whole query is a plausible person name with no role/skills, do not apply spelling fixes; use search_type "name" as usual.

1. Extract what the user MEANT, using normalized spellings from section 0 where applicable.
   - Do not infer extra requirements, domains, or experience that are not in the query.
   - Never use knowledge from examples to invent new filters.
   - Never decide that something is a "domain", "company", "acronym", "location", etc. unless it is very clearly used that way in the sentence.

2. Designation (job title / role)
   - Set designation for ANY phrase that is clearly a job title or role, including in AND queries.
   - Do NOT split a job title into skills. Example: "Python Developer" → designation = "python developer" (NOT skills = ["python"])
   - "Senior QA Automation Engineer" → designation = "senior qa automation engineer"
   - In "X AND Y" style: if one term is a role/title (e.g. QA, PM, Developer, Manager, Tester, Analyst) and another is a skill/tech, put the ROLE in designation and the SKILL in must_have_all.
   - Examples: "Python AND QA" → designation = "qa", must_have_all = ["python"]. "sql AND python full stack developer" → designation = "python full stack developer", must_have_all = ["sql"]
   - Single-token roles: "QA", "PM", "BA", "Dev", "Tester", "Manager", "Analyst" etc. in AND with skills → set as designation, NOT in must_have_all.
   - Multi-word role phrases (e.g. "python full stack developer", "front end python developer") → always designation, never split into skills.
   - Treat "IT" in role phrases as role context (e.g. "IT Infrastructure and Software Solutions Expert") and keep the phrase in designation, not in must_have_all.
   - OR role-phrase rule: if query is role phrase OR role phrase (e.g. "python developer OR java developer"), treat both sides as alternative titles, NOT as skills. Do not put "python"/"java"/"developer" into must_have_all.
   - Slash role-phrase rule: if query is "role phrase / role phrase" (e.g. "sr software automation test engineer/selenium automation tester"), treat it like OR between two titles.
   - Slash-title pattern rule: when query looks like "X/Y/Z Developer" (or Engineer/Analyst/etc.), treat the whole query as title context and keep only the base role as designation (e.g. designation="developer"). Do NOT create skill filters from X/Y/Z for this pattern.

3. Skills / must_have_all
   - Only include words or short phrases that are clearly listed as required skills/technologies/tools.
   - Do NOT interpret nouns as skills unless they are in a clear skills list context.
   - "in pharmaceutical domain" → do NOT add "pharmaceutical domain" to skills unless the query explicitly says it is a skill/requirement.
   - Role modifiers are not skills. Words like "independent", "senior/sr", "junior/jr", "lead", "principal", "staff", "contract", "freelance" should stay in title context, not must_have_all.

4. Domain / industry
   - Use filters.domain ONLY when the query clearly describes a business/industry domain for the candidate, e.g. "in financial technology domain", "fintech industry", "healthcare domain".
   - When domain/industry is explicitly mentioned, put a short, lowercase phrase in filters.domain (e.g. "financial technology", "fintech", "healthcare").
   - Do NOT also add that domain phrase to must_have_all unless the query explicitly lists it as a required skill/tool.

5. Experience
   - ONLY extract when there are clear numeric patterns:
     - "15+ years", "5 years", "5-8 years", "at least 10 years", "more than 7 years"
   - Do NOT infer experience from words like: senior, lead, principal, experienced, expert, veteran, etc.

6. Location
   - ONLY when clearly used as a place: "in Bangalore", "located in Texas", "New York"
   - Do NOT treat company names, project names, or product names as locations.

7. Boolean logic – be literal
   - Parse AND / OR / parentheses / quotes exactly as written.
   - "A OR B" → must_have_one_of_groups = [["a"], ["b"]]
   - "(A AND B) OR C" → must_have_one_of_groups = [["a", "b"], ["c"]]
   - Terms without operators → treat as AND → must_have_all
   - Comma-separated lists without AND/OR (e.g. "python, sql, django") → every term is required → must_have_all must list ALL terms (AND semantics); apply section 0 spelling normalization to each segment.
   - Quoted phrases stay together as one term.
   - Slash "/" between terms/phrases should be interpreted as OR intent (A/B -> A OR B), especially for role-style queries.

8. text_for_embedding – mandatory
   - ALWAYS create it.
   - Build it from the query words with **spelling normalized** per section 0 (so embeddings match real resume wording).
   - Order preference: main role phrase → listed skills/tools → experience phrase → location phrase
   - Keep it natural but lowercase.

9. search_type
   - "name" → only when query is clearly just a full name with no other content
   - "hybrid" → when there is clear designation/role + at least one of: numeric experience OR list of skills/tools
   - "semantic" → everything else

10. Absolute prohibitions – YOU MUST NOT:
   - Assume any word is a skill just because it often is in resumes
   - Split job titles into skills
   - Infer experience level from seniority words
   - Decide something is a "domain" or "industry" unless it is explicitly phrased that way in the query (e.g. uses words like "domain", "industry", "sector", "vertical").
   - Use meaning or context from the examples below to invent filters for the current query
   - Add, remove, or rephrase **intent** (new requirements) beyond what the query implies — spelling normalization per section 0 is allowed and required

Output format – STRICT JSON only – nothing else:

{
  "search_type": "semantic" | "name" | "hybrid",
  "text_for_embedding": "lowercase text from the query with tech/role spellings normalized",
  "filters": {
    "designation": null | "exact lowercase phrase",
    "must_have_all": ["exact lowercase term", "another term"],
    "must_have_one_of_groups": [["term1"], ["term2 and phrase"]],
    "min_experience": null | integer,
    "max_experience": null | integer,
    "domain": null | "exact lowercase domain phrase",
    "location": null | "city" | ["city1", "city2"],
    "candidate_name": null | "full name lowercase"
  }
}

Examples are ONLY for understanding the output format — NEVER use their logic, assumptions or categorizations on a new query.
"""


class AISearch2QueryParser:
    """Service for parsing search queries using OLLAMA LLM."""
    
    def __init__(self):
        self.ollama_host = settings.ollama_host
        self.model = "llama3.1"
        # QueryCategoryIdentifier removed - category is now provided explicitly in payload

    def _recompute_role_only(self, parsed_data: Dict) -> None:
        """Recompute role_only after post-processing (e.g. comma AND merge)."""
        filters = parsed_data.get("filters", {})
        designation = filters.get("designation") or ""
        must_all = filters.get("must_have_all") or []
        must_groups = filters.get("must_have_one_of_groups") or []
        has_skills = bool(must_all or (must_groups and any(g for g in must_groups)))
        parsed_data["role_only"] = (
            bool(designation and str(designation).strip())
            and not has_skills
            and filters.get("min_experience") is None
            and filters.get("max_experience") is None
            and not filters.get("location")
            and not filters.get("candidate_name")
        )

    def _merge_comma_separated_and_must_have_all(self, query: str, parsed_data: Dict) -> None:
        """
        Comma-separated segments (without explicit AND/OR) → AND semantics: every segment
        must appear in must_have_all (skills/tools), with simple location routing for
        known city tokens. Merges with LLM output (union of required terms).
        """
        q = (query or "").strip()
        if not q or "," not in q:
            return
        if re.search(r"\bOR\b", q, re.I) or re.search(r"\bAND\b", q, re.I):
            return
        parts = [p.strip() for p in q.split(",") if p.strip()]
        if len(parts) < 2:
            return
        if any(len(p) > 56 for p in parts):
            return

        filters = parsed_data.setdefault("filters", {})
        designation_value = (filters.get("designation") or "").strip().lower()

        # Common city / region tokens → location (first unset wins)
        location_hints = {
            "bangalore", "bengaluru", "hyderabad", "chennai", "mumbai", "pune", "noida",
            "gurgaon", "gurugram", "delhi", "kolkata", "ahmedabad", "kochi", "coimbatore",
            "new york", "seattle", "nashville", "austin", "boston", "chicago", "dallas",
            "texas", "california", "florida", "duncansville",
        }

        merged_skills: List[str] = []
        for p in parts:
            pl = p.strip().lower()
            if not pl:
                continue
            primary_loc = pl.split(",", 1)[0].strip() if pl else ""
            if primary_loc in location_hints or pl in location_hints:
                if not filters.get("location"):
                    filters["location"] = pl if pl in location_hints else primary_loc
                continue
            if designation_value and (
                pl == designation_value
                or (len(pl) <= len(designation_value) + 2 and pl in designation_value)
                or (len(designation_value) <= len(pl) + 2 and designation_value in pl)
            ):
                continue
            merged_skills.extend(normalize_skill_list([p]))

        if not merged_skills:
            return

        existing = list(filters.get("must_have_all") or [])
        combined = normalize_skill_list(existing + merged_skills)
        filters["must_have_all"] = combined
        # ai-search-2: comma lists require AND — downstream must drop candidates missing any term
        parsed_data["comma_strict_and"] = True

    def _apply_slash_role_or_skill_groups(self, query: str, parsed_data: Dict) -> None:
        """
        Pattern: "<x>/<y>/<z> <role>" should be treated as a role-title phrase only.
        Example: "SQL/ETL/SSIS Developer" -> designation=developer, no skill filters,
        and slash_role_tokens=["sql","etl","ssis"] for title-level narrowing downstream.
        """
        q = (query or "").strip()
        if not q or "/" not in q:
            return

        # Avoid overriding explicit boolean intent.
        if re.search(r"\bOR\b|\bAND\b", q, re.I):
            return

        role_heads = {
            "developer", "engineer", "analyst", "architect", "tester",
            "consultant", "administrator", "manager", "scientist",
        }

        # Case A: slash separates full role phrases -> treat as role OR role.
        slash_parts = [part.strip() for part in q.split("/") if part.strip()]
        if len(slash_parts) >= 2:
            normalized_role_phrases: List[str] = []
            all_role_phrases = True
            for part in slash_parts:
                phrase = re.sub(r"[^a-z0-9\s]+", " ", part.lower())
                phrase = re.sub(r"\s+", " ", phrase).strip()
                tokens = phrase.split()
                if len(tokens) < 2 or tokens[-1] not in role_heads:
                    all_role_phrases = False
                    break
                normalized_role_phrases.append(phrase)
            if all_role_phrases and len(normalized_role_phrases) >= 2:
                filters = parsed_data.setdefault("filters", {})
                filters["designation"] = None
                filters["must_have_all"] = []
                filters["must_have_one_of_groups"] = []
                parsed_data["comma_strict_and"] = False
                parsed_data["role_or_phrases"] = normalized_role_phrases
                parsed_data["role_only"] = False
                # Ensure slash token narrowing does not run for role OR phrases.
                parsed_data["slash_role_tokens"] = []
                return

        role_match = re.search(
            r"\b(developer|engineer|analyst|architect|tester|consultant|administrator|manager)\b\s*$",
            q,
            re.I,
        )
        if not role_match:
            return

        role_word = role_match.group(1).lower()
        prefix = q[: role_match.start()].strip()
        if not prefix or "/" not in prefix:
            return

        raw_terms = [part.strip() for part in prefix.split("/") if part.strip()]
        if len(raw_terms) < 2:
            return
        normalized_terms = normalize_skill_list(raw_terms)
        if not normalized_terms:
            return

        filters = parsed_data.setdefault("filters", {})

        # Keep the role explicit when slash-role pattern is detected.
        if not (filters.get("designation") or "").strip():
            filters["designation"] = role_word

        # Slash-title pattern is designation-only: clear skill constraints.
        filters["must_have_all"] = []
        filters["must_have_one_of_groups"] = []
        parsed_data["comma_strict_and"] = False
        # Keep slash tokens for title-only narrowing (e.g., sql/etl developer).
        parsed_data["slash_role_tokens"] = normalized_terms

    def _apply_or_role_phrase_override(self, query: str, parsed_data: Dict) -> None:
        """
        Handle "role phrase OR role phrase" as title alternatives, not skill filters.
        Example: "python developer OR java developer" -> role_or_phrases set,
        designation/skill filters cleared for downstream title-only filtering.
        """
        q = (query or "").strip().lower()
        if not q or " or " not in q:
            return
        parts = [p.strip() for p in re.split(r"\bOR\b", q, flags=re.I) if p.strip()]
        if len(parts) < 2:
            return

        role_heads = {
            "developer", "engineer", "analyst", "architect", "tester",
            "consultant", "administrator", "manager", "scientist",
        }
        normalized_phrases: List[str] = []
        for p in parts:
            if "," in p or "/" in p:
                return
            phrase = re.sub(r"[^a-z0-9\s]+", " ", p)
            phrase = re.sub(r"\s+", " ", phrase).strip()
            if not phrase:
                return
            tokens = phrase.split()
            if len(tokens) < 2 or len(tokens) > 6:
                return
            if tokens[-1] not in role_heads:
                return
            normalized_phrases.append(phrase)

        if len(normalized_phrases) < 2:
            return

        filters = parsed_data.setdefault("filters", {})
        filters["designation"] = None
        filters["must_have_all"] = []
        filters["must_have_one_of_groups"] = []
        parsed_data["comma_strict_and"] = False
        parsed_data["role_or_phrases"] = normalized_phrases
        parsed_data["role_only"] = False

    def _strip_non_skill_modifiers_from_must_have_all(self, parsed_data: Dict) -> None:
        """
        Remove title/seniority/employment modifiers from must_have_all when designation exists.
        Prevents false Pinecone skill filters like {'skills': {'$in': ['independent']}}.
        """
        filters = parsed_data.get("filters", {})
        designation = (filters.get("designation") or "").strip().lower()
        must_all = normalize_skill_list(filters.get("must_have_all") or [])
        if not designation or not must_all:
            return

        non_skill_modifiers = {
            "independent", "senior", "sr", "junior", "jr", "lead", "principal",
            "staff", "contract", "contractor", "freelance", "intern", "associate",
            "assistant", "head", "chief", "it", "expert",
        }
        designation_tokens = set(re.findall(r"[a-z0-9]+", designation))

        cleaned = []
        for token in must_all:
            t = str(token).strip().lower()
            if not t:
                continue
            if t in non_skill_modifiers:
                continue
            # If token is just a designation word and is not a known tech skill, drop it.
            if t in designation_tokens and t in {
                "developer", "engineer", "analyst", "manager", "architect",
                "accountant", "administrator", "consultant", "specialist", "tester",
            }:
                continue
            cleaned.append(t)

        filters["must_have_all"] = cleaned

    def _apply_role_phrase_fallback_from_query(self, query: str, parsed_data: Dict) -> None:
        """
        If parser misses designation for a clear role phrase query, recover by using the
        full query as designation and clearing skill filters.
        Example: "IT Infrastructure and Software Solutions Expert".
        """
        q = (query or "").strip().lower()
        if not q:
            return
        # Skip explicit boolean/list intents handled elsewhere.
        # Keep natural "and" in title phrases (e.g. "infrastructure and software ...").
        if any(sep in q for sep in ["/", ","]) or re.search(r"\bOR\b", q, re.I):
            return

        filters = parsed_data.setdefault("filters", {})
        if (filters.get("designation") or "").strip():
            return

        # Role phrase endings (include "expert" for non-standard titles).
        role_suffixes = (
            "developer", "engineer", "analyst", "architect", "tester",
            "consultant", "administrator", "manager", "specialist", "expert",
            "scientist", "accountant",
        )
        normalized = re.sub(r"[^a-z0-9\s]+", " ", q)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if not normalized:
            return
        if not any(normalized.endswith(f" {suffix}") or normalized == suffix for suffix in role_suffixes):
            return

        filters["designation"] = normalized
        filters["must_have_all"] = []
        filters["must_have_one_of_groups"] = []
        parsed_data["comma_strict_and"] = False
        parsed_data["role_only"] = True
    
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
    
    def _extract_json(self, text: str) -> Dict:
        """Extract JSON object from LLM response."""
        if not text:
            logger.warning("Empty response from LLM")
            return self._default_response()
        
        # Clean the text - remove markdown code blocks if present
        cleaned_text = text.strip()
        cleaned_text = re.sub(r'```json\s*', '', cleaned_text)
        cleaned_text = re.sub(r'```\s*', '', cleaned_text)
        
        # Try to find JSON object
        json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                return self._validate_response(parsed)
            except json.JSONDecodeError:
                pass
        
        # Try parsing the entire cleaned text
        try:
            parsed = json.loads(cleaned_text)
            return self._validate_response(parsed)
        except json.JSONDecodeError:
            logger.warning(
                f"Failed to parse JSON from LLM response. Response preview: {cleaned_text[:500]}"
            )
            return self._default_response()
    
    def _validate_response(self, parsed: Dict) -> Dict:
        """Validate and normalize parsed response."""
        # Ensure required fields exist
        if "search_type" not in parsed:
            parsed["search_type"] = "semantic"
        
        if "text_for_embedding" not in parsed:
            parsed["text_for_embedding"] = ""
        
        if "filters" not in parsed:
            parsed["filters"] = {}
        
        # Ensure filter fields exist
        filters = parsed["filters"]
        if "designation" not in filters:
            filters["designation"] = None
        if "must_have_all" not in filters:
            filters["must_have_all"] = []
        if "must_have_one_of_groups" not in filters:
            filters["must_have_one_of_groups"] = []
        if "min_experience" not in filters:
            filters["min_experience"] = None
        if "max_experience" not in filters:
            filters["max_experience"] = None
        if "domain" not in filters:
            filters["domain"] = None
        if "location" not in filters:
            filters["location"] = None
        if "candidate_name" not in filters:
            filters["candidate_name"] = None

        # Slot validation: coerce numeric and string types
        try:
            if filters.get("min_experience") is not None:
                v = filters["min_experience"]
                filters["min_experience"] = int(v) if v != "" else None
        except (ValueError, TypeError):
            filters["min_experience"] = None
        try:
            if filters.get("max_experience") is not None:
                v = filters["max_experience"]
                filters["max_experience"] = int(v) if v != "" else None
        except (ValueError, TypeError):
            filters["max_experience"] = None
        if filters.get("location") is not None and isinstance(filters["location"], list):
            filters["location"] = (filters["location"][0] if filters["location"] else None) or None
        if filters.get("candidate_name") is not None and isinstance(filters["candidate_name"], list):
            filters["candidate_name"] = (filters["candidate_name"][0] if filters["candidate_name"] else None) or None

        # Heuristic: if designation is missing but we have role-like content in must_have_all,
        # move it into designation instead of treating it as mandatory skills.
        if not filters["designation"]:
            must_all = filters.get("must_have_all") or []
            if isinstance(must_all, list) and len(must_all) >= 1:
                # Normalize must_have_all tokens once
                norm_tokens = [
                    str(t).strip().lower()
                    for t in must_all
                    if isinstance(t, str) and str(t).strip()
                ]
                if not norm_tokens:
                    pass
                # Single multi-word term → designation (e.g. "computer technician", "python full stack developer")
                if len(norm_tokens) == 1:
                    term = norm_tokens[0]
                    if term and " " in term:
                        filters["designation"] = term
                        filters["must_have_all"] = []
                else:
                    # NEW: Multi-token phrase that matches the overall query text → treat as role phrase.
                    # Example: query="Distribution logistics" → must_have_all=["distribution","logistics"]
                    # joined_term="distribution logistics" matches text_for_embedding → designation.
                    text_for_embedding = str(parsed.get("text_for_embedding", "")).lower().strip()
                    joined_term = " ".join(norm_tokens)
                    if joined_term and text_for_embedding and joined_term == text_for_embedding:
                        filters["designation"] = joined_term
                        filters["must_have_all"] = []
                    else:
                        # Single-token role keyword in must_have_all (e.g. "qa" in ["python", "qa"]) → move "qa" to designation
                        role_keywords = {
                            # IT / generic
                            "qa",
                            "pm",
                            "ba",
                            "dev",
                            "tester",
                            "analyst",
                            "manager",
                            "developer",
                            "engineer",
                            "sre",
                            "sdet",
                            # NON-IT / operations / logistics hints
                            "logistics",
                            "operations",
                            "supervisor",
                            "coordinator",
                            "specialist",
                            "executive",
                            "officer",
                            "associate",
                        }
                        role_terms = [t for t in norm_tokens if t in role_keywords]
                        skill_terms = [t for t in norm_tokens if t not in role_keywords]
                        if len(role_terms) == 1 and skill_terms:
                            filters["designation"] = role_terms[0]
                            filters["must_have_all"] = [s for s in skill_terms if s]
                        elif len(role_terms) == 1 and not skill_terms:
                            filters["designation"] = role_terms[0]
                            filters["must_have_all"] = []
        
        # Category fields are not part of LLM output - they are provided explicitly in payload
        parsed["mastercategory"] = None
        parsed["category"] = None
        
        # Normalize search_type
        search_type = parsed["search_type"].lower()
        if search_type not in ["semantic", "name", "hybrid"]:
            parsed["search_type"] = "semantic"
        
        # Normalize designation: convert list to string if needed, then lowercase
        if filters["designation"]:
            designation_value = filters["designation"]
            if isinstance(designation_value, list):
                # If LLM returns a list, take the first element
                designation_value = designation_value[0] if designation_value else None
            if designation_value:
                designation_norm = str(designation_value).lower().strip()
                filters["designation"] = designation_norm or None
        
        # Normalize domain: convert list to string if needed, then lowercase
        if filters["domain"]:
            domain_value = filters["domain"]
            if isinstance(domain_value, list):
                domain_value = domain_value[0] if domain_value else None
            if domain_value:
                domain_norm = str(domain_value).lower().strip()
                filters["domain"] = domain_norm or None
        
        # Normalize location: handle both list and string forms and lowercase
        if filters["location"]:
            loc_value = filters["location"]
            if isinstance(loc_value, list):
                # Take the first non-empty element
                primary = None
                for item in loc_value:
                    if item:
                        primary = str(item).lower().strip()
                        if primary:
                            break
                filters["location"] = primary or None
            else:
                loc_norm = str(loc_value).lower().strip()
                filters["location"] = loc_norm or None
        
        # Normalize candidate_name: convert list to string if needed, then strip
        if filters["candidate_name"]:
            if isinstance(filters["candidate_name"], list):
                # If LLM returns a list, take the first element
                filters["candidate_name"] = filters["candidate_name"][0] if filters["candidate_name"] else None
            if filters["candidate_name"]:
                filters["candidate_name"] = str(filters["candidate_name"]).strip()
                # Set to None if empty string after normalization
                if not filters["candidate_name"]:
                    filters["candidate_name"] = None
        
        # Role-only intent: designation present, no other strong filters → inclusion by role family only
        designation = filters.get("designation") or ""
        must_all = filters.get("must_have_all") or []
        must_groups = filters.get("must_have_one_of_groups") or []
        has_skills = bool(must_all or (must_groups and any(g for g in must_groups)))
        parsed["role_only"] = (
            bool(designation and str(designation).strip())
            and not has_skills
            and filters.get("min_experience") is None
            and filters.get("max_experience") is None
            and not filters.get("location")
            and not filters.get("candidate_name")
        )
        return parsed
    
    def _default_response(self) -> Dict:
        """Return default response structure."""
        return {
            "search_type": "semantic",
            "text_for_embedding": "",
            "filters": {
                "designation": None,
                "must_have_all": [],
                "must_have_one_of_groups": [],
                "min_experience": None,
                "max_experience": None,
                "domain": None,
                "location": None,
                "candidate_name": None
            },
            # Category fields are not part of LLM output - set to None (will be overridden by controller)
            "mastercategory": None,
            "category": None,
            "role_only": False,
            "comma_strict_and": False,
        }
    
    def _infer_mastercategory_from_query(self, query: str, parsed_data: Dict) -> Optional[str]:
        """
        Infer mastercategory from query when LLM identification fails.
        Uses keyword-based heuristics for common IT/NON-IT indicators.
        
        Args:
            query: Original search query
            parsed_data: Parsed query data (may contain designation, skills, etc.)
        
        Returns:
            "IT" or "NON_IT" or None if cannot infer
        """
        query_lower = query.lower()
        designation = (parsed_data.get("filters", {}).get("designation") or "").lower()
        text_for_embedding = parsed_data.get("text_for_embedding", "").lower()
        
        # Combine all text for analysis
        combined_text = f"{query_lower} {designation} {text_for_embedding}".lower()
        
        # IT domain indicators (strong signals)
        it_keywords = [
            # Job titles
            "engineer", "developer", "programmer", "architect", "qa", "automation",
            "devops", "sre", "sdet", "sde", "data engineer", "data scientist",
            "software", "backend", "frontend", "full stack", "fullstack",
            # Technologies
            "python", "java", "javascript", "typescript", "c#", "c++", "go", "rust",
            "sql", "database", "api", "microservices", "kubernetes", "docker",
            "aws", "azure", "gcp", "cloud", "ai", "ml", "machine learning",
            "selenium", "testing", "test automation", "ci/cd", "jenkins", "git"
        ]
        
        # NON-IT domain indicators (strong signals)
        non_it_keywords = [
            # Job titles (generic - check context)
            "manager", "director", "executive", "consultant",
            "sales", "marketing", "hr", "human resources", "recruiter",
            "finance", "accounting", "accountant", "cfo", "controller",
            "operations", "supply chain", "procurement", "vendor",
            # Functions
            "business development", "customer service", "support"
        ]
        
        # Context-aware NON-IT keywords (only if IT context is weak)
        if "it" not in combined_text and "software" not in combined_text:
            # Add context-specific NON-IT keywords
            if "business" in combined_text and "analyst" in combined_text:
                non_it_keywords.append("business analyst")
            if "project" in combined_text and "manager" in combined_text:
                non_it_keywords.append("project manager")
        
        # Count IT indicators
        it_count = sum(1 for keyword in it_keywords if keyword in combined_text)
        
        # Count NON-IT indicators (but exclude if IT indicators are strong)
        non_it_count = 0
        if it_count < 2:  # Only count NON-IT if IT signals are weak
            non_it_count = sum(1 for keyword in non_it_keywords if keyword in combined_text)
        
        # Decision logic
        if it_count >= 2:
            return "IT"
        elif non_it_count >= 2:
            return "NON_IT"
        elif it_count == 1:
            # Single IT indicator - likely IT
            return "IT"
        else:
            # Cannot infer - return None
            return None
    
    async def parse_query(self, query: str, skip_category_inference: bool = False) -> Dict:
        """
        Parse natural language search query into structured format.
        
        Args:
            query: Natural language search query
            skip_category_inference: If True, skip category identification (use explicit category from payload)
        
        Returns:
            Dict with structured search intent
        
        Raises:
            RuntimeError: If OLLAMA is not available or parsing fails
        """
        query = fix_common_query_typos((query or "").strip())

        # NLP gate: if query looks like a person name only, skip LLM and return name search (faster, cheaper)
        # IMPORTANT: Only apply this when caller did NOT already provide an explicit
        # mastercategory/category. If the UI sends a category like "IT" / "Network & Security"
        # together with a query such as "Computer Support Specialist", we should
        # treat it as a role-based semantic search, not as a person name.
        if is_likely_name_query(query) and not skip_category_inference:
            name_lower = query
            logger.info(
                f"Name-query gate: skipping LLM for likely name search",
                extra={"query": name_lower[:80]}
            )
            return {
                "search_type": "name",
                "text_for_embedding": name_lower,
                "effective_query": query,
                "filters": {
                    "designation": None,
                    "must_have_all": [],
                    "must_have_one_of_groups": [],
                    "min_experience": None,
                    "max_experience": None,
                    "domain": None,
                    "location": None,
                    "candidate_name": name_lower,
                },
                "mastercategory": None,
                "category": None,
                "comma_strict_and": False,
            }

        # Query validation: cap length (already stripped + typo-fixed)
        max_len = getattr(settings, "ai_search_query_max_length", 2000)
        if len(query) > max_len:
            logger.warning(f"Query truncated from {len(query)} to {max_len} characters")
            query = query[:max_len]
        if not query:
            raise RuntimeError("Query is empty after trimming")

        # Prepare prompt (used for both OpenAI and OLLAMA)
        full_prompt = f"{AI_SEARCH_PROMPT}\n\nInput Query: {query}\n\nOutput:"

        # LLM_MODEL=OpenAI: use OpenAI for query parsing
        if use_openai():
            if not getattr(settings, "openai_api_key", None) or not str(settings.openai_api_key).strip():
                raise RuntimeError(
                    "LLM_MODEL=OpenAI but OPENAI_API_KEY is not set. "
                    "Set OPENAI_API_KEY in your .env file."
                )
            try:
                raw_output = await openai_generate(
                    prompt=f"Input Query: {query}\n\nOutput:",
                    system_prompt=AI_SEARCH_PROMPT,
                    temperature=0.1,
                    timeout_seconds=300.0,
                )
                parsed_data = self._extract_json(raw_output or "")
                if parsed_data == self._default_response() and raw_output:
                    cleaned = re.sub(r'[^\{\}\[\]",:\s\w]', '', raw_output)
                    try:
                        parsed_data = json.loads(cleaned)
                        parsed_data = self._validate_response(parsed_data)
                    except Exception:
                        pass
                parsed_data["mastercategory"] = None
                parsed_data["category"] = None
                self._merge_comma_separated_and_must_have_all(query, parsed_data)
                self._apply_slash_role_or_skill_groups(query, parsed_data)
                self._apply_or_role_phrase_override(query, parsed_data)
                self._apply_role_phrase_fallback_from_query(query, parsed_data)
                self._strip_non_skill_modifiers_from_must_have_all(parsed_data)
                self._recompute_role_only(parsed_data)
                parsed_data["effective_query"] = query
                logger.info(
                    f"Query parsed successfully (OpenAI): search_type={parsed_data['search_type']}",
                    extra={"search_type": parsed_data["search_type"], "has_filters": bool(parsed_data["filters"])}
                )
                return parsed_data
            except Exception as e:
                logger.error(f"OpenAI query parsing failed: {e}", extra={"error": str(e)})
                raise RuntimeError(f"OpenAI query parsing failed: {e}") from e

        # LLM_MODEL=OLLAMA (default): use OLLAMA
        # Check OLLAMA connection
        is_connected, available_model = await self._check_ollama_connection()
        if not is_connected:
            raise RuntimeError(
                f"OLLAMA is not accessible at {self.ollama_host}. "
                "Please ensure OLLAMA is running."
            )
        
        # Use available model if llama3.1 not found
        model_to_use = self.model
        if available_model and "llama3.1" not in available_model.lower():
            logger.warning(f"llama3.1 not found, using available model: {available_model}")
            model_to_use = available_model
        
        # Try using OLLAMA Python client first
        result = None
        last_error = None
        
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
                        prompt=full_prompt,
                        options={
                            "temperature": 0.1,
                            "top_p": 0.9,
                        }
                    )
                    return {"response": response.get("response", "")}
                
                result = await loop.run_in_executor(None, _generate)
                logger.debug("Successfully used OLLAMA Python client for query parsing")
            except Exception as e:
                logger.warning(f"OLLAMA Python client failed, falling back to HTTP API: {e}")
                result = None
        
        # Fallback to HTTP API
        if result is None:
            async with httpx.AsyncClient(timeout=Timeout(300.0)) as client:
                # Try /api/generate endpoint
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
                    logger.debug("Successfully used /api/generate endpoint for query parsing")
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 404:
                        # Try /api/chat endpoint
                        logger.debug("OLLAMA /api/generate not found, trying /api/chat endpoint")
                        try:
                            response = await client.post(
                                f"{self.ollama_host}/api/chat",
                                json={
                                    "model": model_to_use,
                                    "messages": [
                                        {"role": "system", "content": AI_SEARCH_PROMPT},
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
                            else:
                                raise ValueError("Unexpected response format from OLLAMA chat API")
                            logger.debug("Successfully used /api/chat endpoint for query parsing")
                        except Exception as e2:
                            last_error = e2
                            logger.error(f"OLLAMA /api/chat also failed: {e2}")
                    else:
                        raise
        
        if result is None:
            raise RuntimeError(
                f"All OLLAMA API endpoints failed. "
                f"OLLAMA is running at {self.ollama_host} but endpoints return errors. "
                f"Last error: {last_error}"
            )
        
        # Extract JSON from response
        raw_output = result.get("response", "")
        parsed_data = self._extract_json(raw_output)
        
        # Retry once if parsing failed
        if parsed_data == self._default_response() and raw_output:
            logger.warning("Initial JSON parsing failed, retrying with cleaned text")
            # Try to fix common JSON issues
            cleaned = re.sub(r'[^\{\}\[\]",:\s\w]', '', raw_output)
            try:
                parsed_data = json.loads(cleaned)
                parsed_data = self._validate_response(parsed_data)
            except:
                pass
        
        # Skip category identification if flag is set (explicit category provided in payload)
        if skip_category_inference:
            logger.debug("Skipping category inference (explicit category provided in payload)")
        else:
            # Category identification removed - always use explicit category from payload
            logger.debug("Category inference is disabled - using explicit category from payload")
        
        # Do not set mastercategory or category - they will be provided from payload
        parsed_data["mastercategory"] = None
        parsed_data["category"] = None

        self._merge_comma_separated_and_must_have_all(query, parsed_data)
        self._apply_slash_role_or_skill_groups(query, parsed_data)
        self._apply_or_role_phrase_override(query, parsed_data)
        self._apply_role_phrase_fallback_from_query(query, parsed_data)
        self._strip_non_skill_modifiers_from_must_have_all(parsed_data)
        self._recompute_role_only(parsed_data)
        parsed_data["effective_query"] = query

        logger.info(
            f"Query parsed successfully: search_type={parsed_data['search_type']}",
            extra={
                "search_type": parsed_data["search_type"],
                "has_filters": bool(parsed_data["filters"]),
                "skip_category_inference": skip_category_inference
            }
        )

        return parsed_data
        