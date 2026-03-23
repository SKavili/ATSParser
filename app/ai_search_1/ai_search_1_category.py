"""Identify a single ATS category from the query, given mastercategory (IT or NON_IT)."""
import json
import re
from typing import Dict, List, Optional

import httpx
from httpx import Timeout

from app.config import settings
from app.category.category_extractor import IT_CATEGORY_PROMPT, NON_IT_CATEGORY_PROMPT
from app.models.ai_search_models import _normalize_mastercategory
from app.services.llm_service import use_openai, generate as openai_generate
from app.services.pinecone_automation import PineconeAutomation
from app.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import ollama

    OLLAMA_CLIENT_AVAILABLE = True
except ImportError:
    OLLAMA_CLIENT_AVAILABLE = False
    logger.warning("OLLAMA Python client not available for ai_search_1 category")

_CATEGORY_SYSTEM = """You classify recruiter search queries for an ATS.

TASK: Pick exactly ONE category from the list provided for the given mastercategory.
The category string MUST match one list entry exactly (same spelling and punctuation).

RULES:
- Do NOT invent categories.
- Use ONLY the allowed category list passed in the prompt.
- Do NOT return numbers, bullets, prefixes, or explanations.
- Do NOT shorten category names.
- If you cannot map confidently, set category to null.

OUTPUT: STRICT JSON only, no markdown:
{"category":"..."|null}
"""


def _extract_categories_from_prompt(prompt_text: str) -> List[str]:
    """Extract numbered categories from category_extractor prompt text."""
    categories: List[str] = []
    lines = prompt_text.split("\n")
    in_category_section = False

    for line in lines:
        line = line.strip()
        if "SAMPLE" in line and "CATEGORIES:" in line:
            in_category_section = True
            continue
        if in_category_section and line and not line[0].isdigit() and "ASSESSMENT" in line:
            break
        if in_category_section and line and line[0].isdigit():
            category = re.sub(r"^\d+\.\s*", "", line)
            if category:
                categories.append(category)
    return categories


IT_ALLOWED_CATEGORIES = _extract_categories_from_prompt(IT_CATEGORY_PROMPT)
NON_IT_ALLOWED_CATEGORIES = _extract_categories_from_prompt(NON_IT_CATEGORY_PROMPT)


def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def infer_category_from_token_match(mastercategory: str, token: str) -> Optional[str]:
    """
    Dynamic, non-static token-to-category match using current allowed lists.
    Used for comma-ordered focus terms before LLM fallback.
    """
    mc = _normalize_mastercategory(mastercategory)
    if mc not in ("IT", "NON_IT"):
        return None
    t = (token or "").strip().lower()
    if not t:
        return None
    # 1) Strict lexical token match against category title.
    token_pattern = re.compile(rf"\b{re.escape(t)}\b")
    allowed = IT_ALLOWED_CATEGORIES if mc == "IT" else NON_IT_ALLOWED_CATEGORIES
    for c in allowed:
        if token_pattern.search(c.lower()):
            return c

    # 2) Semantic token aliases for common, high-signal skills (non-static category names).
    # These aliases map a token to a category *family* and then pick the matching
    # category from the current allowed list.
    alias_to_category_terms = {
        "sql": ["database", "data technologies", "data engineer"],
        "mysql": ["database", "data technologies"],
        "postgres": ["database", "data technologies"],
        "oracle": ["database", "data technologies"],
        "mongodb": ["database", "data technologies"],
        "python": ["python"],
        "java": ["java"],
        "dotnet": [".net", "dotnet"],
        "aws": ["aws", "cloud platforms"],
        "azure": ["azure", "cloud platforms"],
        "gcp": ["cloud"],
        "devops": ["devops", "platform engineering"],
    }

    # Normalize token to a simple key (keep original as fallback).
    token_key = re.sub(r"[^a-z0-9\.\+#]+", "", t)
    terms = alias_to_category_terms.get(token_key) or alias_to_category_terms.get(t) or []
    if terms:
        for c in allowed:
            c_low = c.lower()
            if any(term in c_low for term in terms):
                return c
    return None


def match_allowed_category(raw: Optional[str], allowed: List[str]) -> Optional[str]:
    """Map model output to a canonical label from the allowed list."""
    if not raw or not allowed:
        return None
    raw_stripped = raw.strip()
    # Normalize common model noise: numbering, bullets, quotes, "category:" labels
    raw_stripped = re.sub(r"^\s*(category\s*[:\-]\s*)", "", raw_stripped, flags=re.IGNORECASE)
    raw_stripped = re.sub(r"^\s*[\-\*\u2022]\s*", "", raw_stripped)
    raw_stripped = re.sub(r"^\s*\d+\s*[\.\)\-:]\s*", "", raw_stripped)
    raw_stripped = raw_stripped.strip(" \"'`")

    # Strict exact match first (required behavior)
    for c in allowed:
        if c.strip() == raw_stripped:
            return c

    # Case/punctuation-insensitive exact-equivalent match
    rk = _norm_key(raw)
    for c in allowed:
        if _norm_key(c) == rk:
            return c

    # IMPORTANT: Do not use substring/fuzzy "new category" assumptions.
    return None


def match_override_category(
    mastercategory: str,
    category: str,
    pinecone_automation: PineconeAutomation,
) -> Optional[str]:
    """Map user-provided category string to the canonical label from IT/NON_IT lists."""
    it_cats = IT_ALLOWED_CATEGORIES
    non_it_cats = NON_IT_ALLOWED_CATEGORIES
    mc = _normalize_mastercategory(mastercategory)
    if mc == "IT":
        return match_allowed_category(category, it_cats)
    if mc == "NON_IT":
        return match_allowed_category(category, non_it_cats)
    return None


def _extract_json(raw: str) -> Optional[Dict]:
    if not raw:
        return None
    text = raw.strip()
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


async def _ollama_category(full_prompt: str, system_for_chat: str) -> str:
    host = settings.ollama_host
    model = "llama3.1"
    async with httpx.AsyncClient(timeout=Timeout(120.0)) as client:
        response = await client.post(
            f"{host}/api/generate",
            json={
                "model": model,
                "prompt": full_prompt,
                "stream": False,
                "options": {"temperature": 0.1, "top_p": 0.9},
            },
        )
        if response.status_code == 404:
            response = await client.post(
                f"{host}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_for_chat},
                        {"role": "user", "content": full_prompt.split("Output:", 1)[-1].strip()},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.1, "top_p": 0.9},
                },
            )
            response.raise_for_status()
            data = response.json()
            if "message" in data and "content" in data["message"]:
                return data["message"]["content"]
            return ""
        response.raise_for_status()
        data = response.json()
        return data.get("response", "") or ""


async def infer_category_from_query(
    query: str,
    mastercategory: str,
    pinecone_automation: PineconeAutomation,
) -> Optional[str]:
    """
    Return one canonical category string from category_extractor IT/NON_IT lists, or None.

    mastercategory must be 'IT' or 'NON_IT' (normalized).
    """
    q = (query or "").strip()
    if not q:
        return None

    mc = _normalize_mastercategory(mastercategory)
    if mc not in ("IT", "NON_IT"):
        return None

    it_cats = IT_ALLOWED_CATEGORIES
    non_it_cats = NON_IT_ALLOWED_CATEGORIES
    allowed = it_cats if mc == "IT" else non_it_cats

    block = "\n".join(f"{i+1}. {c}" for i, c in enumerate(allowed))
    user_block = f"""mastercategory: {mc}

Query:
{q}

Allowed categories (pick exactly one):
{block}

Output JSON only."""

    raw_output = ""
    try:
        if use_openai():
            if not getattr(settings, "openai_api_key", None) or not str(settings.openai_api_key).strip():
                raise RuntimeError("LLM_MODEL=OpenAI but OPENAI_API_KEY is not set.")
            raw_output = await openai_generate(
                prompt=user_block,
                system_prompt=_CATEGORY_SYSTEM,
                temperature=0.1,
                timeout_seconds=120.0,
            )
        else:
            full_prompt = f"{_CATEGORY_SYSTEM}\n\n{user_block}\n\nOutput:"
            if OLLAMA_CLIENT_AVAILABLE:
                try:
                    import asyncio

                    loop = asyncio.get_event_loop()

                    def _gen():
                        client = ollama.Client(
                            host=settings.ollama_host.replace("http://", "").replace("https://", "")
                        )
                        r = client.generate(
                            model="llama3.1",
                            prompt=full_prompt,
                            options={"temperature": 0.1, "top_p": 0.9},
                        )
                        return r.get("response", "") or ""

                    raw_output = await loop.run_in_executor(None, _gen)
                except Exception as e:
                    logger.warning(f"OLLAMA client category failed, HTTP fallback: {e}")
                    raw_output = await _ollama_category(full_prompt, _CATEGORY_SYSTEM)
            else:
                raw_output = await _ollama_category(full_prompt, _CATEGORY_SYSTEM)
    except Exception as e:
        logger.error(f"Category LLM failed: {e}", extra={"error": str(e)})
        raw_output = ""

    data = _extract_json(raw_output) or {}
    cat_raw = data.get("category")
    cat = match_allowed_category(cat_raw if isinstance(cat_raw, str) else None, allowed)

    if not cat and use_openai() and getattr(settings, "openai_api_key", None):
        try:
            small_list = "\n".join(f"- {c}" for c in allowed)
            fix = await openai_generate(
                prompt=f'Query:\n{q}\n\nPick exactly one category from:\n{small_list}\n\nReply JSON: {{"category":"..."}}',
                system_prompt="Pick one category from the list only. JSON only.",
                temperature=0.0,
                timeout_seconds=60.0,
            )
            parsed_fix = _extract_json(fix) or {}
            c2 = parsed_fix.get("category")
            cat = match_allowed_category(c2 if isinstance(c2, str) else None, allowed)
        except Exception as e:
            logger.debug(f"Category retry LLM failed: {e}")

    logger.info(
        f"ai_search_1 category: {cat}",
        extra={"category": cat, "mastercategory": mc},
    )
    return cat
