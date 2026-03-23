"""Identify mastercategory (IT vs NON_IT) from a recruiter search query."""
import json
import re
from typing import Dict, Optional

import httpx
from httpx import Timeout

from app.config import settings
from app.models.ai_search_models import _normalize_mastercategory
from app.services.llm_service import use_openai, generate as openai_generate
from app.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import ollama

    OLLAMA_CLIENT_AVAILABLE = True
except ImportError:
    OLLAMA_CLIENT_AVAILABLE = False
    logger.warning("OLLAMA Python client not available for ai_search_1 mastercategory")

_MASTER_SYSTEM = """You classify recruiter search queries for an ATS.

TASK: Decide if the query is looking for candidates in IT (technical / software / engineering)
or NON_IT (business, finance, HR, sales, operations, etc. without primary technology delivery).

RULES:
- "IT" = software development, QA automation, DevOps/SRE, data engineering, cloud engineering,
  security engineering, technical support with strong stack, etc.
- "NON_IT" = HR, recruiting (non-tech), sales, marketing, finance, accounting, operations,
  supply chain, legal, healthcare non-clinical admin, etc.
- Output exactly "IT" or "NON_IT". If truly impossible to tell, use null.

OUTPUT: STRICT JSON only, no markdown:
{"mastercategory":"IT"|"NON_IT"|null}
"""


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


def infer_mastercategory_heuristic(query: str, parsed_data: Dict) -> Optional[str]:
    """Rule-based IT vs NON_IT when LLM output is missing or unclear."""
    query_lower = query.lower()
    designation = (parsed_data.get("filters", {}) or {}).get("designation") or ""
    designation = str(designation).lower()
    text_for_embedding = str(parsed_data.get("text_for_embedding", "")).lower()
    combined_text = f"{query_lower} {designation} {text_for_embedding}"

    it_keywords = [
        "engineer", "developer", "programmer", "architect", "qa", "automation",
        "devops", "sre", "sdet", "sde", "data engineer", "data scientist",
        "software", "backend", "frontend", "full stack", "fullstack",
        "python", "java", "javascript", "typescript", "c#", "c++", "go", "rust",
        "sql", "database", "api", "microservices", "kubernetes", "docker",
        "aws", "azure", "gcp", "cloud", "ai", "ml", "machine learning",
        "selenium", "testing", "test automation", "ci/cd", "jenkins", "git",
    ]
    non_it_keywords = [
        "sales", "marketing", "hr", "human resources", "recruiter",
        "finance", "accounting", "accountant", "cfo", "controller",
        "operations", "supply chain", "procurement", "vendor",
        "business development", "customer service",
    ]
    it_count = sum(1 for k in it_keywords if k in combined_text)
    non_it_count = sum(1 for k in non_it_keywords if k in combined_text)
    if it_count >= 2:
        return "IT"
    if non_it_count >= 2 and it_count < 2:
        return "NON_IT"
    if it_count == 1:
        return "IT"
    return None


async def _ollama_mastercategory(full_prompt: str) -> str:
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
                        {"role": "system", "content": _MASTER_SYSTEM},
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


async def infer_mastercategory_from_query(query: str, parsed_query: Dict) -> Optional[str]:
    """
    Return canonical mastercategory: 'IT', 'NON_IT', or None.

    Uses LLM when configured; falls back to infer_mastercategory_heuristic.
    """
    q = (query or "").strip()
    if not q:
        return None

    user_block = f"Query:\n{q}\n\nOutput JSON only."
    raw_output = ""
    try:
        if use_openai():
            if not getattr(settings, "openai_api_key", None) or not str(settings.openai_api_key).strip():
                raise RuntimeError("LLM_MODEL=OpenAI but OPENAI_API_KEY is not set.")
            raw_output = await openai_generate(
                prompt=user_block,
                system_prompt=_MASTER_SYSTEM,
                temperature=0.1,
                timeout_seconds=120.0,
            )
        else:
            full_prompt = f"{_MASTER_SYSTEM}\n\n{user_block}\n\nOutput:"
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
                    logger.warning(f"OLLAMA client mastercategory failed, HTTP fallback: {e}")
                    raw_output = await _ollama_mastercategory(full_prompt)
            else:
                raw_output = await _ollama_mastercategory(full_prompt)
    except Exception as e:
        logger.error(f"Mastercategory LLM failed: {e}", extra={"error": str(e)})
        raw_output = ""

    data = _extract_json(raw_output) or {}
    mc_raw = data.get("mastercategory")
    mc: Optional[str] = None
    if mc_raw is not None:
        mc = _normalize_mastercategory(str(mc_raw))

    if not mc:
        mc = infer_mastercategory_heuristic(q, parsed_query)

    logger.info(f"ai_search_1 mastercategory: {mc}", extra={"mastercategory": mc})
    return mc
