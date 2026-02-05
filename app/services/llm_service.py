"""Shared LLM service: OpenAI when LLM_MODEL=OpenAI, otherwise callers use OLLAMA."""
from typing import Optional
import httpx
from httpx import Timeout

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"


def use_openai() -> bool:
    """True if LLM provider is OpenAI (from LLM_MODEL env)."""
    return getattr(settings, "llm_model", "OLLAMA").strip().upper() == "OPENAI"


async def generate(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.1,
    timeout_seconds: float = 300.0,
) -> str:
    """
    Generate completion using OpenAI API. Use when LLM_MODEL=OpenAI.
    Callers should check use_openai() first and only call this when True.

    Args:
        prompt: User/content prompt.
        system_prompt: Optional system message for chat.
        temperature: 0.0-1.0.
        timeout_seconds: Request timeout.

    Returns:
        Generated text (content of the first choice).

    Raises:
        ValueError: If OPENAI_API_KEY is missing when OpenAI is configured.
        RuntimeError: On API errors.
    """
    api_key = getattr(settings, "openai_api_key", None)
    if not api_key or not str(api_key).strip():
        raise ValueError(
            "OPENAI_API_KEY is required when LLM_MODEL=OpenAI. "
            "Set OPENAI_API_KEY in your .env file."
        )
    model = getattr(settings, "openai_model", "gpt-3.5-turbo") or "gpt-3.5-turbo"

    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": prompt})

    async with httpx.AsyncClient(timeout=Timeout(timeout_seconds)) as client:
        response = await client.post(
            OPENAI_CHAT_URL,
            headers={
                "Authorization": f"Bearer {api_key.strip()}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
            },
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("OpenAI API returned no choices")
        content = choices[0].get("message", {}).get("content", "")
        return content if content is not None else ""
