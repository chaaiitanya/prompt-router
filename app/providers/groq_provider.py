"""
app/providers/groq_provider.py — Groq provider implementation.

CONCEPT: Why Groq?
  Groq runs open-source models (Llama 3.1, Mixtral) on custom silicon
  called LPUs (Language Processing Units). This makes them:
    - ~10x cheaper than GPT-4o for equivalent tasks
    - Often faster (sub-500ms for many prompts)

  The Groq SDK is intentionally OpenAI-compatible — same message format,
  same response structure. This means the adapter is nearly identical
  to OpenAI's, with just a different client and model names.

  This is a deliberate design choice by Groq: it lets existing OpenAI
  users switch with minimal code changes. Our gateway benefits from the
  same compatibility.
"""

import logging
import time
from typing import Optional

from groq import AsyncGroq

from app.config import settings
from app.models import Message
from app.providers.base import LLMProvider, ProviderResponse

logger = logging.getLogger(__name__)


class GroqProvider(LLMProvider):
    name = "groq"

    def __init__(self) -> None:
        self._client = AsyncGroq(api_key=settings.groq_api_key)
        self._default_model = settings.provider_default_models["groq"]

    async def complete(
        self,
        messages: list[Message],
        max_tokens: int = 1024,
        model: Optional[str] = None,
    ) -> ProviderResponse:
        target_model = model or self._default_model

        # Groq is OpenAI-compatible: same message dict format
        groq_messages = [{"role": m.role, "content": m.content} for m in messages]

        logger.debug("Groq request | model=%s messages=%d", target_model, len(messages))
        start = time.monotonic()

        response = await self._client.chat.completions.create(
            model=target_model,
            messages=groq_messages,
            max_tokens=max_tokens,
        )

        latency = self._elapsed_ms(start)

        choice = response.choices[0]
        usage = response.usage

        logger.debug(
            "Groq response | model=%s in=%d out=%d latency=%.0fms",
            target_model, usage.prompt_tokens, usage.completion_tokens, latency,
        )

        return ProviderResponse(
            content=choice.message.content or "",
            model=target_model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            latency_ms=latency,
        )
