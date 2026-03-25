"""
app/providers/openai_provider.py — OpenAI provider implementation.

CONCEPT: Adapter pattern
  The OpenAI SDK uses its own message format, token counting field names,
  and model identifiers. This file "adapts" OpenAI's specific interface
  into our standard LLMProvider interface.

  The rest of the codebase never imports `openai` directly — it only
  ever calls provider.complete(). That means we could swap OpenAI for
  a different provider with zero changes to the gateway logic.

CONCEPT: AsyncOpenAI vs OpenAI
  openai.OpenAI() is synchronous — it blocks the entire server while
  waiting for a response. openai.AsyncOpenAI() returns a coroutine
  that the event loop can pause, letting other requests proceed.
  Always use the async client in an async FastAPI app.
"""

import logging
import time
from typing import Optional

from openai import AsyncOpenAI

from app.config import settings
from app.models import Message
from app.providers.base import LLMProvider, ProviderResponse

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self) -> None:
        # Create one async client at import time and reuse it.
        # Creating a new client per request would waste time on TCP handshakes.
        # This is the "connection pooling" pattern.
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._default_model = settings.provider_default_models["openai"]

    async def complete(
        self,
        messages: list[Message],
        max_tokens: int = 1024,
        model: Optional[str] = None,
    ) -> ProviderResponse:
        target_model = model or self._default_model

        # Convert our Message objects → OpenAI's dict format
        # Our: Message(role="user", content="Hi")
        # OpenAI expects: {"role": "user", "content": "Hi"}
        openai_messages = [{"role": m.role, "content": m.content} for m in messages]

        logger.debug("OpenAI request | model=%s messages=%d", target_model, len(messages))
        start = time.monotonic()

        response = await self._client.chat.completions.create(
            model=target_model,
            messages=openai_messages,
            max_tokens=max_tokens,
        )

        latency = self._elapsed_ms(start)

        # OpenAI's response structure:
        # response.choices[0].message.content  → the text
        # response.usage.prompt_tokens         → input tokens
        # response.usage.completion_tokens     → output tokens
        choice = response.choices[0]
        usage = response.usage

        logger.debug(
            "OpenAI response | model=%s in=%d out=%d latency=%.0fms",
            target_model, usage.prompt_tokens, usage.completion_tokens, latency,
        )

        return ProviderResponse(
            content=choice.message.content or "",
            model=target_model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            latency_ms=latency,
        )
