"""
app/providers/anthropic_provider.py — Anthropic (Claude) provider implementation.

CONCEPT: Anthropic's API differences from OpenAI
  Anthropic's API is similar but has two key differences:
    1. "system" messages are a separate top-level parameter, not part of the
       messages list. We need to extract them before sending.
    2. Token field names differ: input_tokens / output_tokens (no "prompt_" prefix).

  This is exactly why the adapter pattern pays off — these quirks are
  contained here and invisible to the rest of the app.
"""

import logging
import time
from typing import Optional

import anthropic

from app.config import settings
from app.models import Message
from app.providers.base import LLMProvider, ProviderResponse

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._default_model = settings.provider_default_models["anthropic"]

    async def complete(
        self,
        messages: list[Message],
        max_tokens: int = 1024,
        model: Optional[str] = None,
    ) -> ProviderResponse:
        target_model = model or self._default_model

        # Anthropic requires system prompt as a separate argument.
        # Split messages: pull out system messages, pass the rest normally.
        system_content = " ".join(
            m.content for m in messages if m.role == "system"
        )
        conversation = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role != "system"
        ]

        logger.debug("Anthropic request | model=%s messages=%d", target_model, len(conversation))
        start = time.monotonic()

        kwargs: dict = dict(
            model=target_model,
            max_tokens=max_tokens,
            messages=conversation,
        )
        if system_content:
            kwargs["system"] = system_content

        response = await self._client.messages.create(**kwargs)

        latency = self._elapsed_ms(start)

        # Anthropic's response structure:
        # response.content[0].text          → the text (it's a list of content blocks)
        # response.usage.input_tokens
        # response.usage.output_tokens
        content_text = response.content[0].text if response.content else ""
        usage = response.usage

        logger.debug(
            "Anthropic response | model=%s in=%d out=%d latency=%.0fms",
            target_model, usage.input_tokens, usage.output_tokens, latency,
        )

        return ProviderResponse(
            content=content_text,
            model=target_model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            latency_ms=latency,
        )
