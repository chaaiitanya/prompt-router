"""
app/providers/base.py — The abstract contract every LLM provider must follow.

CONCEPT: Abstract Base Class (ABC)
  An abstract class is a template that defines WHAT a class must do,
  without saying HOW. Any class that inherits from LLMProvider must
  implement complete() or Python will raise a TypeError at startup.

  Why bother? Because the gateway only needs to call provider.complete().
  It doesn't care whether that's OpenAI, Anthropic, or Groq under the hood.
  This is the "program to an interface, not an implementation" principle.

  Analogy: A power socket is an abstract interface. You can plug in a lamp,
  a phone charger, or a fan — they all speak the same protocol (voltage/pins).
  The wall doesn't need to know which device is plugged in.

CONCEPT: dataclass
  A dataclass is a class that mainly holds data. Python auto-generates
  __init__, __repr__, and __eq__ for you based on the field annotations.
  It's like a lightweight Pydantic model, but without HTTP validation —
  used for internal data passing between modules.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from app.models import Message


@dataclass
class ProviderResponse:
    """
    Standardized response from any LLM provider.

    Every provider (OpenAI, Anthropic, Groq) returns data in a different
    format. We map all of them into this single shape so the gateway
    doesn't need provider-specific parsing logic everywhere.
    """
    content: str          # The generated text
    model: str            # Exact model that responded (e.g. "gpt-4o-mini")
    input_tokens: int     # Tokens in the prompt (affects cost)
    output_tokens: int    # Tokens in the response (affects cost)
    latency_ms: float     # Wall-clock time from request to response


class LLMProvider(ABC):
    """
    Abstract base class for all LLM providers.

    To add a new provider (e.g. Cohere, Mistral):
      1. Create app/providers/cohere_provider.py
      2. Subclass LLMProvider
      3. Implement complete()
      4. Register it in the gateway

    That's it — no other files need to change.
    """

    # Subclasses declare their provider name here (used in logs + analytics)
    name: str = "base"

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        max_tokens: int = 1024,
        model: Optional[str] = None,
    ) -> ProviderResponse:
        """
        Call the LLM and return a standardized ProviderResponse.

        Args:
            messages:   Conversation history (same format across providers)
            max_tokens: Hard cap on response length
            model:      Optional override — if None, use provider's default model

        Returns:
            ProviderResponse with content, token counts, and latency
        """
        ...

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        """Helper: milliseconds elapsed since `start` (from time.monotonic())."""
        return round((time.monotonic() - start) * 1000, 2)
