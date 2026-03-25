"""
app/models.py — Request and response shapes for the gateway API.

CONCEPT: Pydantic models
  A Pydantic model is a Python class where every field has a type annotation.
  When FastAPI receives an HTTP request, it automatically:
    1. Parses the JSON body
    2. Validates that each field has the right type
    3. Raises a clear 422 error if something is wrong (before your code even runs)

  This means you never have to write "if 'messages' not in data: raise error" —
  Pydantic does it for you.

CONCEPT: Literal types
  Literal["cheapest", "fastest", "task_type"] means the field can ONLY be one
  of those exact string values. Anything else → validation error automatically.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# Shared building blocks
# ─────────────────────────────────────────────────────────────────────────────

class Message(BaseModel):
    """
    A single chat message — same format as OpenAI's API.
    Using a compatible format means clients written for OpenAI can talk
    to our gateway with zero changes.

    role: who sent this message
      "system"    → instructions to the model (e.g. "You are a helpful assistant")
      "user"      → what the human typed
      "assistant" → a previous model response (for multi-turn conversations)
    """
    role: Literal["system", "user", "assistant"]
    content: str


# ─────────────────────────────────────────────────────────────────────────────
# Gateway request — what clients send to POST /v1/chat/completions
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """
    The payload your client sends to the gateway.

    Why not just forward the raw request to OpenAI?
    Because we added extra fields (strategy, task_type) that control routing.
    The gateway strips those before forwarding to the actual LLM.
    """

    # Required: the conversation history (at least one message)
    messages: list[Message] = Field(
        ...,
        min_length=1,
        description="Conversation messages. At minimum one 'user' message.",
    )

    # Optional: which routing strategy to use
    # "cheapest"  → pick the lowest cost-per-token provider
    # "fastest"   → pick the provider with best recent latency
    # "task_type" → pick based on what kind of task it is (see task_type field)
    strategy: Literal["cheapest", "fastest", "task_type"] = Field(
        default="cheapest",
        description="Routing strategy to apply.",
    )

    # Optional: hint about the task type, used when strategy="task_type"
    task_type: Optional[Literal["summarize", "code", "analysis", "default"]] = Field(
        default="default",
        description="Task hint used by 'task_type' routing strategy.",
    )

    # Optional: force a specific provider (bypasses routing logic)
    # Useful for testing individual providers
    provider_override: Optional[Literal["openai", "anthropic", "groq"]] = Field(
        default=None,
        description="Force a specific provider, skipping routing strategy.",
    )

    # Optional: cap on how many tokens the response can use
    max_tokens: int = Field(
        default=1024,
        ge=1,
        le=4096,
        description="Max tokens in the response. Higher = more expensive.",
    )

    class Config:
        # Shows example values in /docs — great for learning
        json_schema_extra = {
            "example": {
                "messages": [
                    {"role": "user", "content": "Explain quantum entanglement in simple terms."}
                ],
                "strategy": "cheapest",
                "task_type": "default",
                "max_tokens": 512,
            }
        }


# ─────────────────────────────────────────────────────────────────────────────
# Gateway response — what the gateway sends back to the client
# ─────────────────────────────────────────────────────────────────────────────

class ChatResponse(BaseModel):
    """
    The enriched response from the gateway.

    Unlike a raw LLM response, this includes cost/latency metadata so
    callers can understand what happened behind the scenes.
    """

    # The actual text the model generated
    content: str

    # Which provider and model actually handled this request
    provider: str   # e.g. "groq"
    model: str      # e.g. "llama-3.1-8b-instant"

    # Token counts (used to calculate cost)
    input_tokens: int
    output_tokens: int

    # Dollar cost of this request (input + output tokens combined)
    cost_usd: float = Field(description="Total cost in USD for this request.")

    # How long the LLM took to respond (milliseconds)
    latency_ms: float

    # Was this served from cache? If True, cost_usd is effectively $0
    cached: bool = False

    # Which strategy was used to pick this provider
    strategy_used: str

    class Config:
        json_schema_extra = {
            "example": {
                "content": "Quantum entanglement is when two particles...",
                "provider": "groq",
                "model": "llama-3.1-8b-instant",
                "input_tokens": 18,
                "output_tokens": 142,
                "cost_usd": 0.0000134,
                "latency_ms": 312.5,
                "cached": False,
                "strategy_used": "cheapest",
            }
        }


# ─────────────────────────────────────────────────────────────────────────────
# Health check response
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str = "0.1.0"
    environment: str


# ─────────────────────────────────────────────────────────────────────────────
# Spend analytics responses
# ─────────────────────────────────────────────────────────────────────────────

class SpendSummaryResponse(BaseModel):
    """All-time spend totals returned by GET /v1/spend."""
    total_usd: float
    request_count: int
    total_input_tokens: int
    total_output_tokens: int
    by_provider: dict[str, float]
    by_model: dict[str, float]
    recent_requests: Optional[list[dict]] = None


class SpendWindowResponse(BaseModel):
    """Time-windowed spend returned by GET /v1/spend/since/{seconds}."""
    window_seconds: float
    total_usd: float
    request_count: int
    by_provider: dict[str, float]
    by_model: dict[str, float]
