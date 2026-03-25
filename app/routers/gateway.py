"""
app/routers/gateway.py — The main API endpoint (Phase 3: real routing strategies).

CONCEPT: provider_override vs routing strategy
  Two ways to pick a provider:
    1. provider_override: "openai" / "anthropic" / "groq"
       → Caller explicitly forces a provider. Bypasses routing entirely.
       → Useful for testing individual providers.
    2. strategy-based routing (Phase 3: now fully implemented)
       → cheapest  — lowest input cost/token across providers
       → fastest   — lowest rolling-average latency (falls back to cheapest on cold start)
       → task_type — config-driven map (summarize→groq, code→openai, analysis→anthropic)

CONCEPT: HTTPException
  FastAPI's way of returning HTTP error responses.
  raise HTTPException(status_code=400, detail="reason")
  → Client gets JSON: {"detail": "reason"} with HTTP 400.

CONCEPT: try/except for provider errors
  External API calls can fail: network timeout, rate limit, invalid key.
  We catch those here and convert them to a clean 502 (Bad Gateway)
  error rather than a 500 (Internal Server Error) with a stack trace.
  Callers get a useful message; our internals stay private.
"""

import logging

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.models import ChatRequest, ChatResponse
from app.providers.base import LLMProvider
from app.providers.openai_provider import OpenAIProvider
from app.providers.anthropic_provider import AnthropicProvider
from app.providers.groq_provider import GroqProvider
from app.routing import latency_tracker, pick_cheapest, pick_fastest, pick_by_task_type

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Provider registry ─────────────────────────────────────────────────────────
# Instantiated once at module load time so each provider reuses its
# HTTP connection pool across requests (much faster than creating per-request).
#
# CONCEPT: dict as a registry
#   A dict mapping string names to objects is a simple "registry" pattern.
#   To add a new provider: create the class, add one line here.
#   No if/elif chains needed anywhere else.
_PROVIDERS: dict[str, LLMProvider] = {
    "openai":    OpenAIProvider(),
    "anthropic": AnthropicProvider(),
    "groq":      GroqProvider(),
}


def _pick_provider(request: ChatRequest) -> tuple[LLMProvider, str]:
    """
    Decide which provider to use for this request.

    Returns (provider_instance, strategy_label_for_logging).

    Priority:
      1. provider_override  → bypass routing entirely
      2. strategy="cheapest"  → pick_cheapest()
      3. strategy="fastest"   → pick_fastest() (falls back to cheapest on cold start)
      4. strategy="task_type" → pick_by_task_type() using config map
    """
    if request.provider_override:
        provider = _PROVIDERS[request.provider_override]
        return provider, f"override:{request.provider_override}"

    if request.strategy == "cheapest":
        return pick_cheapest(_PROVIDERS)

    if request.strategy == "fastest":
        return pick_fastest(_PROVIDERS)

    # strategy == "task_type"
    return pick_by_task_type(_PROVIDERS, request.task_type or "default")


@router.post(
    "/v1/chat/completions",
    response_model=ChatResponse,
    summary="Route a chat completion request",
    description=(
        "Routes the request to an LLM provider based on the selected strategy, "
        "calls the provider, and returns an enriched response with cost metadata."
    ),
)
async def chat_completions(request: ChatRequest) -> ChatResponse:
    """
    Main gateway handler (Phase 3).

    Flow:
      1. Pick provider via strategy               ← Phase 2
      2. Call provider.complete()                 ← Phase 2
      3. Record latency for future fastest picks  ← Phase 3
      4. Calculate cost from token counts         ← Phase 2
      5. Return enriched ChatResponse             ← Phase 2

    Phase 4 will add: semantic cache check before calling provider
    Phase 5 will add: cost_tracker.record() call
    Phase 6 will add: prometheus metrics
    """
    provider, strategy_label = _pick_provider(request)

    logger.info(
        "Routing request | provider=%s strategy=%s messages=%d",
        provider.name, strategy_label, len(request.messages),
    )

    # ── Call the provider ────────────────────────────────────────────────────
    try:
        result = await provider.complete(
            messages=request.messages,
            max_tokens=request.max_tokens,
        )
    except Exception as exc:
        # Log the real error server-side, return a safe message to the client
        logger.error("Provider %s failed: %s", provider.name, exc, exc_info=True)
        raise HTTPException(
            status_code=502,
            detail=f"Provider '{provider.name}' returned an error. Check server logs.",
        ) from exc

    # ── Record latency for future fastest-strategy picks ─────────────────────
    # We do this before the cost block so the data is available immediately
    # for the next request even if cost calculation somehow raises.
    latency_tracker.record(provider.name, result.latency_ms)

    # ── Calculate cost ───────────────────────────────────────────────────────
    # cost = (input_tokens × input_price_per_token) + (output_tokens × output_price_per_token)
    # settings.cost_per_token() handles the per-million → per-token conversion
    input_cost  = result.input_tokens  * settings.cost_per_token(result.model, "input")
    output_cost = result.output_tokens * settings.cost_per_token(result.model, "output")
    total_cost  = round(input_cost + output_cost, 8)

    logger.info(
        "Request complete | provider=%s model=%s in=%d out=%d cost=$%.6f latency=%.0fms",
        provider.name, result.model,
        result.input_tokens, result.output_tokens,
        total_cost, result.latency_ms,
    )

    return ChatResponse(
        content=result.content,
        provider=provider.name,
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=total_cost,
        latency_ms=result.latency_ms,
        cached=False,
        strategy_used=strategy_label,
    )
