"""
app/routing.py — Phase 3: real provider-selection strategies.

Three strategies:

  cheapest   → pick the provider whose model has the lowest blended cost
               for the expected request size. We estimate input cost only
               (output token count is unknown before the call), which is the
               right signal for routing — the actual output cost is tracked
               after the fact.

  fastest    → pick the provider with the best rolling-average latency.
               Latency is updated after every real request via
               LatencyTracker.record(). Falls back to cheapest if we have
               no observations yet.

  task_type  → delegates to the static config map in settings
               (summarize→groq, code→openai, analysis→anthropic, …).

CONCEPT: rolling average
  Instead of storing every latency observation (unbounded memory), we keep
  a running mean using the online update formula:
      new_avg = old_avg + (new_value - old_avg) / count
  This is O(1) space and O(1) update, and converges quickly.
"""

import logging
import threading
from typing import TYPE_CHECKING, Optional

from app.config import settings

if TYPE_CHECKING:
    from app.providers.base import LLMProvider

logger = logging.getLogger(__name__)


# ── Latency tracker ───────────────────────────────────────────────────────────

class LatencyTracker:
    """
    Thread-safe rolling-average latency per provider.

    CONCEPT: why thread-safe?
      FastAPI runs async handlers, but multiple requests can be in-flight
      at the same time. A threading.Lock ensures two requests don't
      update the same counter simultaneously and corrupt the average.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # provider_name → (rolling_avg_ms, observation_count)
        self._data: dict[str, tuple[float, int]] = {}

    def record(self, provider_name: str, latency_ms: float) -> None:
        """Update the rolling average for a provider after a completed request."""
        with self._lock:
            if provider_name not in self._data:
                self._data[provider_name] = (latency_ms, 1)
            else:
                avg, count = self._data[provider_name]
                count += 1
                # Online mean update — no need to store all past values
                new_avg = avg + (latency_ms - avg) / count
                self._data[provider_name] = (new_avg, count)

        logger.debug(
            "Latency recorded | provider=%s latency=%.0fms new_avg=%.0fms",
            provider_name, latency_ms, self._data[provider_name][0],
        )

    def average_ms(self, provider_name: str) -> Optional[float]:
        """Return the rolling average latency, or None if no data yet."""
        with self._lock:
            entry = self._data.get(provider_name)
            return entry[0] if entry else None

    def snapshot(self) -> dict[str, float]:
        """Return a copy of current averages for all providers (used in /health)."""
        with self._lock:
            return {name: avg for name, (avg, _) in self._data.items()}


# Module-level singleton shared across all requests
latency_tracker = LatencyTracker()


# ── Strategy functions ────────────────────────────────────────────────────────

def pick_cheapest(
    providers: dict[str, "LLMProvider"],
) -> tuple["LLMProvider", str]:
    """
    Pick the provider with the lowest input cost per token.

    Why input-only?
      We don't know the output token count before the call. Input cost is a
      reliable proxy — providers cheap on input are almost always cheap overall.

    Tie-breaking: alphabetical by provider name for determinism.
    """
    best_name: Optional[str] = None
    best_cost: float = float("inf")

    for name in sorted(providers):  # sorted = deterministic tie-breaking
        model = settings.provider_default_models.get(name)
        if not model:
            continue
        cost = settings.cost_per_token(model, "input")
        logger.debug("cheapest | provider=%s model=%s input_cost=%.9f", name, model, cost)
        if cost < best_cost:
            best_cost = cost
            best_name = name

    if best_name is None:
        # Fallback: should never happen unless registry is empty
        best_name = next(iter(providers))

    logger.info("cheapest strategy → %s (input cost/token=%.9f)", best_name, best_cost)
    return providers[best_name], f"cheapest:{best_name}"


def pick_fastest(
    providers: dict[str, "LLMProvider"],
) -> tuple["LLMProvider", str]:
    """
    Pick the provider with the lowest rolling-average latency.

    Falls back to cheapest if no latency data exists yet (cold start).
    Providers with no observations are treated as slower than any observed
    provider — this avoids routing all cold-start requests to an unknown.
    """
    best_name: Optional[str] = None
    best_latency: float = float("inf")

    for name in sorted(providers):
        avg = latency_tracker.average_ms(name)
        if avg is None:
            # No data — treat as very slow so we prefer providers we've measured
            logger.debug("fastest | provider=%s no data yet", name)
            continue
        logger.debug("fastest | provider=%s avg_latency=%.0fms", name, avg)
        if avg < best_latency:
            best_latency = avg
            best_name = name

    if best_name is None:
        # Cold start: no observations for any provider → fall back to cheapest
        logger.info("fastest strategy → no latency data, falling back to cheapest")
        return pick_cheapest(providers)

    logger.info("fastest strategy → %s (avg latency=%.0fms)", best_name, best_latency)
    return providers[best_name], f"fastest:{best_name}"


def pick_by_task_type(
    providers: dict[str, "LLMProvider"],
    task_type: str,
) -> tuple["LLMProvider", str]:
    """
    Pick a provider based on the task_type hint using the config routing map.

    The map lives in settings.task_type_routing so it can be changed via
    environment variables without touching code.
    """
    provider_name = settings.task_type_routing.get(
        task_type,
        settings.task_type_routing["default"],
    )
    logger.info("task_type strategy | task=%s → %s", task_type, provider_name)
    return providers[provider_name], f"task_type:{task_type}:{provider_name}"
