"""
app/cost_tracker.py — Phase 5: persist per-request spend data to Redis.

CONCEPT: two complementary data structures

  1. Hash — spend:totals
     Stores running totals using Redis HINCRBYFLOAT (atomic float increment).
     Fields:
       total              → all-time spend across all providers
       provider:openai    → spend attributed to OpenAI
       provider:anthropic → spend attributed to Anthropic
       provider:groq      → spend attributed to Groq
       model:gpt-4o-mini  → spend attributed to a specific model
       (etc.)

     Why a hash?
       One round-trip to get all totals: HGETALL spend:totals
       Atomic increments: two requests updating simultaneously won't corrupt data.

  2. Sorted set — spend:requests
     Each entry: value=JSON blob, score=Unix timestamp (float).
     This lets us query "all requests in the last 24 hours" with:
       ZRANGEBYSCORE spend:requests <24h_ago> +inf

     Why a sorted set?
       Redis sorted sets are ordered by score. Time-range queries are O(log N + M)
       where M is the number of results — very fast even with millions of entries.

CONCEPT: INCRBYFLOAT
  Redis natively supports atomic float increments on hash fields.
  This is better than read-modify-write (which has a race condition between
  two concurrent requests reading the same value before either has written).

CONCEPT: spend analytics endpoint
  GET /v1/spend returns:
    - all-time totals (total, per-provider, per-model)
    - request_count
    - optional: last N requests for inspection
"""

import json
import logging
import time

from redis.asyncio import Redis

logger = logging.getLogger(__name__)

_TOTALS_KEY = "spend:totals"
_REQUESTS_KEY = "spend:requests"


class CostTracker:
    """
    Records per-request cost to Redis and exposes spend summary queries.

    All writes are fire-and-forget (errors logged, never raised) so a
    Redis hiccup never breaks the main request path.
    """

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    async def record(
        self,
        provider: str,
        model: str,
        cost_usd: float,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        cached: bool,
        strategy: str,
    ) -> None:
        """
        Persist one request's cost data.

        Two writes happen in parallel (asyncio.gather would be cleaner but
        we keep it simple with two awaits — both are fast Redis commands):
          1. HINCRBYFLOAT on spend:totals for total / provider / model buckets
          2. ZADD on spend:requests with the current Unix timestamp as score
        """
        if cached:
            # Cached responses cost $0 and don't consume provider tokens —
            # recording them would inflate request counts without spend meaning.
            logger.debug("Skipping cost record for cached response")
            return

        try:
            pipe = self._redis.pipeline()

            # Atomic increments — all fields updated in one pipeline round-trip
            pipe.hincrbyfloat(_TOTALS_KEY, "total", cost_usd)
            pipe.hincrbyfloat(_TOTALS_KEY, f"provider:{provider}", cost_usd)
            pipe.hincrbyfloat(_TOTALS_KEY, f"model:{model}", cost_usd)
            pipe.hincrby(_TOTALS_KEY, "request_count", 1)
            pipe.hincrby(_TOTALS_KEY, "tokens:input", input_tokens)
            pipe.hincrby(_TOTALS_KEY, "tokens:output", output_tokens)

            # Store request detail in sorted set (score = current Unix time)
            entry = json.dumps({
                "ts": time.time(),
                "provider": provider,
                "model": model,
                "cost_usd": cost_usd,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": latency_ms,
                "strategy": strategy,
            })
            pipe.zadd(_REQUESTS_KEY, {entry: time.time()})

            await pipe.execute()

            logger.debug(
                "Cost recorded | provider=%s model=%s cost=$%.6f",
                provider, model, cost_usd,
            )

        except Exception as exc:
            # Never let tracking failures surface to the caller
            logger.error("CostTracker.record failed: %s", exc, exc_info=True)

    async def summary(self, last_n_requests: int = 0) -> dict:
        """
        Return a spend summary dict suitable for the /v1/spend response.

        Args:
            last_n_requests: if > 0, also return the most recent N request
                             entries from the sorted set (newest first).

        Returns a dict with:
          total_usd          → all-time spend
          request_count      → total non-cached requests recorded
          total_input_tokens
          total_output_tokens
          by_provider        → {provider_name: usd_spent}
          by_model           → {model_name: usd_spent}
          recent_requests    → list of last_n_requests entries (if requested)
        """
        try:
            totals = await self._redis.hgetall(_TOTALS_KEY)
        except Exception as exc:
            logger.error("CostTracker.summary failed reading totals: %s", exc)
            return {"error": "Could not read spend data from Redis"}

        # Redis returns bytes; decode and cast
        def _float(key: bytes) -> float:
            return round(float(totals.get(key, b"0")), 8)

        def _int(key: bytes) -> int:
            return int(totals.get(key, b"0"))

        by_provider: dict[str, float] = {}
        by_model: dict[str, float] = {}

        for raw_key, raw_val in totals.items():
            key = raw_key.decode()
            if key.startswith("provider:"):
                by_provider[key[len("provider:"):]] = round(float(raw_val), 8)
            elif key.startswith("model:"):
                by_model[key[len("model:"):]] = round(float(raw_val), 8)

        result: dict = {
            "total_usd": _float(b"total"),
            "request_count": _int(b"request_count"),
            "total_input_tokens": _int(b"tokens:input"),
            "total_output_tokens": _int(b"tokens:output"),
            "by_provider": by_provider,
            "by_model": by_model,
        }

        if last_n_requests > 0:
            try:
                # ZRANGE with REV + LIMIT returns newest-first entries
                raw_entries = await self._redis.zrange(
                    _REQUESTS_KEY,
                    start=0,
                    end=last_n_requests - 1,
                    rev=True,
                )
                result["recent_requests"] = [
                    json.loads(e) for e in raw_entries
                ]
            except Exception as exc:
                logger.error("CostTracker.summary failed reading requests: %s", exc)
                result["recent_requests"] = []

        return result

    async def since(self, seconds: float) -> dict:
        """
        Return total spend for the last `seconds` seconds.

        Uses ZRANGEBYSCORE on the requests sorted set — O(log N + M).
        Useful for dashboards showing "last hour" or "last 24h" spend.
        """
        since_ts = time.time() - seconds
        try:
            raw_entries = await self._redis.zrangebyscore(
                _REQUESTS_KEY,
                min=since_ts,
                max="+inf",
            )
        except Exception as exc:
            logger.error("CostTracker.since failed: %s", exc)
            return {"error": str(exc)}

        total = 0.0
        count = 0
        by_provider: dict[str, float] = {}
        by_model: dict[str, float] = {}

        for raw in raw_entries:
            entry = json.loads(raw)
            cost = entry["cost_usd"]
            provider = entry["provider"]
            model = entry["model"]
            total += cost
            count += 1
            by_provider[provider] = round(by_provider.get(provider, 0.0) + cost, 8)
            by_model[model] = round(by_model.get(model, 0.0) + cost, 8)

        return {
            "window_seconds": seconds,
            "total_usd": round(total, 8),
            "request_count": count,
            "by_provider": by_provider,
            "by_model": by_model,
        }
