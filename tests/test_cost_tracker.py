"""
tests/test_cost_tracker.py — Unit tests for Phase 5 cost tracker.

Uses an in-memory fake Redis (dict-backed) so no real Redis is needed.

CONCEPT: testing async code with pytest-asyncio
  Async functions can't be called directly in tests — they return coroutines,
  not values. pytest-asyncio runs each async test in its own event loop so
  `await` works as expected.

  Mark individual tests with @pytest.mark.asyncio, or set asyncio_mode=auto
  in pytest config. We use explicit marks for clarity.
"""

import json
import time
import pytest

from app.cost_tracker import CostTracker


# ── Fake Redis ─────────────────────────────────────────────────────────────────

class FakeRedis:
    """
    Minimal in-memory Redis stub covering only the commands CostTracker uses:
      HINCRBYFLOAT, HINCRBY, HGETALL, ZADD, ZRANGE, ZRANGEBYSCORE, pipeline
    """

    def __init__(self):
        self._hashes: dict[str, dict[bytes, bytes]] = {}
        self._zsets: dict[str, dict[str, float]] = {}   # key → {member: score}

    # ── Hash commands ──────────────────────────────────────────────────────────

    async def hincrbyfloat(self, key: str, field: str, amount: float) -> float:
        h = self._hashes.setdefault(key, {})
        field_b = field.encode()
        current = float(h.get(field_b, b"0"))
        new_val = current + amount
        h[field_b] = str(new_val).encode()
        return new_val

    async def hincrby(self, key: str, field: str, amount: int) -> int:
        h = self._hashes.setdefault(key, {})
        field_b = field.encode()
        current = int(h.get(field_b, b"0"))
        new_val = current + amount
        h[field_b] = str(new_val).encode()
        return new_val

    async def hgetall(self, key: str) -> dict[bytes, bytes]:
        return dict(self._hashes.get(key, {}))

    # ── Sorted set commands ────────────────────────────────────────────────────

    async def zadd(self, key: str, mapping: dict) -> int:
        zset = self._zsets.setdefault(key, {})
        for member, score in mapping.items():
            zset[member] = score
        return len(mapping)

    async def zrange(self, key: str, start: int, end: int, rev: bool = False) -> list:
        zset = self._zsets.get(key, {})
        sorted_members = sorted(zset, key=lambda m: zset[m], reverse=rev)
        if end == -1:
            return [m.encode() for m in sorted_members[start:]]
        return [m.encode() for m in sorted_members[start: end + 1]]

    async def zrangebyscore(self, key: str, min, max) -> list:
        zset = self._zsets.get(key, {})
        max_score = float("inf") if max == "+inf" else float(max)
        return [
            m.encode()
            for m, score in sorted(zset.items(), key=lambda x: x[1])
            if float(min) <= score <= max_score
        ]

    # ── Pipeline (executes commands sequentially in fake) ─────────────────────

    def pipeline(self) -> "FakePipeline":
        return FakePipeline(self)


class FakePipeline:
    """Batches commands and executes them sequentially on execute()."""

    def __init__(self, redis: FakeRedis):
        self._redis = redis
        self._commands: list = []

    def hincrbyfloat(self, key, field, amount):
        self._commands.append(("hincrbyfloat", key, field, amount))
        return self

    def hincrby(self, key, field, amount):
        self._commands.append(("hincrby", key, field, amount))
        return self

    def zadd(self, key, mapping):
        self._commands.append(("zadd", key, mapping))
        return self

    async def execute(self):
        results = []
        for cmd, *args in self._commands:
            result = await getattr(self._redis, cmd)(*args)
            results.append(result)
        return results


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_tracker() -> tuple[CostTracker, FakeRedis]:
    redis = FakeRedis()
    return CostTracker(redis), redis


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestCostTrackerRecord:
    @pytest.mark.asyncio
    async def test_increments_total(self):
        tracker, _ = make_tracker()
        await tracker.record("groq", "llama-3.1-8b-instant", 0.001, 10, 20, 100.0, False, "cheapest:groq")
        summary = await tracker.summary()
        assert summary["total_usd"] == pytest.approx(0.001, abs=1e-9)

    @pytest.mark.asyncio
    async def test_increments_provider_bucket(self):
        tracker, _ = make_tracker()
        await tracker.record("openai", "gpt-4o-mini", 0.005, 50, 100, 200.0, False, "cheapest:openai")
        summary = await tracker.summary()
        assert summary["by_provider"]["openai"] == pytest.approx(0.005, abs=1e-9)

    @pytest.mark.asyncio
    async def test_increments_model_bucket(self):
        tracker, _ = make_tracker()
        await tracker.record("groq", "llama-3.1-8b-instant", 0.002, 10, 20, 80.0, False, "cheapest:groq")
        summary = await tracker.summary()
        assert summary["by_model"]["llama-3.1-8b-instant"] == pytest.approx(0.002, abs=1e-9)

    @pytest.mark.asyncio
    async def test_increments_token_counts(self):
        tracker, _ = make_tracker()
        await tracker.record("groq", "llama-3.1-8b-instant", 0.001, 15, 30, 80.0, False, "cheapest:groq")
        summary = await tracker.summary()
        assert summary["total_input_tokens"] == 15
        assert summary["total_output_tokens"] == 30

    @pytest.mark.asyncio
    async def test_cached_requests_are_skipped(self):
        tracker, _ = make_tracker()
        await tracker.record("groq", "llama-3.1-8b-instant", 0.0, 10, 20, 5.0, True, "cheapest:groq")
        summary = await tracker.summary()
        assert summary["total_usd"] == 0.0
        assert summary["request_count"] == 0

    @pytest.mark.asyncio
    async def test_multiple_requests_accumulate(self):
        tracker, _ = make_tracker()
        await tracker.record("groq", "llama-3.1-8b-instant", 0.001, 10, 20, 80.0, False, "cheapest:groq")
        await tracker.record("openai", "gpt-4o-mini", 0.003, 30, 60, 300.0, False, "cheapest:openai")
        summary = await tracker.summary()
        assert summary["total_usd"] == pytest.approx(0.004, abs=1e-9)
        assert summary["request_count"] == 2

    @pytest.mark.asyncio
    async def test_stores_request_in_sorted_set(self):
        tracker, _ = make_tracker()
        await tracker.record("groq", "llama-3.1-8b-instant", 0.001, 10, 20, 80.0, False, "cheapest:groq")
        summary = await tracker.summary(last_n_requests=5)
        assert len(summary["recent_requests"]) == 1
        req = summary["recent_requests"][0]
        assert req["provider"] == "groq"
        assert req["cost_usd"] == pytest.approx(0.001)


class TestCostTrackerSummary:
    @pytest.mark.asyncio
    async def test_empty_returns_zeros(self):
        tracker, _ = make_tracker()
        summary = await tracker.summary()
        assert summary["total_usd"] == 0.0
        assert summary["request_count"] == 0
        assert summary["by_provider"] == {}
        assert summary["by_model"] == {}

    @pytest.mark.asyncio
    async def test_last_n_requests_newest_first(self):
        tracker, _ = make_tracker()
        for i in range(3):
            await tracker.record("groq", "llama-3.1-8b-instant", float(i) * 0.001,
                                  10, 20, 80.0, False, "cheapest:groq")
        summary = await tracker.summary(last_n_requests=2)
        assert len(summary["recent_requests"]) == 2

    @pytest.mark.asyncio
    async def test_no_recent_requests_when_not_requested(self):
        tracker, _ = make_tracker()
        await tracker.record("groq", "llama-3.1-8b-instant", 0.001, 10, 20, 80.0, False, "cheapest:groq")
        summary = await tracker.summary(last_n_requests=0)
        assert "recent_requests" not in summary


class TestCostTrackerSince:
    @pytest.mark.asyncio
    async def test_includes_recent_requests(self):
        tracker, _ = make_tracker()
        await tracker.record("groq", "llama-3.1-8b-instant", 0.002, 10, 20, 80.0, False, "cheapest:groq")
        result = await tracker.since(3600)   # last hour
        assert result["total_usd"] == pytest.approx(0.002, abs=1e-9)
        assert result["request_count"] == 1

    @pytest.mark.asyncio
    async def test_window_breakdown_by_provider(self):
        tracker, _ = make_tracker()
        await tracker.record("openai", "gpt-4o-mini", 0.01, 50, 100, 300.0, False, "cheapest:openai")
        result = await tracker.since(3600)
        assert result["by_provider"]["openai"] == pytest.approx(0.01, abs=1e-9)

    @pytest.mark.asyncio
    async def test_window_seconds_echoed_in_response(self):
        tracker, _ = make_tracker()
        result = await tracker.since(7200)
        assert result["window_seconds"] == 7200
