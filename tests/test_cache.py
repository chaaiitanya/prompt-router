"""
tests/test_cache.py — Unit tests for Phase 4 semantic cache.

We mock Redis and the OpenAI embedding client so no real services are needed.

CONCEPT: mocking external dependencies
  The cache has two external dependencies: Redis and OpenAI embeddings.
  We replace both with fakes so tests:
    - Run offline (no API keys, no network)
    - Run fast (no real HTTP calls)
    - Are deterministic (we control the vectors)
"""

import json
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from app.cache import SemanticCache, _cosine_similarity, _messages_to_text
from app.models import ChatResponse, Message


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_messages(*contents: str) -> list[Message]:
    return [Message(role="user", content=c) for c in contents]


def make_response(**kwargs) -> ChatResponse:
    defaults = dict(
        content="Hello!",
        provider="groq",
        model="llama-3.1-8b-instant",
        input_tokens=10,
        output_tokens=20,
        cost_usd=0.000001,
        latency_ms=150.0,
        cached=False,
        strategy_used="cheapest:groq",
    )
    defaults.update(kwargs)
    return ChatResponse(**defaults)


def unit_vector(size: int = 8, seed: int = 0) -> np.ndarray:
    """Return a reproducible unit vector of the given size."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(size).astype(np.float32)
    return v / np.linalg.norm(v)


# ── _cosine_similarity ─────────────────────────────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors_return_one(self):
        v = unit_vector()
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_zero_vector_returns_zero(self):
        a = np.zeros(4, dtype=np.float32)
        b = unit_vector(4)
        assert _cosine_similarity(a, b) == 0.0

    def test_result_clipped_to_valid_range(self):
        # Floating-point arithmetic can produce values slightly outside [-1, 1]
        v = unit_vector()
        sim = _cosine_similarity(v, v * (1 + 1e-7))
        assert -1.0 <= sim <= 1.0


# ── _messages_to_text ──────────────────────────────────────────────────────────

class TestMessagesToText:
    def test_single_message(self):
        msgs = [Message(role="user", content="Hi")]
        assert _messages_to_text(msgs) == "user: Hi"

    def test_multi_turn(self):
        msgs = [
            Message(role="system", content="Be concise"),
            Message(role="user", content="What is Python?"),
        ]
        result = _messages_to_text(msgs)
        assert result == "system: Be concise\nuser: What is Python?"


# ── SemanticCache ──────────────────────────────────────────────────────────────

@pytest.fixture
def fake_redis():
    """In-memory fake for redis.asyncio.Redis using a dict."""
    store: dict = {}

    redis = AsyncMock()

    async def hset(key, mapping):
        store[key] = {k.encode() if isinstance(k, str) else k: v for k, v in mapping.items()}

    async def hgetall(key):
        return store.get(key, {})

    async def keys(pattern):
        prefix = pattern.rstrip("*")
        return [k for k in store if k.startswith(prefix)]

    async def expire(key, ttl):
        pass  # TTL not enforced in fake

    redis.hset = hset
    redis.hgetall = hgetall
    redis.keys = keys
    redis.expire = expire
    return redis, store


def make_cache(fake_redis_fixture, embed_fn):
    """Build a SemanticCache with a patched embed method."""
    redis, store = fake_redis_fixture
    sc = SemanticCache(redis)
    sc._embed = embed_fn
    return sc, store


class TestSemanticCacheGet:
    @pytest.mark.asyncio
    async def test_miss_on_empty_cache(self, fake_redis):
        async def embed(text):
            return unit_vector(8, seed=1)

        cache, _ = make_cache(fake_redis, embed)
        result = await cache.get(make_messages("Hello"))
        assert result is None

    @pytest.mark.asyncio
    async def test_hit_on_identical_prompt(self, fake_redis):
        vec = unit_vector(8, seed=42)

        async def embed(text):
            return vec

        cache, _ = make_cache(fake_redis, embed)
        messages = make_messages("What is Python?")
        response = make_response(content="Python is a language.")

        await cache.set(messages, response)
        hit = await cache.get(messages)

        assert hit is not None
        assert hit.content == "Python is a language."
        assert hit.cached is True
        assert hit.cost_usd == 0.0

    @pytest.mark.asyncio
    async def test_miss_below_threshold(self, fake_redis):
        # Two orthogonal vectors → similarity = 0, well below threshold
        vec_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        vec_b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        call_count = 0

        async def embed(text):
            nonlocal call_count
            v = vec_a if call_count == 0 else vec_b
            call_count += 1
            return v

        cache, _ = make_cache(fake_redis, embed)
        cache._threshold = 0.95

        await cache.set(make_messages("Original question"), make_response())
        hit = await cache.get(make_messages("Completely different question"))
        assert hit is None

    @pytest.mark.asyncio
    async def test_hit_marks_cost_as_zero(self, fake_redis):
        vec = unit_vector(8, seed=7)

        async def embed(text):
            return vec

        cache, _ = make_cache(fake_redis, embed)
        response = make_response(cost_usd=0.005)

        await cache.set(make_messages("Q"), response)
        hit = await cache.get(make_messages("Q"))

        assert hit.cost_usd == 0.0
        assert hit.cached is True

    @pytest.mark.asyncio
    async def test_malformed_entry_is_skipped(self, fake_redis):
        redis, store = fake_redis
        # Manually insert a bad entry (missing 'response' field)
        store["llmcache:bad"] = {b"embedding": b"not_valid_bytes_xxx"}

        async def embed(text):
            return unit_vector(8)

        cache = SemanticCache(redis)
        cache._embed = embed
        # Should not raise — bad entries are logged and skipped
        result = await cache.get(make_messages("Anything"))
        assert result is None


class TestSemanticCacheSet:
    @pytest.mark.asyncio
    async def test_set_stores_embedding_and_response(self, fake_redis):
        vec = unit_vector(8, seed=3)

        async def embed(text):
            return vec

        cache, store = make_cache(fake_redis, embed)
        response = make_response(content="Stored!")

        await cache.set(make_messages("Store me"), response)

        assert len(store) == 1
        entry = next(iter(store.values()))
        assert b"embedding" in entry
        assert b"response" in entry

        # Verify response JSON round-trips correctly
        data = json.loads(entry[b"response"])
        assert data["content"] == "Stored!"
