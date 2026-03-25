"""
app/cache.py — Phase 4: Semantic cache backed by Redis.

CONCEPT: semantic cache vs exact cache
  A normal (exact) cache stores responses keyed by the exact request string.
  "What is Python?" and "What is Python programming language?" would be two
  different cache keys — even though the answer is almost identical.

  A semantic cache stores an embedding vector alongside each response.
  On a new request we embed the prompt, then find the closest stored vector
  using cosine similarity. If similarity ≥ threshold, we return the cached
  response even though the phrasing is different.

  This dramatically improves cache hit rates for real-world usage.

CONCEPT: embeddings
  An embedding is a list of floating-point numbers (a vector) that represents
  the "meaning" of a piece of text. Texts with similar meaning end up close
  together in vector space, even if the words differ.
  We use OpenAI's text-embedding-3-small model (1536 dimensions).

CONCEPT: cosine similarity
  Measures the angle between two vectors. Result is in [-1, 1]:
    1.0 → identical direction (same meaning)
    0.0 → orthogonal (unrelated)
   -1.0 → opposite (rare for text)
  We use 0.95 as the hit threshold — strict enough to avoid wrong answers.

CONCEPT: Redis as a vector store (simple version)
  We're not using a dedicated vector DB (like Pinecone or pgvector).
  Instead we store each entry as a Redis hash:
    - "embedding" field: numpy array serialised to bytes
    - "response"  field: JSON-serialised ChatResponse

  On a lookup we load all entries and do cosine similarity in Python/numpy.
  This is fine for thousands of entries. Phase 5+ could add Redis Vector
  Search (RediSearch) if the cache grows very large.

CONCEPT: cache key prefix
  We prefix every Redis key with "llmcache:" so our keys don't clash with
  any other data stored in the same Redis instance.
"""

import json
import logging
import time
import uuid
from typing import Optional

import numpy as np
from openai import AsyncOpenAI
from redis.asyncio import Redis

from app.config import settings
from app.models import ChatResponse, Message

logger = logging.getLogger(__name__)

_EMBED_MODEL = "text-embedding-3-small"
_KEY_PREFIX = "llmcache:"


def _messages_to_text(messages: list[Message]) -> str:
    """
    Flatten a conversation into a single string for embedding.

    We embed the full conversation so that multi-turn context is captured.
    System prompts are included because they change the intended output.

    Example:
      [system: "Be concise", user: "What is Python?"]
      → "system: Be concise\nuser: What is Python?"
    """
    return "\n".join(f"{m.role}: {m.content}" for m in messages)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two 1-D vectors.

    cos(θ) = (a · b) / (‖a‖ × ‖b‖)

    We clip to [-1, 1] to guard against tiny floating-point errors that
    could push the value just outside the valid range.
    """
    dot = float(np.dot(a, b))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm == 0:
        return 0.0
    return float(np.clip(dot / norm, -1.0, 1.0))


class SemanticCache:
    """
    Semantic similarity cache using Redis for storage and OpenAI for embeddings.

    Usage:
        cache = SemanticCache(redis_client)
        await cache.connect()  # call once at startup (warms up embed client)

        hit = await cache.get(messages)
        if hit:
            return hit  # cached ChatResponse

        result = await provider.complete(...)
        response = ChatResponse(...)
        await cache.set(messages, response)
    """

    def __init__(self, redis: Redis) -> None:
        self._redis = redis
        # Reuse the same AsyncOpenAI client as the gateway to share connection pool
        self._embed_client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._threshold = settings.cache_similarity_threshold
        self._ttl = settings.cache_ttl_seconds

    async def _embed(self, text: str) -> np.ndarray:
        """
        Get an embedding vector for `text` from OpenAI.

        Returns a normalised numpy array of shape (1536,).
        We normalise so that dot-product == cosine similarity (faster scan).
        """
        start = time.monotonic()
        response = await self._embed_client.embeddings.create(
            model=_EMBED_MODEL,
            input=text,
        )
        latency = round((time.monotonic() - start) * 1000, 1)
        vector = np.array(response.data[0].embedding, dtype=np.float32)
        # Normalise to unit length so dot product == cosine similarity
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        logger.debug("Embedding generated | len=%d latency=%.0fms", len(vector), latency)
        return vector

    async def get(self, messages: list[Message]) -> Optional[ChatResponse]:
        """
        Look up a semantically similar cached response.

        Returns a ChatResponse with cached=True if a hit is found,
        or None on a miss.

        Steps:
          1. Embed the incoming prompt
          2. Scan all cache keys in Redis
          3. For each entry, load its stored embedding and compute similarity
          4. If best similarity ≥ threshold, return the stored response
        """
        text = _messages_to_text(messages)
        query_vec = await self._embed(text)

        keys = await self._redis.keys(f"{_KEY_PREFIX}*")
        if not keys:
            logger.debug("Cache miss (empty cache)")
            return None

        best_sim = 0.0
        best_response: Optional[ChatResponse] = None

        for key in keys:
            entry = await self._redis.hgetall(key)
            if not entry:
                continue

            try:
                stored_vec = np.frombuffer(entry[b"embedding"], dtype=np.float32)
                sim = _cosine_similarity(query_vec, stored_vec)
            except Exception as exc:
                logger.warning("Skipping malformed cache entry %s: %s", key, exc)
                continue

            if sim > best_sim:
                best_sim = sim
                if sim >= self._threshold:
                    try:
                        response_data = json.loads(entry[b"response"])
                        best_response = ChatResponse(**response_data)
                    except Exception as exc:
                        logger.warning("Could not deserialise cached response: %s", exc)
                        best_response = None

        if best_response is not None:
            logger.info(
                "Cache HIT | similarity=%.4f threshold=%.4f",
                best_sim, self._threshold,
            )
            # Return a copy marked as cached (cost is effectively $0)
            return best_response.model_copy(update={"cached": True, "cost_usd": 0.0})

        logger.debug("Cache miss | best_similarity=%.4f threshold=%.4f", best_sim, self._threshold)
        return None

    async def set(self, messages: list[Message], response: ChatResponse) -> None:
        """
        Store a prompt+response pair in the cache.

        We store:
          - embedding: raw bytes of the float32 numpy vector
          - response:  JSON of the ChatResponse

        The entry expires after cache_ttl_seconds (default 1 hour).
        """
        text = _messages_to_text(messages)
        vector = await self._embed(text)

        key = f"{_KEY_PREFIX}{uuid.uuid4().hex}"
        await self._redis.hset(key, mapping={
            "embedding": vector.tobytes(),
            "response": response.model_dump_json(),
        })
        await self._redis.expire(key, self._ttl)

        logger.info(
            "Cache SET | key=%s ttl=%ds",
            key, self._ttl,
        )

    async def close(self) -> None:
        """Close the embedding client connection pool."""
        await self._embed_client.close()
