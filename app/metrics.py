"""
app/metrics.py — Phase 6: Prometheus metrics definitions.

CONCEPT: Prometheus
  Prometheus is a time-series metrics database. It works by "scraping" —
  periodically sending an HTTP GET to your app's /metrics endpoint and
  storing the returned data. Grafana then queries Prometheus to draw graphs.

  Unlike logging (which records individual events), metrics aggregate data
  into counters and distributions that are cheap to store and fast to query.

CONCEPT: metric types used here

  Counter  — only goes up, resets on restart.
    Use for: total requests, total errors, total tokens consumed.
    Example: llm_requests_total{provider="groq"} 42

  Histogram — records observations in configurable buckets.
    Use for: latency, request size — anything where you want percentiles.
    Prometheus automatically computes p50/p95/p99 from histogram data.
    Example: llm_request_duration_seconds_bucket{le="0.5"} 38

CONCEPT: labels
  Labels are key-value pairs attached to a metric that allow slicing:
    llm_cost_usd_total{provider="openai", model="gpt-4o-mini"}
    llm_cost_usd_total{provider="groq",   model="llama-3.1-8b-instant"}
  Grafana can sum, filter, or compare these in dashboards.

  Rule of thumb: labels should have low cardinality (few distinct values).
  Never use user IDs or free-text as labels — that creates millions of
  time series and kills Prometheus performance.

CONCEPT: module-level singletons
  Prometheus metrics must be registered exactly once. We define them here
  at module import time. Any module that needs to record a metric imports
  from here — no re-registration, no duplication.
"""

from prometheus_client import Counter, Histogram, REGISTRY

# ── Request counters ──────────────────────────────────────────────────────────

REQUEST_TOTAL = Counter(
    "llm_requests_total",
    "Total number of LLM requests handled by the gateway.",
    labelnames=["provider", "model", "cached"],
)
"""
Increment on every completed request (hit or miss).

Labels:
  provider — "openai" / "anthropic" / "groq" (or "cache" for cache hits)
  model    — exact model string e.g. "gpt-4o-mini"
  cached   — "true" / "false"

Usage in Grafana:
  rate(llm_requests_total[5m])          → requests/sec
  sum by (provider)(llm_requests_total) → total per provider
"""

PROVIDER_ERRORS_TOTAL = Counter(
    "llm_provider_errors_total",
    "Total number of errors returned by LLM providers.",
    labelnames=["provider"],
)
"""
Increment when provider.complete() raises an exception.

Grafana alert example:
  rate(llm_provider_errors_total[5m]) > 0.1  → page on-call
"""

CACHE_HITS_TOTAL = Counter(
    "llm_cache_hits_total",
    "Total number of requests served from the semantic cache.",
)

CACHE_MISSES_TOTAL = Counter(
    "llm_cache_misses_total",
    "Total number of requests that missed the semantic cache.",
)

# ── Latency histogram ─────────────────────────────────────────────────────────

REQUEST_DURATION = Histogram(
    "llm_request_duration_seconds",
    "End-to-end request latency from gateway receipt to response sent.",
    labelnames=["provider", "model"],
    # Buckets chosen to capture LLM latency range (50ms – 30s):
    # sub-second fast models, multi-second complex models
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)
"""
Observe latency_ms / 1000 on every non-cached request.

Grafana queries:
  histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m]))
    → p95 latency over the last 5 minutes
  histogram_quantile(0.50, ...) → median latency
"""

# ── Token counters ────────────────────────────────────────────────────────────

TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Total tokens consumed, split by direction.",
    labelnames=["provider", "model", "type"],  # type = "input" | "output"
)
"""
Increment input_tokens with type="input", output_tokens with type="output".

Grafana: rate(llm_tokens_total{type="output"}[1h]) → output tokens/hour
"""

# ── Cost counter ──────────────────────────────────────────────────────────────

COST_USD_TOTAL = Counter(
    "llm_cost_usd_total",
    "Cumulative LLM spend in USD.",
    labelnames=["provider", "model"],
)
"""
Increment by cost_usd on every non-cached request.

Grafana: increase(llm_cost_usd_total[24h]) → spend in the last 24 hours
"""


# ── Instrumentation helpers ───────────────────────────────────────────────────

def record_cache_hit() -> None:
    """Call when a request is served from the semantic cache."""
    CACHE_HITS_TOTAL.inc()
    REQUEST_TOTAL.labels(provider="cache", model="cache", cached="true").inc()


def record_cache_miss() -> None:
    """Call when a cache lookup returns nothing."""
    CACHE_MISSES_TOTAL.inc()


def record_provider_error(provider: str) -> None:
    """Call when a provider raises an exception."""
    PROVIDER_ERRORS_TOTAL.labels(provider=provider).inc()


def record_request(
    provider: str,
    model: str,
    latency_ms: float,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
) -> None:
    """
    Record all metrics for a successful (non-cached) provider response.

    Bundles the four metric updates into one call so the gateway handler
    stays readable.
    """
    REQUEST_TOTAL.labels(provider=provider, model=model, cached="false").inc()
    REQUEST_DURATION.labels(provider=provider, model=model).observe(latency_ms / 1000)
    TOKENS_TOTAL.labels(provider=provider, model=model, type="input").inc(input_tokens)
    TOKENS_TOTAL.labels(provider=provider, model=model, type="output").inc(output_tokens)
    COST_USD_TOTAL.labels(provider=provider, model=model).inc(cost_usd)
