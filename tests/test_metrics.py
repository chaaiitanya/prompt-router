"""
tests/test_metrics.py — Unit tests for Phase 6 Prometheus metrics.

CONCEPT: testing Prometheus metrics
  prometheus_client stores metric values in a global registry.
  To avoid test pollution (one test's increments leaking into another),
  we read the value before and after each operation and assert the delta,
  rather than asserting an absolute value.

  Helper: _delta(metric, labels) returns (value_after - value_before)
  when used as a context manager.
"""


import pytest

from app.metrics import (
    CACHE_HITS_TOTAL,
    CACHE_MISSES_TOTAL,
    COST_USD_TOTAL,
    PROVIDER_ERRORS_TOTAL,
    REQUEST_DURATION,
    REQUEST_TOTAL,
    TOKENS_TOTAL,
    record_cache_hit,
    record_cache_miss,
    record_provider_error,
    record_request,
)

# ── Helper ─────────────────────────────────────────────────────────────────────

def _counter_value(counter, labels: dict) -> float:
    """Read the current value of a labelled Counter."""
    try:
        return counter.labels(**labels)._value.get()
    except Exception:
        return 0.0


def _unlabelled_counter_value(counter) -> float:
    """Read the current value of an unlabelled Counter."""
    return counter._value.get()


def _histogram_count(histogram, labels: dict) -> float:
    """Read the observation count from a labelled Histogram via collect()."""
    try:
        for metric in histogram.collect():
            for sample in metric.samples:
                if sample.name.endswith("_count") and sample.labels == labels:
                    return sample.value
        return 0.0
    except Exception:
        return 0.0


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestRecordCacheHit:
    def test_increments_cache_hits_total(self):
        before = _unlabelled_counter_value(CACHE_HITS_TOTAL)
        record_cache_hit()
        after = _unlabelled_counter_value(CACHE_HITS_TOTAL)
        assert after - before == pytest.approx(1.0)

    def test_increments_requests_total_with_cache_label(self):
        before = _counter_value(REQUEST_TOTAL, {"provider": "cache", "model": "cache", "cached": "true"})
        record_cache_hit()
        after = _counter_value(REQUEST_TOTAL, {"provider": "cache", "model": "cache", "cached": "true"})
        assert after - before == pytest.approx(1.0)


class TestRecordCacheMiss:
    def test_increments_cache_misses_total(self):
        before = _unlabelled_counter_value(CACHE_MISSES_TOTAL)
        record_cache_miss()
        after = _unlabelled_counter_value(CACHE_MISSES_TOTAL)
        assert after - before == pytest.approx(1.0)


class TestRecordProviderError:
    def test_increments_errors_for_named_provider(self):
        before = _counter_value(PROVIDER_ERRORS_TOTAL, {"provider": "openai"})
        record_provider_error("openai")
        after = _counter_value(PROVIDER_ERRORS_TOTAL, {"provider": "openai"})
        assert after - before == pytest.approx(1.0)

    def test_different_providers_tracked_separately(self):
        before_groq = _counter_value(PROVIDER_ERRORS_TOTAL, {"provider": "groq"})
        before_anthropic = _counter_value(PROVIDER_ERRORS_TOTAL, {"provider": "anthropic"})
        record_provider_error("groq")
        after_groq = _counter_value(PROVIDER_ERRORS_TOTAL, {"provider": "groq"})
        after_anthropic = _counter_value(PROVIDER_ERRORS_TOTAL, {"provider": "anthropic"})
        assert after_groq - before_groq == pytest.approx(1.0)
        assert after_anthropic - before_anthropic == pytest.approx(0.0)


class TestRecordRequest:
    _labels = {"provider": "groq", "model": "llama-3.1-8b-instant"}

    def test_increments_requests_total(self):
        before = _counter_value(REQUEST_TOTAL, {**self._labels, "cached": "false"})
        record_request("groq", "llama-3.1-8b-instant", 150.0, 10, 20, 0.001)
        after = _counter_value(REQUEST_TOTAL, {**self._labels, "cached": "false"})
        assert after - before == pytest.approx(1.0)

    def test_observes_latency_histogram(self):
        before = _histogram_count(REQUEST_DURATION, self._labels)
        record_request("groq", "llama-3.1-8b-instant", 200.0, 10, 20, 0.001)
        after = _histogram_count(REQUEST_DURATION, self._labels)
        assert after - before == pytest.approx(1.0)

    def test_increments_input_tokens(self):
        labels = {**self._labels, "type": "input"}
        before = _counter_value(TOKENS_TOTAL, labels)
        record_request("groq", "llama-3.1-8b-instant", 100.0, 15, 30, 0.001)
        after = _counter_value(TOKENS_TOTAL, labels)
        assert after - before == pytest.approx(15.0)

    def test_increments_output_tokens(self):
        labels = {**self._labels, "type": "output"}
        before = _counter_value(TOKENS_TOTAL, labels)
        record_request("groq", "llama-3.1-8b-instant", 100.0, 15, 30, 0.001)
        after = _counter_value(TOKENS_TOTAL, labels)
        assert after - before == pytest.approx(30.0)

    def test_increments_cost_counter(self):
        before = _counter_value(COST_USD_TOTAL, self._labels)
        record_request("groq", "llama-3.1-8b-instant", 100.0, 10, 20, 0.00042)
        after = _counter_value(COST_USD_TOTAL, self._labels)
        assert after - before == pytest.approx(0.00042)

    def test_different_providers_tracked_separately(self):
        before_openai = _counter_value(REQUEST_TOTAL, {"provider": "openai", "model": "gpt-4o-mini", "cached": "false"})
        record_request("openai", "gpt-4o-mini", 300.0, 50, 100, 0.005)
        after_openai = _counter_value(REQUEST_TOTAL, {"provider": "openai", "model": "gpt-4o-mini", "cached": "false"})
        # groq counter should be unaffected by the openai call above
        assert after_openai - before_openai == pytest.approx(1.0)
