"""
tests/test_routing.py — Unit tests for Phase 3 routing strategies.

These tests use fake/mock providers so no real API keys are needed.

CONCEPT: what to test
  We test the decision logic (which provider gets picked), not the providers
  themselves. Each strategy function is pure logic — given a dict of providers
  and some config, it should always return the same answer.
"""

import pytest

from app.routing import LatencyTracker, pick_cheapest, pick_fastest, pick_by_task_type
from app.providers.base import LLMProvider, ProviderResponse
from app.models import Message


# ── Fake providers ─────────────────────────────────────────────────────────────

class FakeProvider(LLMProvider):
    """Minimal stand-in for a real provider — never makes network calls."""

    def __init__(self, name: str) -> None:
        self.name = name

    async def complete(
        self,
        messages: list[Message],
        max_tokens: int = 1024,
        model=None,
    ) -> ProviderResponse:
        return ProviderResponse(
            content="fake",
            model=self.name + "-model",
            input_tokens=10,
            output_tokens=20,
            latency_ms=100.0,
        )


# ── LatencyTracker ─────────────────────────────────────────────────────────────

class TestLatencyTracker:
    def test_no_data_returns_none(self):
        tracker = LatencyTracker()
        assert tracker.average_ms("groq") is None

    def test_first_observation_becomes_average(self):
        tracker = LatencyTracker()
        tracker.record("groq", 200.0)
        assert tracker.average_ms("groq") == 200.0

    def test_rolling_average_converges(self):
        tracker = LatencyTracker()
        tracker.record("groq", 100.0)
        tracker.record("groq", 300.0)
        # (100 + 300) / 2 = 200
        assert tracker.average_ms("groq") == pytest.approx(200.0)

    def test_multiple_providers_tracked_independently(self):
        tracker = LatencyTracker()
        tracker.record("groq", 100.0)
        tracker.record("openai", 500.0)
        assert tracker.average_ms("groq") == pytest.approx(100.0)
        assert tracker.average_ms("openai") == pytest.approx(500.0)

    def test_snapshot_returns_all_averages(self):
        tracker = LatencyTracker()
        tracker.record("groq", 100.0)
        tracker.record("openai", 400.0)
        snap = tracker.snapshot()
        assert "groq" in snap and "openai" in snap


# ── pick_cheapest ──────────────────────────────────────────────────────────────

class TestPickCheapest:
    def setup_method(self):
        self.providers = {
            "openai":    FakeProvider("openai"),
            "anthropic": FakeProvider("anthropic"),
            "groq":      FakeProvider("groq"),
        }

    def test_returns_groq_as_cheapest(self):
        # Groq (llama-3.1-8b-instant) has the lowest input cost in config
        provider, label = pick_cheapest(self.providers)
        assert provider.name == "groq"

    def test_label_contains_cheapest_and_provider(self):
        _, label = pick_cheapest(self.providers)
        assert label.startswith("cheapest:")
        assert "groq" in label

    def test_single_provider_returns_it(self):
        provider, _ = pick_cheapest({"groq": FakeProvider("groq")})
        assert provider.name == "groq"


# ── pick_fastest ───────────────────────────────────────────────────────────────

class TestPickFastest:
    def setup_method(self):
        self.providers = {
            "openai":    FakeProvider("openai"),
            "anthropic": FakeProvider("anthropic"),
            "groq":      FakeProvider("groq"),
        }

    def test_cold_start_falls_back_to_cheapest(self):
        # No latency data → should behave like pick_cheapest
        tracker = LatencyTracker()
        # Temporarily patch the module-level tracker
        import app.routing as routing_module
        original = routing_module.latency_tracker
        routing_module.latency_tracker = tracker
        try:
            provider, label = pick_fastest(self.providers)
            assert provider.name == "groq"  # cheapest fallback
        finally:
            routing_module.latency_tracker = original

    def test_picks_fastest_provider(self):
        import app.routing as routing_module
        tracker = LatencyTracker()
        tracker.record("openai", 500.0)
        tracker.record("anthropic", 800.0)
        tracker.record("groq", 150.0)

        original = routing_module.latency_tracker
        routing_module.latency_tracker = tracker
        try:
            provider, label = pick_fastest(self.providers)
            assert provider.name == "groq"
            assert "fastest" in label
        finally:
            routing_module.latency_tracker = original

    def test_prefers_measured_over_unmeasured(self):
        """A provider with one observation should beat providers with none."""
        import app.routing as routing_module
        tracker = LatencyTracker()
        tracker.record("openai", 9999.0)  # very slow but measured

        original = routing_module.latency_tracker
        routing_module.latency_tracker = tracker
        try:
            # Only openai has data — should be picked over unmeasured providers
            provider, _ = pick_fastest({"openai": FakeProvider("openai"),
                                        "groq": FakeProvider("groq")})
            assert provider.name == "openai"
        finally:
            routing_module.latency_tracker = original


# ── pick_by_task_type ──────────────────────────────────────────────────────────

class TestPickByTaskType:
    def setup_method(self):
        self.providers = {
            "openai":    FakeProvider("openai"),
            "anthropic": FakeProvider("anthropic"),
            "groq":      FakeProvider("groq"),
        }

    def test_code_routes_to_openai(self):
        provider, label = pick_by_task_type(self.providers, "code")
        assert provider.name == "openai"
        assert "code" in label

    def test_analysis_routes_to_anthropic(self):
        provider, label = pick_by_task_type(self.providers, "analysis")
        assert provider.name == "anthropic"

    def test_summarize_routes_to_groq(self):
        provider, label = pick_by_task_type(self.providers, "summarize")
        assert provider.name == "groq"

    def test_unknown_task_type_uses_default(self):
        provider, _ = pick_by_task_type(self.providers, "unknown_task")
        # default → groq per config
        assert provider.name == "groq"

    def test_label_includes_task_and_provider(self):
        _, label = pick_by_task_type(self.providers, "code")
        assert "task_type" in label
        assert "code" in label
        assert "openai" in label
