"""
app/config.py — Central configuration for the gateway.

CONCEPT: pydantic-settings
  Pydantic is a Python library that validates data types at runtime.
  pydantic-settings extends it to read values from environment variables
  and .env files automatically. Instead of scattered os.getenv() calls,
  you define one Settings class and every part of the app imports it.

  When you access `settings.openai_api_key`, pydantic-settings looks for
  OPENAI_API_KEY in the environment (or .env file) and returns it — already
  type-checked as a string.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """
    All configuration lives here. Add a field → it's automatically read
    from the matching env var (uppercased field name).
    """

    # ── LLM API Keys ──────────────────────────────────────────────────────
    openai_api_key: str = Field(..., description="OpenAI API key (required)")
    anthropic_api_key: str = Field(..., description="Anthropic API key (required)")
    groq_api_key: str = Field(..., description="Groq API key (required)")

    # ── Redis ──────────────────────────────────────────────────────────────
    redis_url: str = Field("redis://localhost:6379", description="Redis connection URL")

    # ── Semantic Cache ─────────────────────────────────────────────────────
    # How similar does a cached prompt need to be to count as a hit?
    # 0.95 means 95% cosine similarity — very strict, avoids wrong answers
    cache_similarity_threshold: float = Field(0.95)
    cache_ttl_seconds: int = Field(3600)  # 1 hour default TTL

    # ── App ────────────────────────────────────────────────────────────────
    app_env: str = Field("development")
    log_level: str = Field("INFO")

    # ── Model Cost Table ───────────────────────────────────────────────────
    # Cost in USD per 1,000,000 tokens (per-million pricing is standard)
    # This dict maps model name → {input cost, output cost}
    # Output tokens cost more because they require more compute
    model_costs: dict = Field(
        default={
            # OpenAI
            "gpt-4o": {"input": 5.00, "output": 15.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            # Anthropic
            "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
            "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
            # Groq (runs open-source models at very low cost)
            "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
            "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
        },
        description="Cost per million tokens by model",
    )

    # ── Provider → Default Model Mapping ──────────────────────────────────
    # When we pick a provider, which model do we use by default?
    provider_default_models: dict = Field(
        default={
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-5-haiku-20241022",
            "groq": "llama-3.1-8b-instant",
        }
    )

    # ── Task-type → Provider Mapping (used by task_type routing strategy) ──
    task_type_routing: dict = Field(
        default={
            "summarize": "groq",      # Fast + cheap for simple tasks
            "code": "openai",         # GPT-4o excels at code
            "analysis": "anthropic",  # Claude excels at long-form analysis
            "default": "groq",        # Fallback
        }
    )

    # Tell pydantic-settings to load from a .env file if present
    # env_file_encoding ensures special characters in keys are read correctly
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,  # OPENAI_API_KEY and openai_api_key both work
    )

    def cost_per_token(self, model: str, token_type: str) -> float:
        """
        Return the cost for a single token of the given type.

        Example: cost_per_token("gpt-4o", "input") → 0.000005
          (that's $5 / 1,000,000 tokens)

        Args:
            model: model name string (must exist in model_costs)
            token_type: "input" or "output"
        """
        costs = self.model_costs.get(model, {})
        per_million = costs.get(token_type, 0.0)
        return per_million / 1_000_000  # convert to per-token cost


# Module-level singleton — import this everywhere instead of creating new instances
# CONCEPT: singleton pattern — one shared instance so config is consistent app-wide
settings = Settings()
