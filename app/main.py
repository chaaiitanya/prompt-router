"""
app/main.py — FastAPI application entry point.

CONCEPT: FastAPI
  FastAPI is a modern Python web framework for building APIs.
  It's "async-first" (handles many requests concurrently), automatically
  generates interactive API docs at /docs, and validates request/response
  data using Pydantic models you define.

CONCEPT: lifespan
  The lifespan context manager runs code at startup and shutdown.
  It's the right place to:
    - Connect to databases / Redis
    - Load expensive models into memory
    - Close connections cleanly when the server shuts down

  Think of it like __enter__ / __exit__ for the whole application.

CONCEPT: logging
  Python's built-in logging module lets you emit structured log messages
  at different severity levels: DEBUG, INFO, WARNING, ERROR.
  We configure it once here and all modules use `logger = logging.getLogger(__name__)`.
"""

import logging
import logging.config
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from redis.asyncio import Redis

from app.cache import SemanticCache
from app.config import settings
from app.cost_tracker import CostTracker
from app.models import HealthResponse
from app.routers import gateway as gateway_module
from app.routers import spend as spend_module
from app.routers.gateway import router as gateway_router
from app.routers.spend import router as spend_router

# ── Logging setup ─────────────────────────────────────────────────────────────

def configure_logging() -> None:
    """Set up structured logging for the whole application."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


configure_logging()
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code before `yield` runs at startup.
    Code after `yield` runs at shutdown.
    """
    logger.info("Starting LLM Gateway (env=%s)", settings.app_env)

    # ── Startup ──
    redis = Redis.from_url(settings.redis_url, decode_responses=False)
    cache = SemanticCache(redis)
    tracker = CostTracker(redis)
    # Inject into route modules so handlers can use them
    gateway_module.cache = cache
    gateway_module.cost_tracker = tracker
    spend_module.cost_tracker = tracker
    logger.info("Redis connected | url=%s", settings.redis_url)
    logger.info("Gateway ready")

    yield  # ← the app runs while we're paused here

    # ── Shutdown ──
    logger.info("Shutting down gateway")
    await cache.close()
    await redis.aclose()


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="LLM Gateway",
    description=(
        "Self-hosted API gateway that routes prompts across OpenAI, Anthropic, "
        "and Groq based on cost, latency, and task type. Includes semantic caching "
        "and spend analytics."
    ),
    version="0.1.0",
    lifespan=lifespan,
    # /docs and /redoc are the interactive API documentation UIs
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS middleware ───────────────────────────────────────────────────────────
# CORS (Cross-Origin Resource Sharing) allows browser apps on other domains
# to call this API. Required if you want a frontend to talk to the gateway.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.app_env == "development" else [],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────

# Include the gateway router — this adds POST /v1/chat/completions
app.include_router(gateway_router, tags=["Gateway"])
app.include_router(spend_router)


@app.get("/metrics", tags=["Ops"], include_in_schema=False)
async def metrics() -> Response:
    """
    Prometheus scrape endpoint.

    CONCEPT: include_in_schema=False
      Hides this endpoint from /docs — it's only useful for Prometheus,
      not for API consumers. Keeps the docs clean.

    Prometheus scrapes this URL every N seconds (configured in prometheus.yml).
    The response is plain text in the Prometheus exposition format:
      # HELP llm_requests_total Total number of LLM requests ...
      # TYPE llm_requests_total counter
      llm_requests_total{cached="false",model="...",provider="groq"} 42.0
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health", response_model=HealthResponse, tags=["Ops"])
async def health_check() -> HealthResponse:
    """
    Simple health check endpoint.

    Load balancers and container orchestrators (Kubernetes, ECS) call this
    to check if the service is alive. Must return 200 quickly.

    In later phases we'll also check Redis connectivity here.
    """
    return HealthResponse(
        status="ok",
        environment=settings.app_env,
    )
