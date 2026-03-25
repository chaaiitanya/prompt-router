"""
app/routers/spend.py — Phase 5: spend analytics endpoints.

Endpoints:
  GET /v1/spend                    → all-time totals + optional recent requests
  GET /v1/spend/since/{seconds}    → spend within a rolling time window

CONCEPT: query parameters
  FastAPI automatically parses query parameters from the URL:
    GET /v1/spend?last_n=10
  declares `last_n: int = Query(default=0)` in the function signature.
  No manual parsing needed — FastAPI + Pydantic handle it.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Query, HTTPException

from app.cost_tracker import CostTracker
from app.models import SpendSummaryResponse, SpendWindowResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Injected at startup by main.py (same pattern as cache in gateway.py)
cost_tracker: Optional[CostTracker] = None


def _require_tracker() -> CostTracker:
    """Raise 503 if the cost tracker isn't available yet (Redis not connected)."""
    if cost_tracker is None:
        raise HTTPException(
            status_code=503,
            detail="Spend tracking unavailable — Redis not connected.",
        )
    return cost_tracker


@router.get(
    "/v1/spend",
    response_model=SpendSummaryResponse,
    summary="All-time spend summary",
    description=(
        "Returns total spend broken down by provider and model. "
        "Pass `last_n` to also include the most recent N individual requests."
    ),
    tags=["Analytics"],
)
async def spend_summary(
    last_n: int = Query(default=0, ge=0, le=500, description="Include last N requests"),
) -> SpendSummaryResponse:
    tracker = _require_tracker()
    data = await tracker.summary(last_n_requests=last_n)
    if "error" in data:
        raise HTTPException(status_code=503, detail=data["error"])
    return SpendSummaryResponse(**data)


@router.get(
    "/v1/spend/since/{seconds}",
    response_model=SpendWindowResponse,
    summary="Spend within a rolling time window",
    description=(
        "Returns spend for the last `seconds` seconds. "
        "Useful for dashboards: pass 3600 for last hour, 86400 for last 24h."
    ),
    tags=["Analytics"],
)
async def spend_since(seconds: float) -> SpendWindowResponse:
    if seconds <= 0:
        raise HTTPException(status_code=422, detail="seconds must be positive")
    tracker = _require_tracker()
    data = await tracker.since(seconds)
    if "error" in data:
        raise HTTPException(status_code=503, detail=data["error"])
    return SpendWindowResponse(**data)
