"""Pure savings math — no I/O, no dependencies on the ledger."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Savings(BaseModel):
    """Result of comparing baseline-tier cost against actual routed cost."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    actual_cost_usd: float = Field(ge=0.0)
    baseline_cost_usd: float = Field(ge=0.0)
    saved_usd: float
    saved_pct: float = Field(description="0..100 — negative if autopilot was more expensive.")


def compute_savings(*, actual_cost_usd: float, baseline_cost_usd: float) -> Savings:
    """Return per-request savings as a ``Savings`` record.

    ``baseline_cost_usd`` is what the request *would have cost* if it had
    been routed to the standard model. ``actual_cost_usd`` is what the
    autopilot's choice cost. When the baseline is zero (e.g. a degenerate
    empty request), ``saved_pct`` is reported as 0 rather than raising.
    """
    if actual_cost_usd < 0:
        raise ValueError("actual_cost_usd must be >= 0")
    if baseline_cost_usd < 0:
        raise ValueError("baseline_cost_usd must be >= 0")

    saved_usd = baseline_cost_usd - actual_cost_usd
    saved_pct = (saved_usd / baseline_cost_usd * 100.0) if baseline_cost_usd > 0 else 0.0
    return Savings(
        actual_cost_usd=actual_cost_usd,
        baseline_cost_usd=baseline_cost_usd,
        saved_usd=saved_usd,
        saved_pct=saved_pct,
    )
