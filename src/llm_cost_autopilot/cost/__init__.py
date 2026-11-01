"""Cost tracking and savings math.

Public surface:
    Savings           frozen result of baseline-vs-actual comparison.
    compute_savings   pure function: baseline cost minus actual cost.
    CostTracker       builds a RoutingLogEntry and persists it via the ledger.
"""

from llm_cost_autopilot.cost.calculator import Savings, compute_savings
from llm_cost_autopilot.cost.tracker import CostTracker

__all__ = ["CostTracker", "Savings", "compute_savings"]
