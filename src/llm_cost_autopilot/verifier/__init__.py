"""Async quality verifier and feedback loop.

After every routed request, the verifier replays the prompt against the
standard model and scores the two responses for semantic similarity. If they
diverge beyond ``verifier_divergence_threshold``, the routing decision is
recorded as a failure in the correction dataset. ``correction_feedback``
adds each failure to the classifier's correction memory so similar prompts
are escalated one tier.

The verifier is fire-and-forget: failures must never break the user response.

Public surface:
    SimilarityScorer    Protocol for cosine-similarity backends.
    CorrectionStore     Protocol for persistence of routing failures.
    RoutingFailure      frozen record of one failed routing decision.
    VerificationOutcome enum of every verification outcome.
    Verifier            orchestrator with the public ``verify_in_background``.
"""

from llm_cost_autopilot.verifier.async_verifier import (
    CorrectionStore,
    SimilarityScorer,
    Verifier,
)
from llm_cost_autopilot.verifier.feedback import RoutingFailure, VerificationOutcome

__all__ = [
    "CorrectionStore",
    "RoutingFailure",
    "SimilarityScorer",
    "VerificationOutcome",
    "Verifier",
]
