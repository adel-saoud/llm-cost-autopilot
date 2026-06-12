"""Runtime settings — environment-driven, validated by pydantic-settings.

All configuration is exposed via `AUTOPILOT_*` environment variables. Tests can
override fields by instantiating `Settings(**overrides)` directly.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-wide configuration for the autopilot."""

    model_config = SettingsConfigDict(
        env_prefix="AUTOPILOT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # ── Classifier ──────────────────────────────────────────────────────────
    stage1_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HF model id for stage-1 embedding similarity.",
    )
    stage2_model: str = Field(
        default="MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33",
        description="HF model id for stage-2 zero-shot classification.",
    )
    stage1_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description=(
            "Cosine-similarity confidence below which stage 1 escalates to stage 2. "
            "Lowering this makes stage 2 a hot path; raising it drops accuracy."
        ),
    )

    # ── Ollama ──────────────────────────────────────────────────────────────
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL of the local Ollama daemon.",
    )
    budget_model: str = Field(
        default="qwen2.5:0.5b",
        description="Ollama model id for the BUDGET tier.",
    )
    standard_model: str = Field(
        default="llama3.2:3b",
        description="Ollama model id for the STANDARD tier.",
    )

    # ── Verifier ────────────────────────────────────────────────────────────
    verifier_enabled: bool = Field(
        default=True,
        description="Whether the async verifier runs after each routed request.",
    )
    verifier_divergence_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description=(
            "Semantic-similarity floor between cheap and standard outputs. "
            "Below this, the routing decision is recorded as a failure."
        ),
    )

    # ── Storage ─────────────────────────────────────────────────────────────
    sqlite_path: str = Field(
        default="autopilot.db",
        description="Path to the SQLite ledger and correction dataset.",
    )

    # ── Cost simulation ─────────────────────────────────────────────────────
    budget_cost_per_1k_input: float = Field(default=0.0001, ge=0.0)
    budget_cost_per_1k_output: float = Field(default=0.0002, ge=0.0)
    standard_cost_per_1k_input: float = Field(default=0.003, ge=0.0)
    standard_cost_per_1k_output: float = Field(default=0.015, ge=0.0)


def get_settings() -> Settings:
    """Return a freshly-constructed `Settings` instance.

    Kept as a function (not a cached singleton) so tests can monkey-patch
    environment variables and observe the change without import-order tricks.
    """

    return Settings()
