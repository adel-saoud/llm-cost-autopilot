"""Runtime settings: environment-driven, validated by pydantic-settings.

Every field is read from an ``AUTOPILOT_<FIELD>`` environment variable (or a
``.env`` file). Tests override fields by instantiating ``Settings(**overrides)``.
The reference table for users is docs/configuration.md.

pydantic-settings ignores names it does not know, so a misspelled or removed
``AUTOPILOT_*`` variable would silently keep its default. Loading settings
therefore raises ``UnknownSettingError`` for any such name, in the
environment or in ``.env``, with the closest valid name.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable
from typing import Final

from pydantic import Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

ENV_PREFIX: Final = "AUTOPILOT_"

_CLOSE_NAME_CUTOFF: Final = 0.6
"""difflib ratio below which a suggested name misleads more than it helps."""


class UnknownSettingError(ValueError):
    """An ``AUTOPILOT_*`` variable that matches no setting."""


def reject_unknown_env_names(names: Iterable[str], known: Iterable[str]) -> None:
    """Raise ``UnknownSettingError`` when an ``AUTOPILOT_*`` name is not a known setting.

    Names are compared upper-cased, as settings are case-insensitive. Names
    without the prefix belong to other programs and are skipped.
    """
    valid = {name.upper() for name in known}
    unknown = sorted(
        {upper for name in names if (upper := name.upper()).startswith(ENV_PREFIX)} - valid
    )
    if not unknown:
        return
    # Match on the part after the prefix: the shared prefix would make every name look close.
    suffixes = sorted(name.removeprefix(ENV_PREFIX) for name in valid)
    hints: list[str] = []
    for name in unknown:
        close = difflib.get_close_matches(
            name.removeprefix(ENV_PREFIX), suffixes, n=1, cutoff=_CLOSE_NAME_CUTOFF
        )
        hint = f"did you mean {ENV_PREFIX}{close[0]}?" if close else "no setting has a close name"
        hints.append(f"{name} ({hint})")
    raise UnknownSettingError(
        f"Unknown {ENV_PREFIX}* variable(s): {', '.join(hints)}. "
        "Valid names are listed in docs/configuration.md."
    )


class Settings(BaseSettings):
    """Process-wide configuration, frozen so a running app cannot drift from its env."""

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Keep the default sources after rejecting unknown names in the env and ``.env``."""
        seen = [
            name
            for source in (env_settings, dotenv_settings)
            if isinstance(source, EnvSettingsSource)
            for name in source.env_vars
        ]
        reject_unknown_env_names(
            seen, (f"{ENV_PREFIX}{field}" for field in settings_cls.model_fields)
        )
        return init_settings, env_settings, dotenv_settings, file_secret_settings

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

    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL of the local Ollama daemon.",
    )
    simple_model: str = Field(
        default="qwen3.5:0.8b",
        description="Ollama model id for the SIMPLE tier. Must have a price in router.pricing.",
    )
    moderate_model: str = Field(
        default="qwen3.5:4b",
        description=(
            "Ollama model id for the MODERATE tier, the cost baseline and the "
            "verifier's reference. Must have a price in router.pricing."
        ),
    )
    ollama_temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description=(
            "Sampling temperature for every Ollama call that does not set its own. "
            "None keeps Ollama's default. Evaluations set 0 for reproducible runs."
        ),
    )
    ollama_think: bool | None = Field(
        default=False,
        description=(
            "Ollama's think flag on every chat call. False turns off Qwen3.5's "
            "reasoning trace so answers, token counts and costs cover the served "
            "text only. None keeps the model's default."
        ),
    )
    ollama_seed: int | None = Field(
        default=None,
        description="Sampling seed for every Ollama call. None keeps Ollama's random seed.",
    )

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
    verifier_max_concurrency: int = Field(
        default=2,
        ge=1,
        description=(
            "Verifier calls to the standard model allowed at once, so background "
            "checks cannot starve user-facing requests on the same Ollama daemon."
        ),
    )
    verifier_max_pending: int = Field(
        default=100,
        ge=1,
        description=(
            "Verifications allowed to wait for a slot. Beyond this, new ones are "
            "skipped and counted instead of queued without bound."
        ),
    )

    correction_feedback_enabled: bool = Field(
        default=True,
        description="Whether past corrections can escalate similar prompts to a stronger tier.",
    )
    correction_match_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum cosine similarity between a prompt and a past correction for "
            "the correction to apply. It must also beat the prompt's best anchor "
            "score for the tier it was misrouted to."
        ),
    )
    correction_max_per_tenant: int = Field(
        default=500,
        ge=1,
        description="Most recent corrections kept in memory per tenant; older ones are dropped.",
    )

    retention_days: int | None = Field(
        default=30,
        ge=1,
        description=(
            "Routing-log and correction rows older than this are deleted at startup. "
            "Both tables hold raw prompts. None keeps rows forever."
        ),
    )
    api_token: SecretStr | None = Field(
        default=None,
        description=(
            "Bearer token required on every /v1 endpoint when set. "
            "None leaves the API open, which only suits a local demo."
        ),
    )

    sqlite_path: str = Field(
        default="autopilot.db",
        description="Path to the SQLite ledger and correction dataset.",
    )


def get_settings() -> Settings:
    """Return a freshly-constructed `Settings` instance.

    Kept as a function (not a cached singleton) so tests can monkey-patch
    environment variables and observe the change without import-order tricks.
    """

    return Settings()
