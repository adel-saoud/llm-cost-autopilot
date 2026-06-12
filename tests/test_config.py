"""Config defaults and environment-variable overrides."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_cost_autopilot.config import Settings, get_settings


def test_defaults_match_documented_values() -> None:
    s = Settings()
    assert s.stage1_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert s.stage2_model.startswith("MoritzLaurer/deberta-v3-xsmall")
    assert s.stage1_threshold == pytest.approx(0.75)
    assert s.budget_model == "qwen2.5:0.5b"
    assert s.standard_model == "llama3.2:3b"
    assert s.verifier_enabled is True


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOPILOT_STAGE1_THRESHOLD", "0.6")
    monkeypatch.setenv("AUTOPILOT_BUDGET_MODEL", "qwen2.5:1.5b")

    s = get_settings()
    assert s.stage1_threshold == pytest.approx(0.6)
    assert s.budget_model == "qwen2.5:1.5b"


def test_settings_is_frozen() -> None:
    s = Settings()
    with pytest.raises(ValidationError, match="frozen"):
        s.stage1_threshold = 0.1  # type: ignore[misc]


def test_threshold_must_be_in_range(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOPILOT_STAGE1_THRESHOLD", "1.5")
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        get_settings()
