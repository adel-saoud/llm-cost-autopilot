"""Config defaults and environment-variable overrides."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_cost_autopilot.config import Settings, UnknownSettingError, get_settings


def test_defaults_match_documented_values() -> None:
    s = Settings()
    assert s.stage1_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert s.stage2_model.startswith("MoritzLaurer/deberta-v3-xsmall")
    assert s.stage1_threshold == pytest.approx(0.75)
    assert s.simple_model == "qwen3.5:0.8b"
    assert s.moderate_model == "qwen3.5:4b"
    assert s.verifier_enabled is True


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOPILOT_STAGE1_THRESHOLD", "0.6")
    monkeypatch.setenv("AUTOPILOT_SIMPLE_MODEL", "qwen3.5:2b")

    s = get_settings()
    assert s.stage1_threshold == pytest.approx(0.6)
    assert s.simple_model == "qwen3.5:2b"


def test_settings_is_frozen() -> None:
    s = Settings()
    with pytest.raises(ValidationError, match="frozen"):
        s.stage1_threshold = 0.1  # type: ignore[misc]


def test_threshold_must_be_in_range(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOPILOT_STAGE1_THRESHOLD", "1.5")
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        get_settings()


def test_misspelled_variable_stops_startup_with_the_closest_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTOPILOT_STAGE1_TRESHOLD", "0.6")
    with pytest.raises(UnknownSettingError, match="did you mean AUTOPILOT_STAGE1_THRESHOLD"):
        get_settings()


@pytest.mark.parametrize(
    "removed_name",
    ["AUTOPILOT_BUDGET_COST_PER_1K_INPUT", "AUTOPILOT_BUDGET_MODEL", "AUTOPILOT_STANDARD_MODEL"],
)
def test_removed_variable_stops_startup(monkeypatch: pytest.MonkeyPatch, removed_name: str) -> None:
    monkeypatch.setenv(removed_name, "0.1")
    with pytest.raises(UnknownSettingError, match=removed_name):
        get_settings()


def test_unknown_name_in_dotenv_stops_startup(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("AUTOPILOT_QWERTY=1\n", encoding="utf-8")
    with pytest.raises(UnknownSettingError, match="no setting has a close name"):
        Settings(_env_file=env_file)  # type: ignore[call-arg]


def test_other_programs_variables_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11434")
    assert get_settings().simple_model == "qwen3.5:0.8b"
