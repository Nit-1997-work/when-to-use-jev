from __future__ import annotations

import os
from pathlib import Path

import pytest

from jevbench.settings import LLM_SAFETY_SETTINGS, Prices, Settings, load_settings

REQUIRED = {
    "OPENAI_API_BASE": "https://gateway.invalid/v1",
    "OPENAI_API_KEY": "gateway-secret",
    "BASELINE_MODEL": "google/gemini-lite",
    "TYPESAFE_API_KEY": "typesafe-secret",
}
PREFIXES = ("OPENAI_", "BASELINE_", "TYPESAFE_", "JEV_", "PHOENIX_", "REQUEST_TIMEOUT", "CLIENT_LOCATION")


@pytest.fixture
def no_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A clean environment and an absent .env, so only what a test sets is visible."""
    for name in list(os.environ):
        if name.startswith(PREFIXES):
            monkeypatch.delenv(name)
    return tmp_path / "absent.env"


def _load(env_file: Path, monkeypatch: pytest.MonkeyPatch, **values: str) -> Settings:
    for name, value in {**REQUIRED, **values}.items():
        monkeypatch.setenv(name, value)
    return load_settings(env_file)


def test_defaults(no_env_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _load(no_env_file, monkeypatch)
    assert settings.typesafe_model == "jev-1.13.0"
    assert settings.typesafe_base_url == "https://api.typesafe.ai"
    assert settings.jev_input_price == 0.042
    assert settings.request_timeout_s == 30.0
    assert settings.baseline_2 is None
    assert settings.baseline.prices == Prices(None, None, None)
    assert settings.phoenix_enabled is False
    assert settings.phoenix_base_url == "http://localhost:6006"


def test_values_are_parsed_and_quotes_stripped(no_env_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _load(
        no_env_file,
        monkeypatch,
        BASELINE_INPUT_PRICE_PER_MTOK="0.30",
        BASELINE_OUTPUT_PRICE_PER_MTOK="2.50",
        BASELINE_MODEL_2="google/gemini-flash",
        BASELINE_2_REASONING_EFFORT="'low'",
        BASELINE_2_INPUT_PRICE_PER_MTOK="0.75",
        CLIENT_LOCATION='"lab, wired"',
        REQUEST_TIMEOUT_SECONDS="12",
        PHOENIX_COLLECTOR_ENDPOINT="http://localhost:6006",
    )
    assert settings.baseline.prices == Prices(0.30, 2.50, None)
    assert settings.baseline.prices.effective_thinking == 2.50  # thinking billed at the output rate
    assert settings.baseline_2 is not None
    assert settings.baseline_2.reasoning_effort == "low"
    assert settings.client_location == "lab, wired"
    assert settings.request_timeout_s == 12.0
    assert settings.phoenix_enabled is True


def test_missing_or_placeholder_credentials_stop_a_run(no_env_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        _load(no_env_file, monkeypatch, OPENAI_API_KEY="changeme")


def test_commands_without_model_calls_need_no_credentials(no_env_file: Path) -> None:
    settings = load_settings(no_env_file, require_credentials=False)
    assert (settings.gateway_api_key, settings.typesafe_api_key, settings.baseline.model) == ("", "", "")


def test_a_non_numeric_price_is_rejected(no_env_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit, match="must be a number"):
        _load(no_env_file, monkeypatch, JEV_INPUT_PRICE_PER_MTOK="cheap")


def test_public_snapshot_has_no_secrets_or_gateway_url(no_env_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _load(no_env_file, monkeypatch, JEV_PRICE_SOURCE_URL="https://jev/pricing")
    snapshot = repr(settings.public_snapshot()) + repr(settings)
    for private in ("gateway-secret", "typesafe-secret", "gateway.invalid"):
        assert private not in snapshot
    assert settings.public_snapshot()["llm_safety_settings"] == LLM_SAFETY_SETTINGS
    assert settings.public_snapshot()["jev_price_source_url"] == "https://jev/pricing"


def test_prices_for_each_system(no_env_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _load(no_env_file, monkeypatch, BASELINE_INPUT_PRICE_PER_MTOK="0.3")
    assert settings.prices_for("jev") == Prices(0.042, 0.0, 0.0)
    assert settings.prices_for("baseline") == settings.prices_for("baseline-conf") == settings.baseline.prices
    assert settings.prices_for("baseline-2") is None  # not configured
    assert settings.prices_for("unknown") is None


def test_prices_round_trip_through_manifests() -> None:
    prices = Prices(0.3, 2.5, None)
    assert prices.as_dict() == {"input": 0.3, "output": 2.5, "thinking": 2.5}
    assert Prices.from_dict(prices.as_dict()) == Prices(0.3, 2.5, 2.5)
    assert Prices.from_dict({"input": True, "output": "2", "thinking": 10**400}) == Prices(None, None, None)
