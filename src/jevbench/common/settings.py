"""Configuration loaded from `.env` (see `.env.example`). Secrets never leave this module unredacted."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from jevbench.common.paths import ENV_FILE

# The harness never sends provider safety settings, so every LLM call uses the provider's defaults.
LLM_SAFETY_SETTINGS = "provider defaults (not overridden)"


def load_env(env_file: Path | None = None) -> None:
    """Load `.env` into the process environment. Values already set in the environment win."""
    load_dotenv(env_file or ENV_FILE, override=False)


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, "").strip().strip("'\"")
    return value or default


def _float(name: str) -> float | None:
    value = _env(name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        raise SystemExit(f"Setting {name} must be a number, got {value!r}.") from None


def _float_or(name: str, default: float) -> float:
    value = _float(name)
    return default if value is None else value


def _optional_price(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        return float(value)
    except OverflowError:  # an int too large for a float is not a usable price
        return None


@dataclass(frozen=True)
class Prices:
    """USD per 1M tokens. `None` means not configured; cost is then reported as unknown, never as 0."""

    input: float | None
    output: float | None
    thinking: float | None

    @property
    def effective_thinking(self) -> float | None:
        # Providers usually bill reasoning tokens at the output rate.
        return self.thinking if self.thinking is not None else self.output

    def as_dict(self) -> dict[str, float | None]:
        """The form recorded in run manifests (thinking already resolved to the rate actually applied)."""
        return {"input": self.input, "output": self.output, "thinking": self.effective_thinking}

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> Prices:
        return cls(
            input=_optional_price(raw.get("input")),
            output=_optional_price(raw.get("output")),
            thinking=_optional_price(raw.get("thinking")),
        )


@dataclass(frozen=True)
class LLMSystemConfig:
    model: str
    reasoning_effort: str | None
    prices: Prices


@dataclass(frozen=True)
class Settings:
    # Private: the gateway host is deployment-specific, so it stays out of reprs, logs, and tracebacks.
    gateway_base_url: str = field(repr=False)
    gateway_api_key: str = field(repr=False)
    baseline: LLMSystemConfig
    baseline_2: LLMSystemConfig | None
    typesafe_api_key: str = field(repr=False)
    typesafe_base_url: str
    typesafe_model: str
    jev_input_price: float
    jev_price_source_url: str | None
    price_source_url: str | None
    request_timeout_s: float
    client_location: str | None
    phoenix_collector_endpoint: str | None
    phoenix_base_url: str | None
    phoenix_project_prefix: str

    @property
    def phoenix_enabled(self) -> bool:
        return self.phoenix_collector_endpoint is not None

    @property
    def jev_prices(self) -> Prices:
        return Prices(input=self.jev_input_price, output=0.0, thinking=0.0)  # Jev output is free

    def public_snapshot(self) -> dict[str, object]:
        """Everything needed to reproduce a run, with secrets removed."""

        def llm(cfg: LLMSystemConfig | None) -> dict[str, object] | None:
            if cfg is None:
                return None
            return {
                "model": cfg.model,
                "reasoning_effort": cfg.reasoning_effort,
                "price_per_mtok": cfg.prices.as_dict(),
            }

        return {
            "gateway": "LiteLLM-based, OpenAI Chat Completions compatible",
            "baseline": llm(self.baseline),
            "baseline_2": llm(self.baseline_2),
            "llm_safety_settings": LLM_SAFETY_SETTINGS,
            "typesafe_base_url": self.typesafe_base_url,
            "typesafe_model": self.typesafe_model,
            "jev_input_price_per_mtok": self.jev_input_price,
            "jev_price_source_url": self.jev_price_source_url,
            "price_source_url": self.price_source_url,
            "request_timeout_s": self.request_timeout_s,
            "client_location": self.client_location,
        }


def _credential(name: str, *, required: bool) -> str:
    value = _env(name)
    if value is None or value == "changeme":
        if required:
            raise SystemExit(f"Missing required setting {name}. Copy .env.example to .env and fill it in.")
        return ""
    return value


def _prices(prefix: str) -> Prices:
    return Prices(
        input=_float(f"{prefix}_INPUT_PRICE_PER_MTOK"),
        output=_float(f"{prefix}_OUTPUT_PRICE_PER_MTOK"),
        thinking=_float(f"{prefix}_THINKING_PRICE_PER_MTOK"),
    )


def load_settings(env_file: Path | None = None, *, require_credentials: bool = True) -> Settings:
    """Settings from the environment and `.env`.

    `require_credentials=False` is for commands that never call a model (Phoenix setup and sync): missing keys,
    gateway URL, and baseline model are then left empty instead of stopping the command.
    """
    load_env(env_file)

    baseline_2_model = _env("BASELINE_MODEL_2")
    baseline_2 = (
        LLMSystemConfig(
            model=baseline_2_model,
            reasoning_effort=_env("BASELINE_2_REASONING_EFFORT"),
            prices=_prices("BASELINE_2"),
        )
        if baseline_2_model
        else None
    )

    return Settings(
        gateway_base_url=_credential("OPENAI_API_BASE", required=require_credentials),
        gateway_api_key=_credential("OPENAI_API_KEY", required=require_credentials),
        baseline=LLMSystemConfig(
            model=_credential("BASELINE_MODEL", required=require_credentials),
            reasoning_effort=_env("BASELINE_REASONING_EFFORT"),
            prices=_prices("BASELINE"),
        ),
        baseline_2=baseline_2,
        typesafe_api_key=_credential("TYPESAFE_API_KEY", required=require_credentials),
        typesafe_base_url=_env("TYPESAFE_BASE_URL") or "https://api.typesafe.ai",
        typesafe_model=_env("TYPESAFE_DEFAULT_MODEL") or "jev-1.13.0",
        jev_input_price=_float_or("JEV_INPUT_PRICE_PER_MTOK", 0.042),
        jev_price_source_url=_env("JEV_PRICE_SOURCE_URL"),
        price_source_url=_env("BASELINE_PRICE_SOURCE_URL"),
        request_timeout_s=_float_or("REQUEST_TIMEOUT_SECONDS", 30.0),
        client_location=_env("CLIENT_LOCATION"),
        phoenix_collector_endpoint=_env("PHOENIX_COLLECTOR_ENDPOINT"),
        phoenix_base_url=_env("PHOENIX_BASE_URL") or "http://localhost:6006",
        phoenix_project_prefix=_env("PHOENIX_PROJECT_PREFIX") or "jevbench",
    )
