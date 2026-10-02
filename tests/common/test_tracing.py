from __future__ import annotations

from dataclasses import replace

import pytest
from opentelemetry.trace import StatusCode

import jevbench.common.tracing as tracing
from jevbench.common.settings import Settings


def test_tracing_is_off_without_a_collector(settings: Settings) -> None:
    assert tracing.setup_tracing(settings, "jev") is None


def test_setup_registers_one_batched_project_per_process(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    import openinference.instrumentation.openai as openai_instrumentation
    import phoenix.otel

    registered: list[dict[str, object]] = []
    instrumented: list[object] = []

    class FakeInstrumentor:
        def instrument(self, *, tracer_provider: object) -> None:
            instrumented.append(tracer_provider)

    monkeypatch.setattr(tracing, "_configured_project", None)
    monkeypatch.setattr(phoenix.otel, "register", lambda **kwargs: registered.append(kwargs) or "provider")
    monkeypatch.setattr(openai_instrumentation, "OpenAIInstrumentor", FakeInstrumentor)
    enabled = replace(settings, phoenix_collector_endpoint="http://localhost:6006/")

    assert tracing.setup_tracing(enabled, "jev") == "test-jev"
    assert tracing.setup_tracing(enabled, "baseline") == "test-jev"  # one project per process
    assert registered == [
        {"endpoint": "http://localhost:6006/v1/traces", "project_name": "test-jev", "batch": True, "verbose": False}
    ]
    assert instrumented == ["provider"]


class _Span:
    def __init__(self) -> None:
        self.attributes: dict[str, object] = {}
        self.status: object = None

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def set_status(self, status: object) -> None:
        self.status = status


def test_outcomes_and_usage_are_recorded_on_spans() -> None:
    span = _Span()
    tracing.record_result(span, output="", outcome="refused", attributes={"jevbench.correct": False})  # pyright: ignore[reportArgumentType]
    assert span.attributes["jevbench.outcome"] == "refused"
    assert span.attributes["jevbench.correct"] is False
    assert span.status is not None
    assert span.status.status_code == StatusCode.ERROR  # pyright: ignore[reportAttributeAccessIssue]

    usage = _Span()
    tracing.record_jev_usage(usage, model_reported="jev-1", input_tokens=10, output_tokens=2, answer={"choice": "a"})  # pyright: ignore[reportArgumentType]
    assert usage.attributes["llm.token_count.total"] == 12
    assert usage.attributes["llm.model_name"] == "jev-1"


def test_spans_are_no_ops_without_a_provider() -> None:
    with tracing.example_span("classify", "jev", "e1", "main", 1, "hi"), tracing.jev_llm_span("jev-1", {"text": "hi"}):
        assert tracing.current_trace_id() is None
    tracing.shutdown_tracing()
