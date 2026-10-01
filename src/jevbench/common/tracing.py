"""Phoenix / OpenInference tracing. Optional: without PHOENIX_COLLECTOR_ENDPOINT every call here is a no-op."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode, Tracer
from opentelemetry.util.types import AttributeValue

from jevbench.common.settings import Settings

_TRACER_NAME = "jevbench"
_configured_project: str | None = None


def setup_tracing(settings: Settings, system_name: str) -> str | None:
    """Send spans for this process to the Phoenix project for `system_name`. Returns the project name."""
    global _configured_project
    if not settings.phoenix_enabled:
        return None
    if _configured_project is not None:
        return _configured_project

    from openinference.instrumentation.openai import OpenAIInstrumentor
    from phoenix.otel import register

    project = f"{settings.phoenix_project_prefix}-{system_name}"
    # batch=True: spans are exported in the background, never on the timed request path.
    provider = register(
        endpoint=f"{settings.phoenix_collector_endpoint.rstrip('/')}/v1/traces"
        if settings.phoenix_collector_endpoint
        else None,
        project_name=project,
        batch=True,
        verbose=False,
    )
    OpenAIInstrumentor().instrument(tracer_provider=provider)
    _configured_project = project
    return project


def shutdown_tracing() -> None:
    provider = trace.get_tracer_provider()
    shutdown = getattr(provider, "shutdown", None)
    if callable(shutdown):
        shutdown()  # flushes the batch processor


def tracer() -> Tracer:
    return trace.get_tracer(_TRACER_NAME)


def current_trace_id() -> str | None:
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else None


@contextmanager
def example_span(name: str, system: str, example_id: str, split: str, repeat: int, input_value: str) -> Iterator[Span]:
    """Parent span for one example (one classification, one re-ranked search); provider calls are child spans."""
    with tracer().start_as_current_span(
        name,
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
            SpanAttributes.INPUT_VALUE: input_value,
            "jevbench.system": system,
            "jevbench.example_id": example_id,
            "jevbench.split": split,
            "jevbench.repeat": repeat,
        },
    ) as span:
        yield span


def record_result(span: Span, *, output: str, outcome: str, attributes: Mapping[str, AttributeValue]) -> None:
    """The example's output, outcome, and experiment-specific attributes; a non-ok outcome marks the span an error."""
    span.set_attribute(SpanAttributes.OUTPUT_VALUE, output)
    for key, value in attributes.items():
        span.set_attribute(key, value)
    span.set_attribute("jevbench.outcome", outcome)
    if outcome != "ok":
        span.set_status(Status(StatusCode.ERROR, outcome))


@contextmanager
def jev_llm_span(model: str, state: Mapping[str, object]) -> Iterator[Span]:
    """Manual OpenInference LLM span for Jev (there is no auto-instrumentation for the TypeSafe SDK)."""
    with tracer().start_as_current_span(
        "typesafe.system_one",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
            SpanAttributes.LLM_MODEL_NAME: model,
            SpanAttributes.LLM_PROVIDER: "typesafe",
            SpanAttributes.INPUT_VALUE: json.dumps(state, ensure_ascii=False),
            SpanAttributes.INPUT_MIME_TYPE: "application/json",
        },
    ) as span:
        yield span


def record_jev_usage(
    span: Span,
    *,
    model_reported: str,
    input_tokens: int | None,
    output_tokens: int | None,
    answer: Mapping[str, object],
) -> None:
    span.set_attribute(SpanAttributes.LLM_MODEL_NAME, model_reported)
    if input_tokens is not None:
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, input_tokens)
    if output_tokens is not None:
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, output_tokens)
    if input_tokens is not None and output_tokens is not None:
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, input_tokens + output_tokens)
    span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps(answer, ensure_ascii=False))
    span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "application/json")
