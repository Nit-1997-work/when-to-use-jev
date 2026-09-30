from __future__ import annotations

from dataclasses import replace

import pytest

from jevbench.classifiers import SYSTEMS, Classifier, build_classifier
from jevbench.classifiers.jev import JevClassifier
from jevbench.classifiers.llm import GatewayLLMClassifier
from jevbench.intents import load_catalog
from jevbench.settings import LLMSystemConfig, Prices, Settings


@pytest.fixture
def two_models(settings: Settings) -> Settings:
    second = LLMSystemConfig(model="google/test-model-2", reasoning_effort="low", prices=Prices(None, None, None))
    return replace(settings, baseline_2=second)


def _llm(classifier: Classifier) -> GatewayLLMClassifier:
    assert isinstance(classifier, GatewayLLMClassifier)
    return classifier


def _required(classifier: GatewayLLMClassifier) -> object:
    response_format = classifier.request_kwargs("hi")["response_format"]
    return response_format["json_schema"]["schema"]["required"]  # pyright: ignore[reportIndexIssue]


async def test_each_system_gets_its_own_model_and_output_contract(two_models: Settings) -> None:
    catalog = load_catalog()
    built = {name: build_classifier(name, two_models, catalog) for name in SYSTEMS}
    try:
        assert isinstance(built["jev"], JevClassifier)
        assert built["jev"].model_requested == "jev-test"
        baseline, conf, second = (_llm(built[name]) for name in ("baseline", "baseline-conf", "baseline-2"))
        assert (baseline.name, conf.name, second.name) == ("baseline", "baseline-conf", "baseline-2")
        assert baseline.model_requested == conf.model_requested == "google/test-model"
        assert second.model_requested == "google/test-model-2"
        assert _required(baseline) == ["intent"]
        assert _required(conf) == ["intent", "confidence"]  # only the -conf system asks for a confidence
        assert second.request_kwargs("hi")["reasoning_effort"] == "low"
        assert "reasoning_effort" not in baseline.request_kwargs("hi")
    finally:
        for classifier in built.values():
            await classifier.aclose()


def test_the_second_baseline_needs_a_model(settings: Settings) -> None:
    with pytest.raises(SystemExit, match="BASELINE_MODEL_2"):
        build_classifier("baseline-2", settings, load_catalog())


def test_unknown_systems_are_rejected(settings: Settings) -> None:
    with pytest.raises(SystemExit, match="Unknown system"):
        build_classifier("gpt", settings, load_catalog())
