"""The LLM strategies must flag themselves as experimental at runtime."""

from __future__ import annotations

import json

import pytest

from jaxfolio.data.synthetic import generate_returns
from jaxfolio.llm import (
    ExperimentalWarning,
    llm_agent_portfolio,
    llm_black_litterman,
    llm_sentiment_portfolio,
)
from jaxfolio.llm._experimental import _reset_experimental_warning
from jaxfolio.llm.client import FakeLLM

ASSETS = [f"ASSET_{i:02d}" for i in range(4)]


@pytest.fixture
def returns():
    return generate_returns(n_assets=4, n_days=260, seed=1)


@pytest.fixture(autouse=True)
def _rearm_warning():
    """Re-arm the one-time warning before each test so it fires deterministically."""
    _reset_experimental_warning()
    yield
    _reset_experimental_warning()


def _fake():
    """A FakeLLM returning valid per-asset JSON so every strategy runs offline."""
    payload = json.dumps({a: 0.01 for a in ASSETS})
    return FakeLLM(lambda prompt, i: payload)


@pytest.mark.parametrize(
    "call",
    [
        lambda r: llm_black_litterman(r, client=_fake(), samples=1),
        lambda r: llm_sentiment_portfolio(r, {"ASSET_00": "up"}, client=_fake()),
        lambda r: llm_agent_portfolio(r, client=_fake()),
    ],
    ids=["black_litterman", "sentiment", "agent"],
)
def test_llm_strategy_warns_experimental(returns, call):
    with pytest.warns(ExperimentalWarning, match="investment advice"):
        call(returns)


def test_warning_fires_only_once_per_process(returns):
    import warnings

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        llm_black_litterman(returns, client=_fake(), samples=1)
        llm_black_litterman(returns, client=_fake(), samples=1)
    experimental = [w for w in record if issubclass(w.category, ExperimentalWarning)]
    assert len(experimental) == 1, "the experimental warning should fire once, not per call"
