"""Tests for the local-LLM strategies — fully offline via FakeLLM.

No network or Ollama server is required: FakeLLM returns canned JSON. A single
optional test exercises a real local Ollama server when one is reachable.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import jaxfolio as jf
from jaxfolio.llm import (
    FakeLLM,
    debate,
    llm_agent_portfolio,
    llm_black_litterman,
    llm_sentiment,
    llm_sentiment_portfolio,
    llm_views,
    parse_json,
    sentiment_to_views,
)
from jaxfolio.llm.client import OllamaClient


@pytest.fixture
def universe():
    r = jf.generate_returns(n_assets=5, n_days=300, seed=5)
    return r, list(r.columns)


def _views_responder(assets, mu=0.01, sd=0.004):
    """A FakeLLM responder emitting per-asset view JSON that varies per call."""

    def responder(prompt, i):
        rng = np.random.default_rng(i)
        return json.dumps({a: round(float(rng.normal(mu, sd)), 4) for a in assets})

    return responder


# --------------------------------------------------------------------------- #
# parse_json
# --------------------------------------------------------------------------- #
def test_parse_json_plain():
    assert parse_json('{"A": 0.01, "B": -0.02}') == {"A": 0.01, "B": -0.02}


def test_parse_json_fenced_with_prose():
    text = 'Here are my views:\n```json\n{"A": 0.03}\n```\nHope that helps!'
    assert parse_json(text) == {"A": 0.03}


def test_parse_json_embedded():
    assert parse_json('blah {"A": 1} trailing') == {"A": 1}


def test_parse_json_garbage_returns_empty():
    assert parse_json("no json here") == {}
    assert parse_json("") == {}


# --------------------------------------------------------------------------- #
# views
# --------------------------------------------------------------------------- #
def test_llm_views_parses_and_calibrates(universe):
    r, assets = universe
    llm = FakeLLM(_views_responder(assets))
    vs = llm_views(r, llm, samples=6)
    assert set(vs.views) == set(assets)
    assert 0.0 < vs.confidence <= 1.0
    # With 6 samples every asset should have a dispersion estimate.
    assert len(vs.per_asset_std) == len(assets)
    assert len(vs.raw_samples) == 6


def test_llm_views_tight_agreement_higher_confidence(universe):
    r, assets = universe
    # Identical responses => zero dispersion => high confidence.
    fixed = json.dumps({a: 0.01 for a in assets})
    tight = llm_views(r, FakeLLM([fixed]), samples=5)
    noisy = llm_views(r, FakeLLM(_views_responder(assets, sd=0.02)), samples=5)
    assert tight.confidence >= noisy.confidence


# --------------------------------------------------------------------------- #
# sentiment
# --------------------------------------------------------------------------- #
def test_llm_sentiment_scores_clipped(universe):
    _, assets = universe
    news = {a: "great earnings" for a in assets}
    # Include an out-of-range value to confirm clipping.
    llm = FakeLLM([json.dumps({assets[0]: 5.0, **{a: 0.5 for a in assets[1:]}})])
    scores = llm_sentiment(news, llm)
    assert scores[assets[0]] == 1.0  # clipped to [-1, 1]
    assert all(-1.0 <= v <= 1.0 for v in scores.values())


def test_sentiment_to_views_scaling():
    views = sentiment_to_views({"A": 1.0, "B": -0.5}, strength=0.02)
    assert views["A"] == pytest.approx(0.02)
    assert views["B"] == pytest.approx(-0.01)


def test_llm_sentiment_empty_news():
    assert llm_sentiment({}, FakeLLM(["{}"])) == {}


# --------------------------------------------------------------------------- #
# agents
# --------------------------------------------------------------------------- #
def test_debate_aggregates_agents(universe):
    r, assets = universe
    llm = FakeLLM(_views_responder(assets))
    result = debate(r, llm)
    assert set(result.agent_views) == {"bull", "bear", "risk"}
    assert set(result.views) == set(assets)
    assert 0.0 < result.confidence <= 1.0


# --------------------------------------------------------------------------- #
# strategies (end-to-end, offline)
# --------------------------------------------------------------------------- #
def test_llm_black_litterman_valid(universe):
    r, assets = universe
    res = llm_black_litterman(r, client=FakeLLM(_views_responder(assets)), samples=5)
    assert np.isclose(res.weights.sum(), 1.0, atol=1e-3)
    assert (res.weights >= -1e-4).all()
    assert res.method == "LLM Black-Litterman"
    assert "llm_views" in res.metadata
    assert res.metadata["samples"] == 5


def test_llm_sentiment_portfolio_valid(universe):
    r, assets = universe
    news = {a: ["strong guidance", "analyst upgrade"] for a in assets}
    llm = FakeLLM([json.dumps({a: 0.4 for a in assets})])
    res = llm_sentiment_portfolio(r, news, client=llm)
    assert np.isclose(res.weights.sum(), 1.0, atol=1e-3)
    assert res.method == "LLM Sentiment Tilt"
    assert res.metadata["sentiment_scores"]


def test_llm_agent_portfolio_valid(universe):
    r, assets = universe
    res = llm_agent_portfolio(r, client=FakeLLM(_views_responder(assets)))
    assert np.isclose(res.weights.sum(), 1.0, atol=1e-3)
    assert res.method == "LLM Multi-Agent Debate"
    assert set(res.metadata["agent_views"]) == {"bull", "bear", "risk"}


def test_llm_strategy_works_in_backtest(universe):
    import functools

    r, assets = universe
    strat = functools.partial(
        llm_black_litterman, client=FakeLLM(_views_responder(assets)), samples=3
    )
    from jaxfolio.backtest import backtest

    res = backtest(r, strat, lookback=120, rebalance_every=40)
    assert len(res.returns) > 0


# --------------------------------------------------------------------------- #
# client error path + optional live test
# --------------------------------------------------------------------------- #
def test_ollama_unreachable_raises_clear_error():
    # Point at a port nothing is listening on; expect an actionable ConnectionError.
    client = OllamaClient(host="http://127.0.0.1:1", timeout=1.0)
    with pytest.raises(ConnectionError) as exc:
        client.complete("hi")
    assert "Ollama" in str(exc.value)


def _ollama_models() -> list[str]:
    """Return the list of models a local Ollama server has pulled (empty if none)."""
    try:
        import requests

        resp = requests.get("http://localhost:11434/api/tags", timeout=0.5)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


@pytest.mark.skipif(not _ollama_models(), reason="no local Ollama server with a pulled model")
def test_llm_black_litterman_live(universe):  # pragma: no cover - needs a server + model
    r, _ = universe
    model = _ollama_models()[0]
    res = llm_black_litterman(r, client=OllamaClient(model), samples=2)
    assert np.isclose(res.weights.sum(), 1.0, atol=1e-2)
