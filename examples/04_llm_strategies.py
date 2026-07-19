"""LLM-based portfolio strategies on **local** models.

Runs fully offline by default using a deterministic FakeLLM, so you can see the
three strategies end-to-end with no model installed. Pass ``--live`` to drive a
real local Ollama model instead (``ollama serve`` + ``ollama pull llama3.1``).

    uv run python examples/04_llm_strategies.py                        # offline (FakeLLM)
    uv run python examples/04_llm_strategies.py --live                 # local Ollama (llama3.1)
    uv run python examples/04_llm_strategies.py --live --model gemma3  # a specific model
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

import jaxfolio as jf
from jaxfolio import viz
from jaxfolio.llm import (
    FakeLLM,
    OllamaClient,
    llm_agent_portfolio,
    llm_black_litterman,
    llm_sentiment_portfolio,
)

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)


def _live_model() -> str:
    """Resolve the Ollama model from ``--model``, ``$JAXFOLIO_OLLAMA_MODEL``, or default."""
    if "--model" in sys.argv:
        return sys.argv[sys.argv.index("--model") + 1]
    return os.environ.get("JAXFOLIO_OLLAMA_MODEL", "llama3.1")


def make_client(assets: list[str], live: bool):
    """Return a real Ollama client (``--live``) or an offline FakeLLM."""
    if live:
        return OllamaClient(_live_model())

    # Offline: a deterministic responder that emits per-asset view/score JSON,
    # jittered per call so the K-sample confidence calibration has signal.
    def responder(prompt: str, i: int) -> str:
        rng = np.random.default_rng(i + 1)
        if "sentiment" in prompt.lower() or "bullish" in prompt.lower():
            return json.dumps({a: round(float(rng.uniform(-0.5, 0.8)), 3) for a in assets})
        return json.dumps({a: round(float(rng.normal(0.01, 0.006)), 4) for a in assets})

    return FakeLLM(responder)


def main() -> None:
    live = "--live" in sys.argv
    returns = jf.generate_returns(n_assets=8, n_days=756, seed=7)
    assets = list(returns.columns)
    print(f"Mode: {f'LIVE (Ollama {_live_model()})' if live else 'offline (FakeLLM)'}\n")

    # 1 — LLM-enhanced Black-Litterman (ICLR 2025): sampled views + calibrated confidence.
    bl = llm_black_litterman(returns, client=make_client(assets, live), samples=5)
    print("LLM Black-Litterman")
    print("  views:", {k: round(v, 4) for k, v in bl.metadata["llm_views"].items()})
    print(f"  confidence: {bl.metadata['llm_confidence']:.2f}")
    print("  top holdings:", {k: f"{v:.1%}" for k, v in bl.top(3).items()})

    # 2 — News-sentiment tilt.
    news = {
        a: f"{a} posts record quarterly revenue and raises full-year guidance" for a in assets[:4]
    }
    sent = llm_sentiment_portfolio(returns, news, client=make_client(assets, live))
    print("\nLLM Sentiment Tilt")
    print("  scores:", {k: round(v, 2) for k, v in sent.metadata["sentiment_scores"].items()})
    print("  top holdings:", {k: f"{v:.1%}" for k, v in sent.top(3).items()})

    # 3 — Multi-agent debate (bull / bear / risk).
    agent = llm_agent_portfolio(returns, client=make_client(assets, live))
    print("\nLLM Multi-Agent Debate")
    print("  agents:", list(agent.metadata["agent_views"]))
    print("  top holdings:", {k: f"{v:.1%}" for k, v in agent.top(3).items()})

    # Visualize on the efficient frontier.
    highlight = {
        "LLM Black-Litterman": bl,
        "LLM Sentiment": sent,
        "LLM Agents": agent,
        "Max Sharpe": jf.maximum_sharpe(returns),
    }
    fig = viz.plot_efficient_frontier(returns, highlight=highlight)
    viz.save(fig, str(OUT / "llm_strategies_frontier.png"))
    viz.save(viz.plot_weights(bl), str(OUT / "llm_bl_weights.png"))
    print(f"\nSaved plots -> {OUT}/")


if __name__ == "__main__":
    main()
