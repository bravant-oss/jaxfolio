"""LLM-driven portfolio strategies (local models).

Three strategies, all with the standard ``method(returns, ...) -> PortfolioResult``
shape so they drop into the backtester and plots like any built-in. Each elicits
per-asset **views** from a local LLM and routes them through the existing
:func:`jaxfolio.optimizers.classical.black_litterman`, so they inherit its
equilibrium prior, constraints, and diagnostics for free.

By default they construct an :class:`~jaxfolio.llm.client.OllamaClient`; pass
``client=FakeLLM(...)`` (or any :class:`~jaxfolio.llm.client.LLMClient`) to run
offline or in tests. To use one inside :func:`jaxfolio.backtest.compare`, bind the
non-returns arguments with :func:`functools.partial`.
"""

from __future__ import annotations

from jaxfolio.llm.agents import DEFAULT_ROLES, debate
from jaxfolio.llm.client import LLMClient, default_client
from jaxfolio.llm.sentiment import llm_sentiment, sentiment_to_views
from jaxfolio.llm.views import llm_views
from jaxfolio.optimizers.classical import black_litterman
from jaxfolio.types import OptimizerConfig, PortfolioResult


def _resolve_client(client: LLMClient | None) -> LLMClient:
    return client if client is not None else default_client()


def llm_black_litterman(
    returns,
    client: LLMClient | None = None,
    *,
    samples: int = 5,
    extra_context: str | None = None,
    tau: float = 0.05,
    risk_aversion: float = 2.5,
    config: OptimizerConfig | None = None,
) -> PortfolioResult:
    """LLM-enhanced Black-Litterman (ICLR 2025).

    Samples the local LLM ``samples`` times for per-asset return views, calibrates
    the view confidence from the sampling dispersion, and feeds both into
    Black-Litterman. Metadata records the raw views and confidence.
    """
    client = _resolve_client(client)
    viewset = llm_views(returns, client, samples=samples, extra_context=extra_context)
    result = black_litterman(
        returns,
        views=viewset.views,
        view_confidence=viewset.confidence,
        tau=tau,
        risk_aversion=risk_aversion,
        config=config,
    )
    result.method = "LLM Black-Litterman"
    result.metadata.update(
        {
            "llm_views": viewset.views,
            "llm_confidence": viewset.confidence,
            "llm_view_std": viewset.per_asset_std,
            "samples": samples,
        }
    )
    return result


def llm_sentiment_portfolio(
    returns,
    news: dict,
    client: LLMClient | None = None,
    *,
    strength: float = 0.03,
    view_confidence: float = 0.5,
    tau: float = 0.05,
    risk_aversion: float = 2.5,
    config: OptimizerConfig | None = None,
) -> PortfolioResult:
    """News-sentiment-tilted portfolio via a local LLM.

    The LLM scores per-asset sentiment from ``news`` (``{asset: text | [headlines]}``),
    which is converted to return-view tilts and combined with the market
    equilibrium through Black-Litterman.
    """
    client = _resolve_client(client)
    scores = llm_sentiment(news, client)
    views = sentiment_to_views(scores, strength=strength)
    result = black_litterman(
        returns,
        views=views,
        view_confidence=view_confidence,
        tau=tau,
        risk_aversion=risk_aversion,
        config=config,
    )
    result.method = "LLM Sentiment Tilt"
    result.metadata.update({"sentiment_scores": scores, "sentiment_views": views})
    return result


def llm_agent_portfolio(
    returns,
    client: LLMClient | None = None,
    *,
    news: dict | None = None,
    roles=DEFAULT_ROLES,
    tau: float = 0.05,
    risk_aversion: float = 2.5,
    config: OptimizerConfig | None = None,
) -> PortfolioResult:
    """Multi-agent-debate portfolio (bull / bear / risk agents).

    Role-specialized LLM agents each argue per-asset views; the moderator
    aggregates them (with agreement-based confidence) and Black-Litterman turns
    the consensus into weights.
    """
    client = _resolve_client(client)
    result_debate = debate(returns, client, roles=roles, news=news)
    result = black_litterman(
        returns,
        views=result_debate.views,
        view_confidence=result_debate.confidence,
        tau=tau,
        risk_aversion=risk_aversion,
        config=config,
    )
    result.method = "LLM Multi-Agent Debate"
    result.metadata.update(
        {
            "agent_views": result_debate.agent_views,
            "consensus_views": result_debate.views,
            "consensus_confidence": result_debate.confidence,
        }
    )
    return result
