"""Multi-agent LLM debate for portfolio views (AlphaAgents / HARLF style).

Several role-specialized agents — a bull, a bear, and a risk manager by default —
each argue a per-asset stance from the same data using a local LLM. A moderator
step aggregates their (confidence-weighted) stances into a single set of
expected-return views, which then flow into Black-Litterman.

The value over a single prompt is *structured disagreement*: the bull looks for
upside, the bear for downside, the risk agent penalizes volatility, and the
aggregation reflects where they converge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from jaxfolio.llm.client import LLMClient, parse_json
from jaxfolio.llm.views import _format_stats, calibrate_confidence

# Each role: (name, system-prompt persona, view-tilt sign preference).
DEFAULT_ROLES: tuple[tuple[str, str, float], ...] = (
    (
        "bull",
        "You are a bullish growth-oriented analyst. You look for upside catalysts and "
        "momentum. Score each asset's next-period expected return (decimal), leaning "
        "optimistic where justified.",
        1.0,
    ),
    (
        "bear",
        "You are a bearish, skeptical short-seller. You look for downside risks, "
        "over-extension, and mean reversion. Score each asset's next-period expected "
        "return (decimal), leaning cautious where justified.",
        1.0,
    ),
    (
        "risk",
        "You are a risk manager. You prioritize capital preservation and penalize "
        "high-volatility assets. Score each asset's next-period expected return "
        "(decimal), discounting for risk.",
        1.0,
    ),
)

_JSON_INSTRUCTION = (
    "Respond with ONLY a JSON object mapping each ticker to your expected "
    "next-period return as a decimal (e.g. 0.02 for +2%)."
)


@dataclass
class DebateResult:
    """Aggregated views plus each agent's individual stance."""

    views: dict[str, float]
    confidence: float
    agent_views: dict[str, dict[str, float]] = field(default_factory=dict)


def _agent_prompt(stats: str, persona: str, news: str | None) -> str:
    ctx = f"\n\nRelevant news:\n{news}" if news else ""
    return f"{persona}\n\nUniverse statistics:\n{stats}{ctx}\n\n{_JSON_INSTRUCTION}"


def debate(
    returns,
    client: LLMClient,
    *,
    roles=DEFAULT_ROLES,
    news: dict | None = None,
    temperature: float = 0.7,
    clip: tuple[float, float] = (-0.2, 0.2),
) -> DebateResult:
    """Run a one-round multi-agent debate and aggregate into views.

    Each role produces per-asset return estimates; the moderator averages them
    per asset (equal weight across agents), and confidence reflects cross-agent
    agreement (assets the agents agree on carry more weight).
    """
    stats, names = _format_stats(returns)
    news_block = None
    if news:
        news_block = "\n".join(
            f"- {a}: {v if isinstance(v, str) else '; '.join(map(str, v))}" for a, v in news.items()
        )

    lo, hi = clip
    agent_views: dict[str, dict[str, float]] = {}
    stacked: dict[str, list[float]] = {n: [] for n in names}

    for role_name, persona, _pref in roles:
        resp = client.complete(
            _agent_prompt(stats, persona, news_block),
            system=persona,
            temperature=temperature,
        )
        obj = parse_json(resp)
        view: dict[str, float] = {}
        for name in names:
            if name in obj:
                try:
                    val = float(np.clip(float(obj[name]), lo, hi))
                except (TypeError, ValueError):
                    continue
                view[name] = val
                stacked[name].append(val)
        agent_views[role_name] = view

    views = {n: float(np.mean(v)) for n, v in stacked.items() if v}
    # Confidence: high when the agents agree, relative to the view magnitude.
    per_asset_std = {n: float(np.std(v)) for n, v in stacked.items() if len(v) > 1}
    confidence = calibrate_confidence(views, per_asset_std)

    return DebateResult(views=views, confidence=confidence, agent_views=agent_views)
