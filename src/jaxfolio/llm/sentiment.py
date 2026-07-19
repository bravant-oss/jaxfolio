"""Local-LLM news-sentiment scoring and its conversion to return views.

A local model reads per-asset news/context text and emits a sentiment score in
``[-1, 1]``; :func:`sentiment_to_views` turns those scores into small
expected-return tilts suitable for the Black-Litterman ``views`` argument.
"""

from __future__ import annotations

import numpy as np

from jaxfolio.llm.client import LLMClient, parse_json

_SYSTEM = (
    "You are a financial news sentiment analyst. For each asset, read the provided "
    "headlines/notes and score sentiment for the asset's forward returns on a scale "
    "from -1.0 (very bearish) to +1.0 (very bullish), with 0.0 neutral. Respond "
    "with ONLY a JSON object mapping each ticker to its numeric score."
)


def _format_news(news: dict) -> str:
    """Render the per-asset news block for the prompt.

    ``news`` maps each asset to either one string or a list of headlines.
    """
    blocks = []
    for asset, items in news.items():
        text = items if isinstance(items, str) else "; ".join(str(x) for x in items)
        blocks.append(f"- {asset}: {text}")
    return "\n".join(blocks)


def llm_sentiment(
    news: dict,
    client: LLMClient,
    *,
    temperature: float = 0.3,
) -> dict[str, float]:
    """Score per-asset news sentiment in ``[-1, 1]`` with a local LLM.

    ``news`` maps asset -> a string or list of headlines. Assets whose score the
    model omits or malforms are skipped (treated as neutral downstream).
    """
    if not news:
        return {}
    prompt = (
        "Score sentiment for each asset based on its news:\n"
        f"{_format_news(news)}\n\n"
        "Return a JSON object mapping each ticker to a score in [-1, 1]."
    )
    resp = client.complete(prompt, system=_SYSTEM, temperature=temperature)
    obj = parse_json(resp)
    scores: dict[str, float] = {}
    for asset in news:
        if asset in obj:
            try:
                scores[asset] = float(np.clip(float(obj[asset]), -1.0, 1.0))
            except (TypeError, ValueError):
                continue
    return scores


def sentiment_to_views(
    scores: dict[str, float],
    *,
    strength: float = 0.03,
) -> dict[str, float]:
    """Convert sentiment scores to expected-return views.

    Each score in ``[-1, 1]`` becomes a view of ``score * strength`` (so
    ``strength`` is the per-period return magnitude a maximally bullish call
    implies, e.g. ``0.03`` = +3%).
    """
    return {asset: float(score * strength) for asset, score in scores.items()}
