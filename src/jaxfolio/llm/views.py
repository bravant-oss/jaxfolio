"""LLM-generated return views for the Black-Litterman model.

Implements the core of the ICLR-2025 "Integrating LLM-Generated Views into
Mean-Variance Optimization" method: prompt a local LLM with each asset's recent
return statistics, sample it ``k`` times, parse per-asset expected-return views,
and turn the *dispersion across samples* into a confidence — uncertain views are
automatically down-weighted. The output plugs straight into
:func:`jaxfolio.optimizers.classical.black_litterman`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from jaxfolio.llm.client import LLMClient, parse_json
from jaxfolio.moments.estimators import as_matrix

_SYSTEM = (
    "You are a quantitative equity analyst. Given recent return statistics for a "
    "set of assets, output a JSON object mapping each asset ticker to your view of "
    "its expected return over the next period, as a decimal (e.g. 0.02 for +2%, "
    "-0.01 for -1%). Respond with ONLY a JSON object, no commentary."
)


@dataclass
class ViewSet:
    """LLM-generated views and their calibrated confidences.

    Attributes
    ----------
    views:
        ``{asset: mean expected return}`` across the LLM samples.
    confidence:
        Scalar in ``(0, 1]`` summarizing overall view confidence (higher = the
        samples agreed more closely). Suitable for ``black_litterman``'s
        ``view_confidence`` argument.
    per_asset_std:
        ``{asset: std of the sampled views}`` — the raw dispersion.
    raw_samples:
        The parsed per-sample view dicts (for inspection / debugging).
    """

    views: dict[str, float]
    confidence: float
    per_asset_std: dict[str, float] = field(default_factory=dict)
    raw_samples: list[dict] = field(default_factory=list)


def calibrate_confidence(views: dict[str, float], per_asset_std: dict[str, float]) -> float:
    """Map sampling dispersion to a Black-Litterman view confidence in ``(0, 1]``.

    Confidence is high when the samples agree tightly *relative to the typical
    view magnitude* — normalizing by magnitude (not by cross-asset spread) keeps
    it well-behaved for single-asset universes and when all views agree, where a
    cross-asset-spread denominator would collapse to the floor.
    """
    if not per_asset_std:
        return 0.5  # single sample or no dispersion signal
    avg_std = float(np.mean(list(per_asset_std.values())))
    ref_scale = float(np.mean([abs(v) for v in views.values()])) + 1e-4
    return float(np.clip(1.0 / (1.0 + avg_std / ref_scale), 0.05, 1.0))


def _format_stats(returns) -> tuple[str, list[str]]:
    """Build the per-asset statistics block shown to the LLM."""
    mat, names = as_matrix(returns)
    arr = np.asarray(mat)
    ppy = 252
    lines = []
    for j, name in enumerate(names):
        col = arr[:, j]
        ann_ret = float(np.mean(col) * ppy)
        ann_vol = float(np.std(col) * np.sqrt(ppy))
        cum = float(np.prod(1.0 + col) - 1.0)
        lines.append(
            f"- {name}: annualized_return={ann_ret:+.1%}, annualized_vol={ann_vol:.1%}, "
            f"period_cumulative={cum:+.1%}"
        )
    return "\n".join(lines), names


def build_prompt(returns, extra_context: str | None = None) -> tuple[str, list[str]]:
    """Construct the view-elicitation prompt and return ``(prompt, asset_names)``."""
    stats, names = _format_stats(returns)
    ctx = f"\n\nAdditional context:\n{extra_context}" if extra_context else ""
    prompt = (
        "Recent statistics for the investable universe:\n"
        f"{stats}{ctx}\n\n"
        "Return a JSON object mapping each ticker to your expected next-period "
        "return (decimal). Example: {"
        f'"{names[0]}": 0.01' + "}."
    )
    return prompt, names


def llm_views(
    returns,
    client: LLMClient,
    *,
    samples: int = 5,
    temperature: float = 0.8,
    extra_context: str | None = None,
    clip: tuple[float, float] = (-0.2, 0.2),
) -> ViewSet:
    """Elicit per-asset return views from a local LLM and calibrate confidence.

    Parameters
    ----------
    client:
        A local LLM client (``OllamaClient`` in production, ``FakeLLM`` in tests).
    samples:
        Number of independent samples ``k``. More samples give a better estimate
        of both the mean view and its uncertainty.
    clip:
        Per-period return views are clipped to this range to reject outliers /
        hallucinated magnitudes.

    Returns
    -------
    ViewSet
        Mean views, a variance-derived confidence, and diagnostics.
    """
    prompt, names = build_prompt(returns, extra_context)
    responses = client.sample(prompt, samples, system=_SYSTEM, temperature=temperature)

    lo, hi = clip
    parsed: list[dict[str, float]] = []
    stacked: dict[str, list[float]] = {n: [] for n in names}
    for resp in responses:
        obj = parse_json(resp)
        clean: dict[str, float] = {}
        for name in names:
            if name in obj:
                try:
                    val = float(np.clip(float(obj[name]), lo, hi))
                except (TypeError, ValueError):
                    continue
                clean[name] = val
                stacked[name].append(val)
        parsed.append(clean)

    views = {n: float(np.mean(v)) for n, v in stacked.items() if v}
    per_asset_std = {n: float(np.std(v)) for n, v in stacked.items() if len(v) > 1}
    confidence = calibrate_confidence(views, per_asset_std)

    return ViewSet(
        views=views,
        confidence=confidence,
        per_asset_std=per_asset_std,
        raw_samples=parsed,
    )
