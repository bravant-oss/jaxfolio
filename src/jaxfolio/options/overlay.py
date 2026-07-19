"""Options overlays on top of an optimized equity portfolio.

Bridges the optimizer output (:class:`PortfolioResult`) and the options layer:
apply a covered-call or collar overlay to the largest holdings, aggregate the net
Greeks of the resulting book, and produce a combined at-expiry payoff for the
overlaid position. This is what makes the two halves of the library compose.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jaxfolio.options.strategies import (
    OptionStrategy,
    collar,
    covered_call,
)
from jaxfolio.types import PortfolioResult


@dataclass
class OverlayBook:
    """A basket of per-asset option strategies scaled by portfolio weights.

    Attributes
    ----------
    strategies:
        Mapping ``asset -> OptionStrategy`` for the overlaid holdings.
    weights:
        Portfolio weight of each overlaid asset (used to scale payoffs/Greeks).
    spots:
        Reference spot price per asset.
    """

    strategies: dict[str, OptionStrategy]
    weights: dict[str, float]
    spots: dict[str, float]

    def net_greeks(self, vol: float = 0.25, rate: float = 0.0) -> dict[str, float]:
        """Weighted net Greeks across the whole overlay book."""
        agg = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
        for asset, strat in self.strategies.items():
            w = self.weights[asset]
            g = strat.greeks(self.spots[asset], vol, rate)
            for k in agg:
                agg[k] += w * g[k]
        return agg

    def payoff_curve(self, shock_grid: np.ndarray) -> np.ndarray:
        """Weighted portfolio P&L across a grid of *relative* spot shocks.

        ``shock_grid`` is a multiplier applied to every asset's spot (e.g.
        ``np.linspace(0.7, 1.3, 200)`` sweeps -30% to +30%). Returns the
        weight-scaled aggregate P&L of the overlaid holdings.
        """
        total = np.zeros_like(shock_grid, dtype=float)
        for asset, strat in self.strategies.items():
            w = self.weights[asset]
            spots = self.spots[asset] * shock_grid
            total = total + w * strat.payoff_at_expiry(spots)
        return total


def covered_call_overlay(
    result: PortfolioResult,
    spots: dict[str, float],
    *,
    top_n: int = 5,
    moneyness: float = 1.05,
    expiry: float = 0.25,
    vol: float = 0.25,
    rate: float = 0.0,
) -> OverlayBook:
    """Write covered calls on the ``top_n`` largest holdings.

    ``moneyness`` sets the call strike relative to spot (``1.05`` = 5% OTM).
    Assets without a provided spot price are skipped.
    """
    holdings = result.top(top_n)
    strategies: dict[str, OptionStrategy] = {}
    weights: dict[str, float] = {}
    used_spots: dict[str, float] = {}
    for asset, w in holdings.items():
        if asset not in spots or w <= 0:
            continue
        s = spots[asset]
        strategies[asset] = covered_call(s, s * moneyness, expiry=expiry, vol=vol, rate=rate)
        weights[asset] = float(w)
        used_spots[asset] = s
    return OverlayBook(strategies, weights, used_spots)


def collar_overlay(
    result: PortfolioResult,
    spots: dict[str, float],
    *,
    top_n: int = 5,
    put_moneyness: float = 0.95,
    call_moneyness: float = 1.05,
    expiry: float = 0.25,
    vol: float = 0.25,
    rate: float = 0.0,
) -> OverlayBook:
    """Wrap the ``top_n`` largest holdings in protective collars.

    Bounds each holding's outcome between ``put_moneyness`` and
    ``call_moneyness`` times spot — capping both tail loss and upside.
    """
    holdings = result.top(top_n)
    strategies: dict[str, OptionStrategy] = {}
    weights: dict[str, float] = {}
    used_spots: dict[str, float] = {}
    for asset, w in holdings.items():
        if asset not in spots or w <= 0:
            continue
        s = spots[asset]
        strategies[asset] = collar(
            s, s * put_moneyness, s * call_moneyness, expiry=expiry, vol=vol, rate=rate
        )
        weights[asset] = float(w)
        used_spots[asset] = s
    return OverlayBook(strategies, weights, used_spots)
