"""Tests for options overlays and (smoke) tests for the plotting layer."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend for tests

import numpy as np  # noqa: E402

import jaxfolio as jf  # noqa: E402
from jaxfolio.options.overlay import collar_overlay, covered_call_overlay  # noqa: E402
from jaxfolio.options.strategies import straddle  # noqa: E402


def test_collar_overlay_book(returns):
    res = jf.maximum_sharpe(returns)
    spots = {a: 100.0 for a in res.assets}
    book = collar_overlay(res, spots, top_n=4)
    assert len(book.strategies) <= 4
    greeks = book.net_greeks()
    assert set(greeks) == {"delta", "gamma", "vega", "theta", "rho"}


def test_covered_call_overlay_payoff_curve(returns):
    res = jf.risk_parity(returns)
    spots = {a: 100.0 for a in res.assets}
    book = covered_call_overlay(res, spots, top_n=3)
    grid = np.linspace(0.8, 1.2, 50)
    pnl = book.payoff_curve(grid)
    assert pnl.shape == (50,)
    assert np.isfinite(pnl).all()


def test_plot_weights_smoke(returns):
    from jaxfolio import viz

    res = jf.minimum_variance(returns)
    fig = viz.plot_weights(res)
    assert fig is not None


def test_plot_payoff_smoke():
    from jaxfolio import viz

    strat = straddle(100.0, 100.0, expiry=0.25, vol=0.2)
    fig = viz.plot_payoff(strat, spot=100.0)
    assert fig is not None


def test_plot_dendrogram_smoke(returns):
    from jaxfolio import viz

    res = jf.hierarchical_risk_parity(returns)
    fig = viz.plot_dendrogram(res)
    assert fig is not None
