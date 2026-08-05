"""Tests for options overlays and (smoke) tests for the plotting layer."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend for tests

import numpy as np  # noqa: E402
import pytest  # noqa: E402

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


# --------------------------------------------------------------------------- #
# Multi-period plots
# --------------------------------------------------------------------------- #
def _mp_result(returns, **costs):
    n = len(jf.asset_columns(returns))
    w_prev = np.eye(n)[0]
    return jf.multi_period_mean_variance(
        returns,
        horizon=5,
        w_prev=w_prev,
        risk_aversion=3.0,
        costs=jf.TradingCosts(**costs) if costs else None,
    )


def test_plot_weight_path_smoke(small_returns):
    from jaxfolio import viz

    fig = viz.plot_weight_path(_mp_result(small_returns, spread_bps=10.0, impact_bps=200.0))
    ax = fig.axes[0]
    # Period 0 must be included, so the plan reads against the book it starts from.
    assert ax.get_xlim() == (0.0, 5.0)
    assert "Planned Weight Path" in ax.get_title()


def test_plot_weight_path_folds_to_top_n(small_returns):
    from jaxfolio import viz

    res = _mp_result(small_returns, spread_bps=10.0, impact_bps=200.0)
    fig = viz.plot_weight_path(res, top_n=2)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "Other" in labels, labels
    assert len(labels) == 3, labels


def test_plot_weight_path_handles_shorting(returns):
    """A signed path cannot be stacked, so it must fall back to lines."""
    from jaxfolio import viz

    n = len(jf.asset_columns(returns))
    res = jf.multi_period_mean_variance(
        returns,
        horizon=4,
        w_prev=np.eye(n)[0],
        risk_aversion=3.0,
        costs=jf.TradingCosts(spread_bps=10.0, impact_bps=200.0),
        config=jf.OptimizerConfig(long_only=False, tol=1e-6),
    )
    assert res.trajectory.min() < -1e-3, "fixture should actually short something"
    ax = viz.plot_weight_path(res).axes[0]
    assert "long / short" in ax.get_ylabel()
    assert len(ax.lines) >= n  # one line per asset plus the zero baseline


def test_plot_turnover_schedule_omits_tautological_reference(small_returns):
    """Without an explicit reference there must be no one-shot line.

    For a monotone path, total turnover equals ``|w_T - w_prev|`` *identically*, so
    drawing the path's own terminal as a "one-shot" reference would look like a
    comparison while measuring nothing. It is opt-in via ``reference_weights``.
    """
    from jaxfolio import viz

    res = _mp_result(small_returns, spread_bps=10.0, impact_bps=200.0)
    labels = [
        t.get_text() for t in viz.plot_turnover_schedule(res).axes[0].get_legend().get_texts()
    ]
    assert not any("one-shot" in lbl.lower() for lbl in labels), labels

    myopic = jf.mean_variance(small_returns, risk_aversion=3.0).weights
    with_ref = viz.plot_turnover_schedule(res, reference_weights=myopic)
    labels = [t.get_text() for t in with_ref.axes[0].get_legend().get_texts()]
    assert any("one-shot" in lbl.lower() for lbl in labels), labels


def test_plot_path_convergence_smoke(small_returns):
    from jaxfolio import viz

    fig = viz.plot_path_convergence(_mp_result(small_returns, spread_bps=10.0, impact_bps=200.0))
    assert "Convergence" in fig.axes[0].get_title()


def test_plot_cost_comparison_smoke(small_returns):
    from jaxfolio import viz

    regimes = {
        "free": _mp_result(small_returns),
        "spread": _mp_result(small_returns, spread_bps=50.0),
        "impact": _mp_result(small_returns, spread_bps=10.0, impact_bps=500.0),
    }
    fig = viz.plot_cost_comparison(regimes)
    assert len(fig.axes) == 2  # schedule + total-traded bars


def test_plot_cost_comparison_rejects_empty():
    from jaxfolio import viz

    with pytest.raises(ValueError, match="at least one"):
        viz.plot_cost_comparison({})


def test_multiperiod_dashboard_smoke(small_returns):
    from jaxfolio import viz

    res = _mp_result(small_returns, spread_bps=10.0, impact_bps=200.0)
    assert viz.multiperiod_dashboard(res) is not None
    # With both optional panels, and an external reference for the turnover KPI.
    myopic = jf.mean_variance(small_returns, risk_aversion=3.0).weights
    fig = viz.multiperiod_dashboard(res, comparison={"impact": res}, reference_weights=myopic)
    assert fig is not None


def test_multiperiod_plots_reject_single_period_results(returns):
    """A single-period result has no path; say so rather than failing obscurely."""
    from jaxfolio import viz

    single = jf.minimum_variance(returns)
    for plot in (viz.plot_weight_path, viz.plot_turnover_schedule, viz.plot_path_convergence):
        with pytest.raises(ValueError, match="no weight path"):
            plot(single)
