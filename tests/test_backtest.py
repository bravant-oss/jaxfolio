"""Tests for the backtester and metrics, plus an end-to-end integration test."""

from __future__ import annotations

import functools
import math

import numpy as np
import pandas as pd
import pytest

import jaxfolio as jf
from jaxfolio.backtest import metrics as M
from jaxfolio.backtest.engine import _accepts_holdings, backtest, compare, metrics_table


def test_metrics_on_known_series():
    # Constant positive return: no drawdown, infinite-ish Sharpe handled gracefully.
    r = np.full(252, 0.001)
    assert M.max_drawdown(r) == 0.0
    assert M.annualized_return(r) > 0
    assert M.hit_rate(r) == 1.0


def test_drawdown_is_nonpositive(returns):
    r = returns.iloc[:, 0]
    dd = M.drawdown_series(r)
    assert (dd <= 1e-12).all()


def test_cumulative_returns_start_positive(returns):
    r = returns.iloc[:, 0]
    cum = M.cumulative_returns(r)
    assert cum[0] > 0


def test_summary_keys(returns):
    s = M.summary(returns.iloc[:, 0])
    for k in ("sharpe", "sortino", "max_drawdown", "calmar", "var_95", "cvar_95"):
        assert k in s


def test_backtest_runs_end_to_end(returns):
    res = backtest(returns, jf.equal_weight, lookback=100, rebalance_every=20)
    assert len(res.returns) > 0
    assert res.weights.shape[1] == len(returns.columns)
    assert "sharpe" in res.metrics
    # Equal-weight weights should each be ~1/N right after a rebalance.
    assert np.isclose(res.weights.iloc[0].sum(), 1.0, atol=1e-6)


def test_compare_and_metrics_table(returns):
    results = compare(
        returns,
        {"1/N": jf.equal_weight, "MinVar": jf.minimum_variance},
        lookback=100,
        rebalance_every=25,
    )
    table = metrics_table(results)
    assert set(table.index) == {"1/N", "MinVar"}
    assert "sharpe" in table.columns


def test_transaction_costs_reduce_returns(returns):
    no_cost = backtest(
        returns, jf.maximum_sharpe, lookback=100, rebalance_every=10, transaction_cost=0.0
    )
    with_cost = backtest(
        returns, jf.maximum_sharpe, lookback=100, rebalance_every=10, transaction_cost=0.01
    )
    # Costs can only lower cumulative performance.
    assert with_cost.equity_curve.iloc[-1] <= no_cost.equity_curve.iloc[-1] + 1e-9


# --------------------------------------------------------------------------- #
# Holdings injection and trajectory execution
# --------------------------------------------------------------------------- #
def test_new_flags_are_inert_for_legacy_optimizers(returns):
    """The two new keywords must be exact no-ops for trajectory-less optimizers.

    Every existing strategy returns no path and declares no ``w_prev``, so
    enabling the new machinery must not perturb a single number. This is the
    regression guard for the loop rewrite.
    """
    kw = dict(lookback=100, rebalance_every=20, transaction_cost=0.002)
    base = backtest(returns, jf.maximum_sharpe, **kw)
    explicit = backtest(
        returns, jf.maximum_sharpe, holdings_aware=False, follow_trajectory=False, **kw
    )
    default = backtest(returns, jf.maximum_sharpe, **kw)
    for other in (explicit, default):
        pd.testing.assert_series_equal(other.returns, base.returns)
        pd.testing.assert_frame_equal(other.weights, base.weights)
        pd.testing.assert_series_equal(other.turnover, base.turnover)


def test_legacy_turnover_is_recorded_only_at_rebalances(returns):
    """For a path-less optimizer, turnover appears on exactly the rebalance dates."""
    lookback, every = 100, 20
    res = backtest(returns, jf.equal_weight, lookback=lookback, rebalance_every=every)
    expected = [
        returns.index[t] for t in range(lookback, len(returns)) if (t - lookback) % every == 0
    ]
    assert list(res.turnover.index) == expected
    assert len(res.turnover) == math.ceil((len(returns) - lookback) / every)


def test_holdings_are_passed_to_optimizer(returns):
    """A ``w_prev``-declaring optimizer receives the live book at each rebalance."""
    seen: list[np.ndarray] = []

    def spy(window, *, w_prev=None):
        seen.append(np.array(w_prev, dtype=float))
        return jf.equal_weight(window)

    backtest(returns, spy, lookback=100, rebalance_every=20)
    assert len(seen) >= 2
    # Documented contract: the book is flat (all-zero) before the first rebalance.
    assert np.allclose(seen[0], 0.0), f"first call saw {seen[0]}"
    assert np.isclose(seen[1].sum(), 1.0, atol=1e-9), f"second call saw sum={seen[1].sum()}"


def test_lambda_wrapper_falls_back_to_legacy(returns):
    """A lambda hides the parameter, so detection fails *closed* — the safe way."""
    calls: list[bool] = []

    def spy(window, *, w_prev=None):
        calls.append(w_prev is not None)
        return jf.equal_weight(window)

    res = backtest(returns, lambda w: spy(w), lookback=100, rebalance_every=20)
    assert calls and not any(calls), "a lambda-wrapped optimizer must not receive w_prev"
    baseline = backtest(returns, jf.equal_weight, lookback=100, rebalance_every=20)
    pd.testing.assert_series_equal(res.returns, baseline.returns)


def test_partial_preserves_holdings_detection(returns):
    """``functools.partial`` keeps unbound parameters visible, so detection works."""
    seen: list[object] = []

    def spy(window, *, scale=1.0, w_prev=None):
        seen.append(w_prev)
        return jf.equal_weight(window)

    backtest(returns, functools.partial(spy, scale=2.0), lookback=100, rebalance_every=20)
    assert seen and all(x is not None for x in seen)


def test_holdings_aware_override(returns):
    """``True`` forces injection; on an incompatible optimizer it fails loudly."""
    seen: list[object] = []

    def kwargs_only(window, **kwargs):  # **kwargs deliberately does not auto-detect
        seen.append(kwargs.get("w_prev"))
        return jf.equal_weight(window)

    backtest(returns, kwargs_only, lookback=100, rebalance_every=20)
    assert seen and all(x is None for x in seen), "**kwargs must not auto-detect"
    seen.clear()
    backtest(returns, kwargs_only, lookback=100, rebalance_every=20, holdings_aware=True)
    assert seen and all(x is not None for x in seen), "the override must inject"

    with pytest.raises(TypeError):
        backtest(returns, jf.equal_weight, lookback=100, rebalance_every=20, holdings_aware=True)


def test_accepts_holdings_on_uninspectable_callable():
    """An uninspectable callable falls back to legacy behavior rather than raising."""

    class NoSignature:
        __call__ = None  # inspect.signature raises on this

    assert _accepts_holdings(NoSignature()) is False
    assert _accepts_holdings(jf.equal_weight) is False


def _path_optimizer(rows):
    """Return an optimizer emitting a fixed weight path, for execution tests."""

    def opt(window):
        assets = list(window.columns)
        path = np.zeros((len(rows), len(assets)))
        for i, row in enumerate(rows):
            path[i, : len(row)] = row
        return jf.PortfolioResult(
            weights=path[0], assets=assets, method="fake-path", trajectory=path
        )

    return opt


def test_trajectory_is_executed_between_rebalances(returns):
    """Each planned row is traded into on its own period, paying cost each time."""
    n = len(returns.columns)
    rows = [[1.0], [0.0, 1.0], [0.0, 0.0, 1.0]]
    res = backtest(
        returns, _path_optimizer(rows), lookback=100, rebalance_every=20, transaction_cost=0.001
    )
    dates = returns.index[100:103]
    for k, date in enumerate(dates):
        expected = np.zeros(n)
        expected[k] = 1.0
        assert np.allclose(res.weights.loc[date].to_numpy(), expected, atol=1e-12), (
            f"period {k} held {res.weights.loc[date].to_numpy()}"
        )
        assert date in res.turnover.index, f"no turnover recorded on path step {k}"


def test_follow_trajectory_false_trades_only_the_first_row(returns):
    """The comparison arm: take row 0, then drift, discarding the rest of the path."""
    rows = [[1.0], [0.0, 1.0], [0.0, 0.0, 1.0]]
    res = backtest(
        returns,
        _path_optimizer(rows),
        lookback=100,
        rebalance_every=20,
        transaction_cost=0.001,
        follow_trajectory=False,
    )
    d1, d2 = returns.index[101], returns.index[102]
    assert not np.allclose(res.weights.loc[d1].to_numpy()[1], 1.0)
    assert d1 not in res.turnover.index and d2 not in res.turnover.index


def test_trajectory_truncated_by_next_rebalance(returns):
    """A path longer than the rebalance cadence is cut off by the next solve."""
    rows = [[1.0]] + [[0.0, 1.0]] * 8
    res = backtest(
        returns, _path_optimizer(rows), lookback=100, rebalance_every=3, transaction_cost=0.001
    )
    for t in range(100, 110, 3):  # every rebalance date resets to row 0
        date = returns.index[t]
        assert np.isclose(res.weights.loc[date].to_numpy()[0], 1.0, atol=1e-12), (
            f"rebalance on {date} did not reset to row 0"
        )


def test_total_cost_metric(returns):
    res = backtest(
        returns, jf.maximum_sharpe, lookback=100, rebalance_every=20, transaction_cost=0.002
    )
    assert np.isclose(res.metrics["total_cost"], 0.002 * res.turnover.sum(), atol=1e-12)
    free = backtest(
        returns, jf.maximum_sharpe, lookback=100, rebalance_every=20, transaction_cost=0.0
    )
    assert free.metrics["total_cost"] == 0.0


def test_multiperiod_end_to_end_lowers_cost_drag(returns):
    """The integration claim: pricing costs inside the optimizer cuts cost drag.

    Asserted on ``total_cost`` rather than Sharpe on purpose — the cost reduction
    is what the optimizer actually optimizes, whereas net Sharpe on synthetic data
    is not guaranteed to improve and would make this a flaky test.
    """
    kw = dict(lookback=100, rebalance_every=20, transaction_cost=0.005)
    myopic = backtest(
        returns, functools.partial(jf.mean_variance, risk_aversion=3.0), name="myopic", **kw
    )
    path_aware = backtest(
        returns,
        functools.partial(
            jf.multi_period_mean_variance,
            horizon=5,
            risk_aversion=3.0,
            costs=jf.TradingCosts(spread_bps=50.0, impact_bps=500.0),
        ),
        name="multi-period",
        **kw,
    )
    assert path_aware.metrics["total_cost"] < myopic.metrics["total_cost"], (
        f"multi-period cost {path_aware.metrics['total_cost']:.5f} not below "
        f"myopic {myopic.metrics['total_cost']:.5f}"
    )
