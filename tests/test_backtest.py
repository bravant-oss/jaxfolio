"""Tests for the backtester and metrics, plus an end-to-end integration test."""

from __future__ import annotations

import numpy as np

import jaxfolio as jf
from jaxfolio.backtest import metrics as M
from jaxfolio.backtest.engine import backtest, compare, metrics_table


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
