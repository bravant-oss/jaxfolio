"""A vectorized walk-forward backtester.

The backtester is optimizer-agnostic: it takes any callable
``optimizer(returns_window) -> PortfolioResult`` and rebalances on a fixed
schedule, applying linear transaction costs on turnover. It returns a
:class:`BacktestResult` holding the strategy return series, weight history, and a
metrics summary — ready for the plotting utilities.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from jaxfolio.backtest import metrics as M
from jaxfolio.types import PortfolioResult

Optimizer = Callable[[pd.DataFrame], PortfolioResult]


@dataclass
class BacktestResult:
    """Output of a backtest run."""

    name: str
    returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def equity_curve(self) -> pd.Series:
        return pd.Series(M.cumulative_returns(self.returns), index=self.returns.index)

    @property
    def drawdown(self) -> pd.Series:
        return pd.Series(M.drawdown_series(self.returns), index=self.returns.index)


def backtest(
    returns: pd.DataFrame,
    optimizer: Optimizer,
    *,
    name: str | None = None,
    lookback: int = 252,
    rebalance_every: int = 21,
    transaction_cost: float = 0.0010,
    periods_per_year: int = 252,
    risk_free: float = 0.0,
) -> BacktestResult:
    """Walk-forward backtest of a single optimizer.

    Parameters
    ----------
    returns:
        Asset return panel (rows = periods, columns = assets).
    optimizer:
        Callable mapping a trailing return window to a :class:`PortfolioResult`.
    lookback:
        Number of trailing periods handed to the optimizer at each rebalance.
    rebalance_every:
        Rebalance frequency in periods (21 ≈ monthly for daily data).
    transaction_cost:
        Proportional cost per unit of turnover (one-way), e.g. ``0.0010`` = 10bps.

    Returns
    -------
    BacktestResult
        Net-of-cost strategy returns, weight history, turnover, and metrics.
    """
    assets = list(returns.columns)
    n = len(assets)
    dates = returns.index
    ret_vals = returns.to_numpy()

    weights = np.zeros(n)
    weight_hist: dict[pd.Timestamp, np.ndarray] = {}
    turnover_hist: dict[pd.Timestamp, float] = {}
    strat_returns = np.zeros(len(returns))

    for t in range(lookback, len(returns)):
        # Rebalance at the cadence; otherwise let weights drift with returns.
        if (t - lookback) % rebalance_every == 0:
            window = returns.iloc[t - lookback : t]
            result = optimizer(window)
            target = _align_weights(result, assets)
            cost = transaction_cost * np.abs(target - weights).sum()
            turnover_hist[dates[t]] = float(np.abs(target - weights).sum())
            weights = target
        else:
            cost = 0.0

        period_ret = ret_vals[t]
        gross = float(np.dot(weights, period_ret))
        strat_returns[t] = gross - cost
        weight_hist[dates[t]] = weights.copy()

        # Drift weights with realized returns (buy-and-hold between rebalances).
        grown = weights * (1.0 + period_ret)
        total = grown.sum()
        if total > 0:
            weights = grown / total

    ret_series = pd.Series(strat_returns, index=dates, name=name or "strategy")
    ret_series = ret_series.iloc[lookback:]
    w_df = pd.DataFrame.from_dict(weight_hist, orient="index", columns=assets)
    to_series = pd.Series(turnover_hist, name="turnover")

    summary = M.summary(ret_series, risk_free=risk_free, periods_per_year=periods_per_year)
    summary["avg_turnover"] = float(to_series.mean()) if len(to_series) else 0.0
    return BacktestResult(
        name=name or "strategy",
        returns=ret_series,
        weights=w_df,
        turnover=to_series,
        metrics=summary,
    )


def compare(
    returns: pd.DataFrame,
    optimizers: dict[str, Optimizer],
    **kwargs,
) -> dict[str, BacktestResult]:
    """Backtest several optimizers on the same data and return a name->result map."""
    return {name: backtest(returns, opt, name=name, **kwargs) for name, opt in optimizers.items()}


def metrics_table(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """Assemble a tidy metrics comparison table across backtest results."""
    rows = {name: res.metrics for name, res in results.items()}
    return pd.DataFrame(rows).T


def _align_weights(result: PortfolioResult, assets: list[str]) -> np.ndarray:
    """Reorder/pad a result's weights to the backtest's asset ordering."""
    lookup = dict(zip(result.assets, result.weights, strict=True))
    return np.array([lookup.get(a, 0.0) for a in assets], dtype=float)
