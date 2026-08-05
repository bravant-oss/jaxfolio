"""A vectorized walk-forward backtester.

The backtester is optimizer-agnostic: it takes any callable
``optimizer(returns_window) -> PortfolioResult`` and rebalances on a fixed
schedule, applying linear transaction costs on turnover. It returns a
:class:`BacktestResult` holding the strategy return series, weight history, and a
metrics summary — ready for the plotting utilities.

Two optional capabilities extend that contract for cost-aware strategies, both
inert for optimizers that do not use them:

* **Holdings injection.** An optimizer that declares a ``w_prev`` keyword is
  handed the *current* portfolio at each rebalance, so it can price the trade it
  is about to recommend instead of optimizing in a vacuum.
* **Trajectory execution.** An optimizer that returns a
  :attr:`~jaxfolio.types.PortfolioResult.trajectory` has its planned weight path
  executed step by step over the following periods, paying cost on each step,
  rather than having every row but the first discarded.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from jaxfolio.backtest import metrics as M
from jaxfolio.data.returns import asset_columns, date_column
from jaxfolio.types import PortfolioResult

Optimizer = Callable[[pl.DataFrame], PortfolioResult]
"""The minimum optimizer contract: a trailing return window in, a result out.

A callable may additionally declare a keyword-only ``w_prev`` parameter, which
:func:`backtest` fills with the currently held weights (see
:func:`_accepts_holdings`). Note that vector **does not necessarily sum to one**
— it is the all-zero flat book before the first rebalance, and drifts between
rebalances thereafter — so a holdings-aware optimizer must not assume it is a
valid portfolio.
"""


@dataclass
class BacktestResult:
    """Output of a backtest run."""

    name: str
    returns: pl.DataFrame
    weights: pl.DataFrame
    turnover: pl.DataFrame
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def equity_curve(self) -> pl.DataFrame:
        date_col = date_column(self.returns)
        values = self.returns.get_column(self.name)
        data = {"equity": M.cumulative_returns(values)}
        if date_col:
            data = {date_col: self.returns.get_column(date_col), **data}
        return pl.DataFrame(data)

    @property
    def drawdown(self) -> pl.DataFrame:
        date_col = date_column(self.returns)
        values = self.returns.get_column(self.name)
        data = {"drawdown": M.drawdown_series(values)}
        if date_col:
            data = {date_col: self.returns.get_column(date_col), **data}
        return pl.DataFrame(data)


def backtest(
    returns: pl.DataFrame,
    optimizer: Optimizer,
    *,
    name: str | None = None,
    lookback: int = 252,
    rebalance_every: int = 21,
    transaction_cost: float = 0.0010,
    periods_per_year: int = 252,
    risk_free: float = 0.0,
    holdings_aware: bool | None = None,
    follow_trajectory: bool = True,
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
    holdings_aware:
        Whether to pass the current holdings to the optimizer as ``w_prev``.
        ``None`` (default) auto-detects it from the optimizer's signature.
        ``True`` forces injection — use this for a ``lambda`` wrapper that hides
        the parameter, and note it raises ``TypeError`` if the optimizer does not
        accept it. ``False`` forces the plain single-argument call.
    follow_trajectory:
        When the optimizer returns a :attr:`PortfolioResult.trajectory`, execute
        that planned path step by step over the periods after the rebalance,
        paying transaction cost on each step. ``False`` trades to the first row
        only and then drifts — the useful comparison arm for measuring what
        path-execution actually buys. Inert for optimizers returning no path.

    Returns
    -------
    BacktestResult
        Net-of-cost strategy returns, weight history, turnover, and metrics.

    Notes
    -----
    ``turnover`` is indexed by *trading* period. For an optimizer without a
    trajectory those are exactly the rebalance dates (unchanged behavior); when a
    path is being executed, the intermediate steps appear too.

    Realized turnover here will not equal a multi-period optimizer's planned
    ``metadata["turnover_path"]``: the plan assumes no drift, whereas the engine
    lets weights drift with returns and then trades from the drifted book.
    """
    assets = asset_columns(returns)
    n = len(assets)
    date_col = date_column(returns)
    dates = returns.get_column(date_col).to_list() if date_col else list(range(returns.height))
    ret_vals = returns.select(assets).to_numpy()

    weights = np.zeros(n)
    weight_hist: dict[object, np.ndarray] = {}
    turnover_hist: dict[object, float] = {}
    strat_returns = np.zeros(len(returns))
    total_cost = 0.0

    pass_holdings = _accepts_holdings(optimizer) if holdings_aware is None else holdings_aware
    traj: np.ndarray | None = None  # planned path still awaiting execution
    step = 0  # next row of ``traj`` to trade into

    for t in range(lookback, len(returns)):
        target: np.ndarray | None = None

        # Rebalance at the cadence; otherwise execute a planned step, else drift.
        if (t - lookback) % rebalance_every == 0:
            window = returns.slice(t - lookback, lookback)
            result = (
                optimizer(window, w_prev=weights.copy()) if pass_holdings else optimizer(window)
            )
            # Row 0 of any trajectory *is* ``result.weights`` by contract, so the
            # rebalance-period trade is identical with or without a path.
            target = _align_weights(result, assets)
            traj = _align_trajectory(result, assets) if follow_trajectory else None
            step = 1
            if traj is not None and traj.shape[0] <= 1:
                traj = None  # nothing left to execute
        elif traj is not None and step < traj.shape[0]:
            # Trade from the *drifted* book toward the planned row: the drift
            # happened over the period, the trade happens now, so the turnover
            # actually paid is |row_k - drifted|, not |row_k - row_{k-1}|.
            target = traj[step]
            step += 1
            if step >= traj.shape[0]:
                traj = None  # path exhausted -> drift until the next rebalance

        if target is not None:
            turn = float(np.abs(target - weights).sum())
            cost = transaction_cost * turn
            turnover_hist[dates[t]] = turn
            weights = target
        else:
            cost = 0.0
        total_cost += cost

        period_ret = ret_vals[t]
        gross = float(np.dot(weights, period_ret))
        strat_returns[t] = gross - cost
        weight_hist[dates[t]] = weights.copy()

        # Drift weights with realized returns (buy-and-hold between rebalances).
        grown = weights * (1.0 + period_ret)
        total = grown.sum()
        if total > 0:
            weights = grown / total

    result_name = name or "strategy"
    result_dates = dates[lookback:]
    ret_data = {result_name: strat_returns[lookback:]}
    if date_col:
        ret_data = {date_col: result_dates, **ret_data}
    ret_series = pl.DataFrame(ret_data)
    hist_dates = list(weight_hist)
    hist_values = np.vstack(list(weight_hist.values())) if weight_hist else np.empty((0, n))
    weight_data = dict(zip(assets, hist_values.T, strict=True))
    if date_col:
        weight_data = {date_col: hist_dates, **weight_data}
    w_df = pl.DataFrame(weight_data)
    turn_dates = list(turnover_hist)
    turn_data = {"turnover": list(turnover_hist.values())}
    if date_col:
        turn_data = {date_col: turn_dates, **turn_data}
    to_series = pl.DataFrame(turn_data)

    summary = M.summary(ret_series, risk_free=risk_free, periods_per_year=periods_per_year)
    summary["avg_turnover"] = (
        float(to_series.get_column("turnover").mean()) if len(to_series) else 0.0
    )
    summary["total_cost"] = float(total_cost)
    return BacktestResult(
        name=result_name,
        returns=ret_series,
        weights=w_df,
        turnover=to_series,
        metrics=summary,
    )


def compare(
    returns: pl.DataFrame,
    optimizers: dict[str, Optimizer],
    **kwargs,
) -> dict[str, BacktestResult]:
    """Backtest several optimizers on the same data and return a name->result map."""
    return {name: backtest(returns, opt, name=name, **kwargs) for name, opt in optimizers.items()}


def metrics_table(results: dict[str, BacktestResult]) -> pl.DataFrame:
    """Assemble a tidy metrics comparison table across backtest results."""
    return pl.DataFrame([{"strategy": name, **res.metrics} for name, res in results.items()])


def _align_weights(result: PortfolioResult, assets: list[str]) -> np.ndarray:
    """Reorder/pad a result's weights to the backtest's asset ordering."""
    lookup = dict(zip(result.assets, result.weights, strict=True))
    return np.array([lookup.get(a, 0.0) for a in assets], dtype=float)


def _align_trajectory(result: PortfolioResult, assets: list[str]) -> np.ndarray | None:
    """Reorder/pad a result's weight *path* to the backtest's asset ordering.

    Returns ``None`` for any result without a trajectory, which is every
    single-period optimizer — hence read via ``getattr`` so a duck-typed result
    object works too.
    """
    traj = getattr(result, "trajectory", None)
    if traj is None:
        return None
    traj = np.asarray(traj, dtype=float)
    pos = {a: i for i, a in enumerate(result.assets)}
    out = np.zeros((traj.shape[0], len(assets)), dtype=float)
    for j, a in enumerate(assets):
        i = pos.get(a)
        if i is not None:
            out[:, j] = traj[:, i]
    return out


def _accepts_holdings(optimizer: Callable) -> bool:
    """True if ``optimizer`` declares a ``w_prev`` keyword the engine should fill.

    Signature introspection rather than an opt-in marker attribute, because
    ``functools.partial`` does not proxy attributes but *does* preserve unbound
    parameters — so introspection handles the wrapper case a marker would miss,
    while staying visible in the rendered docs.

    Two deliberate negatives. A bare ``**kwargs`` does **not** count: a wrapper
    that swallows unknown keywords would drop ``w_prev`` silently, which is worse
    than never passing it. And an uninspectable callable (some C builtins) falls
    back to ``False``, i.e. to legacy behavior — failing closed.
    """
    try:
        params = inspect.signature(optimizer).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return False
    p = params.get("w_prev")
    return p is not None and p.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )
