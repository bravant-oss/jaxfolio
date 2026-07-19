"""Performance and risk metrics for return series.

All functions accept a 1-D array-like of periodic returns and return plain
floats. Annualization uses ``periods_per_year`` (252 for daily by default).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_PPY = 252


def _to_array(returns) -> np.ndarray:
    if isinstance(returns, pd.Series | pd.DataFrame):
        returns = returns.to_numpy()
    return np.asarray(returns, dtype=float).reshape(-1)


def annualized_return(returns, periods_per_year: int = _PPY) -> float:
    """Geometric annualized return (CAGR) of a periodic return series."""
    r = _to_array(returns)
    if r.size == 0:
        return 0.0
    growth = np.prod(1.0 + r)
    years = r.size / periods_per_year
    if growth <= 0:
        return -1.0
    return float(growth ** (1.0 / years) - 1.0)


def annualized_volatility(returns, periods_per_year: int = _PPY) -> float:
    """Annualized standard deviation of returns."""
    r = _to_array(returns)
    return float(np.std(r, ddof=1) * np.sqrt(periods_per_year)) if r.size > 1 else 0.0


def sharpe_ratio(returns, risk_free: float = 0.0, periods_per_year: int = _PPY) -> float:
    """Annualized Sharpe ratio (``risk_free`` is a per-period rate)."""
    r = _to_array(returns) - risk_free
    sd = np.std(r, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(r) / sd * np.sqrt(periods_per_year))


def sortino_ratio(returns, risk_free: float = 0.0, periods_per_year: int = _PPY) -> float:
    """Annualized Sortino ratio (downside-deviation denominator).

    With no downside (zero denominator), returns ``+inf`` for positive mean
    excess return, ``-inf`` for negative, and ``0`` for a flat series — so a
    downside-free strategy ranks above one with drawdowns rather than tying at 0.
    """
    r = _to_array(returns) - risk_free
    downside = r[r < 0]
    dd = np.sqrt(np.mean(downside**2)) if downside.size else 0.0
    mean = float(np.mean(r)) if r.size else 0.0
    if dd == 0:
        return float(np.sign(mean) * np.inf) if mean != 0 else 0.0
    return float(mean / dd * np.sqrt(periods_per_year))


def cumulative_returns(returns) -> np.ndarray:
    """Cumulative wealth path starting from 1.0 (equity curve)."""
    r = _to_array(returns)
    return np.cumprod(1.0 + r)


def drawdown_series(returns) -> np.ndarray:
    """Drawdown at each point: ``equity / running_peak - 1`` (<= 0)."""
    equity = cumulative_returns(returns)
    peak = np.maximum.accumulate(equity)
    return equity / peak - 1.0


def max_drawdown(returns) -> float:
    """Maximum peak-to-trough drawdown (a negative number)."""
    dd = drawdown_series(returns)
    return float(dd.min()) if dd.size else 0.0


def calmar_ratio(returns, periods_per_year: int = _PPY) -> float:
    """Annualized return divided by the absolute max drawdown.

    With no drawdown (zero denominator), returns ``+inf`` for a positive
    annualized return, ``-inf`` for negative, and ``0`` for flat — so a
    drawdown-free strategy is not mis-ranked as the worst performer.
    """
    ann = annualized_return(returns, periods_per_year)
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return float(np.sign(ann) * np.inf) if ann != 0 else 0.0
    return float(ann / mdd)


def value_at_risk(returns, alpha: float = 0.95) -> float:
    """Historical Value-at-Risk at confidence ``alpha`` (a positive loss)."""
    r = _to_array(returns)
    if r.size == 0:
        return 0.0
    return float(-np.quantile(r, 1.0 - alpha))


def conditional_value_at_risk(returns, alpha: float = 0.95) -> float:
    """Historical CVaR / expected shortfall at confidence ``alpha``."""
    r = _to_array(returns)
    if r.size == 0:
        return 0.0
    var = -value_at_risk(r, alpha)
    tail = r[r <= var]
    return float(-tail.mean()) if tail.size else float(-var)


def hit_rate(returns) -> float:
    """Fraction of periods with a positive return."""
    r = _to_array(returns)
    return float(np.mean(r > 0)) if r.size else 0.0


def summary(returns, risk_free: float = 0.0, periods_per_year: int = _PPY) -> dict[str, float]:
    """Bundle the headline metrics into a single dict for reporting/plots."""
    return {
        "annual_return": annualized_return(returns, periods_per_year),
        "annual_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, risk_free, periods_per_year),
        "sortino": sortino_ratio(returns, risk_free, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns, periods_per_year),
        "var_95": value_at_risk(returns, 0.95),
        "cvar_95": conditional_value_at_risk(returns, 0.95),
        "hit_rate": hit_rate(returns),
    }
