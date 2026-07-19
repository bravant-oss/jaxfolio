"""Backtesting engine and performance metrics."""

from jaxfolio.backtest.engine import (
    BacktestResult,
    backtest,
    compare,
    metrics_table,
)
from jaxfolio.backtest.metrics import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    conditional_value_at_risk,
    cumulative_returns,
    drawdown_series,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    summary,
    value_at_risk,
)

__all__ = [
    "backtest",
    "compare",
    "metrics_table",
    "BacktestResult",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "value_at_risk",
    "conditional_value_at_risk",
    "cumulative_returns",
    "drawdown_series",
    "summary",
]
