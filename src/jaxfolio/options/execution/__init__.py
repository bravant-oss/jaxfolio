"""Options execution framework (research simulation).

Simulate trading and rolling option positions against a *modeled* volatility
surface / Black-Scholes mid, with transaction costs and full mark-to-market
accounting. This is a backtesting / research tool — **not** a live broker,
order-management system, or connection to any exchange or venue. Simulated fills,
costs, and P&L will differ from live trading. See ``DISCLAIMER.md``.
"""

from jaxfolio.options.execution.book import (
    BUY,
    SELL,
    ExecutionSimulator,
    Fill,
    Instrument,
    OptionBook,
    Order,
    Position,
)
from jaxfolio.options.execution.costs import CostModel
from jaxfolio.options.execution.rolling import (
    OverlaySimResult,
    simulate_covered_call_roll,
)

__all__ = [
    "CostModel",
    "Instrument",
    "Order",
    "Fill",
    "Position",
    "OptionBook",
    "ExecutionSimulator",
    "BUY",
    "SELL",
    "OverlaySimResult",
    "simulate_covered_call_roll",
]
