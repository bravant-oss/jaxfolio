"""Walk-forward simulation of a rolled option overlay.

Ties the execution layer to a price path: hold the underlying and repeatedly
write / roll a short call (a covered-call overlay), marking the combined book to
market every step and charging transaction costs at each roll. This is the
options analogue of the walk-forward equity backtester in
:mod:`jaxfolio.backtest` — a research simulation, not live trading.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jaxfolio.options.execution.book import ExecutionSimulator, Instrument, Order
from jaxfolio.options.execution.costs import CostModel


@dataclass
class OverlaySimResult:
    """Result of a rolled-overlay simulation.

    Attributes
    ----------
    equity:
        Total equity (stock + option book + cash) at each step.
    stock_only:
        Buy-and-hold stock equity at each step, for comparison.
    times:
        Elapsed time in years at each step.
    total_costs:
        Total commissions paid across all rolls.
    n_rolls:
        Number of times the call was written/rolled.
    """

    equity: np.ndarray
    stock_only: np.ndarray
    times: np.ndarray
    total_costs: float
    n_rolls: int

    @property
    def total_return(self) -> float:
        """Total return of the overlaid book over the path."""
        return float(self.equity[-1] / self.equity[0] - 1.0)

    @property
    def stock_return(self) -> float:
        """Total return of plain buy-and-hold stock over the path."""
        return float(self.stock_only[-1] / self.stock_only[0] - 1.0)


def simulate_covered_call_roll(
    prices,
    *,
    moneyness: float = 1.05,
    tenor: float = 0.25,
    roll_every: int = 21,
    vol: float = 0.25,
    shares: float = 1.0,
    cost_model: CostModel | None = None,
    rate: float = 0.0,
    div: float = 0.0,
    periods_per_year: int = 252,
) -> OverlaySimResult:
    """Simulate a rolling covered-call overlay over a price path.

    Holds ``shares`` of the underlying throughout and writes one short call per
    share at ``moneyness * spot`` with time-to-expiry ``tenor``. Every
    ``roll_every`` steps the existing call is bought back and a fresh one written
    at the current spot, incurring costs from ``cost_model``.

    Parameters
    ----------
    prices:
        1-D array (or pandas Series) of underlying prices along the path.
    moneyness:
        Call strike relative to spot (``1.05`` = 5% out-of-the-money).
    tenor:
        Time-to-expiry (years) of each written call.
    roll_every:
        Number of steps between rolls. Should satisfy
        ``roll_every / periods_per_year < tenor`` so calls are rolled before
        expiry.
    vol:
        Flat implied vol used to price and mark the options.
    shares:
        Underlying shares held (and calls written) — the overlay is fully covered.

    Returns
    -------
    OverlaySimResult
        Equity path of the overlaid book, the stock-only benchmark, costs, and
        the number of rolls.
    """
    prices = np.asarray(prices, dtype=float).reshape(-1)
    if prices.size == 0:
        raise ValueError("prices path is empty")
    dt = 1.0 / periods_per_year
    sim = ExecutionSimulator(cost_model=cost_model or CostModel(), rate=rate, div=div, cash=0.0)

    equity = np.empty(prices.size)
    times = np.arange(prices.size) * dt
    current: Instrument | None = None
    write_index = 0
    n_rolls = 0

    for i, spot in enumerate(prices):
        need_roll = current is None or (i - write_index) >= roll_every
        if need_roll:
            if current is not None:
                # Buy to close the existing short call at its remaining maturity.
                shift = (i - write_index) * dt
                sim.execute(Order(current, +shares), spot, vol, shift)
            current = Instrument("call", moneyness * spot, tenor)
            sim.execute(Order(current, -shares), spot, vol, 0.0)  # sell to open
            write_index = i
            n_rolls += 1

        shift = (i - write_index) * dt
        book_equity = sim.equity(spot, vol, shift)  # cash + option MTM (short call is a liability)
        equity[i] = book_equity + shares * spot  # + stock market value

    stock_only = shares * prices
    return OverlaySimResult(
        equity=equity,
        stock_only=stock_only,
        times=times,
        total_costs=sim.total_costs,
        n_rolls=n_rolls,
    )
