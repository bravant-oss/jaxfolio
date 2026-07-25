"""Order/fill accounting and a mark-to-market option book.

This is the core of the execution layer: an :class:`ExecutionSimulator` prices an
option at Black-Scholes mid, applies a :class:`~jaxfolio.options.execution.costs.CostModel`
to get a fill, updates an :class:`OptionBook` of :class:`Position` objects and a
cash balance, and can mark the whole book to market at any spot/vol/horizon.

Scope: this simulates fills against a *modeled* price. It is a research /
backtesting tool — **not** a live broker, order-management system, or connection
to any exchange. See ``DISCLAIMER.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jaxfolio.options.execution.costs import CostModel
from jaxfolio.options.greeks import all_greeks
from jaxfolio.options.pricing import black_scholes_price

BUY = 1
SELL = -1


@dataclass(frozen=True)
class Instrument:
    """A vanilla option contract identity (what is traded)."""

    kind: str  # "call" or "put"
    strike: float
    expiry: float  # time-to-expiry in years at the reference/valuation date

    @property
    def is_call(self) -> bool:
        return self.kind.lower() == "call"

    def key(self) -> tuple[str, float, float]:
        return (self.kind.lower(), float(self.strike), float(self.expiry))


@dataclass
class Order:
    """An instruction to trade ``quantity`` contracts of ``instrument``.

    ``quantity`` is signed: positive = buy (long), negative = sell (short).
    """

    instrument: Instrument
    quantity: float

    @property
    def side(self) -> int:
        return BUY if self.quantity >= 0 else SELL


@dataclass
class Fill:
    """The realized result of executing an :class:`Order`."""

    instrument: Instrument
    quantity: float
    mid: float
    fill_price: float
    commission: float

    @property
    def cash_flow(self) -> float:
        """Signed cash impact: buying pays (negative), selling receives (positive)."""
        return -self.quantity * self.fill_price - self.commission


@dataclass
class Position:
    """A net position in one instrument, tracking quantity and average cost."""

    instrument: Instrument
    quantity: float = 0.0
    avg_price: float = 0.0

    def apply(self, fill: Fill) -> None:
        """Fold a fill into the position, updating the average entry price."""
        new_qty = self.quantity + fill.quantity
        if self.quantity == 0 or (self.quantity > 0) == (fill.quantity > 0):
            # Opening or adding in the same direction: blend the average price.
            total = self.quantity + fill.quantity
            if total != 0:
                self.avg_price = (
                    self.avg_price * self.quantity + fill.fill_price * fill.quantity
                ) / total
        elif abs(fill.quantity) > abs(self.quantity):
            # Flipped through zero: the residual carries the new fill's price.
            self.avg_price = fill.fill_price
        # else: partial close, avg_price of the remaining lot is unchanged.
        self.quantity = new_qty
        if abs(self.quantity) < 1e-12:
            self.quantity = 0.0
            self.avg_price = 0.0

    def market_value(
        self, spot: float, vol: float, rate: float, div: float, ttm_shift: float
    ) -> float:
        """Mark-to-model value of the position at a horizon."""
        if self.quantity == 0.0:
            return 0.0
        t = max(self.instrument.expiry - ttm_shift, 1e-6)
        px = float(
            black_scholes_price(
                spot, self.instrument.strike, t, vol, rate, div, self.instrument.is_call
            )
        )
        return self.quantity * px


@dataclass
class OptionBook:
    """A collection of option positions keyed by instrument identity."""

    positions: dict[tuple, Position] = field(default_factory=dict)

    def apply(self, fill: Fill) -> None:
        key = fill.instrument.key()
        pos = self.positions.get(key)
        if pos is None:
            pos = Position(fill.instrument)
            self.positions[key] = pos
        pos.apply(fill)
        if pos.quantity == 0.0:
            del self.positions[key]

    def market_value(self, spot, vol, rate=0.0, div=0.0, ttm_shift=0.0) -> float:
        """Total mark-to-model value of all open positions."""
        return sum(p.market_value(spot, vol, rate, div, ttm_shift) for p in self.positions.values())

    def net_greeks(self, spot, vol, rate=0.0, div=0.0, ttm_shift=0.0) -> dict[str, float]:
        """Net position Greeks across the book."""
        agg = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
        for p in self.positions.values():
            if p.quantity == 0.0:
                continue
            t = max(p.instrument.expiry - ttm_shift, 1e-6)
            g = all_greeks(spot, p.instrument.strike, t, vol, rate, div, p.instrument.is_call)
            for k in agg:
                agg[k] += p.quantity * g[k]
        return agg


@dataclass
class ExecutionSimulator:
    """Executes orders against a modeled mid price and books the results.

    Parameters
    ----------
    cost_model:
        The commission/slippage model applied to every fill.
    rate, div:
        Rate and dividend yield used both to price the option mid and to mark the
        book.
    cash:
        Starting cash balance (updated by every fill's cash flow).
    """

    cost_model: CostModel = field(default_factory=CostModel)
    rate: float = 0.0
    div: float = 0.0
    cash: float = 0.0
    book: OptionBook = field(default_factory=OptionBook)
    fills: list[Fill] = field(default_factory=list)

    def mid_price(
        self, instrument: Instrument, spot: float, vol: float, ttm_shift: float = 0.0
    ) -> float:
        """Black-Scholes mid price of ``instrument`` at the given market state."""
        t = max(instrument.expiry - ttm_shift, 1e-6)
        return float(
            black_scholes_price(
                spot, instrument.strike, t, vol, self.rate, self.div, instrument.is_call
            )
        )

    def execute(self, order: Order, spot: float, vol: float, ttm_shift: float = 0.0) -> Fill:
        """Execute ``order`` at the current market state, updating book and cash."""
        mid = self.mid_price(order.instrument, spot, vol, ttm_shift)
        fill_price = self.cost_model.fill_price(mid, order.side)
        commission = self.cost_model.commission(order.quantity)
        fill = Fill(order.instrument, order.quantity, mid, fill_price, commission)
        self.book.apply(fill)
        self.cash += fill.cash_flow
        self.fills.append(fill)
        return fill

    def equity(self, spot: float, vol: float, ttm_shift: float = 0.0) -> float:
        """Total equity = cash + mark-to-market value of the option book."""
        return self.cash + self.book.market_value(spot, vol, self.rate, self.div, ttm_shift)

    @property
    def total_costs(self) -> float:
        """Total commission paid across all fills so far."""
        return sum(f.commission for f in self.fills)
