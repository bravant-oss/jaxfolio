"""Multi-leg options strategies: payoffs, P&L, and Greeks.

An :class:`OptionLeg` describes a single position (call/put, long/short, strike,
expiry, quantity). An :class:`OptionStrategy` is a collection of legs (optionally
plus a stock leg) with helpers to evaluate payoff at expiry, mark-to-model P&L at
a horizon, net Greeks, and the break-even points. A library of preset
constructors builds the common structures (covered call, collar, spreads,
straddle, iron condor, butterfly, calendar).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from jaxfolio.options.greeks import all_greeks
from jaxfolio.options.pricing import black_scholes_price


@dataclass
class OptionLeg:
    """A single option position.

    Attributes
    ----------
    kind:
        ``"call"`` or ``"put"``.
    strike:
        Strike price.
    quantity:
        Signed contract count — positive = long, negative = short.
    expiry:
        Time to expiry in years (used for mark-to-model and Greeks).
    premium:
        Premium paid (long) or received (short) per contract. If ``None`` it is
        filled from the Black-Scholes price at construction time by the presets.
    """

    kind: str
    strike: float
    quantity: float = 1.0
    expiry: float = 0.25
    premium: float | None = None

    @property
    def is_call(self) -> bool:
        return self.kind.lower() == "call"

    def intrinsic(self, spot: np.ndarray) -> np.ndarray:
        """Intrinsic value of the option at ``spot`` (per contract, unsigned)."""
        if self.is_call:
            return np.maximum(spot - self.strike, 0.0)
        return np.maximum(self.strike - spot, 0.0)

    def payoff_at_expiry(self, spot: np.ndarray) -> np.ndarray:
        """Signed P&L of this leg at expiry, net of premium."""
        prem = self.premium if self.premium is not None else 0.0
        return self.quantity * (self.intrinsic(spot) - prem)


@dataclass
class StockLeg:
    """A linear underlying position held alongside options (e.g. covered call)."""

    quantity: float = 1.0
    entry_price: float = 100.0

    def payoff_at_expiry(self, spot: np.ndarray) -> np.ndarray:
        return self.quantity * (spot - self.entry_price)


@dataclass
class OptionStrategy:
    """A named collection of option (and optional stock) legs."""

    name: str
    legs: list[OptionLeg] = field(default_factory=list)
    stock: StockLeg | None = None

    def payoff_at_expiry(self, spot: np.ndarray) -> np.ndarray:
        """Total strategy P&L at expiry across a grid of terminal spot prices."""
        spot = np.asarray(spot, dtype=float)
        total = np.zeros_like(spot)
        for leg in self.legs:
            total = total + leg.payoff_at_expiry(spot)
        if self.stock is not None:
            total = total + self.stock.payoff_at_expiry(spot)
        return total

    def net_premium(self) -> float:
        """Net premium: negative = net debit (paid), positive = net credit."""
        return float(-sum(leg.quantity * (leg.premium or 0.0) for leg in self.legs))

    def value_at(
        self, spot: float, vol: float, rate: float = 0.0, div: float = 0.0, ttm_shift: float = 0.0
    ) -> float:
        """Mark-to-model value of all option legs at a horizon.

        ``ttm_shift`` is subtracted from each leg's expiry to advance time (e.g.
        ``ttm_shift=1/12`` marks the book one month forward).
        """
        total = 0.0
        for leg in self.legs:
            t = max(leg.expiry - ttm_shift, 1e-6)
            price = float(black_scholes_price(spot, leg.strike, t, vol, rate, div, leg.is_call))
            total += leg.quantity * price
        if self.stock is not None:
            total += self.stock.quantity * spot
        return total

    def pnl_at(
        self, spot: float, vol: float, rate: float = 0.0, div: float = 0.0, ttm_shift: float = 0.0
    ) -> float:
        """Mark-to-model P&L relative to the entry cost of the option legs."""
        entry = sum(leg.quantity * (leg.premium or 0.0) for leg in self.legs)
        if self.stock is not None:
            entry += self.stock.quantity * self.stock.entry_price
        return self.value_at(spot, vol, rate, div, ttm_shift) - entry

    def greeks(self, spot: float, vol: float, rate: float = 0.0, div: float = 0.0) -> dict:
        """Net position Greeks (sum of per-leg Greeks weighted by quantity)."""
        agg = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
        for leg in self.legs:
            g = all_greeks(spot, leg.strike, leg.expiry, vol, rate, div, leg.is_call)
            for k in agg:
                agg[k] += leg.quantity * g[k]
        if self.stock is not None:
            agg["delta"] += self.stock.quantity  # stock has unit delta
        return agg

    def break_evens(self, spot_grid: np.ndarray | None = None) -> list[float]:
        """Approximate break-even spot prices (payoff sign changes)."""
        if spot_grid is None:
            strikes = [leg.strike for leg in self.legs]
            lo, hi = 0.5 * min(strikes), 1.5 * max(strikes)
            spot_grid = np.linspace(lo, hi, 2000)
        pnl = self.payoff_at_expiry(spot_grid)
        sign = np.sign(pnl)
        crossings = np.where(np.diff(sign) != 0)[0]
        # Linear interpolation at each zero crossing.
        bes = []
        for i in crossings:
            x0, x1 = spot_grid[i], spot_grid[i + 1]
            y0, y1 = pnl[i], pnl[i + 1]
            if y1 != y0:
                bes.append(float(x0 - y0 * (x1 - x0) / (y1 - y0)))
        return bes


# --------------------------------------------------------------------------- #
# Preset strategy builders. Each fills leg premiums from Black-Scholes so the
# payoff diagrams reflect realistic entry costs.
# --------------------------------------------------------------------------- #
def _priced_leg(kind, strike, qty, expiry, spot, vol, rate, div) -> OptionLeg:
    premium = float(black_scholes_price(spot, strike, expiry, vol, rate, div, kind == "call"))
    return OptionLeg(kind=kind, strike=strike, quantity=qty, expiry=expiry, premium=premium)


def covered_call(
    spot: float, strike: float, *, expiry: float = 0.25, vol: float = 0.25, rate: float = 0.0
) -> OptionStrategy:
    """Long stock + short call: income overlay that caps upside."""
    leg = _priced_leg("call", strike, -1.0, expiry, spot, vol, rate, 0.0)
    return OptionStrategy("Covered Call", [leg], StockLeg(1.0, spot))


def protective_put(
    spot: float, strike: float, *, expiry: float = 0.25, vol: float = 0.25, rate: float = 0.0
) -> OptionStrategy:
    """Long stock + long put: downside insurance."""
    leg = _priced_leg("put", strike, 1.0, expiry, spot, vol, rate, 0.0)
    return OptionStrategy("Protective Put", [leg], StockLeg(1.0, spot))


def collar(
    spot: float,
    put_strike: float,
    call_strike: float,
    *,
    expiry: float = 0.25,
    vol: float = 0.25,
    rate: float = 0.0,
) -> OptionStrategy:
    """Long stock + long put + short call: bounded payoff, often near-zero cost."""
    put = _priced_leg("put", put_strike, 1.0, expiry, spot, vol, rate, 0.0)
    call = _priced_leg("call", call_strike, -1.0, expiry, spot, vol, rate, 0.0)
    return OptionStrategy("Collar", [put, call], StockLeg(1.0, spot))


def bull_call_spread(
    spot: float,
    lower: float,
    upper: float,
    *,
    expiry: float = 0.25,
    vol: float = 0.25,
    rate: float = 0.0,
) -> OptionStrategy:
    """Long lower-strike call + short upper-strike call: capped bullish debit."""
    long_c = _priced_leg("call", lower, 1.0, expiry, spot, vol, rate, 0.0)
    short_c = _priced_leg("call", upper, -1.0, expiry, spot, vol, rate, 0.0)
    return OptionStrategy("Bull Call Spread", [long_c, short_c])


def bear_put_spread(
    spot: float,
    upper: float,
    lower: float,
    *,
    expiry: float = 0.25,
    vol: float = 0.25,
    rate: float = 0.0,
) -> OptionStrategy:
    """Long upper-strike put + short lower-strike put: capped bearish debit."""
    long_p = _priced_leg("put", upper, 1.0, expiry, spot, vol, rate, 0.0)
    short_p = _priced_leg("put", lower, -1.0, expiry, spot, vol, rate, 0.0)
    return OptionStrategy("Bear Put Spread", [long_p, short_p])


def straddle(
    spot: float, strike: float, *, expiry: float = 0.25, vol: float = 0.25, rate: float = 0.0
) -> OptionStrategy:
    """Long call + long put at the same strike: a bet on large moves either way."""
    call = _priced_leg("call", strike, 1.0, expiry, spot, vol, rate, 0.0)
    put = _priced_leg("put", strike, 1.0, expiry, spot, vol, rate, 0.0)
    return OptionStrategy("Long Straddle", [call, put])


def strangle(
    spot: float,
    put_strike: float,
    call_strike: float,
    *,
    expiry: float = 0.25,
    vol: float = 0.25,
    rate: float = 0.0,
) -> OptionStrategy:
    """Long OTM put + long OTM call: cheaper volatility bet than a straddle."""
    put = _priced_leg("put", put_strike, 1.0, expiry, spot, vol, rate, 0.0)
    call = _priced_leg("call", call_strike, 1.0, expiry, spot, vol, rate, 0.0)
    return OptionStrategy("Long Strangle", [put, call])


def iron_condor(
    spot: float,
    put_long: float,
    put_short: float,
    call_short: float,
    call_long: float,
    *,
    expiry: float = 0.25,
    vol: float = 0.25,
    rate: float = 0.0,
) -> OptionStrategy:
    """Short strangle wrapped in long protective wings: a range-bound credit trade.

    Strikes must satisfy ``put_long < put_short < call_short < call_long``.
    """
    legs = [
        _priced_leg("put", put_long, 1.0, expiry, spot, vol, rate, 0.0),
        _priced_leg("put", put_short, -1.0, expiry, spot, vol, rate, 0.0),
        _priced_leg("call", call_short, -1.0, expiry, spot, vol, rate, 0.0),
        _priced_leg("call", call_long, 1.0, expiry, spot, vol, rate, 0.0),
    ]
    return OptionStrategy("Iron Condor", legs)


def butterfly(
    spot: float,
    lower: float,
    middle: float,
    upper: float,
    *,
    expiry: float = 0.25,
    vol: float = 0.25,
    rate: float = 0.0,
) -> OptionStrategy:
    """Long butterfly with calls: peak payoff pinned at the middle strike."""
    legs = [
        _priced_leg("call", lower, 1.0, expiry, spot, vol, rate, 0.0),
        _priced_leg("call", middle, -2.0, expiry, spot, vol, rate, 0.0),
        _priced_leg("call", upper, 1.0, expiry, spot, vol, rate, 0.0),
    ]
    return OptionStrategy("Long Butterfly", legs)


def calendar_spread(
    spot: float,
    strike: float,
    *,
    near_expiry: float = 0.08,
    far_expiry: float = 0.33,
    vol: float = 0.25,
    rate: float = 0.0,
) -> OptionStrategy:
    """Short near-dated + long far-dated call at one strike: a theta/vega play.

    Because the legs have different expiries, use :meth:`OptionStrategy.pnl_at`
    (mark-to-model) rather than expiry payoff to analyze this structure.
    """
    short_near = _priced_leg("call", strike, -1.0, near_expiry, spot, vol, rate, 0.0)
    long_far = _priced_leg("call", strike, 1.0, far_expiry, spot, vol, rate, 0.0)
    return OptionStrategy("Calendar Spread", [short_near, long_far])
