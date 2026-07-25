"""Transaction-cost models for the options execution simulator.

A :class:`CostModel` turns a *mid* price into a realistic *fill* price and a
commission. Costs are intentionally simple and transparent (per-contract
commission + a proportional half-spread / slippage in basis points) — enough to
make simulated P&L reflect frictions without pretending to model a real venue's
microstructure.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """A commission + slippage model.

    Attributes
    ----------
    commission_per_contract:
        Flat cash commission charged per contract traded (per unit of absolute
        quantity), applied on both entry and exit.
    slippage_bps:
        Proportional slippage in basis points of the mid price, modeling the
        half-spread / market impact. A buy fills at ``mid * (1 + bps/1e4)``, a
        sell at ``mid * (1 - bps/1e4)``.
    min_commission:
        Floor applied to the per-trade commission.
    """

    commission_per_contract: float = 0.65
    slippage_bps: float = 5.0
    min_commission: float = 0.0

    def fill_price(self, mid: float, side: int) -> float:
        """Fill price for a ``side`` trade (+1 buy, -1 sell) at ``mid``.

        Slippage always works against the trader: buys fill above mid, sells
        below. A negative mid is floored at 0 (options cannot be worth < 0).
        """
        slip = self.slippage_bps / 1e4
        price = mid * (1.0 + side * slip)
        return max(price, 0.0)

    def commission(self, quantity: float) -> float:
        """Commission for trading ``quantity`` contracts (sign-insensitive)."""
        raw = abs(quantity) * self.commission_per_contract
        return max(raw, self.min_commission)
