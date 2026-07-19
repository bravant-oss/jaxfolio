"""Build, price, and visualize options strategies; overlay a collar on a portfolio.

Run with:
    uv run python examples/03_options_strategies.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import jaxfolio as jf
from jaxfolio import viz
from jaxfolio.options import (
    all_greeks,
    black_scholes_price,
    implied_volatility,
    iron_condor,
    straddle,
)
from jaxfolio.options.overlay import collar_overlay

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

SPOT = 100.0


def main() -> None:
    # --- Single-option pricing & Greeks (Greeks via autodiff) --------------- #
    price = float(black_scholes_price(SPOT, 100.0, 0.25, 0.2, 0.03))
    print(f"ATM 3M call price: {price:.3f}")
    print("Greeks:", {k: round(v, 4) for k, v in all_greeks(SPOT, 100, 0.25, 0.2, 0.03).items()})

    iv = float(implied_volatility(price, SPOT, 100.0, 0.25, 0.03))
    print(f"Implied vol recovered from price: {iv:.4f} (input 0.2000)")

    # --- Multi-leg strategies ---------------------------------------------- #
    condor = iron_condor(SPOT, 85, 95, 105, 115, expiry=0.25, vol=0.22)
    strad = straddle(SPOT, 100.0, expiry=0.25, vol=0.22)

    viz.save(viz.plot_payoff(condor, spot=SPOT), str(OUT / "iron_condor_payoff.png"))
    viz.save(viz.plot_payoff(strad, spot=SPOT), str(OUT / "straddle_payoff.png"))
    viz.save(
        viz.plot_greeks_profile(condor, spot=SPOT, vol=0.22),
        str(OUT / "iron_condor_greeks.png"),
    )

    # --- Implied-vol surface (synthetic smile/term structure) --------------- #
    strikes = np.linspace(80, 120, 25)
    expiries = np.linspace(0.05, 1.0, 20)
    # A simple smile: IV rises away from ATM and with maturity.
    moneyness = (strikes[None, :] - SPOT) / SPOT
    ivs = 0.18 + 0.12 * moneyness**2 + 0.04 * expiries[:, None]
    viz.save(
        viz.plot_vol_surface(SPOT, strikes, expiries, ivs),
        str(OUT / "vol_surface.png"),
    )

    # --- Portfolio + options overlay --------------------------------------- #
    returns = jf.generate_returns(n_assets=8, n_days=756, seed=7)
    portfolio = jf.maximum_sharpe(returns)
    spots = {a: SPOT for a in portfolio.assets}
    book = collar_overlay(portfolio, spots, top_n=5, put_moneyness=0.95, call_moneyness=1.08)

    print("\nCollar overlay on the max-Sharpe portfolio:")
    print("  Net Greeks:", {k: round(v, 4) for k, v in book.net_greeks().items()})

    grid = np.linspace(0.7, 1.3, 200)
    pnl = book.payoff_curve(grid)
    print(f"  Overlay P&L at -30% shock: {pnl[0]:+.2f}, at +30% shock: {pnl[-1]:+.2f}")

    print(f"\nSaved options plots -> {OUT}/")


if __name__ == "__main__":
    main()
