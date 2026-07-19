"""Options pricing, Greeks, strategies, and portfolio overlays."""

from jaxfolio.options.greeks import (
    all_greeks,
    chain_greeks,
    delta,
    gamma,
    rho,
    theta,
    vega,
)
from jaxfolio.options.overlay import (
    OverlayBook,
    collar_overlay,
    covered_call_overlay,
)
from jaxfolio.options.pricing import (
    binomial_american,
    black_scholes_price,
    implied_volatility,
    price_call_chain,
)
from jaxfolio.options.strategies import (
    OptionLeg,
    OptionStrategy,
    StockLeg,
    bear_put_spread,
    bull_call_spread,
    butterfly,
    calendar_spread,
    collar,
    covered_call,
    iron_condor,
    protective_put,
    straddle,
    strangle,
)

__all__ = [
    # pricing
    "black_scholes_price",
    "price_call_chain",
    "implied_volatility",
    "binomial_american",
    # greeks
    "delta",
    "gamma",
    "vega",
    "theta",
    "rho",
    "all_greeks",
    "chain_greeks",
    # strategies
    "OptionLeg",
    "StockLeg",
    "OptionStrategy",
    "covered_call",
    "protective_put",
    "collar",
    "bull_call_spread",
    "bear_put_spread",
    "straddle",
    "strangle",
    "iron_condor",
    "butterfly",
    "calendar_spread",
    # overlay
    "OverlayBook",
    "covered_call_overlay",
    "collar_overlay",
]
