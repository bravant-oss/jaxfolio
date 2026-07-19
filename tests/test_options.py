"""Tests for the options module: pricing, Greeks, strategies, overlays."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaxfolio.options import greeks as GK
from jaxfolio.options import strategies as S
from jaxfolio.options.pricing import (
    binomial_american,
    black_scholes_price,
    implied_volatility,
)


def test_put_call_parity():
    """C - P = S e^{-qT} - K e^{-rT} must hold for Black-Scholes."""
    s, k, t, v, r, q = 100.0, 105.0, 0.5, 0.25, 0.03, 0.01
    call = float(black_scholes_price(s, k, t, v, r, q, is_call=True))
    put = float(black_scholes_price(s, k, t, v, r, q, is_call=False))
    lhs = call - put
    rhs = s * np.exp(-q * t) - k * np.exp(-r * t)
    assert np.isclose(lhs, rhs, atol=1e-4)


def test_autodiff_delta_matches_closed_form():
    """Call delta from autodiff should equal e^{-qT} N(d1)."""
    from jax.scipy.stats import norm

    s, k, t, v, r, q = 100.0, 100.0, 0.5, 0.2, 0.02, 0.0
    d1 = (np.log(s / k) + (r - q + 0.5 * v**2) * t) / (v * np.sqrt(t))
    closed = np.exp(-q * t) * float(norm.cdf(d1))
    auto = float(GK.delta(s, k, t, v, r, q, is_call=True))
    assert np.isclose(auto, closed, atol=1e-5)


def test_gamma_positive_and_symmetric_call_put():
    """Gamma is identical for calls and puts and strictly positive ATM."""
    s, k, t, v = 100.0, 100.0, 0.3, 0.2
    g_call = float(GK.gamma(s, k, t, v, is_call=True))
    g_put = float(GK.gamma(s, k, t, v, is_call=False))
    assert g_call > 0
    assert np.isclose(g_call, g_put, atol=1e-6)


def test_implied_vol_roundtrip():
    """Pricing at a known vol then inverting should recover the vol."""
    s, k, t, true_vol, r = 100.0, 110.0, 0.75, 0.32, 0.02
    price = float(black_scholes_price(s, k, t, true_vol, r, is_call=True))
    iv = float(implied_volatility(price, s, k, t, r, is_call=True))
    assert np.isclose(iv, true_vol, atol=1e-4)


def test_binomial_converges_to_black_scholes_for_european_call():
    """A no-dividend American call equals the European call (never early-exercised)."""
    s, k, t, v, r = 100.0, 100.0, 1.0, 0.2, 0.05
    bs = float(black_scholes_price(s, k, t, v, r, is_call=True))
    binom = float(binomial_american(s, k, t, v, r, is_call=True, steps=400))
    assert np.isclose(bs, binom, atol=0.15)


def test_chain_greeks_shapes():
    strikes = jnp.array([90.0, 100.0, 110.0])
    ttms = jnp.array([0.25, 0.25, 0.25])
    vols = jnp.array([0.2, 0.2, 0.2])
    g = GK.chain_greeks(100.0, strikes, ttms, vols)
    for key in ("price", "delta", "gamma", "vega", "theta"):
        assert g[key].shape == (3,)
    # Delta increases with decreasing strike for calls.
    assert g["delta"][0] > g["delta"][2]


def test_covered_call_payoff_caps_upside():
    strat = S.covered_call(100.0, 110.0, expiry=0.25, vol=0.2)
    # Far above the call strike, payoff is flat (capped).
    high = strat.payoff_at_expiry(np.array([150.0, 200.0]))
    assert np.isclose(high[0], high[1], atol=1e-6)


def test_straddle_payoff_is_v_shaped():
    strat = S.straddle(100.0, 100.0, expiry=0.25, vol=0.2)
    pnl = strat.payoff_at_expiry(np.array([70.0, 100.0, 130.0]))
    # Minimum at the strike, higher on both wings.
    assert pnl[1] < pnl[0] and pnl[1] < pnl[2]


def test_bull_call_spread_bounded():
    strat = S.bull_call_spread(100.0, 95.0, 105.0, expiry=0.25, vol=0.2)
    pnl = strat.payoff_at_expiry(np.linspace(80, 120, 200))
    # Bounded above and below.
    assert pnl.max() < 15.0
    assert pnl.min() > -15.0


def test_iron_condor_profit_in_middle():
    strat = S.iron_condor(100.0, 85, 95, 105, 115, expiry=0.25, vol=0.2)
    mid = strat.payoff_at_expiry(np.array([100.0]))[0]
    wing = strat.payoff_at_expiry(np.array([70.0]))[0]
    assert mid > wing  # credit kept in the middle, loss at the wings


def test_strategy_net_greeks_keys():
    strat = S.collar(100.0, 95.0, 105.0, expiry=0.25, vol=0.2)
    g = strat.greeks(100.0, 0.2)
    assert set(g) == {"delta", "gamma", "vega", "theta", "rho"}


def test_break_evens_found():
    strat = S.straddle(100.0, 100.0, expiry=0.25, vol=0.2)
    bes = strat.break_evens()
    assert len(bes) == 2  # a long straddle has two break-evens


@pytest.mark.parametrize("preset", [S.protective_put, S.covered_call])
def test_stock_plus_option_presets(preset):
    strat = preset(100.0, 100.0, expiry=0.25, vol=0.2)
    assert strat.stock is not None
    pnl = strat.payoff_at_expiry(np.array([100.0]))
    assert np.isfinite(pnl).all()
