"""Option Greeks via JAX automatic differentiation.

Rather than re-deriving closed-form Greek formulas (and risking them drifting out
of sync with the pricer), we differentiate :func:`black_scholes_price` directly:

* delta = d price / d spot
* gamma = d^2 price / d spot^2
* vega  = d price / d vol
* theta = -d price / d ttm   (per-year; divide by 365 for per-calendar-day)
* rho   = d price / d rate

All are ``vmap``-friendly, so :func:`chain_greeks` returns every Greek across a
whole option chain in one call.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxfolio.options.pricing import black_scholes_price

Array = jnp.ndarray


def _price(spot, strike, ttm, vol, rate, div, is_call):
    return black_scholes_price(spot, strike, ttm, vol, rate, div, is_call)


def delta(spot, strike, ttm, vol, rate=0.0, div=0.0, is_call=True) -> Array:
    """Delta: sensitivity of price to the underlying spot."""
    f = lambda s: _price(s, strike, ttm, vol, rate, div, is_call)
    return jax.grad(f)(jnp.asarray(float(spot)))


def gamma(spot, strike, ttm, vol, rate=0.0, div=0.0, is_call=True) -> Array:
    """Gamma: second derivative of price w.r.t. spot (same for calls and puts)."""
    f = lambda s: _price(s, strike, ttm, vol, rate, div, is_call)
    return jax.grad(jax.grad(f))(jnp.asarray(float(spot)))


def vega(spot, strike, ttm, vol, rate=0.0, div=0.0, is_call=True) -> Array:
    """Vega: sensitivity of price to volatility (per 1.00 change in vol)."""
    f = lambda v: _price(spot, strike, ttm, v, rate, div, is_call)
    return jax.grad(f)(jnp.asarray(float(vol)))


def theta(spot, strike, ttm, vol, rate=0.0, div=0.0, is_call=True) -> Array:
    """Theta: time decay (per year). Negative for most long options."""
    f = lambda t: _price(spot, strike, t, vol, rate, div, is_call)
    return -jax.grad(f)(jnp.asarray(float(ttm)))


def rho(spot, strike, ttm, vol, rate=0.0, div=0.0, is_call=True) -> Array:
    """Rho: sensitivity of price to the risk-free rate."""
    f = lambda r: _price(spot, strike, ttm, vol, r, div, is_call)
    return jax.grad(f)(jnp.asarray(float(rate)))


def all_greeks(spot, strike, ttm, vol, rate=0.0, div=0.0, is_call=True) -> dict[str, float]:
    """Return every first/second-order Greek for a single option as a dict."""
    return {
        "price": float(_price(spot, strike, ttm, vol, rate, div, is_call)),
        "delta": float(delta(spot, strike, ttm, vol, rate, div, is_call)),
        "gamma": float(gamma(spot, strike, ttm, vol, rate, div, is_call)),
        "vega": float(vega(spot, strike, ttm, vol, rate, div, is_call)),
        "theta": float(theta(spot, strike, ttm, vol, rate, div, is_call)),
        "rho": float(rho(spot, strike, ttm, vol, rate, div, is_call)),
    }


def chain_greeks(
    spot: float,
    strikes: Array,
    ttms: Array,
    vols: Array,
    rate: float = 0.0,
    div: float = 0.0,
    is_call: bool = True,
) -> dict[str, Array]:
    """Vectorized Greeks across a chain of (strike, ttm, vol) triples.

    Returns a dict of arrays (one entry per Greek), each aligned with the input
    chain. Built from ``vmap`` over the single-option autodiff Greeks.
    """
    strikes = jnp.asarray(strikes, dtype=float)
    ttms = jnp.broadcast_to(jnp.asarray(ttms, dtype=float), strikes.shape)
    vols = jnp.broadcast_to(jnp.asarray(vols, dtype=float), strikes.shape)

    # argnums: 0=spot, 2=ttm, 3=vol, 4=rate.
    price_fn = lambda s, k, t, v, r: _price(s, k, t, v, r, div, is_call)

    d_delta = jax.vmap(lambda k, t, v: jax.grad(price_fn, argnums=0)(spot, k, t, v, rate))
    d_gamma = jax.vmap(
        lambda k, t, v: jax.grad(jax.grad(price_fn, argnums=0), argnums=0)(spot, k, t, v, rate)
    )
    d_vega = jax.vmap(lambda k, t, v: jax.grad(price_fn, argnums=3)(spot, k, t, v, rate))
    d_theta = jax.vmap(lambda k, t, v: -jax.grad(price_fn, argnums=2)(spot, k, t, v, rate))
    d_rho = jax.vmap(lambda k, t, v: jax.grad(price_fn, argnums=4)(spot, k, t, v, rate))
    d_price = jax.vmap(lambda k, t, v: price_fn(spot, k, t, v, rate))

    return {
        "price": d_price(strikes, ttms, vols),
        "delta": d_delta(strikes, ttms, vols),
        "gamma": d_gamma(strikes, ttms, vols),
        "vega": d_vega(strikes, ttms, vols),
        "theta": d_theta(strikes, ttms, vols),
        "rho": d_rho(strikes, ttms, vols),
    }
