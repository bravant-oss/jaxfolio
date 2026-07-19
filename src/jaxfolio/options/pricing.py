"""Option pricing in pure JAX.

The Black-Scholes-Merton price is a plain differentiable function, so the Greeks
module gets every sensitivity from ``jax.grad`` without duplicating formulas. A
jitted Newton solver recovers implied volatility, and a binomial (Cox-Ross-
Rubinstein) lattice prices American-style options.

Convention: ``+1`` for calls, ``-1`` for puts (``is_call`` boolean helpers are
provided for readability at call sites).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

Array = jnp.ndarray


def _d1_d2(
    spot: Array, strike: Array, ttm: Array, vol: Array, rate: Array, div: Array
) -> tuple[Array, Array]:
    """Black-Scholes ``d1`` and ``d2`` terms (guarded against ttm/vol → 0)."""
    ttm = jnp.clip(ttm, 1e-8, None)
    vol = jnp.clip(vol, 1e-8, None)
    sqrt_t = jnp.sqrt(ttm)
    d1 = (jnp.log(spot / strike) + (rate - div + 0.5 * vol**2) * ttm) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    return d1, d2


def black_scholes_price(
    spot: Array,
    strike: Array,
    ttm: Array,
    vol: Array,
    rate: Array = 0.0,
    div: Array = 0.0,
    is_call: bool = True,
) -> Array:
    """Black-Scholes-Merton price of a European option.

    All array arguments broadcast, so this prices whole option chains at once.

    Parameters
    ----------
    spot, strike:
        Underlying price and strike.
    ttm:
        Time to maturity in years.
    vol:
        Annualized volatility.
    rate, div:
        Continuous risk-free rate and dividend yield.
    is_call:
        ``True`` for a call, ``False`` for a put.
    """
    d1, d2 = _d1_d2(spot, strike, ttm, vol, rate, div)
    disc = jnp.exp(-rate * ttm)
    carry = jnp.exp(-div * ttm)
    if is_call:
        return spot * carry * norm.cdf(d1) - strike * disc * norm.cdf(d2)
    return strike * disc * norm.cdf(-d2) - spot * carry * norm.cdf(-d1)


# vmapped chain pricer: vectorize over strike, ttm, and vol simultaneously.
_price_over_chain = jax.jit(
    jax.vmap(
        lambda s, k, t, v, r, q: black_scholes_price(s, k, t, v, r, q, True),
        in_axes=(None, 0, 0, 0, None, None),
    ),
    static_argnums=(),
)


def price_call_chain(
    spot: float, strikes: Array, ttms: Array, vols: Array, rate: float = 0.0, div: float = 0.0
) -> Array:
    """Vectorized call prices across a chain of (strike, ttm, vol) triples.

    ``ttms`` and ``vols`` may be scalars (broadcast to every strike) or arrays
    aligned with ``strikes``.
    """
    strikes = jnp.asarray(strikes, dtype=float)
    ttms = jnp.broadcast_to(jnp.asarray(ttms, dtype=float), strikes.shape)
    vols = jnp.broadcast_to(jnp.asarray(vols, dtype=float), strikes.shape)
    return _price_over_chain(spot, strikes, ttms, vols, rate, div)


def implied_volatility(
    price: float,
    spot: float,
    strike: float,
    ttm: float,
    rate: float = 0.0,
    div: float = 0.0,
    is_call: bool = True,
    *,
    initial_vol: float = 0.2,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> Array:
    """Recover implied volatility via jitted Newton iterations.

    Vega (the derivative of price w.r.t. vol) comes from ``jax.grad`` on the
    pricer, so the Newton step needs no hand-coded derivative. Returns ``NaN``
    when no volatility reproduces ``price`` to within ``tol`` (e.g. a price below
    intrinsic value or a deep-OTM option with vanishing vega), so callers can
    distinguish "no solution" from a genuinely tiny implied vol.
    """
    price_of_vol = lambda v: black_scholes_price(spot, strike, ttm, v, rate, div, is_call)
    vega_of_vol = jax.grad(price_of_vol)

    def cond(state):
        i, v, diff = state
        return jnp.logical_and(i < max_iter, jnp.abs(diff) > tol)

    def body(state):
        i, v, _ = state
        diff = price_of_vol(v) - price
        vega = vega_of_vol(v)
        vega = jnp.where(jnp.abs(vega) < 1e-8, 1e-8, vega)
        v_new = jnp.clip(v - diff / vega, 1e-4, 10.0)
        return (i + 1, v_new, v_new - v)

    _, vol, _ = jax.lax.while_loop(cond, body, (0, jnp.asarray(initial_vol), jnp.asarray(jnp.inf)))
    # Flag non-convergence: the recovered vol must actually reprice the option.
    residual = jnp.abs(price_of_vol(vol) - price)
    return jnp.where(residual <= jnp.sqrt(tol) + 1e-6, vol, jnp.nan)


def binomial_american(
    spot: float,
    strike: float,
    ttm: float,
    vol: float,
    rate: float = 0.0,
    div: float = 0.0,
    is_call: bool = True,
    *,
    steps: int = 256,
) -> Array:
    """Cox-Ross-Rubinstein binomial price for an American-style option.

    Backward induction with early-exercise checks at every node. ``steps``
    controls the lattice resolution.
    """
    dt = ttm / steps
    u = jnp.exp(vol * jnp.sqrt(dt))
    d = 1.0 / u
    disc = jnp.exp(-rate * dt)
    p = (jnp.exp((rate - div) * dt) - d) / (u - d)
    p = jnp.clip(p, 0.0, 1.0)
    sign = 1.0 if is_call else -1.0

    j = jnp.arange(steps + 1)
    # Terminal underlying prices and payoffs at maturity (fixed-length buffer).
    terminal_prices = spot * u**j * d ** (steps - j)
    values = jnp.maximum(sign * (terminal_prices - strike), 0.0)

    def body(i, values):
        # Rolling back to layer m = steps - i, which has m + 1 live nodes (0..m).
        m = steps - i
        node_prices = spot * u**j * d ** (m - j)  # valid for j <= m; tail is unused
        # cont[j] = disc * (p * values[j+1] + (1-p) * values[j]); roll gives values[j+1].
        up = jnp.roll(values, -1)
        cont = disc * (p * up + (1.0 - p) * values)
        exercise = jnp.maximum(sign * (node_prices - strike), 0.0)
        return jnp.maximum(cont, exercise)

    values = jax.lax.fori_loop(1, steps + 1, body, values)
    return values[0]
