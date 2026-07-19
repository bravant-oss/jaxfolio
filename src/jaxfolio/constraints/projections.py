"""Constraint projections used by the projected-gradient optimizers.

All projections are pure JAX and jit/vmap-safe. The two workhorses are:

* :func:`project_simplex` — Euclidean projection onto the probability simplex
  ``{w : w >= 0, sum(w) = 1}`` (long-only, fully-invested).
* :func:`project_box_budget` — projection onto a box ``[lo, hi]`` intersected
  with the budget hyperplane ``sum(w) = budget`` (allows bounded shorting).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jnp.ndarray


def project_simplex(v: Array, budget: float = 1.0) -> Array:
    """Euclidean projection of ``v`` onto the scaled simplex summing to ``budget``.

    Implements the classic Duchi et al. (2008) sort-based algorithm, which is
    exact and differentiable almost everywhere.
    """
    n = v.shape[0]
    u = jnp.sort(v)[::-1]
    cssv = jnp.cumsum(u) - budget
    ind = jnp.arange(1, n + 1)
    cond = u - cssv / ind > 0
    # rho = number of positive-threshold coordinates.
    rho = jnp.sum(cond)
    theta = cssv[rho - 1] / rho
    return jnp.maximum(v - theta, 0.0)


def project_box_budget(
    v: Array,
    lower: float = 0.0,
    upper: float = 1.0,
    budget: float = 1.0,
    *,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> Array:
    """Project ``v`` onto ``{w : lower <= w <= upper, sum(w) = budget}``.

    Solves for the Lagrange multiplier ``tau`` on the budget constraint by
    bisection: ``w(tau) = clip(v - tau, lower, upper)`` is monotone decreasing in
    ``tau``, so we bisect until ``sum(w(tau)) == budget``. Feasible whenever
    ``n*lower <= budget <= n*upper``.
    """
    n = v.shape[0]

    def sum_at(tau: Array) -> Array:
        return jnp.sum(jnp.clip(v - tau, lower, upper))

    # Bracket tau: sum decreases as tau grows.
    lo = jnp.min(v) - upper
    hi = jnp.max(v) - lower

    def body(_, bounds):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        s = sum_at(mid)
        # If sum too big, need larger tau -> move lo up; else move hi down.
        too_big = s > budget
        lo = jnp.where(too_big, mid, lo)
        hi = jnp.where(too_big, hi, mid)
        return (lo, hi)

    lo, hi = jax.lax.fori_loop(0, max_iter, body, (lo, hi))
    tau = 0.5 * (lo + hi)
    w = jnp.clip(v - tau, lower, upper)
    # Correct tiny residual so weights sum exactly to budget.
    w = w + (budget - jnp.sum(w)) / n
    return w


def normalize_weights(w: Array, budget: float = 1.0) -> Array:
    """Rescale weights to sum to ``budget`` (assumes a non-zero sum)."""
    s = jnp.sum(w)
    return jnp.where(jnp.abs(s) > 1e-12, w * (budget / s), jnp.full_like(w, budget / w.shape[0]))


def softmax_weights(logits: Array, budget: float = 1.0) -> Array:
    """Map unconstrained logits to long-only weights via softmax (sums to budget)."""
    return budget * jax.nn.softmax(logits)
