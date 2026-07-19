"""Projected-gradient solver shared by the constrained classical optimizers.

Every constrained objective (min-variance, mean-variance, max-Sharpe,
max-diversification, CVaR, ...) is minimized by the same routine: take an optax
gradient step on an unconstrained ``objective(w)``, then project ``w`` back onto
the feasible set. The loop runs inside ``jax.lax.while_loop`` so the whole solve
is a single jit-compiled kernel.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax

from jaxfolio.constraints.projections import project_box_budget, project_simplex

Array = jnp.ndarray
Objective = Callable[[Array], Array]


def make_projection(
    long_only: bool,
    weight_bounds: tuple[float, float],
    budget: float = 1.0,
) -> Callable[[Array], Array]:
    """Return the projection matching the requested constraint set."""
    lo, hi = weight_bounds
    if long_only and lo <= 0.0 and hi >= 1.0:
        # Plain probability simplex is cheaper and exact.
        return lambda w: project_simplex(w, budget)
    return lambda w: project_box_budget(w, lo, hi, budget)


def solve_projected_gradient(
    objective: Objective,
    w0: Array,
    projection: Callable[[Array], Array],
    *,
    learning_rate: float = 1e-2,
    max_iter: int = 2000,
    tol: float = 1e-8,
) -> tuple[Array, dict[str, Array]]:
    """Minimize ``objective`` over the feasible set defined by ``projection``.

    Uses projected Adam with a fixed step budget and an early-stop tolerance on
    the successive-weight change. Returns ``(weights, info)`` where ``info`` holds
    the iteration count and final step size.
    """
    optimizer = optax.adam(learning_rate)
    grad_fn = jax.grad(objective)

    w0 = projection(w0)
    opt_state0 = optimizer.init(w0)

    def cond(state):
        i, _, _, delta = state
        return jnp.logical_and(i < max_iter, delta > tol)

    def body(state):
        i, w, opt_state, _ = state
        grads = grad_fn(w)
        updates, opt_state = optimizer.update(grads, opt_state, w)
        w_new = projection(optax.apply_updates(w, updates))
        delta = jnp.linalg.norm(w_new - w)
        return (i + 1, w_new, opt_state, delta)

    init = (jnp.asarray(0), w0, opt_state0, jnp.asarray(jnp.inf))
    i, w, _, delta = jax.lax.while_loop(cond, body, init)
    return w, {"iterations": i, "final_step": delta}


@jax.jit
def portfolio_return(weights: Array, mu: Array) -> Array:
    """Expected portfolio return ``w . mu``."""
    return jnp.dot(weights, mu)


@jax.jit
def portfolio_variance(weights: Array, cov: Array) -> Array:
    """Portfolio variance ``w' Sigma w``."""
    return weights @ cov @ weights


@jax.jit
def portfolio_volatility(weights: Array, cov: Array) -> Array:
    """Portfolio volatility ``sqrt(w' Sigma w)``."""
    return jnp.sqrt(jnp.clip(portfolio_variance(weights, cov), 1e-18, None))


@jax.jit
def sharpe_ratio(weights: Array, mu: Array, cov: Array, risk_free: float = 0.0) -> Array:
    """Sharpe ratio of a weight vector given moments."""
    excess = portfolio_return(weights, mu) - risk_free
    return excess / portfolio_volatility(weights, cov)
