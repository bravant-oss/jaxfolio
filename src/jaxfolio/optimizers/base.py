"""Projected-gradient solver shared by the constrained classical optimizers.

Every constrained objective (min-variance, mean-variance, max-Sharpe,
max-diversification, CVaR, ...) is minimized by the same routine: take a
gradient step on an unconstrained ``objective(w)``, then project ``w`` back onto
the feasible set. The loop runs inside ``jax.lax.while_loop`` so the whole solve
is a single jit-compiled kernel.

Two solvers are available:

* ``"spg"`` (default) — spectral projected gradient with Barzilai-Borwein step
  sizes. It needs no learning-rate tuning: the step adapts to the local
  curvature every iteration, so it converges to the constrained optimum
  (matching a dedicated QP solver) in far fewer iterations than a fixed-step
  method. Convergence is measured by the projected-gradient (KKT stationarity)
  norm ``||w - P(w - ∇f)||``.
* ``"adam"`` — the classic fixed-step optax-Adam projected-gradient loop. Its
  smooth dynamics are preferable when differentiating *through* the optimizer to
  train an allocation policy, and it handles the non-smooth CVaR objective (with
  its free auxiliary variable) more gracefully than a single scalar BB step.

Performance note: :func:`solve_constrained` is jit-cached on the *identity* of
the objective/projection functions, so calling a built-in optimizer repeatedly
(e.g. at every backtest rebalance) compiles once and reuses the kernel. The
public :func:`solve_projected_gradient` accepts an arbitrary Python closure and
therefore re-traces per call — fine for one-off custom strategies.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import optax

from jaxfolio.constraints.projections import project_box_budget, project_simplex

Array = jnp.ndarray
Objective = Callable[[Array], Array]

_ALPHA_MIN, _ALPHA_MAX = 1e-10, 1e10  # Barzilai-Borwein step-size safeguards


# --------------------------------------------------------------------------- #
# Projections
# --------------------------------------------------------------------------- #
def make_projection(
    long_only: bool,
    weight_bounds: tuple[float, float],
    budget: float = 1.0,
) -> Callable[[Array], Array]:
    """Return the ``w -> w`` projection matching the requested constraint set."""
    lo, hi = weight_bounds
    if long_only and lo <= 0.0 and hi >= 1.0:
        # Plain probability simplex is cheaper and exact.
        return lambda w: project_simplex(w, budget)
    return lambda w: project_box_budget(w, lo, hi, budget)


# Module-level ``projection(x, pparams)`` variants used by the *cached* kernel.
# Keeping them at module scope gives them a stable identity, so the jit cache
# hits across calls (a lambda would be a fresh object each call and re-compile).
# ``pparams`` is a traced tuple, so bounds/budget can change without recompiling.
def _proj_simplex(x: Array, pparams) -> Array:
    (budget,) = pparams
    return project_simplex(x, budget)


def _proj_box(x: Array, pparams) -> Array:
    lo, hi, budget = pparams
    return project_box_budget(x, lo, hi, budget)


def _proj_simplex_aux(z: Array, pparams) -> Array:
    """Project only the leading weight-block of a packed ``[w, aux]`` vector."""
    (budget,) = pparams
    return jnp.concatenate([project_simplex(z[:-1], budget), z[-1:]])


def _proj_box_aux(z: Array, pparams) -> Array:
    lo, hi, budget = pparams
    return jnp.concatenate([project_box_budget(z[:-1], lo, hi, budget), z[-1:]])


def select_projection(long_only: bool, weight_bounds: tuple[float, float], *, aux: bool = False):
    """Return ``(projection_fn, pparams)`` for the cached kernel.

    ``aux=True`` selects the variant that projects only the weight-block of a
    packed ``[w, tau]`` vector (used by the CVaR optimizer). ``pparams`` is a
    tuple of Python floats — passed as a traced argument to the kernel.
    """
    lo, hi = weight_bounds
    simplex = long_only and lo <= 0.0 and hi >= 1.0
    if simplex:
        return (_proj_simplex_aux if aux else _proj_simplex), (1.0,)
    return (_proj_box_aux if aux else _proj_box), (float(lo), float(hi), 1.0)


# --------------------------------------------------------------------------- #
# Solver loops (pure — emit JAX ops, run under jit or eagerly)
# --------------------------------------------------------------------------- #
def _estimate_lipschitz(f: Objective, w: Array, iters: int = 20) -> Array:
    """Power-iterate the Hessian of ``f`` at ``w`` to estimate ``L = λ_max(∇²f)``.

    Uses Hessian-vector products (``jvp`` of ``grad``), so it never materializes
    the Hessian and stays jit/vmap-safe. Returns a scalar used to size the
    Barzilai-Borwein solver's *initial* step (``1/L``); the BB updates take over
    from there, so a rough estimate is enough.
    """
    grad_f = jax.grad(f)

    def hvp(v: Array) -> Array:
        return jax.jvp(grad_f, (w,), (v,))[1]

    v0 = jnp.ones_like(w) / jnp.sqrt(w.size)

    def body(_, state):
        v, _lam = state
        hv = hvp(v)
        lam = jnp.linalg.norm(hv)
        return hv / (lam + 1e-18), lam

    _, lam = jax.lax.fori_loop(0, iters, body, (v0, jnp.asarray(1.0, dtype=w.dtype)))
    return lam


def _spg_loop(
    f: Objective,
    projection: Callable[[Array], Array],
    w0: Array,
    alpha0: Array,
    tol: Array,
    max_iter: int,
) -> tuple[Array, Array, Array]:
    """Spectral projected gradient (Barzilai-Borwein). Returns ``(w, iters, pgnorm)``."""
    grad_f = jax.grad(f)

    def pgnorm(w: Array, g: Array) -> Array:
        # Projected-gradient stationarity: zero iff w is a KKT point of the
        # constrained problem. Scale-free unit step makes this a proper stop test.
        return jnp.linalg.norm(w - projection(w - g))

    w = projection(w0)
    g = grad_f(w)

    def cond(state):
        i, _w, _g, _a, pg = state
        return jnp.logical_and(i < max_iter, pg > tol)

    def body(state):
        i, w, g, alpha, _pg = state
        w_new = projection(w - alpha * g)
        g_new = grad_f(w_new)
        s = w_new - w
        y = g_new - g
        sy = jnp.dot(s, y)
        ss = jnp.dot(s, s)
        # BB1 step; hold the previous step when curvature is non-positive.
        alpha_new = jnp.where(sy > 1e-18, jnp.clip(ss / sy, _ALPHA_MIN, _ALPHA_MAX), alpha)
        return (i + 1, w_new, g_new, alpha_new, pgnorm(w_new, g_new))

    init = (jnp.asarray(0), w, g, alpha0, pgnorm(w, g))
    i, w, _g, _a, pg = jax.lax.while_loop(cond, body, init)
    return w, i, pg


def _adam_loop(
    f: Objective,
    projection: Callable[[Array], Array],
    w0: Array,
    learning_rate: float,
    tol: Array,
    max_iter: int,
) -> tuple[Array, Array, Array]:
    """Fixed-step projected Adam. Returns ``(w, iters, final_step)``."""
    optimizer = optax.adam(learning_rate)
    grad_fn = jax.grad(f)

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
    return w, i, delta


def _run_solver(
    f: Objective,
    projection: Callable[[Array], Array],
    w0: Array,
    *,
    solver: str,
    learning_rate: float | None,
    max_iter: int,
    tol,
) -> tuple[Array, Array]:
    """Dispatch to the requested solver. Returns ``(weights, iterations)``."""
    tol = jnp.asarray(tol)
    if solver == "adam":
        lr = 1e-2 if learning_rate is None else learning_rate
        w, iters, _ = _adam_loop(f, projection, w0, lr, tol, max_iter)
        return w, iters
    if solver == "spg":
        if learning_rate is not None:
            alpha0 = jnp.asarray(learning_rate, dtype=w0.dtype)
        else:
            lipschitz = _estimate_lipschitz(f, projection(w0))
            alpha0 = jnp.clip(1.0 / (lipschitz + 1e-12), 1e-8, 1e6)
        w, iters, _ = _spg_loop(f, projection, w0, alpha0, tol, max_iter)
        return w, iters
    raise ValueError(f"unknown solver {solver!r}; expected 'spg' or 'adam'")


# --------------------------------------------------------------------------- #
# Cached kernel (built-ins) and public closure API (custom strategies)
# --------------------------------------------------------------------------- #
@partial(
    jax.jit,
    static_argnames=("objective", "projection", "solver", "learning_rate", "max_iter"),
)
def _solve_cached(
    objective,
    projection,
    obj_params,
    proj_params,
    w0,
    tol,
    *,
    solver: str = "spg",
    learning_rate: float | None = None,
    max_iter: int = 2000,
):
    """Jit-cached solve for built-in optimizers.

    ``objective(w, obj_params)`` and ``projection(x, proj_params)`` are
    module-level functions (static, stable identity); ``obj_params`` /
    ``proj_params`` / ``w0`` / ``tol`` are traced. The cache therefore keys on
    (objective kind, projection kind, solver, learning_rate, n-assets shape) —
    so repeated calls on same-shaped data reuse the compiled kernel.
    """
    f = lambda w: objective(w, obj_params)  # noqa: E731
    proj = lambda w: projection(w, proj_params)  # noqa: E731
    return _run_solver(
        f, proj, w0, solver=solver, learning_rate=learning_rate, max_iter=max_iter, tol=tol
    )


def solve_constrained(
    objective,
    obj_params,
    w0: Array,
    projection,
    proj_params,
    *,
    solver: str = "spg",
    learning_rate: float | None = None,
    max_iter: int = 2000,
    tol: float = 1e-7,
) -> tuple[Array, dict[str, Array]]:
    """Cached entry point for the built-in optimizers.

    ``objective`` and ``projection`` must be *module-level* functions (see
    :mod:`jaxfolio.optimizers.classical`) so the underlying jit cache hits.
    Returns ``(weights, info)`` with ``info["iterations"]``.
    """
    w, iters = _solve_cached(
        objective,
        projection,
        obj_params,
        proj_params,
        w0,
        jnp.asarray(tol),
        solver=solver,
        learning_rate=learning_rate,
        max_iter=max_iter,
    )
    return w, {"iterations": iters}


def solve_projected_gradient(
    objective: Objective,
    w0: Array,
    projection: Callable[[Array], Array],
    *,
    learning_rate: float | None = None,
    max_iter: int = 2000,
    tol: float = 1e-7,
    solver: str = "spg",
) -> tuple[Array, dict[str, Array]]:
    """Minimize ``objective`` over the feasible set defined by ``projection``.

    Accepts arbitrary Python closures ``objective(w) -> scalar`` and
    ``projection(w) -> w`` (the shape used by custom strategies and the
    ``toolkit``). Because the closure identity changes per call this re-traces
    each time — for hot loops over a built-in optimizer, prefer the cached
    :func:`solve_constrained`. Returns ``(weights, info)``.
    """
    w, iters = _run_solver(
        objective, projection, w0, solver=solver, learning_rate=learning_rate, max_iter=max_iter,
        tol=tol,
    )
    return w, {"iterations": iters}


# --------------------------------------------------------------------------- #
# Small jitted portfolio helpers (used across the package)
# --------------------------------------------------------------------------- #
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
