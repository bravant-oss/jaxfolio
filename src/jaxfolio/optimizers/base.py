"""Projected-gradient solver shared by the constrained classical optimizers.

Every constrained objective (min-variance, mean-variance, max-Sharpe,
max-diversification, CVaR, ...) is minimized by the same routine: take a
gradient step on an unconstrained ``objective(w)``, then project ``w`` back onto
the feasible set. The loop runs inside ``jax.lax.while_loop`` so the whole solve
is a single jit-compiled kernel.

Two families of solver are available:

* ``"spg"`` (default) — spectral projected gradient with Barzilai-Borwein step
  sizes. It needs no learning-rate tuning: the step adapts to the local
  curvature every iteration, so it converges to the constrained optimum
  (matching a dedicated QP solver) in far fewer iterations than a fixed-step
  method. Convergence is measured by the projected-gradient (KKT stationarity)
  norm ``||w - P(w - ∇f)||``.
* **any optax optimizer** — by name (``"adam"``, ``"adamw"``, ``"sgd"``,
  ``"rmsprop"``, ...), by factory (``optax.adamw``), with hyperparameters passed
  as ``solver_options={"weight_decay": 1e-3}``. See :mod:`jaxfolio.solvers`.
  These run a fixed-step projected loop whose smooth dynamics are preferable
  when differentiating *through* the optimizer to train an allocation policy,
  and which handle the non-smooth CVaR objective (with its free auxiliary
  variable) more gracefully than a single scalar BB step. Convergence is
  measured by the weight-update norm ``||w_{k+1} - w_k||`` — a proxy for
  optimality rather than a KKT test, so optimizers whose step does not shrink
  near the optimum (``sign_sgd``, ``lion``, plain ``sgd``) may run to
  ``max_iter`` or stall early when the budget projection cancels a uniform step.

Performance note: :func:`solve_constrained` is jit-cached on the *identity* of
the objective/projection functions, so calling a built-in optimizer repeatedly
(e.g. at every backtest rebalance) compiles once and reuses the kernel. The
public :func:`solve_projected_gradient` accepts an arbitrary Python closure and
therefore re-traces per call — fine for one-off custom strategies.

The solver is part of that cache key, which is why it travels as a
:class:`~jaxfolio.solvers.SolverSpec` (a hashable, equality-stable tuple) rather
than as a live ``optax.GradientTransformation``: names and module-level factories
hash identically across calls, a freshly built transformation does not. Changing
a ``solver_options`` value costs exactly one recompile.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import optax

from jaxfolio.constraints.projections import project_box_budget, project_simplex
from jaxfolio.solvers import SolverSpec, build_optimizer, resolve_solver

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


def _proj_simplex_path(W: Array, pparams) -> Array:
    """Project each *row* of a ``(T, N)`` weight path onto the simplex.

    The multi-period feasible set is the Cartesian product of the per-period
    sets, and the projection onto a product is the product of the projections —
    hence a plain row-wise ``vmap``. ``project_simplex``'s internal
    ``v.shape[0]`` is the *post-vmap* row length ``N``, which stays static, so no
    shape information has to travel in ``pparams``.
    """
    (budget,) = pparams
    return jax.vmap(project_simplex, in_axes=(0, None))(W, budget)


def _proj_box_path(W: Array, pparams) -> Array:
    """Row-wise box+budget projection for a ``(T, N)`` weight path."""
    lo, hi, budget = pparams
    return jax.vmap(project_box_budget, in_axes=(0, None, None, None))(W, lo, hi, budget)


def select_projection(
    long_only: bool,
    weight_bounds: tuple[float, float],
    *,
    aux: bool = False,
    path: bool = False,
):
    """Return ``(projection_fn, pparams)`` for the cached kernel.

    ``aux=True`` selects the variant that projects only the weight-block of a
    packed ``[w, tau]`` vector (used by the CVaR optimizer). ``path=True``
    selects the row-wise variant for a ``(T, N)`` weight path (used by the
    multi-period optimizer). ``pparams`` is a tuple of Python floats — passed as
    a traced argument to the kernel, and identical for all three variants.
    """
    if aux and path:
        raise ValueError(
            "select_projection: aux and path are mutually exclusive — aux packs a "
            "scalar tail onto a 1-D vector, path batches rows of a 2-D weight path"
        )
    lo, hi = weight_bounds
    simplex = long_only and lo <= 0.0 and hi >= 1.0
    if simplex:
        if path:
            return _proj_simplex_path, (1.0,)
        return (_proj_simplex_aux if aux else _proj_simplex), (1.0,)
    bounds = (float(lo), float(hi), 1.0)
    if path:
        return _proj_box_path, bounds
    return (_proj_box_aux if aux else _proj_box), bounds


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
    """Spectral projected gradient (Barzilai-Borwein). Returns ``(w, iters, pgnorm)``.

    The loop is *rank-agnostic*: the variable may be a 1-D weight vector or an
    ``(T, N)`` weight path. Inner products are taken on the flattened variable and
    ``jnp.linalg.norm`` is the Frobenius (= flattened Euclidean) norm, so both
    the Barzilai-Borwein step and the stopping test read the same on either
    shape. ``jnp.vdot(x.ravel(), y.ravel())`` is used rather than
    ``jnp.sum(x * y)`` because it lowers to the same ``dot_general`` the previous
    ``jnp.dot(x, y)`` did and is therefore bit-for-bit identical for 1-D inputs;
    the multiply-then-reduce spelling differs in the last bits at small ``n`` and
    would perturb existing solutions.
    """
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
        sy = jnp.vdot(s.ravel(), y.ravel())
        ss = jnp.vdot(s.ravel(), s.ravel())
        # BB1 step; hold the previous step when curvature is non-positive.
        alpha_new = jnp.where(sy > 1e-18, jnp.clip(ss / sy, _ALPHA_MIN, _ALPHA_MAX), alpha)
        return (i + 1, w_new, g_new, alpha_new, pgnorm(w_new, g_new))

    init = (jnp.asarray(0), w, g, alpha0, pgnorm(w, g))
    i, w, _g, _a, pg = jax.lax.while_loop(cond, body, init)
    return w, i, pg


def _optax_loop(
    f: Objective,
    projection: Callable[[Array], Array],
    w0: Array,
    optimizer: optax.GradientTransformation,
    tol: Array,
    max_iter: int,
) -> tuple[Array, Array, Array]:
    """Fixed-step projected optax loop. Returns ``(w, iters, final_step)``."""
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
    solver: str | Callable[..., Any] | SolverSpec,
    learning_rate: float | None,
    max_iter: int,
    tol,
    solver_options: Mapping[str, Any] | None = None,
) -> tuple[Array, Array, Array]:
    """Dispatch to the requested solver. Returns ``(weights, iterations, residual)``.

    ``residual`` is the solver's own convergence measure — the projected-gradient
    (KKT stationarity) norm for ``"spg"``, the weight-update norm for the optax
    loops — so callers can report whether ``tol`` was actually reached rather
    than only how many iterations were spent.
    """
    tol = jnp.asarray(tol)
    spec = resolve_solver(solver, solver_options)  # idempotent on a SolverSpec
    if spec.kind == "spg":
        if learning_rate is not None:
            alpha0 = jnp.asarray(learning_rate, dtype=w0.dtype)
        else:
            lipschitz = _estimate_lipschitz(f, projection(w0))
            alpha0 = jnp.clip(1.0 / (lipschitz + 1e-12), 1e-8, 1e6)
        return _spg_loop(f, projection, w0, alpha0, tol, max_iter)
    optimizer = build_optimizer(spec, learning_rate)
    return _optax_loop(f, projection, w0, optimizer, tol, max_iter)


# --------------------------------------------------------------------------- #
# Cached kernel (built-ins) and public closure API (custom strategies)
# --------------------------------------------------------------------------- #
_SPG = resolve_solver("spg")  # module-level singleton → one stable default cache key


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
    solver: SolverSpec = _SPG,
    learning_rate: float | None = None,
    max_iter: int = 2000,
):
    """Jit-cached solve for built-in optimizers.

    ``objective(w, obj_params)`` and ``projection(x, proj_params)`` are
    module-level functions (static, stable identity); ``obj_params`` /
    ``proj_params`` / ``w0`` / ``tol`` are traced. The cache therefore keys on
    (objective kind, projection kind, solver, learning_rate, n-assets shape) —
    so repeated calls on same-shaped data reuse the compiled kernel.

    ``solver`` must already be a :class:`~jaxfolio.solvers.SolverSpec`: the public
    wrappers resolve it before crossing the jit boundary so the static key is
    always hashable and equality-stable.
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
    solver: str | Callable[..., Any] | SolverSpec = "spg",
    learning_rate: float | None = None,
    max_iter: int = 2000,
    tol: float = 1e-7,
    solver_options: Mapping[str, Any] | None = None,
) -> tuple[Array, dict[str, Array]]:
    """Cached entry point for the built-in optimizers.

    ``objective`` and ``projection`` must be *module-level* functions (see
    :mod:`jaxfolio.optimizers.classical`) so the underlying jit cache hits.
    ``solver`` is ``"spg"``, any optax optimizer name or factory, or a
    pre-resolved :class:`~jaxfolio.solvers.SolverSpec`.
    Returns ``(weights, info)`` with ``info["iterations"]`` and
    ``info["residual"]`` (the solver's convergence measure at the final iterate).
    """
    w, iters, resid = _solve_cached(
        objective,
        projection,
        obj_params,
        proj_params,
        w0,
        jnp.asarray(tol),
        solver=resolve_solver(solver, solver_options),
        learning_rate=learning_rate,
        max_iter=max_iter,
    )
    return w, {"iterations": iters, "residual": resid}


def solve_projected_gradient(
    objective: Objective,
    w0: Array,
    projection: Callable[[Array], Array],
    *,
    learning_rate: float | None = None,
    max_iter: int = 2000,
    tol: float = 1e-7,
    solver: str | Callable[..., Any] | SolverSpec = "spg",
    solver_options: Mapping[str, Any] | None = None,
) -> tuple[Array, dict[str, Array]]:
    """Minimize ``objective`` over the feasible set defined by ``projection``.

    Accepts arbitrary Python closures ``objective(w) -> scalar`` and
    ``projection(w) -> w`` (the shape used by custom strategies and the
    ``toolkit``). Because the closure identity changes per call this re-traces
    each time — for hot loops over a built-in optimizer, prefer the cached
    :func:`solve_constrained`. ``solver`` is ``"spg"``, any optax optimizer name
    or factory, or a pre-resolved :class:`~jaxfolio.solvers.SolverSpec`.
    Returns ``(weights, info)`` with ``info["iterations"]`` and
    ``info["residual"]``.
    """
    w, iters, resid = _run_solver(
        objective,
        projection,
        w0,
        solver=solver,
        learning_rate=learning_rate,
        max_iter=max_iter,
        tol=tol,
        solver_options=solver_options,
    )
    return w, {"iterations": iters, "residual": resid}


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
