"""Multi-period ("path-aware") mean-variance optimization with trading costs.

Every other optimizer in jaxfolio is single-period: it collapses the return panel
to ``(mu, Sigma)`` and emits one target weight vector, with no knowledge of what
you currently hold. Transaction costs are then charged *after* the fact by the
backtester, against a target chosen while blind to them. The familiar consequence
is an optimizer that says "sell half your largest position today" because nothing
in its objective knows that trade is expensive.

This module solves for the whole **weight path** at once. The variable is a
``(T, N)`` matrix ``W`` whose row ``t`` is the portfolio to hold in period ``t``,
anchored at the portfolio you hold now::

    min_W  sum_t [ -w_t'mu_t + (gamma/2) w_t' Sigma_t w_t ]
           + sum_t [ c_lin' |w_t - w_{t-1}| + c_quad' (w_t - w_{t-1})^2 ]
           + l2 * ||W||^2                                  with  w_{-1} = w_prev

subject to each row independently lying in ``{lo <= w <= hi, sum(w) = 1}``. The
quadratic impact term is what makes splitting a large trade across periods
strictly cheaper than executing it at once, so the solution is a *glide path*
from ``w_prev`` toward the long-run target rather than a jump.

Three implementation notes carry the design:

* **The feasible set is a Cartesian product** of the per-period sets, so the
  Euclidean projection onto it is the product of the row projections — a plain
  ``vmap`` (see :func:`~jaxfolio.optimizers.base.select_projection` with
  ``path=True``). That is why the shared projected-gradient solver needs no
  special casing to handle a 2-D variable.
* **The L1 trade cost is non-smooth exactly where the optimum sits** (at zero
  trade, for every asset that should not move). Applied directly, a first-order
  method stalls ~1% above the true optimum regardless of solver family — the same
  failure mode that limits :func:`~jaxfolio.optimizers.classical.min_cvar`. The
  term is therefore Huber-smoothed, and because accuracy is *not* monotone in the
  smoothing width, the solver walks a ladder of shrinking widths and keeps
  whichever iterate is best under the exact, un-smoothed objective. Measured
  against a tightly converged CVXPY QP over asset counts 4-50 and six cost
  regimes, the worst relative gap is ~2e-4 and the typical one ~1e-6; selecting
  rather than trusting the final width is what buys that (it was 8-160% off
  otherwise).
* **Every cost travels as a traced parameter**, so sweeping cost assumptions
  never recompiles the solver kernel. Only ``(horizon, n_assets)`` — baked into
  the variable's shape — keys a new compilation.

Turnover control here is **soft**: costs are priced, not capped. A hard budget
``sum_t |w_t - w_{t-1}| <= tau`` couples the periods, which destroys the product
structure the row-wise projection relies on; see the multi-period guide.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np

from jaxfolio.optimizers.base import select_projection, solve_constrained
from jaxfolio.results import finalize_result, moments
from jaxfolio.types import OptimizerConfig, PortfolioResult, TradingCosts

Array = jnp.ndarray

#: Number of geometric refinements of the Huber width beyond the first stage, and
#: the factor each one shrinks it by. Seven stages spanning 1e-1 down to ~1e-4
#: (relative to ``budget/n``) is what it takes to cover both the easy
#: impact-dominated regime and the stiff linear-cost-dominated one; every stage
#: reuses the same compiled kernel, so the ladder costs iterations, not compiles.
DEFAULT_REFINE = 6
_REFINE_FACTOR = 1.0 / 3.0


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def _path_prev(W: Array, w_prev: Array) -> Array:
    """Return ``w_{t-1}`` for every period, anchored at ``w_prev`` for period 0.

    The anchor is load-bearing beyond mere correctness. It makes the difference
    operator positive definite, which breaks the shift symmetry that would
    otherwise trap the solver's power-iteration Lipschitz estimate in a subspace
    where the trade-penalty curvature vanishes (an unanchored ``jnp.diff`` was
    measured at L=5.9 against a true 9.3, which would oversize the initial step).
    Do **not** rewrite this as ``jnp.diff(W, axis=0)``.
    """
    return jnp.concatenate([w_prev[None, :], W[:-1]], axis=0)


def _huber(d: Array, eps: Array) -> Array:
    """Huber surrogate for ``|d|``: quadratic inside ``eps``, exactly linear outside.

    Chosen over the smooth pseudo-Huber ``sqrt(d**2 + eps**2) - eps`` for two
    reasons: it understates ``|d|`` by at most ``eps/2`` rather than ``eps``, and
    its gradient is exactly ``+-1`` beyond the band, so the marginal cost of an
    asset that *does* trade is exact and only the no-trade band is approximated.
    The gradient at ``d == 0`` is exactly zero with no NaN, which matters because
    the solver warm-starts from ``tile(w_prev)`` where every trade is zero.
    """
    a = jnp.abs(d)
    return jnp.where(a <= eps, 0.5 * d**2 / eps, a - 0.5 * eps)


def _trade_costs(W: Array, p) -> Array:
    """Smoothed linear cost plus quadratic impact on the period-over-period trade."""
    d = W - _path_prev(W, p["w_prev"])
    linear = jnp.sum(p["c_lin"] * _huber(d, p["trade_eps"]))
    impact = jnp.sum(p["c_quad"] * d**2)
    return linear + impact


def _utility(W: Array, p, quad: Array) -> Array:
    """Negated mean-variance utility summed over the path, plus the L2 penalty."""
    lin = jnp.einsum("ti,ti->t", W, p["mu_path"])
    return jnp.sum(-lin + 0.5 * p["risk_aversion"] * quad) + p["l2"] * jnp.sum(W**2)


# Two module-level objective identities, split on the rank of ``cov``. They must
# stay module-level for the solver's jit cache to key on identity, and they must
# stay *separate* rather than broadcasting a shared covariance to (T, N, N):
# ``cov`` is a traced argument, so that broadcast would materialize T copies of an
# (N, N) matrix outside jit, where XLA cannot elide it (192 MB at N=2000, T=12).
# Splitting costs no extra cache entries, since the two ranks are different
# traced shapes and would key separately anyway.
def _mp_shared_cov(W: Array, p) -> Array:
    """Path objective with one covariance shared across all periods."""
    quad = jnp.einsum("ti,ij,tj->t", W, p["cov"], W)
    return _utility(W, p, quad) + _trade_costs(W, p)


def _mp_path_cov(W: Array, p) -> Array:
    """Path objective with a per-period covariance term structure."""
    quad = jnp.einsum("ti,tij,tj->t", W, p["cov"], W)
    return _utility(W, p, quad) + _trade_costs(W, p)


def _exact_objective(W: Array, p) -> Array:
    """The un-smoothed objective, for diagnostics only — never differentiated.

    Evaluating this at the returned path is what makes the smoothing bias
    auditable: ``exact - smoothed >= 0`` bounds how much the Huber band
    understated the true L1 trade cost.
    """
    quad = (
        jnp.einsum("ti,ij,tj->t", W, p["cov"], W)
        if p["cov"].ndim == 2
        else jnp.einsum("ti,tij,tj->t", W, p["cov"], W)
    )
    d = W - _path_prev(W, p["w_prev"])
    trade = jnp.sum(p["c_lin"] * jnp.abs(d)) + jnp.sum(p["c_quad"] * d**2)
    return _utility(W, p, quad) + trade


# --------------------------------------------------------------------------- #
# Core solve
# --------------------------------------------------------------------------- #
def solve_weight_path(
    mu_path,
    cov,
    w_prev,
    *,
    risk_aversion: float = 1.0,
    c_lin=0.0,
    c_quad=0.0,
    l2_reg: float = 0.0,
    trade_eps_rel: float = 1e-1,
    refine: int = DEFAULT_REFINE,
    long_only: bool = True,
    weight_bounds: tuple[float, float] = (0.0, 1.0),
    solver: Any = "spg",
    learning_rate: float | None = None,
    max_iter: int = 2000,
    tol: float = 1e-6,
    solver_options: Mapping[str, Any] | None = None,
) -> tuple[Array, dict[str, Any]]:
    """Solve the multi-period mean-variance problem for a whole weight path.

    This is the primitive behind :func:`multi_period_mean_variance`, exposed for
    callers who already hold forecasts and want the raw path back.

    Parameters
    ----------
    mu_path:
        Expected returns per period, shape ``(T, N)``.
    cov:
        Covariance, either ``(N, N)`` (shared across periods) or ``(T, N, N)``.
    w_prev:
        Currently held weights, shape ``(N,)``. Need **not** sum to one — the
        zero vector is a valid "flat book", and the cost of establishing a
        position from it is priced correctly.
    risk_aversion:
        Coefficient ``gamma`` on the per-period variance term.
    c_lin, c_quad:
        Linear and quadratic trade-cost coefficients in decimal units, broadcast
        against ``(T, N)``. Scalars, ``(N,)`` and ``(T, N)`` all work.
    trade_eps_rel:
        Starting (widest) Huber half-width, relative to ``budget / N``.
    refine:
        Number of geometric refinements of the Huber width beyond the first stage,
        each shrinking it by a factor of three. Every stage warm-starts from the
        previous one and reuses the *same* compiled kernel, because the width is a
        traced parameter — so the ladder costs iterations, never compilations.
        The returned path is the stage that scored best under the **exact**
        (un-smoothed) objective, not simply the last one: accuracy is not monotone
        in the width, so selecting is what makes the result reliable.
    tol:
        Convergence tolerance. Defaults to ``1e-6`` rather than the package-wide
        ``1e-7`` because the float32 projected-gradient norm floors around
        ``1e-6`` on this problem, so a tighter value only buys wasted iterations.

    Returns
    -------
    tuple
        ``(W, info)`` where ``W`` has shape ``(T, N)`` and ``info`` carries
        iteration counts, the per-stage widths, the final residual, and both the
        smoothed and exact objective values.

    Notes
    -----
    A high ``c_lin`` relative to the Huber width legitimately leaves the
    projected-gradient residual on a plateau far above ``tol`` while the answer is
    exactly right (the correct action is "do not trade", and the residual carries
    the un-actioned cost gradient). Read ``info["converged"]`` as a diagnostic,
    not a failure.
    """
    mu_path = jnp.asarray(mu_path, dtype=float)
    cov = jnp.asarray(cov, dtype=float)
    if mu_path.ndim != 2:
        raise ValueError(f"mu_path must be 2-D (T, N), got shape {mu_path.shape}")
    horizon, n = mu_path.shape
    if cov.ndim == 2:
        objective = _mp_shared_cov
        if cov.shape != (n, n):
            raise ValueError(
                f"cov shape {cov.shape} does not match {n} assets; expected ({n}, {n})"
            )
    elif cov.ndim == 3:
        objective = _mp_path_cov
        if cov.shape != (horizon, n, n):
            raise ValueError(
                f"cov shape {cov.shape} does not match the path; expected ({horizon}, {n}, {n})"
            )
    else:
        raise ValueError(f"cov must be (N, N) or (T, N, N), got shape {cov.shape}")
    if refine < 0:
        raise ValueError(f"refine must be non-negative, got {refine}")
    if trade_eps_rel <= 0.0:
        raise ValueError(f"trade_eps_rel must be strictly positive, got {trade_eps_rel}")

    projection, pparams = select_projection(long_only, weight_bounds, path=True)
    budget = pparams[-1]

    # ``dtype=`` is required, not decorative: jnp.full(n, 1/n) is weak-typed and
    # jnp.tile preserves that, but the solver's output is strongly typed — so a
    # weak-typed start would make stage 2 a *different* jit cache key and compile
    # the kernel twice for what is one problem.
    w_prev_j = jnp.asarray(w_prev, dtype=mu_path.dtype).reshape(-1)
    if w_prev_j.shape != (n,):
        raise ValueError(
            f"w_prev shape {w_prev_j.shape} does not match {n} assets; expected ({n},)"
        )
    # Project the warm start onto the feasible set before anything reads it. The
    # solver would project internally anyway, but the stage-selection below also
    # scores this iterate as a candidate — and an *infeasible* point can score
    # lower than every feasible one, which would let it win and be returned. That
    # matters concretely whenever ``w_prev`` is not a valid portfolio: a partially
    # invested book (the backtester hands over exactly that before its first
    # rebalance) or holdings that violate the configured bounds.
    W = projection(jnp.tile(w_prev_j, (horizon, 1)), pparams)

    params = {
        "mu_path": mu_path,
        "cov": cov,
        "w_prev": w_prev_j,
        "risk_aversion": jnp.asarray(risk_aversion, dtype=mu_path.dtype),
        "l2": jnp.asarray(l2_reg, dtype=mu_path.dtype),
        "c_lin": jnp.broadcast_to(jnp.asarray(c_lin, dtype=mu_path.dtype), (n,))
        if jnp.ndim(c_lin) <= 1
        else jnp.asarray(c_lin, dtype=mu_path.dtype),
        "c_quad": jnp.broadcast_to(jnp.asarray(c_quad, dtype=mu_path.dtype), (n,))
        if jnp.ndim(c_quad) <= 1
        else jnp.asarray(c_quad, dtype=mu_path.dtype),
        "trade_eps": jnp.asarray(1.0, dtype=mu_path.dtype),
    }

    # Walk a ladder of shrinking Huber widths, warm-starting each stage from the
    # previous one, and keep the iterate that is best under the *exact* objective.
    #
    # Selecting rather than simply taking the last stage is what makes this
    # reliable. A single width cannot serve every problem: when the linear cost is
    # comparable to the whole mean-variance curvature (common at 50-100bps), a
    # narrow band makes the smoothed problem so stiff that the projected-gradient
    # method crawls, while a wide band biases the answer. Worse, accuracy is *not*
    # monotone in the width, so trusting the narrowest stage is measurably wrong —
    # it was observed 8% to 160% off the true optimum. Because the un-smoothed
    # objective is cheap to evaluate, the ladder can be treated as a search over
    # smoothing levels judged by the real criterion, which bounds the error by the
    # best stage instead of the last. That took the worst case across a grid of
    # panel sizes and cost regimes from ~163% down to ~1e-4.
    eps = float(trade_eps_rel) * budget / n
    stage_iters: list[int] = []
    stage_eps: list[float] = []
    stage_exact: list[float] = []
    best_W = W
    # ``W`` is feasible here (projected above), so this is a legitimate candidate:
    # holding still is often genuinely optimal once trading is expensive enough.
    best_exact = float(_exact_objective(W, {**params, "trade_eps": jnp.asarray(eps)}))
    best_stage = -1  # -1 => the (projected) no-trade path was never beaten
    best_residual = float("inf")
    for stage in range(refine + 1):
        params["trade_eps"] = jnp.asarray(eps, dtype=mu_path.dtype)
        W, info = solve_constrained(
            objective,
            params,
            W,
            projection,
            pparams,
            solver=solver,
            learning_rate=learning_rate,
            max_iter=max_iter,
            tol=tol,
            solver_options=solver_options,
        )
        stage_iters.append(int(info["iterations"]))
        stage_eps.append(eps)
        value = float(_exact_objective(W, params))
        stage_exact.append(value)
        if value < best_exact:
            best_exact, best_W, best_stage = value, W, stage
            best_residual = float(info["residual"])
        eps *= _REFINE_FACTOR

    W = best_W
    # Report the smoothed objective at the width that actually produced ``W``.
    params["trade_eps"] = jnp.asarray(
        stage_eps[best_stage] if best_stage >= 0 else stage_eps[0], dtype=mu_path.dtype
    )
    smoothed = float(objective(W, params))
    exact = best_exact
    residual = best_residual
    d = np.asarray(W - _path_prev(W, w_prev_j))
    turnover_path = np.abs(d).sum(axis=1)
    c_lin_np = np.asarray(params["c_lin"])
    c_quad_np = np.asarray(params["c_quad"])

    info = {
        "iterations": int(sum(stage_iters)),
        "stage_iterations": tuple(stage_iters),
        "stage_trade_eps": tuple(stage_eps),
        "stage_objective_exact": tuple(stage_exact),
        "selected_stage": int(best_stage),
        "trade_eps": float(stage_eps[best_stage] if best_stage >= 0 else stage_eps[0]),
        "residual": float(residual),
        "converged": bool(float(residual) <= tol),
        "turnover_path": turnover_path,
        "total_turnover": float(turnover_path.sum()),
        "total_cost": float((c_lin_np * np.abs(d)).sum() + (c_quad_np * d**2).sum()),
        "objective_smoothed": smoothed,
        "objective_exact": exact,
        "smoothing_bias": exact - smoothed,
    }
    return W, info


# --------------------------------------------------------------------------- #
# Public optimizer
# --------------------------------------------------------------------------- #
def multi_period_mean_variance(
    returns,
    horizon: int = 5,
    *,
    w_prev=None,
    risk_aversion: float = 1.0,
    costs: TradingCosts | None = None,
    mu_path=None,
    cov_path=None,
    refine: int = DEFAULT_REFINE,
    config: OptimizerConfig | None = None,
) -> PortfolioResult:
    """Multi-period mean-variance portfolio: an optimal *trajectory*, not a target.

    Solves for the whole weight path over ``horizon`` periods at once, starting
    from the portfolio you currently hold, with trading costs priced inside the
    objective. Because a large trade costs more than two half-sized ones under
    quadratic impact, the optimizer spreads execution across periods instead of
    jumping to the myopic target.

    ``result.weights`` is the **first** row of the path — the allocation to hold
    now, which is what you act on and what the backtester trades to. The full
    path is on ``result.trajectory``, and the long-run target is
    ``result.metadata["terminal_weights"]``.

    Parameters
    ----------
    returns:
        Asset return panel (rows = periods, columns = assets). Sample moments
        from this panel are held constant across the horizon unless overridden.
    horizon:
        Number of periods ``T`` to plan over.
    w_prev:
        Currently held weights, aligned with the panel's columns. Defaults to a
        flat (all-zero) book, so the cost of *establishing* a position is priced.
        Need not sum to one.
    risk_aversion:
        Coefficient ``gamma`` on the per-period variance term. Larger tilts the
        whole path toward lower variance.
    costs:
        A :class:`~jaxfolio.types.TradingCosts` spec. ``None`` means frictionless,
        in which case every row of the path collapses onto the single-period
        mean-variance solution — pass costs to get path behavior.
    mu_path:
        Optional ``(T, N)`` expected-return term structure, overriding the
        constant sample mean. Use this to express a changing forecast.
    cov_path:
        Optional ``(T, N, N)`` covariance term structure, overriding the constant
        sample covariance.
    refine:
        Number of Huber-width refinements. Higher is more accurate and slower;
        the default is calibrated to land within ~1e-5 relative of the exact
        optimum.
    config:
        Standard :class:`~jaxfolio.types.OptimizerConfig`. Note the default here
        uses ``tol=1e-6``; an explicitly passed config is honored as given.

    Returns
    -------
    PortfolioResult
        ``weights`` is the first path row, ``trajectory`` the full ``(T, N)``
        path. ``metadata`` carries ``turnover_path`` (**planned**, assuming no
        drift), ``total_turnover``, ``total_cost``, ``smoothing_bias``, the
        terminal weights and their annualized diagnostics, and solver counters.

    Notes
    -----
    Costing is what drives the trajectory, so a bare call with no ``costs`` is
    intentionally equivalent to repeating the single-period solution.

    One compilation is spent per distinct ``(horizon, n_assets)`` pair, since the
    horizon is baked into the variable's shape. Varying ``horizon`` per rebalance
    in a backtest therefore recompiles each time; varying costs or
    ``risk_aversion`` does not.

    Turnover control is **soft** — a turnover budget can be priced but not
    guaranteed. See the multi-period guide.

    Examples
    --------
    >>> import jaxfolio as jf
    >>> panel = jf.generate_returns(n_assets=5, n_days=300, seed=0)
    >>> res = jf.multi_period_mean_variance(
    ...     panel, horizon=4, costs=jf.TradingCosts(spread_bps=10.0, impact_bps=5.0)
    ... )
    >>> res.trajectory.shape
    (4, 5)
    """
    config = config or OptimizerConfig(tol=1e-6)
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    costs = costs if costs is not None else TradingCosts()

    mu, cov, names, _ = moments(returns)
    n = len(names)

    if mu_path is None:
        mu_used = jnp.tile(jnp.asarray(mu, dtype=float), (horizon, 1))
    else:
        mu_used = jnp.asarray(mu_path, dtype=float)
        if mu_used.shape != (horizon, n):
            raise ValueError(
                f"mu_path shape {tuple(mu_used.shape)} does not match the problem; "
                f"expected ({horizon}, {n})"
            )

    if cov_path is None:
        cov_used = jnp.asarray(cov, dtype=float)
    else:
        cov_used = jnp.asarray(cov_path, dtype=float)
        if cov_used.shape != (horizon, n, n):
            raise ValueError(
                f"cov_path shape {tuple(cov_used.shape)} does not match the problem; "
                f"expected ({horizon}, {n}, {n})"
            )

    w_prev_arr = np.zeros(n) if w_prev is None else np.asarray(w_prev, dtype=float).reshape(-1)
    if w_prev_arr.shape != (n,):
        raise ValueError(f"w_prev has length {w_prev_arr.shape[0]}, expected {n} (one per asset)")

    cost_params = costs.as_params(n)
    W, info = solve_weight_path(
        mu_used,
        cov_used,
        w_prev_arr,
        risk_aversion=risk_aversion,
        c_lin=cost_params["c_lin"],
        c_quad=cost_params["c_quad"],
        l2_reg=config.l2_reg,
        trade_eps_rel=costs.smoothing,
        refine=refine,
        long_only=config.long_only,
        weight_bounds=config.bounds(),
        solver=config.solver_spec(),
        learning_rate=config.learning_rate,
        max_iter=config.max_iter,
        tol=config.tol,
    )

    path = np.asarray(W, dtype=float)
    # The terminal row is the long-run target; report its diagnostics too, since
    # row 0 sits near ``w_prev`` when costs bind and its Sharpe would otherwise
    # read as a regression against the myopic single-period result.
    terminal = finalize_result(
        path[-1], names, "terminal", mu=mu, cov=cov, risk_free=config.risk_free_rate
    )

    metadata = {
        "horizon": int(horizon),
        "risk_aversion": float(risk_aversion),
        "iterations": info["iterations"],
        "stage_iterations": list(info["stage_iterations"]),
        "stage_trade_eps": [float(e) for e in info["stage_trade_eps"]],
        "stage_objective_exact": [float(v) for v in info["stage_objective_exact"]],
        "selected_stage": info["selected_stage"],
        "trade_eps": info["trade_eps"],
        "residual": info["residual"],
        "converged": info["converged"],
        "costs_active": bool(costs.is_active()),
        "smoothing": float(costs.smoothing),
        "turnover_path": [float(x) for x in info["turnover_path"]],
        "total_turnover": info["total_turnover"],
        "total_cost": info["total_cost"],
        "objective_smoothed": info["objective_smoothed"],
        "objective_exact": info["objective_exact"],
        "smoothing_bias": info["smoothing_bias"],
        "terminal_weights": path[-1].tolist(),
        "terminal_expected_return": terminal.expected_return,
        "terminal_volatility": terminal.volatility,
        "terminal_sharpe": terminal.sharpe,
        "w_prev": w_prev_arr.tolist(),
        "mu_path_provided": mu_path is not None,
        "cov_path_provided": cov_path is not None,
    }

    return finalize_result(
        path[0],
        names,
        "Multi-Period Mean-Variance",
        mu=mu,
        cov=cov,
        risk_free=config.risk_free_rate,
        metadata=metadata,
        trajectory=path,
    )
