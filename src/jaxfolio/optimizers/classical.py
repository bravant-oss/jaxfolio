"""Traditional / classical portfolio optimizers.

All optimizers share the ``method(returns, ...) -> PortfolioResult`` shape and
delegate constrained problems to the projected-gradient solver in
:mod:`jaxfolio.optimizers.base`. Objectives are defined as *module-level*
functions ``objective(w, params)`` (rather than per-call closures) so the
solver's jit cache hits across repeated calls — e.g. every rebalance of a
backtest recompiles nothing after the first solve.

Closed-form / exact solvers are used where they exist (inverse-volatility,
risk-parity coordinate descent) and reported alongside the constrained result.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jaxfolio.constraints.compile import compile_constraints
from jaxfolio.moments.estimators import (
    as_matrix,
    mean_returns,
    sample_covariance,
)
from jaxfolio.optimizers.base import (
    sharpe_ratio,
    solve_constrained,
)
from jaxfolio.results import finalize_result, moments
from jaxfolio.solvers import resolve_solver
from jaxfolio.types import OptimizerConfig, PortfolioResult

Array = jnp.ndarray


# --------------------------------------------------------------------------- #
# Module-level objectives — objective(w, params) -> scalar to minimize.
# ``params`` holds only traced arrays, so changing l2/risk-aversion/rf never
# forces a recompile; only the objective identity and n-assets shape do.
# --------------------------------------------------------------------------- #
def _minvar_objective(w: Array, p) -> Array:
    return w @ p["cov"] @ w + p["l2"] * jnp.sum(w**2)


def _meanvar_objective(w: Array, p) -> Array:
    util = w @ p["mu"] - 0.5 * p["risk_aversion"] * (w @ p["cov"] @ w)
    return -util + p["l2"] * jnp.sum(w**2)


def _sharpe_objective(w: Array, p) -> Array:
    return -sharpe_ratio(w, p["mu"], p["cov"], p["rf"]) + p["l2"] * jnp.sum(w**2)


def _maxdiv_objective(w: Array, p) -> Array:
    weighted_vol = w @ p["vol"]
    port_vol = jnp.sqrt(jnp.clip(w @ p["cov"] @ w, 1e-18, None))
    return -(weighted_vol / port_vol) + p["l2"] * jnp.sum(w**2)


def _kelly_objective(w: Array, p) -> Array:
    growth = jnp.log1p(jnp.clip(p["mat"] @ w, -0.999, None))
    return -jnp.mean(growth) + p["l2"] * jnp.sum(w**2)


def _cvar_objective(z: Array, p) -> Array:
    # z packs (weights, tau); project only the weight-block (aux projection).
    w = z[:-1]
    tau = z[-1]
    losses = -(p["mat"] @ w)  # loss = negative return
    cvar = tau + p["scale"] * jnp.sum(jnp.maximum(losses - tau, 0.0))
    return cvar + p["l2"] * jnp.sum(w**2)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _finalize(
    w: Array,
    names: list[str],
    method: str,
    mu: Array,
    cov: Array,
    risk_free: float,
    metadata: dict | None = None,
    attribution=None,
) -> PortfolioResult:
    """Assemble a PortfolioResult with annualized diagnostics (see toolkit)."""
    return finalize_result(
        w,
        names,
        method,
        mu=mu,
        cov=cov,
        risk_free=risk_free,
        metadata=metadata,
        attribution=attribution,
    )


def _moments(returns, cov_estimator=None):
    """Return ``(mu, cov, names, matrix)`` from a returns object."""
    return moments(returns, cov_estimator)


def _solver_kwargs(config: OptimizerConfig) -> dict:
    """Common solver settings pulled from an OptimizerConfig."""
    return {
        "solver": config.solver_spec(),
        "learning_rate": config.learning_rate,
        "max_iter": config.max_iter,
        "tol": config.tol,
    }


def _cfg(config: OptimizerConfig | None, constraints) -> OptimizerConfig:
    """Resolve the ``config`` / ``constraints`` pair into a single config.

    ``constraints=`` is sugar for ``config.constraints``; supplying both is an
    error rather than a silent precedence rule, because which one wins would be a
    coin flip to the reader.
    """
    config = config or OptimizerConfig()
    if constraints is None:
        return config
    if config.constraints:
        raise ValueError(
            "constraints were given both on OptimizerConfig and as a keyword argument; "
            "pass one or the other"
        )
    return config.with_constraints(constraints)


def _compile(config: OptimizerConfig, names: list[str]):
    """Resolve ``config``'s constraint set against the asset universe."""
    return compile_constraints(
        config.constraints,
        names,
        long_only=config.long_only,
        weight_bounds=config.bounds(),
    )


def _projection(config: OptimizerConfig, names: list[str], *, aux: bool = False):
    """Return ``(projection, pparams, compiled)`` honoring any named constraints.

    Always goes through the compiler, even with no named constraints: the compiler
    guarantees that case returns the *identical* projection object and Python-float
    tuple the legacy path did, and having ``compiled`` in hand is what lets the
    attribution payload name the constraints afterwards.
    """
    compiled = _compile(config, names)
    projection, pparams = compiled.projection(aux=aux)
    return projection, pparams, compiled


def _duals(
    config, objective, params, compiled, w, *, name, sense, smooth=True, convex=True, aux_dim=0
):
    """Capture the KKT payload for ``explain``, unless the caller opted out."""
    if not config.attribution:
        return None
    from jaxfolio.attribution import solver_duals

    return solver_duals(
        objective,
        params,
        compiled,
        w,
        objective_name=name,
        sense=sense,
        smooth=smooth,
        convex=convex,
        l2_reg=config.l2_reg,
        solver_kind=config.solver_spec().kind,
        tol=config.tol,
        aux_dim=aux_dim,
    )


def _reject_constraints(config: OptimizerConfig, method: str, alternative: str) -> None:
    """Raise if named constraints were passed to a closed-form optimizer.

    These methods build their weights analytically rather than solving a
    constrained program, so a cap could not be honored. Failing loudly beats
    returning weights that quietly violate it.
    """
    if config.constraints:
        named = ", ".join(repr(c.name) for c in config.constraints)
        raise NotImplementedError(
            f"{method} is a closed-form weighting and cannot honor named constraints "
            f"({named}). It has no optimization program for a cap to constrain. "
            f"Use {alternative} instead, which solves a constrained problem."
        )


# --------------------------------------------------------------------------- #
# Heuristic / closed-form weightings
# --------------------------------------------------------------------------- #
def equal_weight(returns) -> PortfolioResult:
    """Equal-weight (1/N) portfolio — a famously hard-to-beat baseline."""
    mu, cov, names, _ = _moments(returns)
    n = len(names)
    w = jnp.full(n, 1.0 / n)
    return _finalize(w, names, "Equal Weight", mu, cov, 0.0)


def inverse_volatility(returns) -> PortfolioResult:
    """Inverse-volatility (naive risk parity) weighting."""
    mu, cov, names, _ = _moments(returns)
    vol = jnp.sqrt(jnp.clip(jnp.diag(cov), 1e-18, None))  # floor guards zero-variance assets
    inv = 1.0 / vol
    w = inv / jnp.sum(inv)
    return _finalize(w, names, "Inverse Volatility", mu, cov, 0.0)


def minimum_variance(
    returns,
    config: OptimizerConfig | None = None,
    *,
    constraints=None,
) -> PortfolioResult:
    """Global minimum-variance portfolio.

    Solves ``min w' Sigma w`` subject to the configured constraints via the
    spectral projected-gradient solver, which converges to the exact
    minimum-variance frontier vertex.
    """
    config = _cfg(config, constraints)
    mu, cov, names, _ = _moments(returns)
    n = len(names)

    projection, pparams, compiled = _projection(config, names)
    params = {"cov": cov, "l2": jnp.asarray(config.l2_reg)}
    w, info = solve_constrained(
        _minvar_objective,
        params,
        jnp.full(n, 1.0 / n),
        projection,
        pparams,
        **_solver_kwargs(config),
    )
    return _finalize(
        w,
        names,
        "Minimum Variance",
        mu,
        cov,
        config.risk_free_rate,
        {"iterations": int(info["iterations"])},
        attribution=_duals(
            config,
            _minvar_objective,
            params,
            compiled,
            w,
            name="variance",
            sense="minimize",
        ),
    )


def mean_variance(
    returns,
    risk_aversion: float = 1.0,
    config: OptimizerConfig | None = None,
    *,
    constraints=None,
) -> PortfolioResult:
    """Markowitz mean-variance portfolio.

    Maximizes ``w'mu - (risk_aversion/2) w'Sigma w`` (equivalently minimizes its
    negative) under the configured constraints. Larger ``risk_aversion`` tilts
    toward lower variance.
    """
    config = _cfg(config, constraints)
    mu, cov, names, _ = _moments(returns)
    n = len(names)

    projection, pparams, compiled = _projection(config, names)
    params = {
        "mu": mu,
        "cov": cov,
        "risk_aversion": jnp.asarray(risk_aversion),
        "l2": jnp.asarray(config.l2_reg),
    }
    w, info = solve_constrained(
        _meanvar_objective,
        params,
        jnp.full(n, 1.0 / n),
        projection,
        pparams,
        **_solver_kwargs(config),
    )
    return _finalize(
        w,
        names,
        "Mean-Variance",
        mu,
        cov,
        config.risk_free_rate,
        {"risk_aversion": risk_aversion, "iterations": int(info["iterations"])},
        attribution=_duals(
            config,
            _meanvar_objective,
            params,
            compiled,
            w,
            name="mean-variance utility",
            sense="maximize",
        ),
    )


def maximum_sharpe(
    returns,
    config: OptimizerConfig | None = None,
    *,
    constraints=None,
) -> PortfolioResult:
    """Maximum-Sharpe (tangency) portfolio.

    Minimizes the negative Sharpe ratio under the configured constraints. The
    Sharpe ratio is scale-invariant, so we optimize on the simplex directly.
    """
    config = _cfg(config, constraints)
    mu, cov, names, _ = _moments(returns)
    n = len(names)
    rf = config.risk_free_rate

    projection, pparams, compiled = _projection(config, names)
    params = {"mu": mu, "cov": cov, "rf": jnp.asarray(rf), "l2": jnp.asarray(config.l2_reg)}
    w, info = solve_constrained(
        _sharpe_objective,
        params,
        jnp.full(n, 1.0 / n),
        projection,
        pparams,
        **_solver_kwargs(config),
    )
    return _finalize(
        w,
        names,
        "Maximum Sharpe",
        mu,
        cov,
        rf,
        {"iterations": int(info["iterations"])},
        attribution=_duals(
            config,
            _sharpe_objective,
            params,
            compiled,
            w,
            name="Sharpe ratio",
            sense="maximize",
            convex=False,
        ),
    )


def maximum_diversification(
    returns,
    config: OptimizerConfig | None = None,
    *,
    constraints=None,
) -> PortfolioResult:
    """Maximum-diversification portfolio (Choueifaty & Coignard, 2008).

    Maximizes the diversification ratio ``(w'sigma) / sqrt(w'Sigma w)`` where
    ``sigma`` is the vector of asset volatilities.
    """
    config = _cfg(config, constraints)
    mu, cov, names, _ = _moments(returns)
    n = len(names)
    vol = jnp.sqrt(jnp.diag(cov))

    projection, pparams, compiled = _projection(config, names)
    params = {"cov": cov, "vol": vol, "l2": jnp.asarray(config.l2_reg)}
    w, info = solve_constrained(
        _maxdiv_objective,
        params,
        jnp.full(n, 1.0 / n),
        projection,
        pparams,
        **_solver_kwargs(config),
    )
    return _finalize(
        w,
        names,
        "Maximum Diversification",
        mu,
        cov,
        config.risk_free_rate,
        {"iterations": int(info["iterations"])},
        attribution=_duals(
            config,
            _maxdiv_objective,
            params,
            compiled,
            w,
            name="diversification ratio",
            sense="maximize",
            convex=False,
        ),
    )


def risk_parity(
    returns,
    config: OptimizerConfig | None = None,
    *,
    constraints=None,
) -> PortfolioResult:
    """Equal-risk-contribution (ERC) / risk-parity portfolio.

    Each asset contributes equally to total portfolio risk. Solved with the
    cyclical coordinate descent of Griveau-Billion, Richard & Roncalli (2013) on
    the convex program ``min 0.5 x'Sigma x - (1/N) sum(log x)``. Each coordinate
    update solves a one-dimensional quadratic in closed form,
    ``x_i <- (-beta_i + sqrt(beta_i^2 + 4 sigma_ii b_i)) / (2 sigma_ii)`` with
    ``beta_i = (Sigma x)_i - sigma_ii x_i``, which keeps ``x_i`` strictly
    positive and is provably convergent regardless of covariance scale. Weights
    are the normalized fixed point.

    Named constraints are not supported: the log-barrier keeps every ``x_i``
    strictly positive, so no bound is ever active and the budget is imposed by
    normalizing the fixed point rather than as a constraint — there is no
    constrained program for a cap to enter. Use ``minimum_variance`` with a
    ``GroupCap`` if you need one.
    """
    config = _cfg(config, constraints)
    _reject_constraints(config, "risk_parity", "minimum_variance or mean_variance")
    mu, cov, names, _ = _moments(returns)
    n = len(names)
    b = 1.0 / n
    sigma = np.asarray(cov, dtype=float)
    # Floor the diagonal so a zero-variance asset cannot produce a 0/0 coordinate
    # update or an infinite warm start.
    diag = np.clip(np.diag(sigma), 1e-18, None)

    # Warm start from inverse-volatility weights (a good ERC approximation).
    x = 1.0 / np.sqrt(diag)
    x = x / x.sum()

    iters = 0
    while iters < config.max_iter:
        iters += 1
        x_prev = x.copy()
        for i in range(n):
            beta_i = sigma[i] @ x - diag[i] * x[i]  # (Sigma x)_i without the i-th term
            x[i] = (-beta_i + np.sqrt(beta_i**2 + 4.0 * diag[i] * b)) / (2.0 * diag[i])
        if np.linalg.norm(x / x.sum() - x_prev / x_prev.sum()) < config.tol:
            break

    w = x / x.sum()
    port_var = float(w @ sigma @ w)
    rc = w * (sigma @ w) / port_var
    return _finalize(
        w,
        names,
        "Risk Parity (ERC)",
        mu,
        cov,
        config.risk_free_rate,
        {
            "iterations": iters,
            "risk_contributions": np.asarray(rc, dtype=float).tolist(),
        },
    )


def kelly(
    returns,
    config: OptimizerConfig | None = None,
    *,
    constraints=None,
) -> PortfolioResult:
    """Approximate Kelly-optimal (log-growth) portfolio.

    Maximizes ``E[log(1 + w'r)]`` estimated over the sample paths, which is the
    growth-optimal criterion. Uses the projected-gradient solver over the return
    matrix directly.
    """
    config = _cfg(config, constraints)
    mu, cov, names, mat = _moments(returns)
    n = len(names)

    projection, pparams, compiled = _projection(config, names)
    params = {"mat": mat, "l2": jnp.asarray(config.l2_reg)}
    w, info = solve_constrained(
        _kelly_objective,
        params,
        jnp.full(n, 1.0 / n),
        projection,
        pparams,
        **_solver_kwargs(config),
    )
    return _finalize(
        w,
        names,
        "Kelly (log-growth)",
        mu,
        cov,
        config.risk_free_rate,
        {"iterations": int(info["iterations"])},
        attribution=_duals(
            config,
            _kelly_objective,
            params,
            compiled,
            w,
            name="expected log growth",
            sense="maximize",
        ),
    )


def min_cvar(
    returns,
    alpha: float = 0.95,
    config: OptimizerConfig | None = None,
    *,
    constraints=None,
) -> PortfolioResult:
    """Minimum Conditional-Value-at-Risk portfolio (Rockafellar & Uryasev, 2000).

    Minimizes CVaR at confidence ``alpha`` using the smooth auxiliary-variable
    formulation ``CVaR = tau + 1/((1-alpha)T) sum(max(loss - tau, 0))``, jointly
    optimizing weights ``w`` and the VaR threshold ``tau``.

    The packed ``(w, tau)`` variable (``tau`` unconstrained) and the piecewise
    objective are ill-suited to a single scalar Barzilai-Borwein step, so this
    optimizer substitutes Adam whenever ``config.solver`` is ``"spg"``. An
    explicitly chosen optax optimizer is honored.
    """
    config = _cfg(config, constraints)
    mu, cov, names, mat = _moments(returns)
    n = len(names)
    t = mat.shape[0]
    scale = 1.0 / ((1.0 - alpha) * t)

    projection, pparams, compiled = _projection(config, names, aux=True)
    params = {"mat": mat, "scale": jnp.asarray(scale), "l2": jnp.asarray(config.l2_reg)}

    # Pack (w, tau) into a single vector; tau starts at 0.
    z0 = jnp.concatenate([jnp.full(n, 1.0 / n), jnp.zeros(1)])
    spec = config.solver_spec()
    if spec.kind == "spg":
        spec = resolve_solver("adam")
    z, info = solve_constrained(
        _cvar_objective,
        params,
        z0,
        projection,
        pparams,
        solver=spec,
        learning_rate=config.learning_rate,
        max_iter=config.max_iter,
        tol=config.tol,
    )
    w = z[:-1]
    tau = z[-1]
    losses = -(mat @ w)
    cvar = float(tau + scale * jnp.sum(jnp.maximum(losses - tau, 0.0)))
    return _finalize(
        w,
        names,
        f"Min CVaR ({int(alpha * 100)}%)",
        mu,
        cov,
        config.risk_free_rate,
        {"alpha": alpha, "cvar": cvar, "var": float(tau), "iterations": int(info["iterations"])},
        attribution=_duals(
            config,
            _cvar_objective,
            params,
            compiled,
            z,
            name="CVaR",
            sense="minimize",
            smooth=False,
            aux_dim=1,
        ),
    )


def black_litterman(
    returns,
    views: dict[str, float] | None = None,
    *,
    view_confidence: float = 0.5,
    tau: float = 0.05,
    risk_aversion: float = 2.5,
    market_weights: np.ndarray | None = None,
    config: OptimizerConfig | None = None,
    constraints=None,
) -> PortfolioResult:
    """Black-Litterman portfolio blending market equilibrium with investor views.

    Parameters
    ----------
    views:
        Absolute views mapping asset name -> expected (per-period) return. Only
        assets present in the panel are used. If ``None``, the result reduces to
        the reverse-optimized equilibrium (market) portfolio.
    view_confidence:
        Scalar in ``(0, 1]`` controlling how tightly views are weighted (higher
        = more confident).
    tau:
        Uncertainty scaling on the prior covariance.
    risk_aversion:
        Market risk-aversion used for reverse optimization of equilibrium
        returns.
    market_weights:
        Prior (market-cap) weights; defaults to equal weight.
    """
    config = _cfg(config, constraints)
    mat, names = as_matrix(returns)
    cov = sample_covariance(mat)
    n = len(names)

    w_mkt = (
        jnp.full(n, 1.0 / n) if market_weights is None else jnp.asarray(market_weights, dtype=float)
    )
    # Reverse-optimized (implied equilibrium) excess returns: pi = lambda * Sigma * w_mkt
    pi = risk_aversion * (cov @ w_mkt)

    if views:
        idx = {name: i for i, name in enumerate(names)}
        used = [(idx[a], q) for a, q in views.items() if a in idx]
        if used:
            rows = jnp.zeros((len(used), n))
            q = jnp.zeros(len(used))
            for k, (i, val) in enumerate(used):
                rows = rows.at[k, i].set(1.0)
                q = q.at[k].set(val)
            p = rows
            tau_cov = tau * cov
            # Omega: per-view uncertainty, scaled by confidence. Both the prior
            # view variance (a zero-variance asset) and confidence -> 0 would make
            # Omega singular; floor the confidence and the diagonal to stay stable.
            conf = float(np.clip(view_confidence, 1e-3, 1.0))
            omega_diag = jnp.diag(p @ tau_cov @ p.T) / conf
            omega_diag = jnp.clip(omega_diag, 1e-10, None)
            omega_inv = jnp.diag(1.0 / omega_diag)  # Omega is diagonal by construction
            # Posterior mean (He & Litterman closed form). Ridge-regularize the
            # prior-covariance inverse so a singular Sigma (e.g. a zero-variance
            # asset) does not blow the posterior up to NaN.
            ridge = 1e-10 * jnp.eye(n)
            a = jnp.linalg.inv(tau_cov + ridge)
            b = p.T @ omega_inv @ p
            post_cov = jnp.linalg.inv(a + b + ridge)
            post_mu = post_cov @ (a @ pi + p.T @ omega_inv @ q)
        else:
            post_mu = pi
    else:
        post_mu = pi

    # Mean-variance optimize with the posterior returns (reuse the shared objective).
    projection, pparams, compiled = _projection(config, names)
    params = {
        "mu": post_mu,
        "cov": cov,
        "risk_aversion": jnp.asarray(risk_aversion),
        "l2": jnp.asarray(config.l2_reg),
    }
    w, info = solve_constrained(
        _meanvar_objective, params, w_mkt, projection, pparams, **_solver_kwargs(config)
    )
    mu_sample = mean_returns(mat)
    return _finalize(
        w,
        names,
        "Black-Litterman",
        mu_sample,
        cov,
        config.risk_free_rate,
        {
            "posterior_returns": np.asarray(post_mu, dtype=float).tolist(),
            "n_views": len(views) if views else 0,
            "iterations": int(info["iterations"]),
        },
        attribution=_duals(
            config,
            _meanvar_objective,
            params,
            compiled,
            w,
            name="mean-variance utility (posterior)",
            sense="maximize",
        ),
    )
