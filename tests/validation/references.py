"""Independent reference optimizers for the numerical-validation matrix.

These are deliberately simple, self-contained reimplementations (SciPy SLSQP /
CVXPY LP) of the same problems jaxfolio solves, so agreement is meaningful: the
two code paths share nothing but the math. Kept out of the package (test-only) so
core installs never depend on the heavy reference solvers.
"""

from __future__ import annotations

import numpy as np
import scipy.optimize as opt

from jaxfolio.moments.estimators import as_matrix, mean_returns, sample_covariance


def moments(returns) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample ``(mu, cov, matrix)`` as plain NumPy from a returns object."""
    mat, _ = as_matrix(returns)
    mat = np.asarray(mat, dtype=float)
    return np.asarray(mean_returns(mat)), np.asarray(sample_covariance(mat)), mat


def slsqp_long_only(objective, n: int, *, x0=None) -> np.ndarray:
    """Minimize ``objective`` over the long-only, fully-invested simplex."""
    res = opt.minimize(
        objective,
        np.full(n, 1.0 / n) if x0 is None else x0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    return res.x


def ref_min_variance(mu, cov, mat) -> np.ndarray:
    return slsqp_long_only(lambda w: float(w @ cov @ w), len(mu))


def ref_max_sharpe(mu, cov, mat, rf: float = 0.0) -> np.ndarray:
    def neg_sharpe(w):
        return -float((w @ mu - rf) / np.sqrt(w @ cov @ w + 1e-18))

    return slsqp_long_only(neg_sharpe, len(mu))


def ref_max_diversification(mu, cov, mat) -> np.ndarray:
    vol = np.sqrt(np.diag(cov))

    def neg_div_ratio(w):
        return -float((w @ vol) / np.sqrt(w @ cov @ w + 1e-18))

    return slsqp_long_only(neg_div_ratio, len(mu))


def ref_kelly(mu, cov, mat) -> np.ndarray:
    def neg_log_growth(w):
        return -float(np.mean(np.log1p(np.clip(mat @ w, -0.999, None))))

    return slsqp_long_only(neg_log_growth, len(mu))


def ref_hrp_pypfopt(returns, names):
    """PyPortfolioOpt HRP weights aligned to ``names``.

    Applies a small compatibility shim: PyPortfolioOpt <=1.6 references the
    private ``scipy.cluster.hierarchy._LINKAGE_METHODS``, removed in newer SciPy.
    We restore the expected set so the *independent* reference keeps working
    across the SciPy version range jaxfolio supports (this is exactly the kind of
    transitive-dependency drift the support policy calls out).
    """
    import scipy.cluster.hierarchy as sch

    if not hasattr(sch, "_LINKAGE_METHODS"):
        sch._LINKAGE_METHODS = {
            "single": 0,
            "complete": 1,
            "average": 2,
            "weighted": 3,
            "centroid": 4,
            "median": 5,
            "ward": 6,
        }
    from pypfopt.hierarchical_portfolio import HRPOpt

    weights = HRPOpt(returns=returns).optimize()
    return np.array([weights[name] for name in names], dtype=float)


def ref_min_cvar_cvxpy(mat, alpha: float = 0.95):
    """Rockafellar-Uryasev CVaR LP solved exactly with CVXPY.

    Returns ``(weights, cvar)``. Raises ImportError if CVXPY is unavailable so
    callers can ``importorskip``.
    """
    import cvxpy as cp

    t, n = mat.shape
    w = cp.Variable(n)
    tau = cp.Variable()
    u = cp.Variable(t)
    losses = -(mat @ w)
    cvar = tau + (1.0 / ((1.0 - alpha) * t)) * cp.sum(u)
    constraints = [u >= 0, u >= losses - tau, cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(cp.Minimize(cvar), constraints)
    prob.solve()
    return np.asarray(w.value, dtype=float), float(prob.value)


def ref_multi_period_cvxpy(
    mu,
    cov,
    w_prev,
    *,
    horizon: int,
    risk_aversion: float,
    c_lin,
    c_quad=0.0,
    bounds=(0.0, 1.0),
    long_only: bool = True,
):
    """Exact multi-period mean-variance-with-costs QP solved by CVXPY.

    Maximizes, over a path ``W`` of shape ``(horizon, n)`` anchored at
    ``w_prev``::

        sum_t [ mu' w_t - (gamma/2) w_t' Sigma w_t
                - c_lin * |w_t - w_{t-1}| - c_quad * (w_t - w_{t-1})**2 ]

    subject to ``sum(w_t) == 1`` and the per-asset bounds at *every* period. This
    is the **un-smoothed** problem: the L1 term is exact, so this is the true
    reference that jaxfolio's Huber surrogate approximates.

    The quadratic form is written through a Cholesky factor so convexity is
    unambiguous to CVXPY without ``psd_wrap``, and the solver is driven to a
    tight tolerance — at CVXPY's default the reference is only accurate to ~1e-3
    relative, which is looser than the quantity under test.

    Returns ``(W, objective_value)``. Raises ImportError if CVXPY is unavailable
    so callers can ``importorskip``.
    """
    import cvxpy as cp

    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    w_prev = np.asarray(w_prev, dtype=float)
    n = len(mu)
    lo, hi = bounds
    # A tiny ridge absorbs sample-covariance round-off so the factorization is safe.
    chol = np.linalg.cholesky(cov + 1e-12 * np.eye(n))

    W = cp.Variable((horizon, n))
    # Written with the variable second so ruff's SIM300 does not read a CVXPY
    # constraint expression as a Yoda condition.
    constraints = [cp.sum(W, axis=1) == 1.0, hi >= W]
    constraints.append((max(lo, 0.0) if long_only else lo) <= W)

    objective = 0
    for t in range(horizon):
        w_t = W[t, :]
        d = w_t - (w_prev if t == 0 else W[t - 1, :])
        objective += mu @ w_t - 0.5 * risk_aversion * cp.sum_squares(chol.T @ w_t)
        objective -= c_lin * cp.sum(cp.abs(d))
        if np.any(np.asarray(c_quad, dtype=float) != 0.0):
            objective -= c_quad * cp.sum_squares(d)

    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)
    return np.asarray(W.value, dtype=float), float(prob.value)


def multi_period_objective(path, mu, cov, w_prev, *, risk_aversion, c_lin, c_quad=0.0):
    """Evaluate the **exact** (un-smoothed) path objective, as a maximization.

    Used to score jaxfolio's answer on the same footing as the CVXPY optimum.
    Deliberately independent of the optimizer's own internals — evaluating
    jaxfolio's smoothed surrogate would make the comparison circular.
    """
    path = np.asarray(path, dtype=float)
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    prev = np.vstack([np.asarray(w_prev, dtype=float)[None, :], path[:-1]])
    d = path - prev
    utility = sum(
        mu @ path[t] - 0.5 * risk_aversion * (path[t] @ cov @ path[t]) for t in range(len(path))
    )
    return float(utility - c_lin * np.abs(d).sum() - c_quad * (d**2).sum())


def ref_grouped_projection_cvxpy(v, lower, upper, budget, groups, g_lower, g_upper):
    """Euclidean projection onto box + budget + group rows, solved by CVXPY.

    The independent reference for :func:`jaxfolio.constraints.structured.
    project_grouped_duals`. Returns ``(w, lam, theta)`` with the multipliers already
    translated into jaxfolio's convention: every row is written as ``a'w <= U`` and
    ``a'w >= L`` under ``cp.Minimize``, and ``theta_k`` is the *net* dual
    ``dual(cap) - dual(floor)`` — positive when the cap binds, negative when the
    floor does.

    NOTE: the objective is ``0.5 * sum_squares(w - v)``, matching the projection's
    definition exactly. A stray factor of two here would scale every multiplier by
    two and the test would still "pass" against a consistently wrong kernel.

    Imported lazily so callers can ``importorskip``.
    """
    import cvxpy as cp

    n = len(v)
    w = cp.Variable(n)
    cons = [w >= lower, w <= upper, cp.sum(w) == budget]
    budget_con = cons[2]
    rows = []
    for members, low, high in zip(groups, g_lower, g_upper, strict=True):
        a = np.zeros(n)
        a[list(members)] = 1.0
        cap = cp.sum(cp.multiply(a, w)) <= high
        floor = cp.sum(cp.multiply(a, w)) >= low
        cons += [cap, floor]
        rows.append((cap, floor))
    prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(w - v)), cons)
    prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)
    theta = np.array([float(c.dual_value) - float(f.dual_value) for c, f in rows])
    return np.asarray(w.value), float(budget_con.dual_value), theta


def ref_min_variance_grouped_cvxpy(cov, lower, upper, groups, g_upper, l2: float = 0.0):
    """Long-only minimum variance under group caps, solved by CVXPY.

    Mirrors ``_minvar_objective`` exactly: ``w'Sigma w + l2 * ||w||^2`` with **no**
    factor of one half, so the multipliers are directly comparable.
    """
    import cvxpy as cp

    n = cov.shape[0]
    w = cp.Variable(n)
    cons = [w >= lower, w <= upper, cp.sum(w) == 1.0]
    for members, high in zip(groups, g_upper, strict=True):
        a = np.zeros(n)
        a[list(members)] = 1.0
        cons.append(cp.sum(cp.multiply(a, w)) <= high)
    obj = cp.quad_form(w, cp.psd_wrap(cov)) + l2 * cp.sum_squares(w)
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)
    return np.asarray(w.value), float(prob.value)
