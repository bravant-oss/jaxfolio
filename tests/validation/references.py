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
