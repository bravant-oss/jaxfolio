"""Numerical validation of jaxfolio optimizers against independent reference
solvers on well-conditioned problems.

This is the "method x reference-solver" axis of the validation matrix documented
in ``docs/reference/validation.md``. Each test feeds the *same* sample moments to
jaxfolio and to a reference (SciPy SLSQP, a CVXPY LP, or PyPortfolioOpt) and
asserts they land on the same optimum within a documented tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxfolio.optimizers import classical as C
from jaxfolio.optimizers import graph as G

from . import references as ref


def test_minimum_variance_matches_slsqp(returns):
    mu, cov, mat = ref.moments(returns)
    w_ref = ref.ref_min_variance(mu, cov, mat)
    w = C.minimum_variance(returns).weights
    assert np.allclose(w, w_ref, atol=2e-3), f"Δw_max={np.abs(w - w_ref).max():.4g}"
    assert (w @ cov @ w) <= (w_ref @ cov @ w_ref) * (1 + 1e-3)


def test_maximum_sharpe_matches_slsqp(returns):
    mu, cov, mat = ref.moments(returns)
    w_ref = ref.ref_max_sharpe(mu, cov, mat)
    w = C.maximum_sharpe(returns).weights
    sr = (w @ mu) / np.sqrt(w @ cov @ w)
    sr_ref = (w_ref @ mu) / np.sqrt(w_ref @ cov @ w_ref)
    assert sr >= sr_ref * (1 - 1e-3), f"jaxfolio Sharpe {sr:.5f} < SLSQP {sr_ref:.5f}"


def test_maximum_diversification_matches_slsqp(returns):
    mu, cov, mat = ref.moments(returns)
    vol = np.sqrt(np.diag(cov))
    w_ref = ref.ref_max_diversification(mu, cov, mat)
    w = C.maximum_diversification(returns).weights
    dr = (w @ vol) / np.sqrt(w @ cov @ w)
    dr_ref = (w_ref @ vol) / np.sqrt(w_ref @ cov @ w_ref)
    # jaxfolio should reach at least the reference diversification ratio.
    assert dr >= dr_ref * (1 - 2e-3), f"jaxfolio DR {dr:.5f} < SLSQP {dr_ref:.5f}"


def test_kelly_matches_slsqp(returns):
    mu, cov, mat = ref.moments(returns)
    w_ref = ref.ref_kelly(mu, cov, mat)

    def growth(w):
        return float(np.mean(np.log1p(np.clip(mat @ w, -0.999, None))))

    w = C.kelly(returns).weights
    assert growth(w) >= growth(w_ref) - 1e-5, (
        f"jaxfolio log-growth {growth(w):.6g} < SLSQP {growth(w_ref):.6g}"
    )


def test_risk_parity_equal_risk_contributions(returns):
    """The ERC reference *property*: every asset contributes equal portfolio risk.

    This is the definition risk parity solves for, so matching it exactly (rather
    than a second solver's iterate) is the strongest validation available.
    """
    res = C.risk_parity(returns)
    rc = np.asarray(res.metadata["risk_contributions"], dtype=float)
    n = len(rc)
    assert np.allclose(rc.sum(), 1.0, atol=1e-6)
    # Each contribution within a tight band of 1/n.
    assert np.allclose(rc, 1.0 / n, atol=1e-3), f"max|rc-1/n|={np.abs(rc - 1.0 / n).max():.4g}"


def test_min_cvar_direction_and_regression_ceiling(returns):
    """jaxfolio's CVaR is above the LP optimum (correct direction), and not
    *wildly* so — a guard against the smooth solver regressing further.

    The exact-match contract is asserted (and currently xfails) in
    ``test_min_cvar_matches_cvxpy_lp`` below; see docs/reference/validation.md
    for the documented gap.
    """
    pytest.importorskip("cvxpy")
    _, _, mat = ref.moments(returns)
    alpha = 0.95
    _, cvar_ref = ref.ref_min_cvar_cvxpy(mat, alpha=alpha)
    cvar = float(C.min_cvar(returns, alpha=alpha).metadata["cvar"])
    # LP is the true minimum, so jaxfolio can only be >= it.
    assert cvar >= cvar_ref - 1e-6
    # Regression ceiling: today's gap is ~30%; fail loudly if it worsens past 45%.
    assert cvar <= cvar_ref * 1.45, f"jaxfolio CVaR {cvar:.6g} vs CVXPY optimum {cvar_ref:.6g}"


@pytest.mark.xfail(
    strict=True,
    reason="Smooth Adam-based CVaR solver reaches ~26-33% above the exact CVXPY "
    "LP optimum on multi-asset panels; tracked in docs/reference/validation.md. "
    "Remove this marker when min_cvar converges to the LP optimum.",
)
def test_min_cvar_matches_cvxpy_lp(returns):
    """Aspirational contract: min_cvar should match the exact CVXPY LP optimum.

    Marked ``xfail(strict=True)`` so it documents the intended behavior and will
    flag (xpass) the moment the solver is improved to meet it.
    """
    pytest.importorskip("cvxpy")
    _, _, mat = ref.moments(returns)
    alpha = 0.95
    _, cvar_ref = ref.ref_min_cvar_cvxpy(mat, alpha=alpha)
    cvar = float(C.min_cvar(returns, alpha=alpha).metadata["cvar"])
    assert cvar <= cvar_ref * (1 + 0.02) + 1e-4, (
        f"jaxfolio CVaR {cvar:.6g} vs CVXPY optimum {cvar_ref:.6g}"
    )


def test_hrp_matches_pypfopt(returns):
    """jaxfolio HRP vs PyPortfolioOpt HRP on identical returns.

    Both use the correlation distance ``sqrt(0.5(1-rho))``, single linkage, and
    inverse-variance recursive bisection, so weights should agree closely.
    """
    pytest.importorskip("pypfopt")
    mat, names = _as_named_frame(returns)
    w_ref = ref.ref_hrp_pypfopt(mat, names)

    w = G.hierarchical_risk_parity(returns).weights
    # Same algorithm, so agreement should be tight; allow slack for covariance
    # ddof / linkage tie-breaking differences.
    l1 = np.abs(w - w_ref).sum()
    assert l1 < 0.05, f"HRP L1 weight distance {l1:.4g} too large\njax={w}\nref={w_ref}"


def _as_named_frame(returns):
    """Return ``(DataFrame, column_names)`` for the returns object."""
    import pandas as pd

    if isinstance(returns, pd.DataFrame):
        return returns, list(returns.columns)
    raise TypeError("HRP reference test expects a pandas DataFrame of returns")
