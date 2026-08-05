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
from jaxfolio.optimizers import multiperiod as C_MP
from jaxfolio.types import TradingCosts

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
    """Convert Polars to the pandas frame required by PyPortfolioOpt."""
    import pandas as pd
    import polars as pl

    if isinstance(returns, pl.DataFrame):
        names = [c for c in returns.columns if c != "date"]
        return pd.DataFrame(returns.select(names).to_numpy(), columns=names), names
    raise TypeError("HRP reference test expects a Polars DataFrame of returns")


# --------------------------------------------------------------------------- #
# Multi-period mean-variance with trading costs
# --------------------------------------------------------------------------- #
# The exact problem (true L1 trade cost) is a convex QP in T*N variables, so
# CVXPY solves it to optimality and gives a genuine reference rather than a
# second iterate. jaxfolio instead Huber-smooths the L1 and walks a ladder of
# shrinking widths, keeping whichever stage scores best under the exact
# objective. These tests pin that the resulting gap stays tiny.
#
# Accuracy is NOT uniform across cost regimes, and the tolerances below reflect
# measured behavior rather than a single guess:
#
#   * frictionless / impact-dominated  -> ~1e-8 relative (smooth, strongly convex)
#   * prohibitive linear cost          -> exact (the no-trade path is optimal)
#   * mid-range linear cost            -> up to ~2e-4 relative, the stiff regime
#     where the L1 term rivals the entire mean-variance curvature
#
# Unlike the CVaR gap pinned by a strict xfail above, this needs no xfail: the
# error is a controlled smoothing bias on a strongly convex problem, not a solver
# mismatched to its objective.
_MP_HORIZON = 4
_MP_GAMMA = 3.0
# Measured worst case across n in {4..50} and six cost regimes is 1.9e-4; this
# leaves ~5x headroom without being so loose it would accept a real regression.
_MP_RELGAP = 1e-3


def _mp_setup(returns):
    mu, cov, _ = ref.moments(returns)
    return mu, cov, np.eye(len(mu))[0]


@pytest.mark.parametrize(
    ("spread_bps", "impact_bps"),
    [(0.0, 0.0), (10.0, 0.0), (50.0, 0.0), (10.0, 500.0), (0.0, 2000.0)],
    ids=["frictionless", "linear-10bps", "linear-50bps", "linear+impact", "impact-only"],
)
def test_multi_period_matches_cvxpy_qp(returns, spread_bps, impact_bps):
    """jaxfolio's smoothed path must score within tolerance of the exact QP optimum."""
    pytest.importorskip("cvxpy")
    mu, cov, w_prev = _mp_setup(returns)
    c_lin, c_quad = spread_bps / 1e4, impact_bps / 1e4

    _, obj_ref = ref.ref_multi_period_cvxpy(
        mu,
        cov,
        w_prev,
        horizon=_MP_HORIZON,
        risk_aversion=_MP_GAMMA,
        c_lin=c_lin,
        c_quad=c_quad,
    )
    res = C_MP.multi_period_mean_variance(
        returns,
        horizon=_MP_HORIZON,
        w_prev=w_prev,
        risk_aversion=_MP_GAMMA,
        costs=TradingCosts(spread_bps=spread_bps, impact_bps=impact_bps),
    )
    obj = ref.multi_period_objective(
        res.trajectory,
        mu,
        cov,
        w_prev,
        risk_aversion=_MP_GAMMA,
        c_lin=c_lin,
        c_quad=c_quad,
    )
    relgap = (obj_ref - obj) / abs(obj_ref)
    # CVXPY is the exact maximum, so jaxfolio cannot meaningfully exceed it; the
    # small allowance covers the reference solver's own residual tolerance.
    assert relgap >= -_MP_RELGAP, f"jaxfolio exceeded the exact optimum by {-relgap:.3g}"
    assert relgap <= _MP_RELGAP, (
        f"objective {obj:.10g} vs CVXPY optimum {obj_ref:.10g} "
        f"(relative gap {relgap:.3g}, budget {_MP_RELGAP:.3g})"
    )


def test_multi_period_zero_cost_matches_cvxpy_weights(returns):
    """With no frictions the path separates, so the weights themselves must match.

    Run first as a localizing test: it exercises the mean-variance half of the
    objective independently of the cost machinery, so a failure here points at the
    path construction rather than at the smoothing.
    """
    pytest.importorskip("cvxpy")
    mu, cov, w_prev = _mp_setup(returns)
    W_ref, _ = ref.ref_multi_period_cvxpy(
        mu, cov, w_prev, horizon=_MP_HORIZON, risk_aversion=_MP_GAMMA, c_lin=0.0
    )
    res = C_MP.multi_period_mean_variance(
        returns, horizon=_MP_HORIZON, w_prev=w_prev, risk_aversion=_MP_GAMMA
    )
    assert np.allclose(res.trajectory, W_ref, atol=2e-3), (
        f"Δw_max={np.abs(res.trajectory - W_ref).max():.4g}"
    )


def test_multi_period_impact_glide_path_matches_cvxpy(returns):
    """Quadratic impact ⇒ a decaying execution schedule, matching CVXPY per period.

    This is the economic content of the feature, so it is checked against the
    exact optimum rather than only asserted as a shape. Per-period turnover is the
    quantity a trader reads, which makes it the right thing to compare.
    """
    pytest.importorskip("cvxpy")
    mu, cov, w_prev = _mp_setup(returns)
    c_lin, c_quad = 10.0 / 1e4, 500.0 / 1e4
    W_ref, _ = ref.ref_multi_period_cvxpy(
        mu,
        cov,
        w_prev,
        horizon=_MP_HORIZON,
        risk_aversion=_MP_GAMMA,
        c_lin=c_lin,
        c_quad=c_quad,
    )
    res = C_MP.multi_period_mean_variance(
        returns,
        horizon=_MP_HORIZON,
        w_prev=w_prev,
        risk_aversion=_MP_GAMMA,
        costs=TradingCosts(spread_bps=10.0, impact_bps=500.0),
    )
    prev_ref = np.vstack([w_prev[None, :], W_ref[:-1]])
    turnover_ref = np.abs(W_ref - prev_ref).sum(axis=1)
    turnover = np.asarray(res.metadata["turnover_path"])
    assert np.allclose(turnover, turnover_ref, atol=1e-3), (
        f"turnover schedule {turnover} vs reference {turnover_ref}"
    )
    # And it genuinely decays rather than front-loading everything.
    assert (np.diff(turnover_ref) < 0).all(), f"reference is not a glide path: {turnover_ref}"


def test_multi_period_prohibitive_cost_is_exactly_no_trade(returns):
    """When trading is ruinous the optimum is to hold still — and that is exact.

    No smoothing bias applies here: the Huber band's gradient is exactly zero at
    zero trade, so the no-trade path is a true stationary point of the surrogate.
    """
    pytest.importorskip("cvxpy")
    mu, cov, w_prev = _mp_setup(returns)
    c_lin = 1000.0 / 1e4
    _, obj_ref = ref.ref_multi_period_cvxpy(
        mu, cov, w_prev, horizon=_MP_HORIZON, risk_aversion=_MP_GAMMA, c_lin=c_lin
    )
    res = C_MP.multi_period_mean_variance(
        returns,
        horizon=_MP_HORIZON,
        w_prev=w_prev,
        risk_aversion=_MP_GAMMA,
        costs=TradingCosts(spread_bps=1000.0),
    )
    obj = ref.multi_period_objective(
        res.trajectory, mu, cov, w_prev, risk_aversion=_MP_GAMMA, c_lin=c_lin
    )
    assert abs(obj_ref - obj) / abs(obj_ref) < 1e-6, f"{obj:.10g} vs {obj_ref:.10g}"
    assert np.allclose(res.trajectory, w_prev[None, :], atol=1e-5)


def test_multi_period_stage_selection_beats_the_last_stage(returns):
    """Selecting the best smoothing stage must never be worse than taking the last.

    Accuracy is not monotone in the Huber width, which is exactly why the ladder
    selects on the exact objective instead of trusting its final stage. This pins
    that the selection is actually doing work, and records how much.
    """
    pytest.importorskip("cvxpy")
    mu, cov, w_prev = _mp_setup(returns)
    c_lin = 50.0 / 1e4
    _, obj_ref = ref.ref_multi_period_cvxpy(
        mu, cov, w_prev, horizon=_MP_HORIZON, risk_aversion=_MP_GAMMA, c_lin=c_lin
    )
    res = C_MP.multi_period_mean_variance(
        returns,
        horizon=_MP_HORIZON,
        w_prev=w_prev,
        risk_aversion=_MP_GAMMA,
        costs=TradingCosts(spread_bps=50.0),
    )
    stage_values = res.metadata["stage_objective_exact"]
    selected = res.metadata["selected_stage"]
    # ``stage_objective_exact`` is in minimization form; the chosen one must be the
    # smallest, and at least as good as the final stage.
    assert min(stage_values) <= stage_values[-1] + 1e-12
    if selected >= 0:
        assert np.isclose(stage_values[selected], min(stage_values))
    obj = ref.multi_period_objective(
        res.trajectory, mu, cov, w_prev, risk_aversion=_MP_GAMMA, c_lin=c_lin
    )
    assert (obj_ref - obj) / abs(obj_ref) <= _MP_RELGAP


# --------------------------------------------------------------------------- #
# Named constraints: the projection and its multipliers vs CVXPY
# --------------------------------------------------------------------------- #
# Measured worst deviations over the cases below, in float32 (the package never
# enables x64): weights 3e-6, budget multiplier 4e-6, identified row multipliers
# 5e-6. Unidentified rows are excluded on purpose — see the note in the test.
_PROJ_ATOL = 5e-5


@pytest.mark.parametrize(
    ("long_only", "cap"),
    [(True, 0.30), (True, 0.60), (False, 0.30)],
    ids=["long-only-tight", "long-only-slack", "shortable"],
)
def test_grouped_projection_matches_cvxpy(long_only, cap):
    """The projection and every *identified* multiplier agree with a real QP solver."""
    cvxpy = pytest.importorskip("cvxpy")  # noqa: F841
    import jax.numpy as jnp

    from jaxfolio.constraints.structured import project_grouped_duals

    rng = np.random.default_rng(0)
    n, k = 12, 3
    lo_s, hi_s = (0.0, 0.5) if long_only else (-0.2, 0.5)
    lower, upper = np.full(n, lo_s), np.full(n, hi_s)
    groups = [list(range(0, 4)), list(range(4, 8)), list(range(8, 11))]  # asset 11 ungrouped
    gid = np.full(n, k, dtype=np.int32)
    for j, members in enumerate(groups):
        gid[members] = j
    g_lower = np.zeros(k)
    g_upper = np.full(k, cap)

    for _ in range(5):
        v = rng.normal(0.1, 0.5, n)
        got = project_grouped_duals(
            jnp.asarray(v),
            jnp.asarray(lower),
            jnp.asarray(upper),
            jnp.asarray(1.0),
            jnp.asarray(gid),
            jnp.asarray(np.append(g_lower, 0.0)),
            jnp.asarray(np.append(g_upper, 0.0)),
        )
        w_ref, lam_ref, theta_ref = ref.ref_grouped_projection_cvxpy(
            v, lower, upper, 1.0, groups, g_lower, g_upper
        )
        w = np.asarray(got.weights)
        assert np.allclose(w, w_ref, atol=_PROJ_ATOL), f"Δw_max={np.abs(w - w_ref).max():.4g}"

        # A row's multiplier is only pinned when some member is strictly interior;
        # otherwise any split between the row and the box bounds is a valid KKT
        # certificate and CVXPY's interior-point method picks a different one.
        ident = np.asarray(got.identified)[:k]
        theta = np.asarray(got.rows)[:k]
        if ident.any():
            delta = np.abs(theta[ident] - theta_ref[ident]).max()
            assert delta < _PROJ_ATOL, f"Δtheta_max={delta:.4g} on identified rows"
        if bool(got.budget_identified):
            assert abs(float(got.budget) - lam_ref) < _PROJ_ATOL, (
                f"Δlam={abs(float(got.budget) - lam_ref):.4g}"
            )


def test_grouped_minimum_variance_matches_cvxpy(returns):
    """A capped minimum-variance solve reaches the same optimum as a QP solver."""
    pytest.importorskip("cvxpy")
    from jaxfolio.constraints import GroupCap

    mu, cov, mat = ref.moments(returns)
    n = cov.shape[0]
    names = [c for c in returns.columns if c != "date"]
    groups = [list(range(0, 3)), list(range(3, n))]
    caps = [0.35, 0.80]

    res = C.minimum_variance(
        returns,
        constraints=[
            GroupCap("g0", names[:3], max=caps[0]),
            GroupCap("g1", names[3:], max=caps[1]),
        ],
    )
    w = np.asarray(res.weights)
    w_ref, obj_ref = ref.ref_min_variance_grouped_cvxpy(cov, np.zeros(n), np.ones(n), groups, caps)
    assert w[:3].sum() <= caps[0] + 1e-4, f"cap violated: {w[:3].sum():.5f} > {caps[0]}"
    obj = float(w @ cov @ w)
    assert obj <= obj_ref * (1 + 1e-3), f"jaxfolio objective {obj:.8f} vs CVXPY {obj_ref:.8f}"
    assert np.allclose(w, w_ref, atol=2e-3), f"Δw_max={np.abs(w - w_ref).max():.4g}"
