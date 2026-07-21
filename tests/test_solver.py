"""Tests for the projected-gradient solver: convergence and jit-caching.

The convergence tests pin the constrained optimizers to an independent SciPy
SLSQP reference on the *same* moments — this is the exact gap the benchmark
against other libraries surfaced (the old Adam solver stalled short of the
minimum-variance optimum). The cache test guards the performance fix: repeated
calls on same-shaped data must reuse the compiled kernel rather than re-tracing.
"""

from __future__ import annotations

import numpy as np
import scipy.optimize as opt

from jaxfolio.data.synthetic import generate_returns
from jaxfolio.moments.estimators import as_matrix, mean_returns, sample_covariance
from jaxfolio.optimizers import classical as C
from jaxfolio.optimizers.base import _solve_cached
from jaxfolio.types import OptimizerConfig


def _moments(returns):
    mat, _ = as_matrix(returns)
    return np.asarray(mean_returns(mat)), np.asarray(sample_covariance(mat))


def _slsqp(objective, n):
    """Long-only, fully-invested SLSQP reference optimum."""
    res = opt.minimize(
        objective,
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return res.x


def test_minimum_variance_matches_slsqp(returns):
    mu, cov = _moments(returns)
    n = len(mu)
    w_ref = _slsqp(lambda w: float(w @ cov @ w), n)
    w = C.minimum_variance(returns).weights
    # Strictly convex (PD cov) => unique optimum; weights should match closely.
    assert np.allclose(w, w_ref, atol=2e-3), f"Δw_max={np.abs(w - w_ref).max():.4g}"
    assert (w @ cov @ w) <= (w_ref @ cov @ w_ref) * (1 + 1e-3)


def test_maximum_sharpe_matches_slsqp(returns):
    mu, cov = _moments(returns)
    n = len(mu)
    neg_sharpe = lambda w: -float((w @ mu) / np.sqrt(w @ cov @ w + 1e-18))  # noqa: E731
    w_ref = _slsqp(neg_sharpe, n)
    w = C.maximum_sharpe(returns).weights
    sr = (w @ mu) / np.sqrt(w @ cov @ w)
    sr_ref = (w_ref @ mu) / np.sqrt(w_ref @ cov @ w_ref)
    assert sr >= sr_ref * (1 - 1e-3), f"jaxfolio Sharpe {sr:.5f} < SLSQP {sr_ref:.5f}"


def test_mean_variance_matches_slsqp(returns):
    mu, cov = _moments(returns)
    n = len(mu)
    ra = 3.0
    neg_util = lambda w: -float(w @ mu - 0.5 * ra * (w @ cov @ w))  # noqa: E731
    w_ref = _slsqp(neg_util, n)
    w = C.mean_variance(returns, risk_aversion=ra).weights
    util = w @ mu - 0.5 * ra * (w @ cov @ w)
    util_ref = w_ref @ mu - 0.5 * ra * (w_ref @ cov @ w_ref)
    assert util >= util_ref - 1e-6, f"jaxfolio util {util:.6g} < SLSQP {util_ref:.6g}"


def test_spg_beats_old_adam_default_on_min_variance(returns):
    """The default (SPG) must reach a lower variance than fixed-step Adam did."""
    mu, cov = _moments(returns)
    spg = C.minimum_variance(returns).weights
    adam = C.minimum_variance(returns, config=OptimizerConfig(solver="adam")).weights
    var_spg = spg @ cov @ spg
    var_adam = adam @ cov @ adam
    assert var_spg <= var_adam + 1e-9


def test_adam_solver_still_produces_valid_portfolio(returns):
    cfg = OptimizerConfig(solver="adam", learning_rate=1e-2)
    w = C.maximum_sharpe(returns, config=cfg).weights
    assert np.isclose(w.sum(), 1.0, atol=1e-4)
    assert (w >= -1e-4).all()


def test_jit_cache_reuses_compilation():
    """Same-shaped repeated solves reuse the compiled kernel (the speed fix)."""
    # A panel shape not used elsewhere in the suite, so the cache entry is ours.
    panel = generate_returns(n_assets=13, n_days=300, seed=99)
    before = _solve_cached._cache_size()
    C.minimum_variance(panel)
    after_first = _solve_cached._cache_size()
    C.minimum_variance(panel)
    after_second = _solve_cached._cache_size()
    assert after_first == before + 1, "first solve should compile exactly one variant"
    assert after_second == after_first, "second same-shape solve must reuse the cache"


def test_jit_cache_separates_shapes():
    """Different asset counts get distinct cache entries (correct keying)."""
    p8 = generate_returns(n_assets=8, n_days=300, seed=7)
    p9 = generate_returns(n_assets=9, n_days=300, seed=7)
    C.minimum_variance(p8)
    C.minimum_variance(p9)
    before = _solve_cached._cache_size()
    C.minimum_variance(p8)  # both already compiled -> no growth
    C.minimum_variance(p9)
    assert _solve_cached._cache_size() == before
