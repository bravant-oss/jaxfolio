"""Tests for the projected-gradient solver: convergence, optax solvers, caching.

The convergence tests pin the constrained optimizers to an independent SciPy
SLSQP reference on the *same* moments — this is the exact gap the benchmark
against other libraries surfaced (the old Adam solver stalled short of the
minimum-variance optimum). The solver-selection tests cover the optax spellings
(name, factory, ``solver_options``) and the errors they must raise. The cache
tests guard the performance fix: repeated calls on same-shaped data must reuse
the compiled kernel rather than re-tracing — which is why a solver key has to be
hashable *and* equality-stable across separately constructed configs.
"""

from __future__ import annotations

import numpy as np
import optax
import pytest
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


@pytest.mark.parametrize("name", ["adamw", "sgd", "rmsprop", "lion", "yogi"])
def test_optax_solver_names_produce_valid_portfolios(returns, name):
    """Any optax optimizer must still land on the feasible set."""
    w = C.maximum_sharpe(returns, config=OptimizerConfig(solver=name)).weights
    assert np.isclose(w.sum(), 1.0, atol=1e-4)
    assert (w >= -1e-4).all()


def test_string_and_callable_solver_agree(returns):
    """``solver="adamw"`` and ``solver=optax.adamw`` resolve to the same run."""
    by_name = C.minimum_variance(returns, config=OptimizerConfig(solver="adamw")).weights
    by_factory = C.minimum_variance(returns, config=OptimizerConfig(solver=optax.adamw)).weights
    assert np.array_equal(by_name, by_factory)


def test_solver_options_reach_optax(returns):
    """Hyperparameters in ``solver_options`` actually change the trajectory."""
    plain = OptimizerConfig(solver="sgd", learning_rate=1e-2, max_iter=25)
    momentum = OptimizerConfig(
        solver="sgd", learning_rate=1e-2, max_iter=25, solver_options={"momentum": 0.9}
    )
    w_plain = C.minimum_variance(returns, config=plain).weights
    w_momentum = C.minimum_variance(returns, config=momentum).weights
    assert not np.allclose(w_plain, w_momentum, atol=1e-6)


def test_line_search_solver_rejected(returns):
    """optax.lbfgs needs per-step extra args the loop can't supply — say so."""
    cfg = OptimizerConfig(solver="lbfgs")
    with pytest.raises(ValueError, match="line-search"):
        C.minimum_variance(returns, config=cfg)


def test_prebuilt_transformation_rejected():
    """A pre-built transformation would defeat the jit cache; point at the factory."""
    with pytest.raises(TypeError, match="factory"):
        OptimizerConfig(solver=optax.adam(1e-2))


def test_spg_rejects_solver_options():
    with pytest.raises(ValueError, match="optax solvers only"):
        OptimizerConfig(solver="spg", solver_options={"b1": 0.9})


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


def test_jit_cache_stable_across_optax_configs():
    """Two independently built optax configs must share one cache entry.

    The solver travels as a jit *static* argument, so an unstable key (a fresh
    ``GradientTransformation``, an inline ``partial``) would recompile on every
    solve. This is the regression guard for that.
    """
    panel = generate_returns(n_assets=11, n_days=300, seed=123)
    before = _solve_cached._cache_size()
    C.minimum_variance(
        panel,
        config=OptimizerConfig(solver="adamw", solver_options={"weight_decay": 1e-3}),
    )
    after_first = _solve_cached._cache_size()
    C.minimum_variance(
        panel,  # a *separately constructed* but equal config
        config=OptimizerConfig(solver="adamw", solver_options={"weight_decay": 1e-3}),
    )
    after_second = _solve_cached._cache_size()
    assert after_first == before + 1, "first optax solve should compile exactly one variant"
    assert after_second == after_first, "an equal config must reuse the compiled kernel"
