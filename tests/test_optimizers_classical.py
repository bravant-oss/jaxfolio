"""Tests for classical optimizers."""

from __future__ import annotations

import numpy as np
import pytest

from jaxfolio.moments.estimators import as_matrix, sample_covariance
from jaxfolio.optimizers import classical as C
from jaxfolio.types import OptimizerConfig

ALL_LONG_ONLY = [
    C.equal_weight,
    C.inverse_volatility,
    C.minimum_variance,
    C.maximum_sharpe,
    C.maximum_diversification,
    C.risk_parity,
    C.kelly,
    C.min_cvar,
    C.mean_variance,
    C.black_litterman,
]


@pytest.mark.parametrize("method", ALL_LONG_ONLY)
def test_weights_sum_to_one_long_only(returns, method):
    res = method(returns)
    w = res.weights
    assert np.isclose(w.sum(), 1.0, atol=1e-4), f"{method.__name__} weights sum={w.sum()}"
    assert (w >= -1e-4).all(), f"{method.__name__} has negative weights"
    assert len(res.assets) == len(w)


def test_equal_weight_is_uniform(returns):
    res = C.equal_weight(returns)
    n = len(returns.columns)
    assert np.allclose(res.weights, 1.0 / n)


def test_minimum_variance_has_low_variance(returns):
    """Min-variance should have lower variance than equal weight."""
    mat, _ = as_matrix(returns)
    cov = np.asarray(sample_covariance(mat))
    mv = C.minimum_variance(returns)
    ew = C.equal_weight(returns)
    var_mv = mv.weights @ cov @ mv.weights
    var_ew = ew.weights @ cov @ ew.weights
    assert var_mv <= var_ew + 1e-8


def test_risk_parity_equalizes_risk_contributions(returns):
    res = C.risk_parity(returns)
    rc = np.array(res.metadata["risk_contributions"])
    # All contributions should be close to 1/N.
    n = len(rc)
    assert np.allclose(rc, 1.0 / n, atol=0.03), f"risk contributions not equal: {rc}"


def test_inverse_volatility_favors_low_vol_assets(returns):
    mat, _ = as_matrix(returns)
    cov = np.asarray(sample_covariance(mat))
    vols = np.sqrt(np.diag(cov))
    res = C.inverse_volatility(returns)
    # Lowest-vol asset should get the highest weight.
    assert np.argmax(res.weights) == np.argmin(vols)


def test_min_cvar_reports_cvar(returns):
    res = C.min_cvar(returns, alpha=0.95)
    assert "cvar" in res.metadata
    assert res.metadata["alpha"] == 0.95


def test_black_litterman_with_views(returns):
    assets = list(returns.columns)
    views = {assets[0]: 0.001, assets[1]: -0.0005}
    res = C.black_litterman(returns, views=views)
    assert res.metadata["n_views"] == 2
    assert np.isclose(res.weights.sum(), 1.0, atol=1e-4)


def test_mean_variance_risk_aversion_monotonic(returns):
    """Higher risk aversion => lower portfolio variance."""
    mat, _ = as_matrix(returns)
    cov = np.asarray(sample_covariance(mat))
    low = C.mean_variance(returns, risk_aversion=0.5)
    high = C.mean_variance(returns, risk_aversion=20.0)
    var_low = low.weights @ cov @ low.weights
    var_high = high.weights @ cov @ high.weights
    assert var_high <= var_low + 1e-6


def test_config_long_short_allows_negative(returns):
    cfg = OptimizerConfig(long_only=False, weight_bounds=(-0.5, 1.0))
    res = C.mean_variance(returns, risk_aversion=1.0, config=cfg)
    assert np.isclose(res.weights.sum(), 1.0, atol=1e-3)
