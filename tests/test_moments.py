"""Tests for moment estimators."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jaxfolio.moments.estimators import (
    as_matrix,
    correlation_from_covariance,
    ewma_covariance,
    ledoit_wolf_covariance,
    mean_returns,
    sample_covariance,
)


def test_as_matrix_from_dataframe(returns):
    mat, names = as_matrix(returns)
    assert mat.shape == (len(returns), len(returns.columns))
    assert names == [str(c) for c in returns.columns]


def test_sample_covariance_matches_numpy(returns):
    mat, _ = as_matrix(returns)
    cov = np.asarray(sample_covariance(mat))
    ref = np.cov(np.asarray(mat), rowvar=False)
    assert np.allclose(cov, ref, atol=1e-8)


def test_covariance_is_symmetric_psd(returns):
    mat, _ = as_matrix(returns)
    cov = np.asarray(sample_covariance(mat))
    assert np.allclose(cov, cov.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(cov)
    assert eigvals.min() > -1e-8


def test_ewma_covariance_shape_and_symmetry(returns):
    mat, _ = as_matrix(returns)
    cov = np.asarray(ewma_covariance(mat, halflife=30))
    n = mat.shape[1]
    assert cov.shape == (n, n)
    assert np.allclose(cov, cov.T, atol=1e-8)


def test_ledoit_wolf_shrinkage_in_unit_interval(returns):
    mat, _ = as_matrix(returns)
    cov, intensity = ledoit_wolf_covariance(mat)
    assert 0.0 <= intensity <= 1.0
    cov = np.asarray(cov)
    assert np.allclose(cov, cov.T, atol=1e-8)


def test_correlation_diagonal_is_one(returns):
    mat, _ = as_matrix(returns)
    corr = np.asarray(correlation_from_covariance(sample_covariance(mat)))
    assert np.allclose(np.diag(corr), 1.0, atol=1e-6)
    assert corr.max() <= 1.0 + 1e-6
    assert corr.min() >= -1.0 - 1e-6


def test_mean_annualization():
    mat = jnp.ones((10, 3)) * 0.01
    daily = mean_returns(mat)
    annual = mean_returns(mat, periods_per_year=252)
    assert np.allclose(np.asarray(annual), np.asarray(daily) * 252)
