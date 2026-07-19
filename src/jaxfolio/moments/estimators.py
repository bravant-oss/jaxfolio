"""Moment estimators (mean & covariance) implemented in JAX.

Every estimator takes a ``(T, N)`` return array and returns JAX arrays, so the
downstream optimizers can jit through the whole pipeline. Inputs may be pandas
frames or numpy arrays; :func:`as_matrix` normalizes them.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd

Array = jnp.ndarray


def as_matrix(returns: pd.DataFrame | np.ndarray | Array) -> tuple[Array, list[str]]:
    """Return ``(matrix, asset_names)`` from a returns object.

    Accepts a pandas DataFrame (columns become asset names), or a raw array
    (names default to ``asset_0``...). NaNs are replaced with zeros.
    """
    if isinstance(returns, pd.DataFrame):
        names = [str(c) for c in returns.columns]
        mat = jnp.asarray(returns.to_numpy(dtype=float))
    else:
        mat = jnp.asarray(np.asarray(returns, dtype=float))
        if mat.ndim != 2:
            raise ValueError("returns must be 2-D (T x N)")
        names = [f"asset_{i}" for i in range(mat.shape[1])]
    mat = jnp.nan_to_num(mat, nan=0.0)
    return mat, names


def mean_returns(returns: Array, *, periods_per_year: int | None = None) -> Array:
    """Sample mean of returns; annualized if ``periods_per_year`` is given."""
    mu = jnp.mean(returns, axis=0)
    if periods_per_year is not None:
        mu = mu * periods_per_year
    return mu


def sample_covariance(returns: Array, *, periods_per_year: int | None = None) -> Array:
    """Sample covariance matrix (denominator ``T - 1``); optionally annualized.

    Requires at least two observations; a single row has no defined covariance.
    """
    t = returns.shape[0]
    if t < 2:
        raise ValueError(f"sample_covariance needs at least 2 observations, got {t}")
    demeaned = returns - jnp.mean(returns, axis=0, keepdims=True)
    cov = (demeaned.T @ demeaned) / (t - 1)
    if periods_per_year is not None:
        cov = cov * periods_per_year
    return cov


def ewma_covariance(
    returns: Array,
    *,
    halflife: float = 63.0,
    periods_per_year: int | None = None,
) -> Array:
    """Exponentially-weighted covariance matrix.

    Recent observations receive more weight; ``halflife`` is in periods (default
    ~one quarter of trading days).
    """
    t = returns.shape[0]
    decay = jnp.log(2.0) / halflife
    ages = jnp.arange(t)[::-1]  # most recent row -> age 0
    weights = jnp.exp(-decay * ages)
    weights = weights / jnp.sum(weights)
    wmean = jnp.sum(weights[:, None] * returns, axis=0)
    demeaned = returns - wmean
    cov = (demeaned * weights[:, None]).T @ demeaned
    # Symmetrize against floating point drift.
    cov = 0.5 * (cov + cov.T)
    if periods_per_year is not None:
        cov = cov * periods_per_year
    return cov


def ledoit_wolf_covariance(
    returns: Array,
    *,
    periods_per_year: int | None = None,
) -> tuple[Array, float]:
    """Ledoit-Wolf shrinkage toward a scaled-identity target.

    Returns ``(shrunk_cov, shrinkage_intensity)``. The shrinkage intensity is
    estimated analytically (Ledoit & Wolf, 2004) and clipped to ``[0, 1]``.
    """
    t, n = returns.shape
    x = returns - jnp.mean(returns, axis=0, keepdims=True)
    sample = (x.T @ x) / t

    mu = jnp.trace(sample) / n
    target = mu * jnp.eye(n)

    # pi: sum of asymptotic variances of the sample covariance entries.
    x2 = x**2
    phi_mat = (x2.T @ x2) / t - sample**2
    pi_hat = jnp.sum(phi_mat)

    # rho: for an identity-scaled target the off-diagonal correction is zero;
    # only the diagonal contributes.
    rho_hat = jnp.sum(jnp.diag(phi_mat))

    gamma_hat = jnp.sum((sample - target) ** 2)
    # When the sample already equals the target (e.g. n == 1) gamma_hat is 0 and
    # the shrinkage is undefined; fall back to no shrinkage rather than NaN.
    safe_gamma = jnp.where(gamma_hat > 0, gamma_hat, 1.0)
    kappa = jnp.where(gamma_hat > 0, (pi_hat - rho_hat) / safe_gamma, 0.0)
    shrinkage = jnp.clip(kappa / t, 0.0, 1.0)

    shrunk = shrinkage * target + (1.0 - shrinkage) * sample
    if periods_per_year is not None:
        shrunk = shrunk * periods_per_year
    return shrunk, float(shrinkage)


def correlation_from_covariance(cov: Array) -> Array:
    """Convert a covariance matrix to a correlation matrix."""
    d = jnp.sqrt(jnp.clip(jnp.diag(cov), 1e-18, None))
    corr = cov / jnp.outer(d, d)
    return jnp.clip(corr, -1.0, 1.0)
