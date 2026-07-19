"""Low-level result assembly and moment extraction.

Kept dependency-light on purpose: this module imports only the moment estimators
(never the optimizer package), so both the optimizers and the higher-level
:mod:`jaxfolio.toolkit` can import it without creating an import cycle.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jaxfolio.moments.estimators import as_matrix, mean_returns, sample_covariance
from jaxfolio.types import PortfolioResult

Array = jnp.ndarray

PERIODS_PER_YEAR = 252


def moments(returns, cov_estimator=None) -> tuple[Array, Array, list[str], Array]:
    """Return ``(mu, cov, asset_names, return_matrix)`` from a returns object.

    ``cov_estimator`` optionally overrides the default sample covariance.
    """
    mat, names = as_matrix(returns)
    mu = mean_returns(mat)
    cov = sample_covariance(mat) if cov_estimator is None else cov_estimator(mat)
    return mu, cov, names, mat


def equal_start(n: int) -> Array:
    """A uniform ``1/n`` starting weight vector for the solver."""
    return jnp.full(n, 1.0 / n)


def finalize_result(
    weights,
    assets: list[str],
    method: str,
    *,
    mu: Array | None = None,
    cov: Array | None = None,
    returns=None,
    risk_free: float = 0.0,
    periods_per_year: int = PERIODS_PER_YEAR,
    metadata: dict | None = None,
    clean_eps: float = 1e-10,
) -> PortfolioResult:
    """Assemble a :class:`PortfolioResult` with annualized diagnostics.

    Provide the moments either directly (``mu`` and ``cov``) or implicitly via
    ``returns`` (sample moments are computed from it). Weights are coerced to a
    flat float array and values below ``clean_eps`` in magnitude are zeroed. This
    is the single source of truth for the annualized return / volatility / Sharpe
    reported on every optimizer result — built-in and custom alike.
    """
    w = np.asarray(weights, dtype=float).reshape(-1)
    w = np.where(np.abs(w) < clean_eps, 0.0, w)

    if mu is None or cov is None:
        if returns is None:
            raise ValueError("finalize_result requires either (mu, cov) or returns")
        mat, _ = as_matrix(returns)
        mu = mean_returns(mat) if mu is None else mu
        cov = sample_covariance(mat) if cov is None else cov

    wj = jnp.asarray(w)
    ann_mu = float(jnp.dot(wj, mu) * periods_per_year)
    port_var = float(wj @ cov @ wj)
    ann_vol = float(np.sqrt(max(port_var, 0.0)) * np.sqrt(periods_per_year))
    excess = ann_mu - risk_free * periods_per_year
    sharpe = excess / ann_vol if ann_vol > 0 else None

    return PortfolioResult(
        weights=w,
        assets=list(assets),
        method=method,
        expected_return=ann_mu,
        volatility=ann_vol,
        sharpe=sharpe,
        metadata=metadata or {},
    )
