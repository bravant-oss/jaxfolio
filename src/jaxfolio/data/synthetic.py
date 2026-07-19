"""Synthetic market data generation.

A correlated geometric-Brownian-motion generator that produces realistic price
panels without any network access. Used throughout the tests and examples so
everything is reproducible and offline-friendly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _random_correlation(n: int, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Build a valid correlation matrix with controllable average coupling.

    ``strength`` in ``[0, 1)`` scales how correlated the assets are. We draw a
    low-rank factor loading and add idiosyncratic variance, then normalize to a
    correlation matrix (guaranteed PSD).
    """
    n_factors = max(1, n // 3)
    loadings = rng.normal(size=(n, n_factors)) * np.sqrt(strength)
    cov = loadings @ loadings.T
    cov[np.diag_indices(n)] += 1.0 - strength
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    return corr


def generate_prices(
    n_assets: int = 12,
    n_days: int = 756,
    *,
    seed: int = 0,
    start_price: float = 100.0,
    mean_annual_return: float = 0.08,
    annual_vol_range: tuple[float, float] = (0.15, 0.45),
    correlation_strength: float = 0.5,
    start_date: str = "2021-01-01",
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Generate a panel of correlated daily prices via geometric Brownian motion.

    Parameters
    ----------
    n_assets, n_days:
        Panel shape. ``n_days`` uses ~252 trading days per year.
    seed:
        RNG seed for reproducibility.
    mean_annual_return:
        Center of the per-asset drift (drifts are jittered around this).
    annual_vol_range:
        Range from which per-asset annual volatilities are drawn.
    correlation_strength:
        Average cross-asset correlation in ``[0, 1)``.
    tickers:
        Optional explicit column names; otherwise ``ASSET_00`` ... are used.

    Returns
    -------
    pandas.DataFrame
        Indexed by business day, one column per asset, containing prices.
    """
    rng = np.random.default_rng(seed)
    if tickers is None:
        tickers = [f"ASSET_{i:02d}" for i in range(n_assets)]
    elif len(tickers) != n_assets:
        raise ValueError("len(tickers) must equal n_assets")

    dt = 1.0 / 252.0
    vols = rng.uniform(*annual_vol_range, size=n_assets)
    drifts = rng.normal(mean_annual_return, 0.03, size=n_assets)

    corr = _random_correlation(n_assets, rng, correlation_strength)
    chol = np.linalg.cholesky(corr)

    z = rng.standard_normal(size=(n_days, n_assets)) @ chol.T
    # GBM log-returns: (mu - 0.5 sigma^2) dt + sigma sqrt(dt) dW
    log_rets = (drifts - 0.5 * vols**2) * dt + vols * np.sqrt(dt) * z
    log_price = np.log(start_price) + np.cumsum(log_rets, axis=0)
    prices = np.exp(log_price)

    dates = pd.bdate_range(start=start_date, periods=n_days)
    return pd.DataFrame(prices, index=dates, columns=tickers)


def generate_returns(n_assets: int = 12, n_days: int = 756, **kwargs) -> pd.DataFrame:
    """Convenience wrapper returning simple daily returns from synthetic prices."""
    prices = generate_prices(n_assets=n_assets, n_days=n_days, **kwargs)
    return prices.pct_change().dropna()
