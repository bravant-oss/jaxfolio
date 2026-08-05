"""Reproducible synthetic market data as Polars frames."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from jaxfolio.data.returns import to_returns


def _random_correlation(n: int, rng: np.random.Generator, strength: float) -> np.ndarray:
    n_factors = max(1, n // 3)
    loadings = rng.normal(size=(n, n_factors)) * np.sqrt(strength)
    cov = loadings @ loadings.T
    cov[np.diag_indices(n)] += 1.0 - strength
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)


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
) -> pl.DataFrame:
    """Generate correlated daily GBM prices with an explicit ``date`` column."""
    rng = np.random.default_rng(seed)
    if tickers is None:
        tickers = [f"ASSET_{i:02d}" for i in range(n_assets)]
    elif len(tickers) != n_assets:
        raise ValueError("len(tickers) must equal n_assets")

    dt = 1.0 / 252.0
    vols = rng.uniform(*annual_vol_range, size=n_assets)
    drifts = rng.normal(mean_annual_return, 0.03, size=n_assets)
    chol = np.linalg.cholesky(_random_correlation(n_assets, rng, correlation_strength))
    z = rng.standard_normal(size=(n_days, n_assets)) @ chol.T
    log_rets = (drifts - 0.5 * vols**2) * dt + vols * np.sqrt(dt) * z
    prices = np.exp(np.log(start_price) + np.cumsum(log_rets, axis=0))

    start = np.datetime64(date.fromisoformat(start_date))
    dates = np.busday_offset(start, np.arange(n_days), roll="forward").astype("datetime64[D]")
    return pl.DataFrame({"date": dates, **dict(zip(tickers, prices.T, strict=True))})


def generate_returns(n_assets: int = 12, n_days: int = 756, **kwargs) -> pl.DataFrame:
    """Convenience wrapper returning simple daily returns."""
    return to_returns(generate_prices(n_assets=n_assets, n_days=n_days, **kwargs))
