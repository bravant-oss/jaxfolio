"""Return computation, alignment, and train/test splitting.

All functions accept and return pandas objects so they compose cleanly with the
loaders, and hand off clean float arrays to the JAX optimizers downstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_returns(
    prices: pd.DataFrame,
    *,
    kind: str = "simple",
    dropna: bool = True,
) -> pd.DataFrame:
    """Convert a price panel to returns.

    Parameters
    ----------
    prices:
        Price panel (rows = dates, columns = assets).
    kind:
        ``"simple"`` for arithmetic returns ``p_t / p_{t-1} - 1`` or ``"log"``
        for ``log(p_t / p_{t-1})``.
    dropna:
        Drop the leading NaN row (and any all-NaN rows) if ``True``.
    """
    if kind == "simple":
        rets = prices.pct_change()
    elif kind == "log":
        rets = np.log(prices / prices.shift(1))
    else:
        raise ValueError("kind must be 'simple' or 'log'")
    if dropna:
        rets = rets.dropna(how="all")
    return rets


def align(*frames: pd.DataFrame, how: str = "inner") -> list[pd.DataFrame]:
    """Align multiple frames on their common (or union) index.

    Useful when combining price series pulled from different sources.
    """
    if not frames:
        return []
    index = frames[0].index
    for f in frames[1:]:
        index = index.join(f.index, how=how)
    return [f.reindex(index) for f in frames]


def clean_returns(
    returns: pd.DataFrame,
    *,
    max_missing_frac: float = 0.1,
    fill: str = "zero",
) -> pd.DataFrame:
    """Drop sparse columns and fill remaining gaps.

    Columns missing more than ``max_missing_frac`` of observations are dropped.
    Remaining NaNs are filled with ``0.0`` (``"zero"``) or forward-filled
    (``"ffill"``).
    """
    keep = returns.columns[returns.isna().mean() <= max_missing_frac]
    out = returns[keep].copy()
    if fill == "zero":
        out = out.fillna(0.0)
    elif fill == "ffill":
        out = out.ffill().fillna(0.0)
    else:
        raise ValueError("fill must be 'zero' or 'ffill'")
    return out


def train_test_split(
    returns: pd.DataFrame,
    *,
    test_size: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split into in-sample and out-of-sample sets.

    ``test_size`` is the fraction of the most recent observations held out.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1)")
    n = len(returns)
    cut = int(round(n * (1.0 - test_size)))
    return returns.iloc[:cut], returns.iloc[cut:]


def annualization_factor(returns: pd.DataFrame | pd.Series, periods_per_year: int = 252) -> int:
    """Return the annualization factor (kept explicit for clarity in call sites)."""
    return periods_per_year
