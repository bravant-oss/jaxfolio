"""Return computation, alignment, cleaning, and train/test splitting.

Data panels are Polars frames. A temporal column (conventionally ``date``) is
kept as an ordinary, explicit column; every other numeric column is an asset.
"""

from __future__ import annotations

import polars as pl


def date_column(frame: pl.DataFrame) -> str | None:
    """Return the panel's temporal key, preferring the conventional name."""
    if "date" in frame.columns:
        return "date"
    return next(
        (name for name, dtype in frame.schema.items() if dtype.is_temporal()),
        None,
    )


def asset_columns(frame: pl.DataFrame) -> list[str]:
    """Return numeric asset columns, excluding temporal metadata."""
    date_col = date_column(frame)
    return [
        name
        for name, dtype in frame.schema.items()
        if name != date_col and dtype.is_numeric() and dtype != pl.Boolean
    ]


def to_returns(
    prices: pl.DataFrame,
    *,
    kind: str = "simple",
    dropna: bool = True,
) -> pl.DataFrame:
    """Convert a wide price panel to simple or logarithmic returns."""
    assets = asset_columns(prices)
    if kind == "simple":
        exprs = [(pl.col(c) / pl.col(c).shift(1) - 1.0).alias(c) for c in assets]
    elif kind == "log":
        exprs = [(pl.col(c) / pl.col(c).shift(1)).log().alias(c) for c in assets]
    else:
        raise ValueError("kind must be 'simple' or 'log'")
    out = prices.with_columns(exprs)
    if dropna:
        missing = pl.col(assets).is_null() | pl.col(assets).is_nan()
        out = out.filter(~pl.all_horizontal(missing))
    return out


def align(*frames: pl.DataFrame, how: str = "inner") -> list[pl.DataFrame]:
    """Align frames on their common (``inner``) or combined (``outer``) dates."""
    if not frames:
        return []
    if how not in {"inner", "outer"}:
        raise ValueError("how must be 'inner' or 'outer'")
    keys = [date_column(frame) for frame in frames]
    if any(key is None for key in keys):
        raise ValueError("align requires a date or datetime column in every frame")
    key = keys[0]
    assert key is not None
    normalized = [f.rename({k: key}) if k != key else f for f, k in zip(frames, keys, strict=True)]
    dates = normalized[0].select(key)
    for frame in normalized[1:]:
        join_kind = "full" if how == "outer" else "inner"
        dates = dates.join(frame.select(key), on=key, how=join_kind, coalesce=True)
    dates = dates.unique().sort(key)
    return [dates.join(frame, on=key, how="left") for frame in normalized]


def clean_returns(
    returns: pl.DataFrame,
    *,
    max_missing_frac: float = 0.1,
    fill: str = "zero",
) -> pl.DataFrame:
    """Drop sparse asset columns and fill remaining null/NaN values."""
    assets = asset_columns(returns)
    n = max(returns.height, 1)
    missing = returns.select(
        [((pl.col(c).is_null() | pl.col(c).is_nan()).sum() / n).alias(c) for c in assets]
    ).row(0, named=True)
    keep = [c for c in assets if missing[c] <= max_missing_frac]
    date_col = date_column(returns)
    out = returns.select(([date_col] if date_col else []) + keep)
    if fill == "zero":
        return out.with_columns(pl.col(keep).fill_null(0.0).fill_nan(0.0))
    if fill == "ffill":
        return out.with_columns(
            pl.col(keep).fill_nan(None).fill_null(strategy="forward").fill_null(0.0)
        )
    raise ValueError("fill must be 'zero' or 'ffill'")


def train_test_split(
    returns: pl.DataFrame,
    *,
    test_size: float = 0.25,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Chronological split into in-sample and out-of-sample frames."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1)")
    cut = int(round(returns.height * (1.0 - test_size)))
    return returns.slice(0, cut), returns.slice(cut)


def annualization_factor(returns: pl.DataFrame | pl.Series, periods_per_year: int = 252) -> int:
    """Return the explicit annualization factor used by callers."""
    return periods_per_year
