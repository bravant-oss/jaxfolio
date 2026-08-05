"""Data ingestion into explicit-date Polars frames."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl


def _parse_date(df: pl.DataFrame, column: str) -> pl.DataFrame:
    if column not in df.columns:
        return df
    dtype = df.schema[column]
    if dtype == pl.Utf8:
        return df.with_columns(pl.col(column).str.to_date(strict=False))
    if dtype == pl.Datetime:
        return df.with_columns(pl.col(column).dt.date())
    return df


def _pivot_to_panel(
    df: pl.DataFrame, date_col: str, asset_col: str, price_col: str
) -> pl.DataFrame:
    """Reshape a long frame into a wide date-by-asset panel."""
    df = _parse_date(df, date_col)
    return df.pivot(
        on=asset_col, index=date_col, values=price_col, aggregate_function="first"
    ).sort(date_col)


def load_csv(
    path: str | Path,
    *,
    date_col: str = "date",
    asset_col: str | None = None,
    price_col: str = "close",
    **read_csv_kwargs,
) -> pl.DataFrame:
    """Load a wide or long/tidy CSV price panel."""
    df = pl.read_csv(path, **read_csv_kwargs)
    if asset_col is not None:
        return _pivot_to_panel(df, date_col, asset_col, price_col)
    return _parse_date(df, date_col).sort(date_col)


def load_parquet(
    path: str | Path,
    *,
    date_col: str = "date",
    asset_col: str | None = None,
    price_col: str = "close",
) -> pl.DataFrame:
    """Load a wide or long/tidy Parquet price panel."""
    df = pl.read_parquet(path)
    if asset_col is not None:
        return _pivot_to_panel(df, date_col, asset_col, price_col)
    return _parse_date(df, date_col).sort(date_col) if date_col in df.columns else df


def _provider_panel_to_polars(panel, tickers: list[str]) -> pl.DataFrame:
    """Convert a yfinance provider response without importing its frame library."""
    values = panel.to_numpy(dtype=float)
    dates = [value.date() if hasattr(value, "date") else value for value in panel.index]
    names = [str(c) for c in panel.columns]
    if len(names) == 1:
        names = tickers
    return pl.DataFrame({"date": dates, **dict(zip(names, values.T, strict=True))}).sort("date")


def load_yfinance(
    tickers: list[str] | str,
    *,
    start: str | None = None,
    end: str | None = None,
    period: str | None = "2y",
    interval: str = "1d",
    price_field: str = "Close",
) -> pl.DataFrame:
    """Download a wide price panel from Yahoo Finance as Polars."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "load_yfinance requires the optional 'data' extra: "
            "install with `uv sync --extra data` or `pip install jaxfolio[data]`."
        ) from exc

    if isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.split(",") if t.strip()]
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        period=None if start else period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    # yfinance uses hierarchical columns for multiple tickers. Duck-type that
    # shape so the provider's frame library stays an implementation detail.
    panel = raw[price_field] if getattr(raw.columns, "nlevels", 1) > 1 else raw[[price_field]]
    out = _provider_panel_to_polars(panel, tickers)
    assets = [c for c in out.columns if c != "date"]
    missing = pl.col(assets).is_null() | pl.col(assets).is_nan()
    return out.filter(~pl.all_horizontal(missing))


def load_option_chain(ticker: str, *, expiry: str | None = None) -> pl.DataFrame:
    """Fetch and normalize a Yahoo Finance option chain as a tidy Polars frame."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "load_option_chain requires the optional 'data' extra: "
            "install with `uv sync --extra data`."
        ) from exc

    tk = yf.Ticker(ticker)
    expiries = tk.options
    if not expiries:
        raise ValueError(f"No listed options found for {ticker!r}")
    chosen = expiry or expiries[0]
    chain = tk.option_chain(chosen)

    def _norm(df, kind: str) -> pl.DataFrame:
        source = {
            "strike": "strike",
            "lastPrice": "last",
            "bid": "bid",
            "ask": "ask",
            "volume": "volume",
            "openInterest": "open_interest",
            "impliedVolatility": "implied_vol",
        }
        values = {dest: df[src].to_numpy() for src, dest in source.items()}
        return (
            pl.DataFrame(values)
            .with_columns(
                pl.lit(kind).alias("type"), pl.lit(date.fromisoformat(chosen)).alias("expiry")
            )
            .select(
                "type",
                "strike",
                "expiry",
                "last",
                "bid",
                "ask",
                "volume",
                "open_interest",
                "implied_vol",
            )
        )

    return pl.concat([_norm(chain.calls, "call"), _norm(chain.puts, "put")])
