"""Data ingestion: CSV, Parquet, and (optionally) live market data via yfinance.

The CSV/Parquet loaders have no third-party dependency beyond pandas. The
yfinance loaders are guarded so importing this module never requires the
``[data]`` extra to be installed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _pivot_to_panel(
    df: pd.DataFrame, date_col: str, asset_col: str, price_col: str
) -> pd.DataFrame:
    """Reshape a long (tidy) frame into a wide date-by-asset price panel."""
    panel = df.pivot_table(index=date_col, columns=asset_col, values=price_col)
    panel.index = pd.to_datetime(panel.index)
    return panel.sort_index()


def load_csv(
    path: str | Path,
    *,
    date_col: str = "date",
    asset_col: str | None = None,
    price_col: str = "close",
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Load a price panel from CSV.

    Two layouts are supported:

    * **Wide** (default when ``asset_col`` is ``None``): one column per asset,
      plus a date column used as the index.
    * **Long/tidy** (when ``asset_col`` is given): columns ``date_col``,
      ``asset_col``, ``price_col`` are pivoted into a wide panel.
    """
    df = pd.read_csv(path, **read_csv_kwargs)
    if asset_col is not None:
        return _pivot_to_panel(df, date_col, asset_col, price_col)
    df[date_col] = pd.to_datetime(df[date_col])
    return df.set_index(date_col).sort_index()


def load_parquet(
    path: str | Path,
    *,
    date_col: str = "date",
    asset_col: str | None = None,
    price_col: str = "close",
) -> pd.DataFrame:
    """Load a price panel from Parquet (same layouts as :func:`load_csv`)."""
    df = pd.read_parquet(path)
    if asset_col is not None:
        return _pivot_to_panel(df, date_col, asset_col, price_col)
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    return df.sort_index()


def load_yfinance(
    tickers: list[str] | str,
    *,
    start: str | None = None,
    end: str | None = None,
    period: str | None = "2y",
    interval: str = "1d",
    price_field: str = "Close",
) -> pd.DataFrame:
    """Download a price panel from Yahoo Finance (requires the ``[data]`` extra).

    Returns a wide date-by-ticker panel of ``price_field`` values.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - exercised only without extra
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
    if isinstance(raw.columns, pd.MultiIndex):
        panel = raw[price_field]
    else:  # single ticker
        panel = raw[[price_field]]
        panel.columns = tickers
    return panel.dropna(how="all").sort_index()


def load_option_chain(
    ticker: str,
    *,
    expiry: str | None = None,
) -> pd.DataFrame:
    """Fetch and normalize an option chain from Yahoo Finance (``[data]`` extra).

    Returns a tidy frame with columns ``[type, strike, expiry, last, bid, ask,
    volume, open_interest, implied_vol]`` for calls and puts combined. If
    ``expiry`` is ``None`` the nearest expiry is used.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - exercised only without extra
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

    def _norm(df: pd.DataFrame, kind: str) -> pd.DataFrame:
        cols = {
            "strike": "strike",
            "lastPrice": "last",
            "bid": "bid",
            "ask": "ask",
            "volume": "volume",
            "openInterest": "open_interest",
            "impliedVolatility": "implied_vol",
        }
        out = df[list(cols)].rename(columns=cols)
        out.insert(0, "type", kind)
        out.insert(2, "expiry", pd.to_datetime(chosen))
        return out

    calls = _norm(chain.calls, "call")
    puts = _norm(chain.puts, "put")
    return pd.concat([calls, puts], ignore_index=True)
