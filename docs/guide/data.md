# Data

Optimizers consume a wide **returns panel** — rows are periods, columns are
assets — as a pandas `DataFrame` or a 2-D array. jaxfolio provides a reproducible
synthetic generator plus loaders for CSV, Parquet, and Yahoo Finance, and
utilities to convert prices to returns and split them for out-of-sample testing.

## Synthetic data

[`generate_returns`](../reference/data.md#jaxfolio.data.synthetic.generate_returns)
(and its price-level sibling
[`generate_prices`](../reference/data.md#jaxfolio.data.synthetic.generate_prices))
produce correlated geometric-Brownian-motion panels with **no network access** —
this is what makes the tests, examples, and every snippet in these docs
reproducible and offline-friendly.

```python
import jaxfolio as jf

returns = jf.generate_returns(
    n_assets=12,
    n_days=756,                       # ≈ 3 years of trading days
    seed=7,                           # reproducible
    correlation_strength=0.5,         # average cross-asset coupling in [0, 1)
    annual_vol_range=(0.15, 0.45),    # per-asset vols drawn from this range
    mean_annual_return=0.08,
)
```

The generator builds a valid PSD correlation matrix from a low-rank factor model,
Cholesky-factorizes it, and simulates GBM log-returns. Pass `tickers=[...]` for
explicit column names (otherwise `ASSET_00…`).

## Loading real data

!!! note "Requires the `data` extra"
    The Yahoo Finance and Parquet loaders need `uv add "jaxfolio[data]"`.

=== "Yahoo Finance"

    ```python
    prices = jf.load_yfinance(["AAPL", "MSFT", "NVDA", "JPM"], period="2y", interval="1d")
    returns = jf.to_returns(prices)
    ```

    Returns a wide date-by-ticker panel of adjusted close prices.

=== "CSV"

    ```python
    # Wide layout: one column per asset + a date column
    prices = jf.load_csv("prices.csv", date_col="date")

    # Long/tidy layout: pivot date/asset/price columns into a panel
    prices = jf.load_csv("prices.csv", date_col="date",
                         asset_col="ticker", price_col="close")
    ```

=== "Parquet"

    ```python
    prices = jf.load_parquet("prices.parquet", date_col="date")
    ```

## Prices → returns

[`to_returns`](../reference/data.md#jaxfolio.data.returns.to_returns) converts a
price panel to simple or log returns:

```python
returns = jf.to_returns(prices, kind="simple")   # p_t / p_{t-1} − 1
returns = jf.to_returns(prices, kind="log")       # log(p_t / p_{t-1})
```

## Out-of-sample splitting

[`train_test_split`](../reference/data.md#jaxfolio.data.returns.train_test_split)
splits **chronologically** (never shuffled — this is time-series data), holding
out the most recent `test_size` fraction:

```python
train, test = jf.train_test_split(returns, test_size=0.25)

result = jf.maximum_sharpe(train)     # fit in-sample
# ... then evaluate `result` weights on `test`
```

## Option chains

[`load_option_chain`](../reference/options.md) fetches and normalizes an option
chain from Yahoo Finance (also the `data` extra) into a tidy frame with columns
`[type, strike, expiry, last, bid, ask, volume, open_interest, implied_vol]`:

```python
chain = jf.load_option_chain("AAPL")          # nearest expiry
chain = jf.load_option_chain("AAPL", expiry="2026-01-16")
```

## Covariance estimators

The default moment estimator is the sample covariance, but the
[toolkit](../reference/toolkit.md) exposes robust alternatives you can pass
anywhere moments are computed:

| Estimator | Use when |
|---|---|
| `sample_covariance` | plenty of data relative to assets (default) |
| `ewma_covariance` | recent observations should weigh more (`halflife`) |
| `ledoit_wolf_covariance` | \(N\) is large vs. \(T\); shrinks toward scaled identity |

```python
from jaxfolio import toolkit as tk

mat, _ = tk.as_matrix(returns)
cov_ewma = tk.ewma_covariance(mat, halflife=63.0)
cov_lw, intensity = tk.ledoit_wolf_covariance(mat)   # also returns the shrinkage λ
```
