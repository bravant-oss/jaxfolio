"""Tests for data ingestion and return utilities."""

from __future__ import annotations

import numpy as np
import polars as pl

from jaxfolio.data.returns import align, clean_returns, to_returns, train_test_split
from jaxfolio.data.synthetic import generate_prices, generate_returns


def test_generate_prices_shape_and_positivity():
    prices = generate_prices(n_assets=6, n_days=300, seed=0)
    assert prices.shape == (300, 7)  # explicit date + six assets
    assert (prices.select(pl.exclude("date")).to_numpy() > 0).all()
    assert prices.schema["date"] == pl.Date


def test_generate_returns_reproducible():
    r1 = generate_returns(n_assets=5, n_days=200, seed=3)
    r2 = generate_returns(n_assets=5, n_days=200, seed=3)
    assert r1.equals(r2)


def test_to_returns_simple_vs_log():
    prices = generate_prices(n_assets=3, n_days=100, seed=1)
    simple = to_returns(prices, kind="simple")
    log = to_returns(prices, kind="log")
    assert simple.shape == log.shape
    # For small returns, log ~ simple.
    assert np.allclose(simple.select(pl.exclude("date")), log.select(pl.exclude("date")), atol=1e-2)


def test_clean_returns_drops_sparse_columns():
    df = generate_returns(n_assets=4, n_days=100, seed=2)
    first = next(c for c in df.columns if c != "date")
    df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(first))
    cleaned = clean_returns(df, max_missing_frac=0.1)
    assert cleaned.shape[1] == 4  # explicit date + three retained assets


def test_train_test_split_chronological():
    df = generate_returns(n_assets=3, n_days=100, seed=4)
    train, test = train_test_split(df, test_size=0.2)
    assert len(train) + len(test) == len(df)
    assert train.get_column("date").max() < test.get_column("date").min()


def test_align_uses_explicit_date_column():
    left = generate_prices(n_assets=1, n_days=4, seed=1)
    right = generate_prices(n_assets=1, n_days=4, seed=2).slice(2).rename({"ASSET_00": "B"})
    inner_left, inner_right = align(left, right)
    assert inner_left.get_column("date").equals(inner_right.get_column("date"))
    assert inner_left.height == 2

    outer_left, outer_right = align(left, right, how="outer")
    assert outer_left.columns == ["date", "ASSET_00"]
    assert outer_right.columns == ["date", "B"]
    assert outer_left.height == outer_right.height == 4
