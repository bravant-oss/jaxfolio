"""Tests for data ingestion and return utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from jaxfolio.data.returns import clean_returns, to_returns, train_test_split
from jaxfolio.data.synthetic import generate_prices, generate_returns


def test_generate_prices_shape_and_positivity():
    prices = generate_prices(n_assets=6, n_days=300, seed=0)
    assert prices.shape == (300, 6)
    assert (prices > 0).all().all()
    assert isinstance(prices.index, pd.DatetimeIndex)


def test_generate_returns_reproducible():
    r1 = generate_returns(n_assets=5, n_days=200, seed=3)
    r2 = generate_returns(n_assets=5, n_days=200, seed=3)
    pd.testing.assert_frame_equal(r1, r2)


def test_to_returns_simple_vs_log():
    prices = generate_prices(n_assets=3, n_days=100, seed=1)
    simple = to_returns(prices, kind="simple")
    log = to_returns(prices, kind="log")
    assert simple.shape == log.shape
    # For small returns, log ~ simple.
    assert np.allclose(simple.to_numpy(), log.to_numpy(), atol=1e-2)


def test_clean_returns_drops_sparse_columns():
    df = generate_returns(n_assets=4, n_days=100, seed=2)
    df.iloc[:, 0] = np.nan  # fully missing column
    cleaned = clean_returns(df, max_missing_frac=0.1)
    assert cleaned.shape[1] == 3


def test_train_test_split_chronological():
    df = generate_returns(n_assets=3, n_days=100, seed=4)
    train, test = train_test_split(df, test_size=0.2)
    assert len(train) + len(test) == len(df)
    assert train.index.max() < test.index.min()
