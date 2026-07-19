"""Tests for graph-based and learning-based optimizers."""

from __future__ import annotations

import numpy as np

from jaxfolio.optimizers import graph as G
from jaxfolio.optimizers import learning as L


def test_hrp_weights_valid(returns):
    res = G.hierarchical_risk_parity(returns)
    assert np.isclose(res.weights.sum(), 1.0, atol=1e-6)
    assert (res.weights >= 0).all()
    assert "linkage" in res.metadata
    assert "order" in res.metadata


def test_herc_weights_valid(returns):
    res = G.hierarchical_equal_risk(returns, n_clusters=3)
    assert np.isclose(res.weights.sum(), 1.0, atol=1e-6)
    assert (res.weights >= 0).all()
    assert res.metadata["n_clusters"] == 3


def test_mst_centrality_valid(returns):
    res = G.mst_centrality(returns)
    assert np.isclose(res.weights.sum(), 1.0, atol=1e-6)
    assert (res.weights >= 0).all()
    assert "degree" in res.metadata
    assert len(res.metadata["degree"]) == len(res.assets)


def test_deep_sharpe_trains_and_allocates(small_returns):
    res = L.deep_sharpe(small_returns, lookback=30, epochs=40, hidden=(16,))
    assert np.isclose(res.weights.sum(), 1.0, atol=1e-4)
    assert (res.weights >= 0).all()
    assert res.metadata["epochs"] == 40


def test_online_gradient_valid(returns):
    res = L.online_gradient(returns, eta=0.05)
    assert np.isclose(res.weights.sum(), 1.0, atol=1e-6)
    assert (res.weights >= 0).all()
    assert res.metadata["final_wealth"] > 0
    assert len(res.metadata["wealth_path"]) == len(returns)


def test_hrp_order_is_permutation(returns):
    res = G.hierarchical_risk_parity(returns)
    order = res.metadata["order"]
    assert sorted(order) == list(range(len(res.assets)))
