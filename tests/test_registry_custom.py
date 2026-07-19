"""Tests for the strategy registry and custom-strategy authoring."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jaxfolio as jf
from jaxfolio.backtest import backtest, compare
from jaxfolio.custom import CustomStrategy, custom_strategy
from jaxfolio.registry import (
    get_strategy,
    list_strategies,
    register_strategy,
    strategy_info,
    unregister,
)


def test_builtins_registered():
    names = list_strategies(builtin_only=True)
    assert "maximum_sharpe" in names
    assert "hierarchical_risk_parity" in names
    assert len(names) >= 15
    assert get_strategy("equal_weight") is jf.equal_weight


def test_register_and_lookup():
    @register_strategy("t_dummy", description="dummy", overwrite=True)
    def dummy(returns):
        return None

    assert "t_dummy" in list_strategies()
    assert get_strategy("t_dummy") is dummy
    assert strategy_info("t_dummy").description == "dummy"
    assert "t_dummy" in list_strategies(custom_only=True)
    unregister("t_dummy")
    assert "t_dummy" not in list_strategies()


def test_duplicate_without_overwrite_raises():
    @register_strategy("t_dup", overwrite=True)
    def a(returns):
        return None

    with pytest.raises(ValueError):
        register_strategy("t_dup")(a)
    unregister("t_dup")


def test_unknown_strategy_raises():
    with pytest.raises(KeyError):
        get_strategy("does_not_exist_xyz")


def test_custom_from_weights_valid(returns):
    def momentum(r):
        return np.clip(r.mean().to_numpy(), 0, None)

    strat = custom_strategy("mom", momentum)
    res = strat(returns)
    assert np.isclose(res.weights.sum(), 1.0, atol=1e-6)
    assert (res.weights >= 0).all()
    assert res.method == "mom"
    assert res.metadata["custom"] is True


def test_custom_from_weights_accepts_mapping(returns):
    assets = list(returns.columns)

    def pick_first(r):
        return {assets[0]: 1.0}

    res = custom_strategy("first", pick_first)(returns)
    assert np.isclose(res.weights.sum(), 1.0, atol=1e-6)
    assert res.weights[0] == pytest.approx(1.0)


def test_custom_from_objective_valid(returns):
    def obj(w, ctx):
        # Minimum variance with a small return tilt.
        return w @ ctx.cov @ w - 0.05 * (w @ ctx.mu)

    strat = CustomStrategy.from_objective("minvar_tilt", obj)
    res = strat(returns)
    assert np.isclose(res.weights.sum(), 1.0, atol=1e-4)
    assert (res.weights >= -1e-6).all()
    assert res.metadata["kind"] == "objective"
    assert "iterations" in res.metadata


def test_custom_from_objective_uses_context_n(returns):
    seen = {}

    def obj(w, ctx):
        seen["n"] = ctx.n
        return w @ ctx.cov @ w

    CustomStrategy.from_objective("cap", obj)(returns)
    assert seen["n"] == len(returns.columns)


def test_custom_weight_length_mismatch_raises(returns):
    def bad(r):
        return np.ones(3)  # wrong length

    with pytest.raises(ValueError):
        custom_strategy("bad", bad)(returns)


def test_custom_strategy_in_backtest(returns):
    strat = custom_strategy("inv_price_rank", lambda r: np.ones(r.shape[1]))
    res = backtest(returns, strat, lookback=100, rebalance_every=25)
    assert len(res.returns) > 0
    assert "sharpe" in res.metrics


def test_custom_and_builtin_compare(returns):
    strat = custom_strategy("half_and_half", lambda r: np.ones(r.shape[1]))
    results = compare(
        returns,
        {"Custom": strat, "1/N": jf.equal_weight},
        lookback=100,
        rebalance_every=25,
    )
    assert set(results) == {"Custom", "1/N"}


def test_register_via_from_weights(returns):
    custom_strategy("reg_test", lambda r: np.ones(r.shape[1]), register=True, description="reg")
    assert "reg_test" in list_strategies(custom_only=True)
    # It is callable through the registry.
    res = get_strategy("reg_test")(returns)
    assert np.isclose(res.weights.sum(), 1.0, atol=1e-6)
    unregister("reg_test")


def test_toolkit_finalize_result(returns):
    from jaxfolio import toolkit as tk

    mu, cov, names, _ = tk.moments(returns)
    w = jnp.full(len(names), 1.0 / len(names))
    res = tk.finalize_result(w, names, "tk", mu=mu, cov=cov)
    assert res.expected_return is not None
    assert res.volatility is not None
    assert np.isclose(res.weights.sum(), 1.0, atol=1e-6)
