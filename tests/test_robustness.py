"""Regression tests for edge-case robustness fixes (degenerate inputs, etc.)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jaxfolio as jf
from jaxfolio.backtest import metrics as M
from jaxfolio.llm import FakeLLM, llm_views
from jaxfolio.llm.client import parse_json
from jaxfolio.moments.estimators import ledoit_wolf_covariance, sample_covariance
from jaxfolio.options import black_scholes_price, chain_greeks, implied_volatility, price_call_chain
from jaxfolio.types import OptimizerConfig


@pytest.fixture
def zero_var_returns():
    r = jf.generate_returns(n_assets=4, n_days=300, seed=1)
    r.iloc[:, 0] = 0.0  # a constant, zero-variance column
    return r


# --- degenerate covariance / optimizers ------------------------------------ #
def test_sample_covariance_requires_two_rows():
    with pytest.raises(ValueError):
        sample_covariance(jnp.ones((1, 3)))


def test_ledoit_wolf_single_asset_no_nan():
    r = jf.generate_returns(n_assets=1, n_days=200, seed=0)
    cov, shrink = ledoit_wolf_covariance(jnp.asarray(r.to_numpy()))
    assert not np.isnan(np.asarray(cov)).any()
    assert 0.0 <= shrink <= 1.0


def test_inverse_volatility_zero_variance(zero_var_returns):
    w = jf.inverse_volatility(zero_var_returns).weights
    assert np.isclose(w.sum(), 1.0, atol=1e-6)
    assert not np.isnan(w).any()


def test_risk_parity_zero_variance(zero_var_returns):
    w = jf.risk_parity(zero_var_returns).weights
    assert np.isclose(w.sum(), 1.0, atol=1e-6)
    assert not np.isnan(w).any()


def test_black_litterman_zero_variance_view(zero_var_returns):
    assets = list(zero_var_returns.columns)
    res = jf.black_litterman(zero_var_returns, views={assets[0]: 0.01})
    assert not np.isnan(res.weights).any()
    assert np.isclose(res.weights.sum(), 1.0, atol=1e-4)


def test_black_litterman_zero_confidence_no_div_by_zero():
    r = jf.generate_returns(n_assets=5, seed=2)
    res = jf.black_litterman(r, views={r.columns[0]: 0.002}, view_confidence=0.0)
    assert not np.isnan(res.weights).any()


def test_hrp_herc_single_asset():
    r = jf.generate_returns(n_assets=1, n_days=200, seed=0)
    assert np.isclose(jf.hierarchical_risk_parity(r).weights.sum(), 1.0)
    assert np.isclose(jf.hierarchical_equal_risk(r).weights.sum(), 1.0)


# --- shorting semantics ---------------------------------------------------- #
def test_config_bounds_resolver():
    assert OptimizerConfig(long_only=True).bounds() == (0.0, 1.0)
    assert OptimizerConfig(long_only=False).bounds() == (-1.0, 1.0)
    assert OptimizerConfig(weight_bounds=(-0.3, 0.8)).bounds() == (-0.3, 0.8)


def test_long_short_config_permits_negative_weights():
    # A strongly negative-drift asset should be shortable when long_only=False.
    r = jf.generate_returns(n_assets=5, seed=9, mean_annual_return=0.0)
    r.iloc[:, 0] -= 0.03
    res = jf.mean_variance(r, risk_aversion=5.0, config=OptimizerConfig(long_only=False))
    assert np.isclose(res.weights.sum(), 1.0, atol=1e-3)
    # bounds allow it (the projection is box-budget, not simplex)
    assert res.weights.min() >= -1.0 - 1e-6


# --- custom strategy sign handling ----------------------------------------- #
def test_custom_net_short_not_inverted():
    from jaxfolio.custom import custom_strategy

    r = jf.generate_returns(n_assets=4, seed=2)
    w = custom_strategy("ls", lambda r: np.array([1.0, -2.0, 0.0, 0.0]))(r).weights
    assert w[0] > 0 and w[1] < 0  # stance preserved, not sign-flipped


def test_custom_positive_weights_renormalized():
    from jaxfolio.custom import custom_strategy

    r = jf.generate_returns(n_assets=3, seed=2)
    w = custom_strategy("pos", lambda r: np.array([2.0, 1.0, 1.0]))(r).weights
    assert np.isclose(w.sum(), 1.0, atol=1e-9)


# --- options API ----------------------------------------------------------- #
def test_chain_greeks_includes_rho():
    g = chain_greeks(100.0, jnp.array([90.0, 100.0, 110.0]), 0.25, 0.2)
    assert "rho" in g
    assert g["rho"].shape == (3,)


def test_price_call_chain_scalar_broadcast():
    prices = price_call_chain(100.0, jnp.array([90.0, 100.0, 110.0]), 0.25, 0.2)
    assert prices.shape == (3,)
    assert (np.asarray(prices) > 0).all()


def test_implied_vol_nan_below_intrinsic():
    # Price below intrinsic value has no implied vol -> NaN, not a silent floor.
    iv = float(implied_volatility(0.5, 100.0, 90.0, 0.5, is_call=True))
    assert np.isnan(iv)


def test_implied_vol_valid_still_recovers():
    price = float(black_scholes_price(100, 100, 0.5, 0.3))
    iv = float(implied_volatility(price, 100, 100, 0.5))
    assert np.isclose(iv, 0.3, atol=1e-3)


# --- LLM parsing / confidence ---------------------------------------------- #
def test_parse_json_fenced_failure_falls_through():
    text = '```json\n{oops}\n```\nActually: {"AAPL": 0.02}'
    assert parse_json(text) == {"AAPL": 0.02}


def test_parse_json_brace_inside_string():
    assert parse_json('{"note": "buy } now", "AAPL": 0.02}') == {"note": "buy } now", "AAPL": 0.02}


def test_parse_json_multiple_objects_first_valid_wins():
    assert parse_json('{bad} then {"x": 1}') == {"x": 1}


def test_confidence_single_asset_tight_agreement_high():
    import json

    r = jf.generate_returns(n_assets=1, n_days=200, seed=0)
    llm = FakeLLM([json.dumps({r.columns[0]: v}) for v in (0.010, 0.011, 0.012, 0.0105, 0.0115)])
    vs = llm_views(r, llm, samples=5)
    assert vs.confidence > 0.5  # tight agreement -> high confidence, not the 0.05 floor


# --- metrics denominators -------------------------------------------------- #
def test_calmar_inf_when_no_drawdown():
    good = [0.01, 0.02, 0.005, 0.03]
    assert np.isposinf(M.calmar_ratio(good))
    assert M.calmar_ratio([0.0, 0.0, 0.0]) == 0.0


def test_sortino_inf_when_no_downside():
    good = [0.01, 0.02, 0.005, 0.03]
    assert np.isposinf(M.sortino_ratio(good))


def test_sortino_negative_for_losing_series():
    # A series with downside gives a finite, negative Sortino (sign handling sane).
    val = M.sortino_ratio([-0.01, -0.02])
    assert val < 0 and np.isfinite(val)
