"""Validation on ill-conditioned and degenerate/infeasible inputs.

The "problem-condition" axis of the validation matrix: near-singular covariance
(collinear assets), and degenerate cases (single asset, zero-variance asset).
The contract we validate is *robustness* — jaxfolio must stay feasible and
finite, and match the reference objective where the optimum is well defined.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from jaxfolio.optimizers import classical as C

from . import references as ref


def _collinear_returns(n_assets: int = 5, n_days: int = 500, *, seed: int = 0) -> pd.DataFrame:
    """A near-singular return panel: assets share a common factor with tiny idio.

    The resulting sample covariance has a very high condition number, so the
    minimum-variance optimum is nearly degenerate (non-unique weights, but a
    well-defined minimal variance).
    """
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal(n_days) * 0.01
    cols = {f"A{i}": factor + rng.standard_normal(n_days) * 1e-4 for i in range(n_assets)}
    return pd.DataFrame(cols)


def test_ill_conditioned_covariance_is_high_condition_number():
    """Guard the fixture: the crafted panel really is ill-conditioned."""
    _, cov, _ = ref.moments(_collinear_returns())
    cond = np.linalg.cond(cov)
    assert cond > 1e4, f"expected an ill-conditioned covariance, got cond={cond:.3g}"


def test_min_variance_ill_conditioned_stays_feasible_and_optimal():
    returns = _collinear_returns()
    mu, cov, mat = ref.moments(returns)
    w = C.minimum_variance(returns).weights
    # Feasible: finite, sums to one, long-only.
    assert np.all(np.isfinite(w))
    assert np.isclose(w.sum(), 1.0, atol=1e-4)
    assert (w >= -1e-4).all()
    # Objective: not materially worse than the SLSQP reference despite the
    # near-singular geometry (weights themselves may differ — the optimum is
    # non-unique — so we compare variances, not weights).
    w_ref = ref.ref_min_variance(mu, cov, mat)
    var, var_ref = float(w @ cov @ w), float(w_ref @ cov @ w_ref)
    assert var <= var_ref + 1e-8 + abs(var_ref) * 0.05, f"var {var:.3g} vs ref {var_ref:.3g}"


def test_risk_parity_ill_conditioned_finite():
    """ERC coordinate descent must not divide by zero on near-collinear assets."""
    res = C.risk_parity(_collinear_returns())
    assert np.all(np.isfinite(res.weights))
    assert np.isclose(res.weights.sum(), 1.0, atol=1e-6)


def test_single_asset_degenerate():
    returns = _collinear_returns(n_assets=1, n_days=300)
    res = C.minimum_variance(returns)
    assert res.weights.shape == (1,)
    assert np.isclose(res.weights[0], 1.0, atol=1e-6)


def test_zero_variance_asset_handled():
    """An asset with constant (zero-variance) returns must not produce NaNs."""
    returns = _collinear_returns(n_assets=4, n_days=400)
    returns = returns.copy()
    returns["CASH"] = 0.0  # a riskless, zero-variance column
    for method in (C.minimum_variance, C.risk_parity, C.inverse_volatility):
        w = method(returns).weights
        assert np.all(np.isfinite(w)), f"{method.__name__} produced non-finite weights"
        assert np.isclose(w.sum(), 1.0, atol=1e-4)


def test_infeasible_bounds_raise_or_degrade_gracefully():
    """Bounds whose simplex is empty (each asset capped below 1/n) must not hang.

    With 4 assets capped at 0.1 each, no long-only fully-invested portfolio
    exists. jaxfolio should either raise a clear error or return a finite
    projected point — never NaN or an infinite loop.
    """
    from jaxfolio.types import OptimizerConfig

    returns = _collinear_returns(n_assets=4, n_days=300)
    cfg = OptimizerConfig(weight_bounds=(0.0, 0.1))  # sum of uppers = 0.4 < 1
    try:
        w = C.minimum_variance(returns, config=cfg).weights
    except (ValueError, RuntimeError):
        return  # a clear error is an acceptable outcome
    assert np.all(np.isfinite(w)), "infeasible bounds must not yield non-finite weights"
