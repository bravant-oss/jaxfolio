"""Tests for the multi-period (path-aware, cost-aware) optimizer.

The invariants here fall into three groups. First, *feasibility of every row* —
the multi-period feasible set is the Cartesian product of the per-period sets, so
the single likeliest implementation bug is projecting the whole path onto one
simplex summing to 1 overall rather than one per period; ``test_every_row_is_*``
is the guard for that.

Second, the *economics*. Zero costs must collapse the path onto the single-period
solution, prohibitive costs must pin it at the current holdings, and quadratic
impact must produce a decaying execution schedule that trades less in total than
the myopic one-shot rebalance. That last one is the whole claim of the feature.

Third, the *jit-cache contract*. The Huber width is refined across several solver
stages, and all of them must share one compiled kernel — which additionally
pins the weak-dtype trap, where a weak-typed warm start silently compiles the
kernel twice.

Note on costs: a purely *linear* cost creates a no-trade region but gives no
reason to split a trade, so the path jumps in one step and stays. Only the
**quadratic impact** term rewards spreading execution. Tests about glide paths
therefore set ``impact_bps``; this matches the exact CVXPY optimum (see
``tests/validation/test_reference_solvers.py``) and is not an artifact.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from jaxfolio.data.synthetic import generate_returns
from jaxfolio.moments.estimators import as_matrix, mean_returns, sample_covariance
from jaxfolio.optimizers import classical as C
from jaxfolio.optimizers.base import _solve_cached
from jaxfolio.optimizers.multiperiod import multi_period_mean_variance as mp
from jaxfolio.types import OptimizerConfig, PortfolioResult, TradingCosts

GAMMA = 3.0


# A concentrated starting book, maximally far from any interior optimum, so the
# cost and monotonicity assertions have a large trade to bite on.
def _w_prev(n: int) -> np.ndarray:
    return np.eye(n)[0]


def _moments(returns):
    mat, _ = as_matrix(returns)
    return np.asarray(mean_returns(mat)), np.asarray(sample_covariance(mat))


def _utility(w, mu, cov, gamma=GAMMA):
    """Single-period mean-variance utility (higher is better)."""
    return float(w @ mu - 0.5 * gamma * (w @ cov @ w))


# --------------------------------------------------------------------------- #
# Feasibility — every period independently
# --------------------------------------------------------------------------- #
def test_every_row_is_feasible(small_returns):
    """Each period's weights must sum to the budget and respect the bounds.

    The feasible set is a *product* of per-period sets. Projecting the flattened
    path onto a single simplex would still give ``traj.sum() == 1`` overall while
    each row summed to ``1/T`` — hence the per-row assertion.
    """
    n = len(small_returns.columns)
    res = mp(small_returns, horizon=5, w_prev=_w_prev(n), costs=TradingCosts(spread_bps=10.0))
    traj = res.trajectory
    assert traj.shape == (5, n), f"expected (5, {n}), got {traj.shape}"
    for t, row in enumerate(traj):
        assert np.isclose(row.sum(), 1.0, atol=1e-4), f"row {t} sums to {row.sum()}"
    assert (traj >= -1e-4).all(), f"min weight={traj.min()}"


def test_every_row_respects_weight_bounds(small_returns):
    n = len(small_returns.columns)
    res = mp(
        small_returns,
        horizon=4,
        w_prev=_w_prev(n),
        costs=TradingCosts(spread_bps=10.0),
        config=OptimizerConfig(weight_bounds=(0.0, 0.4), tol=1e-6),
    )
    traj = res.trajectory
    assert traj.max() <= 0.4 + 1e-4, f"max violation={traj.max() - 0.4:.4g}"
    assert np.allclose(traj.sum(axis=1), 1.0, atol=1e-4), f"row sums={traj.sum(axis=1)}"


def test_long_short_allows_negative_rows(returns):
    n = len(returns.columns)
    res = mp(
        returns,
        horizon=3,
        w_prev=_w_prev(n),
        costs=TradingCosts(spread_bps=5.0),
        config=OptimizerConfig(long_only=False, tol=1e-6),
    )
    assert res.trajectory.min() < -1e-3, f"expected shorting, min={res.trajectory.min()}"
    assert np.allclose(res.trajectory.sum(axis=1), 1.0, atol=1e-3)


# --------------------------------------------------------------------------- #
# Result surface
# --------------------------------------------------------------------------- #
def test_first_row_is_reported_weights(small_returns):
    """``weights`` is the allocation to hold *now* — the backtester relies on it."""
    n = len(small_returns.columns)
    res = mp(small_returns, horizon=4, w_prev=_w_prev(n), costs=TradingCosts(impact_bps=500.0))
    assert np.allclose(res.weights, res.trajectory[0], atol=1e-9), (
        f"Δ={np.abs(res.weights - res.trajectory[0]).max():.4g}"
    )


def test_terminal_metadata_matches_last_row(small_returns):
    n = len(small_returns.columns)
    res = mp(small_returns, horizon=4, w_prev=_w_prev(n), costs=TradingCosts(impact_bps=500.0))
    assert np.allclose(res.metadata["terminal_weights"], res.trajectory[-1], atol=1e-12)
    assert res.metadata["terminal_sharpe"] is not None


def test_metadata_contract(small_returns):
    """The documented diagnostics must all be present and JSON-serializable."""
    n = len(small_returns.columns)
    res = mp(small_returns, horizon=5, w_prev=_w_prev(n), costs=TradingCosts(spread_bps=20.0))
    expected = {
        "horizon",
        "risk_aversion",
        "iterations",
        "stage_iterations",
        "stage_trade_eps",
        "trade_eps",
        "residual",
        "converged",
        "costs_active",
        "smoothing",
        "turnover_path",
        "total_turnover",
        "total_cost",
        "objective_smoothed",
        "objective_exact",
        "smoothing_bias",
        "terminal_weights",
        "terminal_expected_return",
        "terminal_volatility",
        "terminal_sharpe",
        "w_prev",
        "mu_path_provided",
        "cov_path_provided",
    }
    missing = expected - set(res.metadata)
    assert not missing, f"missing metadata keys: {sorted(missing)}"
    assert len(res.metadata["turnover_path"]) == 5
    assert np.isclose(sum(res.metadata["turnover_path"]), res.metadata["total_turnover"])
    # metadata is JSON-ish by convention; the bulk array lives on ``trajectory``.
    json.dumps(res.metadata)


def test_turnover_path_matches_trajectory(small_returns):
    """Recompute the planned turnover independently — guards a stale-iterate bug."""
    n = len(small_returns.columns)
    wp = _w_prev(n)
    res = mp(small_returns, horizon=5, w_prev=wp, costs=TradingCosts(impact_bps=400.0))
    prev = np.vstack([wp[None, :], res.trajectory[:-1]])
    expected = np.abs(res.trajectory - prev).sum(axis=1)
    assert np.allclose(res.metadata["turnover_path"], expected, atol=1e-9), (
        f"Δ={np.abs(np.array(res.metadata['turnover_path']) - expected).max():.4g}"
    )


def test_smoothing_bias_is_non_negative(small_returns):
    """Huber understates |d|, so the exact objective can only exceed the smoothed one."""
    n = len(small_returns.columns)
    res = mp(small_returns, horizon=4, w_prev=_w_prev(n), costs=TradingCosts(spread_bps=25.0))
    assert res.metadata["smoothing_bias"] >= -1e-12, res.metadata["smoothing_bias"]


# --------------------------------------------------------------------------- #
# Economics
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("costs", [None, TradingCosts()], ids=["none", "zero"])
def test_zero_cost_reduces_to_single_period(returns, costs):
    """Frictionless ⇒ every period holds the single-period mean-variance optimum."""
    mu, cov = _moments(returns)
    n = len(returns.columns)
    w_sp = C.mean_variance(returns, risk_aversion=GAMMA).weights
    res = mp(returns, horizon=5, w_prev=_w_prev(n), risk_aversion=GAMMA, costs=costs)
    u_sp = _utility(w_sp, mu, cov)
    for t, row in enumerate(res.trajectory):
        assert np.allclose(row, w_sp, atol=2e-3), f"row {t}: Δ={np.abs(row - w_sp).max():.4g}"
        # Domination form: a *better* iterate must never fail the test.
        assert _utility(row, mu, cov) >= u_sp - 1e-6, f"row {t} utility below single-period"


def test_prohibitive_cost_pins_at_w_prev(returns):
    """If trading is ruinously expensive, the optimal path is to hold still."""
    n = len(returns.columns)
    wp = _w_prev(n)
    res = mp(returns, horizon=5, w_prev=wp, risk_aversion=GAMMA, costs=TradingCosts(spread_bps=1e6))
    assert np.allclose(res.trajectory, wp[None, :], atol=1e-3), (
        f"Δ={np.abs(res.trajectory - wp).max():.4g}"
    )
    assert res.metadata["total_turnover"] < 1e-2, res.metadata["total_turnover"]


def test_impact_cost_spreads_execution_monotonically(returns):
    """Quadratic impact ⇒ a decaying glide path toward the terminal target.

    Scoped to the constant-moment default: monotone partial adjustment toward a
    fixed target is the standard no-trade-region result for convex separable
    costs, not a theorem for a time-varying ``mu_path``.
    """
    n = len(returns.columns)
    wp = _w_prev(n)
    res = mp(
        returns,
        horizon=5,
        w_prev=wp,
        risk_aversion=GAMMA,
        costs=TradingCosts(spread_bps=10.0, impact_bps=500.0),
    )
    term = np.asarray(res.metadata["terminal_weights"])
    dist = np.array([np.abs(wp - term).sum()] + [np.abs(w - term).sum() for w in res.trajectory])
    assert (np.diff(dist) <= 1e-6).all(), f"non-monotone approach: diffs={np.diff(dist)}"
    # And it genuinely spreads rather than jumping: several periods trade.
    trading = [x for x in res.metadata["turnover_path"] if x > 1e-3]
    assert len(trading) >= 3, f"expected execution spread over periods, got {trading}"


def test_impact_cost_trades_less_than_one_shot(returns):
    """The economic claim: total turnover is below the myopic single trade."""
    n = len(returns.columns)
    wp = _w_prev(n)
    w_sp = C.mean_variance(returns, risk_aversion=GAMMA).weights
    one_shot = float(np.abs(w_sp - wp).sum())
    res = mp(
        returns,
        horizon=5,
        w_prev=wp,
        risk_aversion=GAMMA,
        costs=TradingCosts(spread_bps=10.0, impact_bps=500.0),
    )
    total = res.metadata["total_turnover"]
    assert total < one_shot, f"multi-period turnover {total:.4f} not below one-shot {one_shot:.4f}"


def test_higher_cost_monotonically_reduces_turnover(returns):
    """Turnover must fall as trading gets more expensive.

    The tolerance is float32-scaled, not cosmetic: at low cost the answer saturates
    at a full rebalance (turnover 2.0 from a single-name book), and consecutive
    settings then agree to a few 1e-6 rather than exactly.
    """
    n = len(returns.columns)
    wp = _w_prev(n)
    totals = [
        mp(
            returns, horizon=4, w_prev=wp, risk_aversion=GAMMA, costs=TradingCosts(spread_bps=bps)
        ).metadata["total_turnover"]
        for bps in (0.0, 5.0, 25.0, 100.0)
    ]
    assert all(totals[i + 1] <= totals[i] + 1e-4 for i in range(len(totals) - 1)), (
        f"turnover not non-increasing in cost: {totals}"
    )
    assert totals[-1] < totals[0] - 1e-3, f"cost had no effect on turnover: {totals}"


def test_horizon_one_is_turnover_penalized(small_returns):
    """``horizon=1`` degenerates to a single-period solve with a trade penalty."""
    n = len(small_returns.columns)
    wp = _w_prev(n)
    res = mp(
        small_returns,
        horizon=1,
        w_prev=wp,
        risk_aversion=GAMMA,
        costs=TradingCosts(spread_bps=50.0),
    )
    assert res.trajectory.shape == (1, n)
    w_sp = C.mean_variance(small_returns, risk_aversion=GAMMA).weights
    # A cost on trading can only shrink the trade relative to the myopic one.
    assert np.abs(res.weights - wp).sum() <= np.abs(w_sp - wp).sum() + 1e-6


# --------------------------------------------------------------------------- #
# w_prev handling
# --------------------------------------------------------------------------- #
def test_w_prev_none_is_a_flat_book(small_returns):
    """Default holdings are the all-zero book, matching the backtester's start.

    Zeros — not ``1/n`` — because that is what the engine holds before its first
    rebalance, and because it correctly charges the cost of *establishing* a
    position rather than claiming one you do not have.
    """
    n = len(small_returns.columns)
    res = mp(small_returns, horizon=3, costs=TradingCosts(spread_bps=10.0))
    assert res.metadata["w_prev"] == [0.0] * n
    explicit = mp(small_returns, horizon=3, w_prev=np.zeros(n), costs=TradingCosts(spread_bps=10.0))
    assert np.allclose(res.trajectory, explicit.trajectory, atol=1e-12)


def test_costless_result_is_independent_of_w_prev(small_returns):
    """With no frictions the problem has no memory, so holdings cannot matter."""
    n = len(small_returns.columns)
    a = mp(small_returns, horizon=3, w_prev=np.zeros(n), risk_aversion=GAMMA)
    b = mp(small_returns, horizon=3, w_prev=_w_prev(n), risk_aversion=GAMMA)
    assert np.allclose(a.trajectory, b.trajectory, atol=1e-4), (
        f"Δ={np.abs(a.trajectory - b.trajectory).max():.4g}"
    )


def test_w_prev_need_not_sum_to_one(small_returns):
    """The engine hands over a drifted, possibly partial book — accept it."""
    n = len(small_returns.columns)
    partial = np.zeros(n)
    partial[0], partial[1] = 0.3, 0.1
    res = mp(small_returns, horizon=3, w_prev=partial, costs=TradingCosts(spread_bps=10.0))
    assert np.isfinite(res.trajectory).all()
    assert np.allclose(res.trajectory.sum(axis=1), 1.0, atol=1e-4)


def test_w_prev_length_is_validated(small_returns):
    n = len(small_returns.columns)
    with pytest.raises(ValueError, match=rf"expected {n}"):
        mp(small_returns, horizon=2, w_prev=np.zeros(n + 1))


# --------------------------------------------------------------------------- #
# Forecast term structure
# --------------------------------------------------------------------------- #
def test_mu_path_shape_validation(small_returns):
    n = len(small_returns.columns)
    with pytest.raises(ValueError, match=rf"mu_path shape \(3, {n}\).*expected \(5, {n}\)"):
        mp(small_returns, horizon=5, mu_path=np.zeros((3, n)))
    with pytest.raises(ValueError, match=rf"cov_path shape \(5, {n}, {n + 1}\)"):
        mp(small_returns, horizon=5, cov_path=np.zeros((5, n, n + 1)))


def test_mu_path_is_consumed_per_period(small_returns):
    """A late-period tilt must show up in the *terminal* weights, not be averaged."""
    mu, _ = _moments(small_returns)
    base = mp(small_returns, horizon=4, risk_aversion=GAMMA)
    tilted = np.tile(mu, (4, 1))
    tilted[-1, 2] += 0.05  # strongly favor asset 2 in the final period only
    res = mp(small_returns, horizon=4, risk_aversion=GAMMA, mu_path=tilted)
    assert res.metadata["mu_path_provided"] is True
    assert res.trajectory[-1, 2] > base.trajectory[-1, 2] + 1e-3, (
        f"terminal weight on asset 2 did not respond: "
        f"{res.trajectory[-1, 2]:.4f} vs {base.trajectory[-1, 2]:.4f}"
    )


def test_cov_path_matches_shared_cov_when_tiled(small_returns):
    """A tiled per-period covariance must reproduce the shared-covariance answer."""
    n = len(small_returns.columns)
    _, cov = _moments(small_returns)
    shared = mp(
        small_returns,
        horizon=3,
        w_prev=_w_prev(n),
        risk_aversion=GAMMA,
        costs=TradingCosts(spread_bps=10.0),
    )
    tiled = mp(
        small_returns,
        horizon=3,
        w_prev=_w_prev(n),
        risk_aversion=GAMMA,
        costs=TradingCosts(spread_bps=10.0),
        cov_path=np.tile(cov, (3, 1, 1)),
    )
    assert np.allclose(shared.trajectory, tiled.trajectory, atol=2e-3), (
        f"Δ={np.abs(shared.trajectory - tiled.trajectory).max():.4g}"
    )


def test_horizon_must_be_positive(small_returns):
    with pytest.raises(ValueError, match="horizon"):
        mp(small_returns, horizon=0)


# --------------------------------------------------------------------------- #
# jit-cache contract
# --------------------------------------------------------------------------- #
def test_jit_cache_does_not_grow():
    """The refinement stages must share one compiled kernel, and costs stay traced.

    This is also the guard for the weak-dtype trap: ``jnp.full``/``jnp.tile``
    produce a *weak-typed* array while the solver's output is strongly typed, so a
    weak-typed warm start would make stage 2 a different cache key and compile the
    kernel twice.
    """
    panel = generate_returns(n_assets=6, n_days=300, seed=4242)  # a shape ours alone
    wp = _w_prev(6)
    costs = TradingCosts(spread_bps=10.0, impact_bps=500.0)

    before = _solve_cached._cache_size()
    mp(panel, horizon=5, w_prev=wp, costs=costs)
    after_first = _solve_cached._cache_size()
    assert after_first == before + 1, "the whole refinement ladder must compile once"

    mp(panel, horizon=5, w_prev=wp, costs=costs)
    assert _solve_cached._cache_size() == after_first, "repeat solve must reuse the kernel"

    # Costs and risk aversion are traced parameters, not cache keys.
    mp(
        panel,
        horizon=5,
        w_prev=wp,
        risk_aversion=9.0,
        costs=TradingCosts(spread_bps=99.0, impact_bps=3.0),
    )
    assert _solve_cached._cache_size() == after_first, "costs must not key the cache"

    # The horizon is baked into the variable's shape, so it *does* key the cache.
    mp(panel, horizon=7, w_prev=wp, costs=costs)
    assert _solve_cached._cache_size() == after_first + 1, "a new horizon compiles once"


# --------------------------------------------------------------------------- #
# TradingCosts and PortfolioResult validation
# --------------------------------------------------------------------------- #
def test_costs_reject_invalid_values():
    with pytest.raises(ValueError, match="spread_bps"):
        TradingCosts(spread_bps=-1.0)
    with pytest.raises(ValueError, match=r"impact_bps\[1\]"):
        TradingCosts(impact_bps=(1.0, -2.0))
    with pytest.raises(ValueError, match="smoothing"):
        TradingCosts(smoothing=0.0)
    with pytest.raises(ValueError, match="turnover_penalty"):
        TradingCosts(turnover_penalty=-0.1)
    with pytest.raises(ValueError, match="1-D"):
        TradingCosts(spread_bps=np.zeros((2, 2)))
    with pytest.raises(TypeError, match="commission_bps"):
        TradingCosts(commission_bps="10bps")


def test_costs_stay_hashable_with_per_asset_values():
    """Frozen dataclasses hash every field, so a stored ndarray would break it."""
    c = TradingCosts(spread_bps=(1.0, 2.0), impact_bps=3.0)
    assert isinstance(hash(c), int)
    assert c.spread_bps == (1.0, 2.0)
    assert c == TradingCosts(spread_bps=(1.0, 2.0), impact_bps=3.0)


def test_costs_broadcast_length_is_reported():
    with pytest.raises(ValueError, match="length 2, expected 1 or 5"):
        TradingCosts(spread_bps=(1.0, 2.0)).as_params(5)


def test_costs_linear_decimal_and_activity():
    c = TradingCosts(spread_bps=5.0, commission_bps=1.0)
    assert np.allclose(c.linear_decimal(3), 6e-4)
    assert c.is_active()
    assert not TradingCosts().is_active()
    assert TradingCosts(turnover_penalty=0.01).is_active()
    # turnover_penalty enters the objective as a linear cost.
    assert np.allclose(np.asarray(TradingCosts(turnover_penalty=0.01).as_params(2)["c_lin"]), 0.01)


def test_result_rejects_bad_trajectory():
    w, assets = np.ones(3) / 3, list("abc")
    with pytest.raises(ValueError, match="trajectory shape"):
        PortfolioResult(weights=w, assets=assets, method="x", trajectory=np.zeros((4, 2)))
    with pytest.raises(ValueError, match="trajectory shape"):
        PortfolioResult(weights=w, assets=assets, method="x", trajectory=np.zeros(3))
    ok = PortfolioResult(weights=w, assets=assets, method="x", trajectory=np.zeros((4, 3)))
    assert ok.trajectory.shape == (4, 3)
    assert ok.trajectory.dtype == float


def test_result_hints_at_flattened_path():
    """Passing the (T, n) path as ``weights`` is silently flattened — say so."""
    with pytest.raises(ValueError, match="did you mean trajectory"):
        PortfolioResult(weights=np.zeros((4, 3)), assets=list("abc"), method="x")


def test_result_without_trajectory_is_unchanged():
    r = PortfolioResult(weights=np.ones(3) / 3, assets=list("abc"), method="x")
    assert r.trajectory is None
    assert "steps=" not in repr(r)
