"""Tests for the options execution framework (costs, book, rolling sim)."""

from __future__ import annotations

import numpy as np
import pytest

from jaxfolio.options import black_scholes_price
from jaxfolio.options.execution import (
    BUY,
    SELL,
    CostModel,
    ExecutionSimulator,
    Instrument,
    Order,
    Position,
    simulate_covered_call_roll,
)


# --------------------------------------------------------------------------- #
# CostModel
# --------------------------------------------------------------------------- #
def test_slippage_works_against_the_trader():
    cm = CostModel(slippage_bps=10.0)
    mid = 5.0
    assert cm.fill_price(mid, BUY) > mid  # buys pay up
    assert cm.fill_price(mid, SELL) < mid  # sells receive less
    assert cm.fill_price(mid, BUY) == pytest.approx(5.0 * (1 + 1e-3))


def test_commission_is_sign_insensitive_and_floored():
    cm = CostModel(commission_per_contract=0.5, min_commission=1.0)
    assert cm.commission(1) == 1.0  # floored
    assert cm.commission(-10) == pytest.approx(5.0)


def test_fill_price_never_negative():
    cm = CostModel(slippage_bps=5.0)
    assert cm.fill_price(-1.0, SELL) == 0.0


# --------------------------------------------------------------------------- #
# Position accounting
# --------------------------------------------------------------------------- #
def test_position_average_price_blends_on_adds():
    from jaxfolio.options.execution.book import Fill

    inst = Instrument("call", 100.0, 0.25)
    pos = Position(inst)
    pos.apply(Fill(inst, 2.0, 3.0, 3.0, 0.0))
    pos.apply(Fill(inst, 2.0, 5.0, 5.0, 0.0))
    assert pos.quantity == 4.0
    assert pos.avg_price == pytest.approx(4.0)


def test_position_closes_to_zero():
    from jaxfolio.options.execution.book import Fill

    inst = Instrument("put", 90.0, 0.25)
    pos = Position(inst)
    pos.apply(Fill(inst, -1.0, 2.0, 2.0, 0.0))
    pos.apply(Fill(inst, 1.0, 2.5, 2.5, 0.0))
    assert pos.quantity == 0.0
    assert pos.avg_price == 0.0


# --------------------------------------------------------------------------- #
# ExecutionSimulator
# --------------------------------------------------------------------------- #
def test_sell_to_open_increases_cash_and_books_short():
    sim = ExecutionSimulator(cost_model=CostModel(commission_per_contract=1.0, slippage_bps=0.0))
    inst = Instrument("call", 105.0, 0.25)
    mid = float(black_scholes_price(100.0, 105.0, 0.25, 0.2))
    fill = sim.execute(Order(inst, -1.0), spot=100.0, vol=0.2)
    assert fill.mid == pytest.approx(mid, rel=1e-6)
    assert sim.cash == pytest.approx(mid - 1.0)  # received premium, paid commission
    assert sim.book.positions[inst.key()].quantity == -1.0


def test_immediate_roundtrip_costs_equal_frictions():
    """Selling then marking at the same state leaves equity at minus the frictions."""
    cm = CostModel(commission_per_contract=1.0, slippage_bps=10.0)
    sim = ExecutionSimulator(cost_model=cm)
    inst = Instrument("call", 105.0, 0.25)
    mid = sim.mid_price(inst, 100.0, 0.2)
    sim.execute(Order(inst, -1.0), spot=100.0, vol=0.2)
    equity = sim.equity(100.0, 0.2)
    # equity = cash (mid*(1-slip) - comm) + bookMTM (-mid) = -(mid*slip + comm)
    assert equity == pytest.approx(-(mid * 1e-3 + 1.0), rel=1e-6)
    assert equity < 0


def test_net_greeks_of_short_call_is_negative_delta():
    sim = ExecutionSimulator(cost_model=CostModel(slippage_bps=0.0))
    inst = Instrument("call", 100.0, 0.25)
    sim.execute(Order(inst, -1.0), spot=100.0, vol=0.2)
    g = sim.book.net_greeks(100.0, 0.2)
    assert g["delta"] < 0  # short a call → negative delta
    assert g["gamma"] < 0  # short gamma


# --------------------------------------------------------------------------- #
# Rolling covered-call simulation
# --------------------------------------------------------------------------- #
def _price_path(drift, n=252, seed=0):
    rng = np.random.default_rng(seed)
    return 100.0 * np.cumprod(1 + rng.normal(drift, 0.01, n))


def test_roll_produces_expected_structure():
    path = _price_path(0.0005)
    res = simulate_covered_call_roll(path, vol=0.2, roll_every=21, tenor=0.25)
    assert res.equity.shape == path.shape
    assert np.all(np.isfinite(res.equity))
    assert res.n_rolls == int(np.ceil(len(path) / 21))
    assert res.total_costs > 0  # every roll pays commission


def test_covered_call_caps_upside_on_rally():
    """On a strong rally the overlay underperforms buy-and-hold (upside capped)."""
    rally = 100.0 * np.cumprod(1 + np.full(252, 0.003))
    res = simulate_covered_call_roll(rally, vol=0.2, roll_every=21, tenor=0.25)
    assert res.total_return < res.stock_return


def test_empty_path_raises():
    with pytest.raises(ValueError, match="empty"):
        simulate_covered_call_roll(np.array([]))
