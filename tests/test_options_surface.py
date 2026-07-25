"""Tests for the implied-volatility surface and discrete-dividend pricing."""

from __future__ import annotations

import numpy as np
import pytest

from jaxfolio.options import (
    SVIParams,
    VolSurface,
    binomial_american,
    black_scholes_price,
    calibrate_svi,
)

STRIKES = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
TTMS = np.array([0.1, 0.25, 0.5])
IV_GRID = np.array(
    [
        [0.30, 0.25, 0.22, 0.24, 0.28],
        [0.29, 0.24, 0.21, 0.23, 0.27],
        [0.28, 0.23, 0.205, 0.225, 0.26],
    ]
)


@pytest.fixture
def surface():
    return VolSurface.from_iv_grid(STRIKES, TTMS, IV_GRID, spot=100.0)


def test_iv_exact_at_grid_nodes(surface):
    for i, t in enumerate(TTMS):
        for j, k in enumerate(STRIKES):
            assert surface.iv(k, t) == pytest.approx(IV_GRID[i, j], abs=1e-9)


def test_price_uses_surface_vol(surface):
    # Price at a grid node equals BS priced at that node's vol.
    expected = float(black_scholes_price(100.0, 100.0, 0.25, 0.21))
    assert surface.price(100.0, 0.25) == pytest.approx(expected, rel=1e-9)


def test_time_interpolation_in_total_variance(surface):
    # A query between two expiries stays within the bracketing vols.
    t = 0.35
    lo = min(surface.iv(100.0, 0.25), surface.iv(100.0, 0.5))
    hi = max(surface.iv(100.0, 0.25), surface.iv(100.0, 0.5))
    assert lo <= surface.iv(100.0, t) <= hi


def test_from_chain_round_trip():
    """Prices generated from a known vol grid invert back to that grid."""
    prices = np.array(
        [
            [float(black_scholes_price(100.0, k, t, IV_GRID[i, j])) for j, k in enumerate(STRIKES)]
            for i, t in enumerate(TTMS)
        ]
    )
    recovered = VolSurface.from_chain(100.0, STRIKES, TTMS, prices)
    assert np.abs(recovered.iv_grid - IV_GRID).max() < 1e-4


def test_svi_calibration_recovers_parameters():
    k = np.linspace(-0.3, 0.3, 15)
    true = SVIParams(a=0.04, b=0.1, rho=-0.4, m=0.0, sigma=0.1)
    w = true.total_variance(k)
    fit = calibrate_svi(k, w)
    assert np.abs(fit.total_variance(k) - w).max() < 1e-6


def test_fit_svi_produces_arbitrage_free_smile(surface):
    fitted = surface.fit_svi()
    assert fitted.svi is not None and len(fitted.svi) == len(TTMS)
    # SVI smiles are smooth/convex, so no butterfly violations.
    assert fitted.arbitrage_report()["butterfly_ok"]


def test_arbitrage_report_flags_calendar_violation():
    # Total variance must not decrease with maturity; invert two slices to break it.
    bad = IV_GRID.copy()
    bad[2] = 0.10  # far-dated vol far below near-dated => calendar arbitrage
    surf = VolSurface.from_iv_grid(STRIKES, TTMS, bad, spot=100.0)
    report = surf.arbitrage_report()
    assert not report["calendar_ok"]
    assert not report["arbitrage_free"]


def test_iv_grid_shape_validation():
    with pytest.raises(ValueError, match="iv_grid shape"):
        VolSurface(STRIKES, TTMS, IV_GRID[:, :-1], spot=100.0)


# --------------------------------------------------------------------------- #
# Discrete-dividend American pricing
# --------------------------------------------------------------------------- #
def test_discrete_dividend_raises_american_put():
    base = float(binomial_american(100, 100, 1.0, 0.2, 0.05, is_call=False, steps=400))
    with_div = float(
        binomial_american(
            100, 100, 1.0, 0.2, 0.05, is_call=False, steps=400, dividends=[(0.5, 3.0)]
        )
    )
    assert with_div > base


def test_discrete_dividend_lowers_american_call():
    base = float(binomial_american(100, 100, 1.0, 0.2, 0.05, is_call=True, steps=400))
    with_div = float(
        binomial_american(100, 100, 1.0, 0.2, 0.05, is_call=True, steps=400, dividends=[(0.5, 3.0)])
    )
    assert with_div < base


def test_no_dividends_matches_default_path():
    """An empty schedule must reproduce the fast-path result.

    The default path runs in JAX (float32) and the discrete-dividend path in
    NumPy (float64), so agreement is to backend precision, not bit-exact.
    """
    a = float(binomial_american(100, 95, 0.75, 0.25, 0.03, is_call=True, steps=300))
    b = float(binomial_american(100, 95, 0.75, 0.25, 0.03, is_call=True, steps=300, dividends=[]))
    assert a == pytest.approx(b, abs=2e-3)


def test_dividend_after_expiry_ignored():
    """A dividend paid after expiry should not affect the price."""
    base = float(binomial_american(100, 100, 0.5, 0.2, 0.05, is_call=True, steps=300))
    later = float(
        binomial_american(100, 100, 0.5, 0.2, 0.05, is_call=True, steps=300, dividends=[(0.9, 5.0)])
    )
    assert later == pytest.approx(base, abs=2e-3)
