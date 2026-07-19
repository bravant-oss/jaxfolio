"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from jaxfolio.data.synthetic import generate_returns


@pytest.fixture(scope="session")
def returns():
    """A reproducible synthetic return panel used across the test suite."""
    return generate_returns(n_assets=8, n_days=500, seed=42)


@pytest.fixture(scope="session")
def small_returns():
    """A tiny panel for fast, edge-case tests."""
    return generate_returns(n_assets=4, n_days=260, seed=1)
