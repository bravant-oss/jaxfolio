"""Tests for weight-constraint projections."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaxfolio.constraints.projections import (
    project_box_budget,
    project_simplex,
    softmax_weights,
)


def test_simplex_sums_to_one_and_nonnegative():
    v = jnp.array([0.3, -1.2, 5.0, 0.1, -0.4])
    w = project_simplex(v)
    assert np.isclose(float(jnp.sum(w)), 1.0, atol=1e-6)
    assert np.all(np.asarray(w) >= -1e-8)


def test_simplex_already_on_simplex_is_fixed_point():
    v = jnp.array([0.25, 0.25, 0.25, 0.25])
    w = project_simplex(v)
    assert np.allclose(np.asarray(w), 0.25, atol=1e-6)


def test_simplex_custom_budget():
    v = jnp.array([1.0, 2.0, 3.0])
    w = project_simplex(v, budget=2.0)
    assert np.isclose(float(jnp.sum(w)), 2.0, atol=1e-6)


def test_box_budget_respects_bounds_and_sum():
    v = jnp.array([2.0, -3.0, 0.5, 0.1, 4.0])
    w = project_box_budget(v, lower=-0.2, upper=0.6, budget=1.0)
    w_np = np.asarray(w)
    assert np.isclose(w_np.sum(), 1.0, atol=1e-6)
    assert w_np.min() >= -0.2 - 1e-6
    assert w_np.max() <= 0.6 + 1e-6


def test_box_budget_allows_shorting():
    v = jnp.array([5.0, -5.0, 0.0])
    w = project_box_budget(v, lower=-1.0, upper=1.0, budget=1.0)
    assert float(jnp.min(w)) < 0.0  # at least one short position


def test_softmax_weights_sum_to_budget():
    logits = jnp.array([1.0, 2.0, -0.5, 0.3])
    w = softmax_weights(logits, budget=1.0)
    assert np.isclose(float(jnp.sum(w)), 1.0, atol=1e-6)
    assert np.all(np.asarray(w) > 0)


@pytest.mark.parametrize("n", [2, 5, 20])
def test_simplex_various_sizes(n):
    rng = np.random.default_rng(n)
    v = jnp.asarray(rng.normal(size=n))
    w = project_simplex(v)
    assert np.isclose(float(jnp.sum(w)), 1.0, atol=1e-6)
