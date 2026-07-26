"""Tests for weight constraints: projections, specifications, and compilation.

Four families of invariant, in increasing strength:

*Feasibility* — the projected point satisfies the constraints. Necessary but far
too weak on its own: ``lambda v: uniform_weights`` passes every feasibility test.

*Optimality* — the projection is the **Euclidean** projection, tested by the
variational inequality ``<v - P(v), w - P(v)> <= 0`` for every feasible ``w``.
This is the actual definition, and it is what the solvers depend on: ``_spg_loop``
stops on ``||w - P(w - grad f)||``, which certifies KKT stationarity only if ``P``
projects exactly. A merely-feasible ``P`` would break the stopping test silently.

*Duality* — the multipliers the kernel reports are the true ones, checked by
finite differences on the projection's own optimal value (``d/d budget = -lam``,
``d/d cap = -theta``) and by the stationarity identity
``v - w == lam + theta[g] + beta``.

*Contract* — the pieces that must not move: an empty constraint set reproduces the
previous behaviour bit-for-bit, and the jit cache grows only on structural change.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxfolio.constraints import (
    Box,
    GroupCap,
    GroupFloor,
    InfeasibleConstraints,
    check_feasible,
    compile_constraints,
)
from jaxfolio.constraints.projections import (
    project_box_budget,
    project_simplex,
    softmax_weights,
)
from jaxfolio.constraints.structured import (
    project_box_budget_vec,
    project_grouped,
    project_grouped_duals,
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


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _grouped_problem(rng, n=12, k=3, long_only=True, cap=0.35):
    """A random feasible (v, lower, upper, budget, gid, g_lower, g_upper)."""
    lo_s, hi_s = (0.0, 0.5) if long_only else (-0.25, 0.5)
    lower = np.full(n, lo_s)
    upper = np.full(n, hi_s)
    gid = np.full(n, k, dtype=np.int32)
    members = np.array_split(rng.permutation(n)[: n - 1], k)  # leave one ungrouped
    for j, part in enumerate(members):
        gid[part] = j
    g_lower = np.zeros(k + 1)
    g_upper = np.concatenate([np.full(k, cap), [0.0]])
    return (
        jnp.asarray(rng.normal(0.1, 0.5, n)),
        jnp.asarray(lower),
        jnp.asarray(upper),
        jnp.asarray(1.0),
        jnp.asarray(gid),
        jnp.asarray(g_lower),
        jnp.asarray(g_upper),
    )


def _random_feasible(rng, lower, upper, budget, gid, g_lower, g_upper, tries=400):
    """Sample a feasible point by rejection — for the variational-inequality test."""
    lower, upper = np.asarray(lower), np.asarray(upper)
    gid, g_lower, g_upper = np.asarray(gid), np.asarray(g_lower), np.asarray(g_upper)
    n_seg = len(g_lower)
    for _ in range(tries):
        w = np.asarray(project_simplex(jnp.asarray(rng.normal(size=len(lower))), float(budget)))
        if (w < lower - 1e-9).any() or (w > upper + 1e-9).any():
            continue
        sums = np.array([w[gid == j].sum() for j in range(n_seg - 1)])
        if (sums > g_upper[:-1] + 1e-9).any() or (sums < g_lower[:-1] - 1e-9).any():
            continue
        return w
    return None


# --------------------------------------------------------------------------- #
# Optimality — the variational inequality that *defines* a Euclidean projection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("long_only", [True, False], ids=["long-only", "shortable"])
def test_box_budget_is_the_euclidean_projection(long_only):
    """``project_box_budget`` minimizes distance, not merely satisfies the bounds.

    Guards the property the SPG stopping test relies on, which feasibility alone
    does not establish. The pre-existing kernel had no such test.
    """
    rng = np.random.default_rng(0)
    lo, hi = (0.0, 0.4) if long_only else (-0.2, 0.4)
    for _ in range(20):
        v = jnp.asarray(rng.normal(0.1, 0.6, 10))
        p = np.asarray(project_box_budget(v, lo, hi, 1.0))
        for _ in range(20):
            w = np.asarray(project_simplex(jnp.asarray(rng.normal(size=10))))
            if (w < lo - 1e-9).any() or (w > hi + 1e-9).any():
                continue
            ip = float(np.dot(np.asarray(v) - p, w - p))
            assert ip <= 1e-5, f"variational inequality violated: <v-P(v), w-P(v)> = {ip:.3e}"


def test_grouped_is_the_euclidean_projection():
    """Same optimality certificate for the grouped kernel."""
    rng = np.random.default_rng(1)
    checked = 0
    for _ in range(12):
        args = _grouped_problem(rng)
        p = np.asarray(project_grouped(*args))
        for _ in range(15):
            w = _random_feasible(rng, *args[1:])
            if w is None:
                continue
            ip = float(np.dot(np.asarray(args[0]) - p, w - p))
            assert ip <= 1e-5, f"variational inequality violated: {ip:.3e}"
            checked += 1
    assert checked > 20, f"too few feasible comparison points ({checked}) to be meaningful"


# --------------------------------------------------------------------------- #
# Feasibility of the structured kernels
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("long_only", [True, False], ids=["long-only", "shortable"])
def test_grouped_respects_every_constraint(long_only):
    rng = np.random.default_rng(2)
    for _ in range(15):
        v, lower, upper, budget, gid, g_lower, g_upper = _grouped_problem(rng, long_only=long_only)
        w = np.asarray(project_grouped(v, lower, upper, budget, gid, g_lower, g_upper))
        assert np.isclose(w.sum(), 1.0, atol=1e-5), f"sum={w.sum()}"
        assert (w >= np.asarray(lower) - 1e-5).all(), f"min={w.min()}"
        assert (w <= np.asarray(upper) + 1e-5).all(), f"max={w.max()}"
        gid_np = np.asarray(gid)
        for j in range(len(g_upper) - 1):
            got = w[gid_np == j].sum()
            cap = float(g_upper[j])
            assert got <= cap + 1e-5, f"group {j} sums to {got} over cap {cap}"


def test_box_budget_vec_matches_scalar_kernel_when_bounds_are_uniform():
    """The per-asset kernel must agree with the scalar one it generalizes."""
    rng = np.random.default_rng(3)
    v = jnp.asarray(rng.normal(0.1, 0.5, 15))
    scalar = np.asarray(project_box_budget(v, -0.1, 0.3, 1.0))
    vec = np.asarray(
        project_box_budget_vec(v, jnp.full(15, -0.1), jnp.full(15, 0.3), jnp.asarray(1.0))
    )
    assert np.allclose(scalar, vec, atol=1e-6), f"max|delta|={np.abs(scalar - vec).max():.3g}"


def test_grouped_reduces_to_box_when_no_row_binds():
    """Slack caps must leave the box+budget answer untouched."""
    rng = np.random.default_rng(4)
    v = jnp.asarray(rng.normal(0.1, 0.5, 9))
    plain = np.asarray(project_box_budget(v, 0.0, 0.4, 1.0))
    grouped = np.asarray(
        project_grouped(
            v,
            jnp.zeros(9),
            jnp.full(9, 0.4),
            jnp.asarray(1.0),
            jnp.asarray(np.repeat([0, 1, 2], 3).astype(np.int32)),
            jnp.zeros(4),
            jnp.asarray([1.0, 1.0, 1.0, 0.0]),  # caps far above anything reachable
        )
    )
    assert np.allclose(plain, grouped, atol=1e-5), f"max|delta|={np.abs(plain - grouped).max():.3g}"


# --------------------------------------------------------------------------- #
# Duality — the multipliers are the real ones
# --------------------------------------------------------------------------- #
def _projection_value(v, lower, upper, budget, gid, g_lower, g_upper):
    """The projection's optimal value ``0.5||w - v||^2`` at the given limits."""
    w = project_grouped(v, lower, upper, budget, gid, g_lower, g_upper)
    return 0.5 * float(jnp.sum((w - v) ** 2))


def test_budget_multiplier_is_the_budget_derivative():
    """``d/d budget of the optimal value == -lam``, by central difference."""
    rng = np.random.default_rng(5)
    v, lower, upper, budget, gid, g_lower, g_upper = _grouped_problem(rng, n=10, k=2)
    d = project_grouped_duals(v, lower, upper, budget, gid, g_lower, g_upper)
    h = 1e-3
    plus = _projection_value(v, lower, upper, budget + h, gid, g_lower, g_upper)
    minus = _projection_value(v, lower, upper, budget - h, gid, g_lower, g_upper)
    fd = (plus - minus) / (2 * h)
    assert np.isclose(fd, -float(d.budget), rtol=2e-2, atol=1e-4), (
        f"finite difference {fd:.6f} vs -lam {-float(d.budget):.6f}"
    )


def test_row_multiplier_is_the_cap_derivative():
    """``d/d cap == -theta`` for a binding row, and ``theta == 0`` for a slack one."""
    rng = np.random.default_rng(6)
    n, k = 10, 2
    v = jnp.asarray(rng.normal(0.3, 0.4, n))
    lower, upper = jnp.zeros(n), jnp.full(n, 0.5)
    gid = jnp.asarray(np.repeat([0, 1], n // 2).astype(np.int32))
    # Row 0 tight enough to bind; row 1 wide open.
    caps = np.array([0.25, 1.0, 0.0])
    g_lower = jnp.zeros(k + 1)
    d = project_grouped_duals(v, lower, upper, jnp.asarray(1.0), gid, g_lower, jnp.asarray(caps))
    w = np.asarray(d.weights)
    assert np.isclose(w[:5].sum(), 0.25, atol=1e-5), f"row 0 should bind, sums to {w[:5].sum()}"
    assert float(d.rows[1]) == 0.0, f"slack row must have theta exactly 0, got {d.rows[1]}"
    assert bool(d.identified[0]), "binding row with interior members must be identified"

    h = 1e-3
    fd = (
        _projection_value(v, lower, upper, 1.0, gid, g_lower, jnp.asarray(caps + [h, 0, 0]))
        - _projection_value(v, lower, upper, 1.0, gid, g_lower, jnp.asarray(caps - [h, 0, 0]))
    ) / (2 * h)
    assert np.isclose(fd, -float(d.rows[0]), rtol=3e-2, atol=1e-4), (
        f"finite difference {fd:.6f} vs -theta {-float(d.rows[0]):.6f}"
    )


def test_duals_satisfy_projection_stationarity():
    """``v - w == lam + theta[gid] + beta`` exactly, for every coordinate.

    The single identity that ties all the reported multipliers together; if any of
    them is mis-signed or mis-scaled, this fails.
    """
    rng = np.random.default_rng(7)
    for _ in range(10):
        v, lower, upper, budget, gid, g_lower, g_upper = _grouped_problem(rng, long_only=False)
        d = project_grouped_duals(v, lower, upper, budget, gid, g_lower, g_upper)
        resid = np.abs(
            (np.asarray(v) - np.asarray(d.weights))
            - (float(d.budget) + np.asarray(d.rows)[np.asarray(gid)] + np.asarray(d.bounds))
        ).max()
        assert resid < 1e-4, f"stationarity residual {resid:.3e}"


def test_unidentified_row_is_flagged_not_guessed():
    """A row whose members all sit on a bound has an unidentifiable multiplier.

    The floor of a group whose every member is pinned at its own lower bound is
    satisfied by any split of the shadow price between the row and the box. The
    kernel must report the minimum-norm choice *and* say it is not identified,
    rather than presenting an arbitrary split as fact.
    """
    n = 6
    # Every asset unattractive, so all land on the lower bound; the group floor at
    # zero is then active simultaneously with all three box bounds.
    v = jnp.asarray([-5.0, -5.0, -5.0, 1.0, 1.0, 1.0])
    gid = jnp.asarray(np.array([0, 0, 0, 1, 1, 1], dtype=np.int32))
    d = project_grouped_duals(
        v,
        jnp.zeros(n),
        jnp.full(n, 0.5),
        jnp.asarray(1.0),
        gid,
        jnp.zeros(3),
        jnp.asarray([1.0, 1.0, 0.0]),
    )
    w = np.asarray(d.weights)
    assert np.allclose(w[:3], 0.0, atol=1e-6), f"expected group 0 at zero, got {w[:3]}"
    assert not bool(d.identified[0]), "row with no interior member must be flagged unidentified"
    assert float(d.rows[0]) == 0.0, "unidentified row must report the minimum-norm theta of 0"


def test_duals_survive_jit_and_vmap():
    """The dual extraction is closed-form, so it must trace and batch.

    A host-side solver (scipy least-squares, an SVD for the rank) would fail this;
    it is the guard that keeps the implementation JAX-native.
    """
    rng = np.random.default_rng(8)
    args = _grouped_problem(rng, n=8, k=2)
    jitted = jax.jit(project_grouped_duals)
    ref = project_grouped_duals(*args)
    assert np.allclose(np.asarray(jitted(*args).weights), np.asarray(ref.weights), atol=1e-6)

    stacked = jnp.stack([args[0], args[0] * 1.5, args[0] - 0.2])
    batched = jax.vmap(lambda v: project_grouped_duals(v, *args[1:]))(stacked)
    assert batched.weights.shape == (3, 8), f"got {batched.weights.shape}"
    assert batched.rows.shape == (3, 3), f"got {batched.rows.shape}"
    assert np.allclose(np.asarray(batched.weights[0]), np.asarray(ref.weights), atol=1e-6)


def test_multipliers_scale_linearly_with_the_problem():
    """The multipliers are homogeneous of degree one in ``(v, bounds, budget)``.

    Scale the whole problem by 100 and every multiplier must scale by 100. This
    catches an absolute tolerance used where a relative one belongs — the
    interior/on-bound test gates which multipliers are reported, so an absolute
    epsilon there would change the answer purely with the unit of denomination.

    Uses a problem whose multipliers are *identified* (an interior asset outside
    any binding row), because an unidentified ``lam`` is only one admissible value
    from an interval and has no obligation to scale.
    """
    n = 8
    # Assets 0-3 form a group whose cap binds; 4-7 are ungrouped and land interior,
    # so they pin lam on its own.
    v = jnp.asarray([0.9, 0.8, 0.7, 0.6, 0.2, 0.2, 0.2, 0.2])
    lower, upper = jnp.zeros(n), jnp.full(n, 0.4)
    gid = jnp.asarray(np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32))
    g_lower = jnp.zeros(2)
    g_upper = jnp.asarray([0.5, 0.0])
    args = (v, lower, upper, jnp.asarray(1.0), gid, g_lower, g_upper)
    a = project_grouped_duals(*args)
    assert bool(a.budget_identified), "test needs an identified lam to be meaningful"
    assert bool(a.identified[0]), "test needs an identified row to be meaningful"

    s = 100.0
    b = project_grouped_duals(
        v * s, lower * s, upper * s, jnp.asarray(1.0 * s), gid, g_lower * s, g_upper * s
    )
    assert np.allclose(np.asarray(b.weights), s * np.asarray(a.weights), rtol=1e-4, atol=1e-4)
    assert np.isclose(float(b.budget), s * float(a.budget), rtol=1e-3), (
        f"lam scaled to {float(b.budget):.5f}, expected {s * float(a.budget):.5f}"
    )
    assert np.allclose(np.asarray(b.rows), s * np.asarray(a.rows), rtol=1e-3, atol=1e-4), (
        f"theta {np.asarray(b.rows)} vs {s * np.asarray(a.rows)}"
    )


def test_budget_multiplier_flagged_unidentified_when_no_free_interior_asset():
    """``lam`` is only pinned by an interior coordinate outside a binding row.

    Here every asset is either on a bound or inside a row that binds, so
    stationarity fixes just ``lam + theta_k``. The kernel must say so rather than
    presenting its bisection's value as the cost of capital.
    """
    n = 4
    v = jnp.asarray([0.9, 0.85, 0.8, 0.75])
    gid = jnp.asarray(np.array([0, 0, 1, 1], dtype=np.int32))
    d = project_grouped_duals(
        v,
        jnp.zeros(n),
        jnp.full(n, 0.4),
        jnp.asarray(1.0),
        gid,
        jnp.zeros(3),
        jnp.asarray([0.5, 0.5, 0.0]),  # both rows bind, together exhausting the budget
    )
    w = np.asarray(d.weights)
    assert np.isclose(w[:2].sum(), 0.5, atol=1e-5), f"row 0 should bind, got {w[:2].sum()}"
    assert np.isclose(w[2:].sum(), 0.5, atol=1e-5), f"row 1 should bind, got {w[2:].sum()}"
    assert not bool(d.budget_identified), "lam cannot be identified when every row binds"


# --------------------------------------------------------------------------- #
# Specifications
# --------------------------------------------------------------------------- #
def test_specs_are_hashable_and_usable_as_cache_keys():
    """Frozen + tuple-coerced, because they ride on a hashable OptimizerConfig."""
    a = GroupCap("tech", ["AAPL", "MSFT"], max=0.30)
    b = GroupCap("tech", ("AAPL", "MSFT"), max=0.30)
    assert a == b and hash(a) == hash(b), "list and tuple inputs must compare equal"
    assert {a: 1}[b] == 1, "specs must work as dict keys"


def test_group_floor_is_a_group_cap_so_limits_cannot_disagree():
    floor = GroupFloor("energy", ["XOM"], min=0.05)
    assert isinstance(floor, GroupCap)
    assert floor.min == 0.05 and floor.max is None


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: GroupCap("t", ["A", "A"], max=0.3), "more than once"),
        (lambda: GroupCap("t", [], max=0.3), "at least one asset"),
        (lambda: GroupCap("t", ["A"]), "neither min nor max"),
        (lambda: GroupCap("t", ["A"], min=0.5, max=0.2), "above max"),
        (lambda: GroupCap("", ["A"], max=0.3), "non-empty string"),
        (lambda: GroupCap("t", ["A"], max=float("inf")), "must be finite"),
        (lambda: Box(lower=0.5, upper=0.2), "exceeds upper bound"),
        (lambda: Box(lower=[0.0, 0.1], upper=[0.2]), "different lengths"),
    ],
    ids=[
        "duplicate-member",
        "empty-group",
        "no-limit",
        "min-above-max",
        "empty-name",
        "non-finite",
        "lower-above-upper",
        "ragged-vectors",
    ],
)
def test_spec_validation_rejects_structural_errors(factory, match):
    with pytest.raises((ValueError, TypeError), match=match):
        factory()


@pytest.mark.parametrize("assets", ["AAPL", ["A", 1]], ids=["bare-string", "mixed-name-and-index"])
def test_group_members_must_be_a_uniform_sequence(assets):
    """A bare string must not silently become a tuple of characters."""
    with pytest.raises(TypeError):
        GroupCap("t", assets, max=0.3)


# --------------------------------------------------------------------------- #
# Compilation and feasibility
# --------------------------------------------------------------------------- #
ASSETS = tuple(f"A{i}" for i in range(6))


def test_compiled_set_reports_membership_by_name():
    c = compile_constraints([GroupCap("tech", ["A0", "A2"], max=0.3)], ASSETS)
    assert c.members("tech") == ("A0", "A2")
    assert c.group_of == (0, 1, 0, 1, 1, 1), f"got {c.group_of}"


def test_row_limits_are_clamped_to_what_the_box_permits():
    """An absent min/max becomes the reachable extreme, keeping every value finite.

    Finiteness is what lets the kernel use an exact ``target == freesum`` slack
    test instead of a tolerance, and it keeps infinities out of float32 arithmetic.
    """
    c = compile_constraints([GroupCap("g", ["A0", "A1"], max=0.3), Box(0.0, 0.25)], ASSETS)
    assert c.row_lower == (0.0,), f"absent min -> sum of member lowers, got {c.row_lower}"
    assert c.row_upper == (0.3,), f"explicit max below reach, got {c.row_upper}"
    wide = compile_constraints([GroupCap("g", ["A0", "A1"], max=0.9), Box(0.0, 0.25)], ASSETS)
    assert wide.row_upper == (0.5,), f"max above reach clamps to 2*0.25, got {wide.row_upper}"


@pytest.mark.parametrize(
    ("cons", "kwargs", "match"),
    [
        (
            [GroupCap("tech", ["A0", "A1"], max=0.3), GroupCap("us", ["A1", "A2"], max=0.4)],
            {},
            "partition the universe",
        ),
        (
            [GroupCap("a", ASSETS[:3], max=0.3), GroupCap("b", ASSETS[3:], max=0.3)],
            {},
            "total weight only in",
        ),
        (
            [GroupFloor("a", ASSETS[:3], min=0.7), GroupFloor("b", ASSETS[3:], min=0.7)],
            {},
            "below the reachable range",
        ),
        ([GroupCap("a", ["A0"], min=0.5), Box(0.0, 0.1)], {}, "cannot be met"),
        (
            [Box(0.5, 0.9, assets=["A0"], name="x"), Box(0.0, 0.2, assets=["A0"], name="y")],
            {},
            "no admissible weight",
        ),
    ],
    ids=["overlap", "caps-below-budget", "floors-above-budget", "row-unreachable", "empty-box"],
)
def test_infeasible_sets_are_rejected_at_compile_time(cons, kwargs, match):
    """Every rejection names the offending constraint and quantifies the gap."""
    with pytest.raises((InfeasibleConstraints, ValueError), match=match):
        compile_constraints(cons, ASSETS, **kwargs)


def test_unknown_asset_suggests_the_nearest_name():
    with pytest.raises(KeyError, match="Did you mean 'A1'"):
        compile_constraints([GroupCap("t", ["A10"], max=0.3)], ASSETS)


def test_duplicate_constraint_names_are_rejected():
    with pytest.raises(ValueError, match="both named"):
        compile_constraints(
            [GroupCap("x", ["A0"], max=0.3), GroupCap("x", ["A1"], max=0.3)], ASSETS
        )


def test_caps_exactly_equal_to_the_budget_are_feasible():
    """The feasibility test is exact, so the boundary case must not be rejected."""
    c = compile_constraints(
        [GroupCap("a", ASSETS[:3], max=0.5), GroupCap("b", ASSETS[3:], max=0.5)], ASSETS
    )
    assert c.n_rows == 2


def test_long_only_can_only_tighten_an_explicit_box():
    """``long_only`` must never license a short an explicit Box forbade."""
    c = compile_constraints([Box(-0.5, 0.5)], ASSETS, long_only=True)
    assert c.lower == (0.0,) * 6, f"long_only should floor at 0, got {c.lower}"
    shortable = compile_constraints(
        [Box(-0.5, 0.5)], ASSETS, long_only=False, weight_bounds=(-1.0, 1.0)
    )
    assert shortable.lower == (-0.5,) * 6


def test_check_feasible_validates_without_solving():
    assert check_feasible([GroupCap("a", ["A0"], max=0.5)], ASSETS) is None
    with pytest.raises(InfeasibleConstraints):
        check_feasible([GroupCap("a", ASSETS, max=0.5)], ASSETS)


def test_perturb_keeps_the_kernel_and_honors_the_original_bounds():
    """A relaxation must move only values — that is what makes a re-solve cheap.

    It must also preserve bounds that came from the config rather than from a Box
    spec, which an earlier version silently reset to the defaults.
    """
    c = compile_constraints(
        [GroupCap("tech", ["A0", "A1"], max=0.3)],
        ASSETS,
        long_only=False,
        weight_bounds=(-0.2, 0.6),
    )
    out = c.perturb("tech", max=0.35)
    assert out.kind == c.kind and out.n_rows == c.n_rows
    assert out.row_upper == (0.35,), f"got {out.row_upper}"
    assert out.lower == c.lower == (-0.2,) * 6, "config bounds must survive a perturb"
    with pytest.raises(KeyError, match="no constraint named"):
        c.perturb("nope", max=0.4)


# --------------------------------------------------------------------------- #
# Contract — what must not move
# --------------------------------------------------------------------------- #
def test_empty_constraint_set_is_byte_identical_to_the_legacy_path():
    """No named constraints must reproduce the *identical* function object and tuple.

    Not "equivalent" — identical. Anything else would give every pre-existing solve
    a fresh jit cache entry, and a Python float replaced by a weak-typed JAX scalar
    would perturb results in the last bits.
    """
    from jaxfolio.optimizers.base import _proj_box, _proj_simplex, select_projection

    for long_only, bounds in [(True, (0.0, 1.0)), (False, (-1.0, 1.0)), (True, (0.0, 0.3))]:
        legacy_fn, legacy_pp = select_projection(long_only, bounds)
        compiled = compile_constraints([], ASSETS, long_only=long_only, weight_bounds=bounds)
        fn, pp = compiled.projection()
        assert fn is legacy_fn, f"{long_only},{bounds}: {fn} is not {legacy_fn}"
        assert pp == legacy_pp, f"{long_only},{bounds}: {pp} != {legacy_pp}"
        assert all(type(x) is float for x in pp), f"pparams must stay Python floats, got {pp}"
    assert select_projection(True, (0.0, 1.0))[0] is _proj_simplex
    assert select_projection(False, (-1.0, 1.0))[0] is _proj_box


@pytest.mark.parametrize(
    ("aux", "path"), [(False, False), (True, False), (False, True)], ids=["plain", "aux", "path"]
)
def test_projection_table_entries_have_stable_identity(aux, path):
    """The jit cache keys on ``id(projection)``, so the table must be built once.

    If this table is ever rebuilt per call, every solve silently recompiles — the
    exact bug the "keep them at module scope" note in ``base.py`` warns about.
    """
    from jaxfolio.optimizers.base import _PROJECTIONS, projection_for

    for kind in ("simplex", "box", "boxvec", "grouped"):
        key = (kind, "aux" if aux else "path" if path else "plain")
        assert _PROJECTIONS[key] is _PROJECTIONS[key], f"{key} identity unstable"
    c = compile_constraints([GroupCap("g", ["A0"], max=0.5)], ASSETS)
    assert projection_for(c, aux=aux, path=path)[0] is projection_for(c, aux=aux, path=path)[0]


def test_aux_and_path_are_mutually_exclusive():
    from jaxfolio.optimizers.base import select_projection

    with pytest.raises(ValueError, match="mutually exclusive"):
        select_projection(True, (0.0, 1.0), aux=True, path=True)


def test_named_constraints_need_the_asset_names():
    from jaxfolio.optimizers.base import make_projection, select_projection

    cons = [GroupCap("g", ["A0"], max=0.5)]
    with pytest.raises(ValueError, match="need `assets`"):
        select_projection(True, (0.0, 1.0), constraints=cons)
    with pytest.raises(ValueError, match="need `assets`"):
        make_projection(True, (0.0, 1.0), constraints=cons)


def test_path_projection_applies_the_constraints_row_wise():
    """The ``(T, N)`` variant must cap each period independently.

    The multi-period feasible set is a Cartesian product over periods, so the
    projection onto it is the product of the per-period projections.
    """
    from jaxfolio.optimizers.base import projection_for

    c = compile_constraints([GroupCap("g", ["A0", "A1"], max=0.3)], ASSETS)
    fn, pp = projection_for(c, path=True)
    rng = np.random.default_rng(11)
    path = jnp.asarray(rng.normal(0.2, 0.5, (4, 6)))
    out = np.asarray(fn(path, pp))
    assert out.shape == (4, 6)
    for t, row in enumerate(out):
        assert np.isclose(row.sum(), 1.0, atol=1e-5), f"row {t} sums to {row.sum()}"
        assert row[:2].sum() <= 0.3 + 1e-5, f"row {t} group sums to {row[:2].sum()}"


def test_aux_projection_leaves_the_auxiliary_tail_untouched():
    """CVaR packs ``[w, tau]``; ``tau`` is unconstrained by construction."""
    from jaxfolio.optimizers.base import projection_for

    c = compile_constraints([GroupCap("g", ["A0", "A1"], max=0.3)], ASSETS)
    fn, pp = projection_for(c, aux=True)
    z = jnp.asarray([0.5, 0.5, 0.1, 0.1, 0.1, 0.1, -7.25])
    out = np.asarray(fn(z, pp))
    assert out[-1] == pytest.approx(-7.25), f"tau moved to {out[-1]}"
    assert np.isclose(out[:-1].sum(), 1.0, atol=1e-5)
    assert out[:2].sum() <= 0.3 + 1e-5, f"group sums to {out[:2].sum()}"


def test_jit_cache_grows_only_on_structural_change():
    """Constraint *values* are traced; only shapes and kinds are static.

    This is what makes a what-if relaxation cheap: changing a cap, or even which
    tickers are in a sector, must not recompile.
    """
    import jaxfolio as jf
    from jaxfolio.data import generate_returns
    from jaxfolio.optimizers.base import _solve_cached

    rets = generate_returns(n_assets=6, n_days=200, seed=3)
    names = list(rets.columns)

    jf.minimum_variance(rets)
    base = _solve_cached._cache_size()
    jf.minimum_variance(rets)
    assert _solve_cached._cache_size() == base, "an unconstrained repeat must not recompile"

    jf.minimum_variance(rets, constraints=[GroupCap("t", names[:3], max=0.3)])
    after_first = _solve_cached._cache_size()
    assert after_first == base + 1, f"first grouped solve should add one entry, got {after_first}"

    for cap in (0.32, 0.35, 0.50):
        jf.minimum_variance(rets, constraints=[GroupCap("t", names[:3], max=cap)])
    assert _solve_cached._cache_size() == after_first, "changing a cap value must not recompile"

    jf.minimum_variance(rets, constraints=[GroupCap("t", names[2:5], max=0.3)])
    assert _solve_cached._cache_size() == after_first, "changing membership must not recompile"

    jf.minimum_variance(
        rets, constraints=[GroupCap("a", names[:2], max=0.4), GroupCap("b", names[2:4], max=0.4)]
    )
    assert _solve_cached._cache_size() == after_first + 1, "a new row count costs one compile"


# --------------------------------------------------------------------------- #
# Integration — the constraints actually bind end to end
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def panel():
    from jaxfolio.data import generate_returns

    return generate_returns(n_assets=6, n_days=400, seed=7)


@pytest.mark.parametrize(
    "method",
    ["minimum_variance", "mean_variance", "maximum_sharpe", "maximum_diversification", "kelly"],
)
def test_group_cap_binds_through_every_solver_based_optimizer(panel, method):
    import jaxfolio as jf

    names = list(panel.columns)
    fn = getattr(jf, method)
    capped = fn(panel, constraints=[GroupCap("tech", names[:3], max=0.30)])
    got = float(capped.weights[:3].sum())
    assert got <= 0.30 + 1e-4, f"{method}: tech exposure {got:.4f} exceeds the 0.30 cap"
    assert np.isclose(capped.weights.sum(), 1.0, atol=1e-5), f"sum={capped.weights.sum()}"


def test_group_cap_binds_through_min_cvar(panel):
    """The packed ``[w, tau]`` optimizer must honor the cap on its weight block."""
    import jaxfolio as jf

    names = list(panel.columns)
    res = jf.min_cvar(panel, constraints=[GroupCap("tech", names[:3], max=0.30)])
    assert float(res.weights[:3].sum()) <= 0.30 + 1e-4, f"tech={res.weights[:3].sum():.4f}"
    assert np.isclose(res.weights.sum(), 1.0, atol=1e-4)


def test_per_asset_cap_binds(panel):
    import jaxfolio as jf

    res = jf.maximum_sharpe(panel, constraints=[Box(0.0, 0.25)])
    assert res.weights.max() <= 0.25 + 1e-5, f"max weight {res.weights.max():.4f}"


def test_group_floor_binds(panel):
    import jaxfolio as jf

    names = list(panel.columns)
    res = jf.minimum_variance(panel, constraints=[GroupFloor("energy", names[4:], min=0.40)])
    assert float(res.weights[4:].sum()) >= 0.40 - 1e-4, f"energy={res.weights[4:].sum():.4f}"


def test_constraints_may_come_from_the_config_or_the_keyword_but_not_both(panel):
    import jaxfolio as jf

    names = list(panel.columns)
    cons = [GroupCap("tech", names[:3], max=0.20)]
    via_kwarg = jf.minimum_variance(panel, constraints=cons)
    via_config = jf.minimum_variance(panel, jf.OptimizerConfig(constraints=cons))
    assert np.allclose(via_kwarg.weights, via_config.weights, atol=1e-6)
    with pytest.raises(ValueError, match="both on OptimizerConfig and as a keyword"):
        jf.minimum_variance(panel, jf.OptimizerConfig(constraints=cons), constraints=cons)


def test_closed_form_optimizers_reject_constraints_rather_than_ignore_them(panel):
    """Silently returning weights that violate a cap would be the worst outcome."""
    import jaxfolio as jf

    names = list(panel.columns)
    cons = [GroupCap("tech", names[:3], max=0.20)]
    with pytest.raises(NotImplementedError, match="closed-form weighting"):
        jf.risk_parity(panel, constraints=cons)
    with pytest.raises(NotImplementedError, match="does not yet support named constraints"):
        jf.multi_period_mean_variance(panel, horizon=3, config=jf.OptimizerConfig(constraints=cons))


def test_custom_strategy_inherits_named_constraints(panel):
    import jaxfolio as jf

    names = list(panel.columns)
    cfg = jf.OptimizerConfig(constraints=[GroupCap("tech", names[:3], max=0.25)])
    strategy = jf.CustomStrategy.from_objective(
        "constrained minvar", lambda w, ctx: w @ ctx.cov @ w, config=cfg
    )
    res = strategy(panel)
    assert float(res.weights[:3].sum()) <= 0.25 + 1e-4, f"tech={res.weights[:3].sum():.4f}"


def test_optimizer_config_validates_constraints_structurally():
    import jaxfolio as jf

    with pytest.raises(TypeError, match="must be jaxfolio.constraints specifications"):
        jf.OptimizerConfig(constraints=["not a constraint"])
    with pytest.raises(ValueError, match="two entries named"):
        jf.OptimizerConfig(
            constraints=[GroupCap("x", ["A0"], max=0.3), GroupCap("x", ["A1"], max=0.3)]
        )
    cfg = jf.OptimizerConfig(constraints=[GroupCap("x", ["A0"], max=0.3)])
    assert isinstance(cfg.constraints, tuple), "must be coerced to a tuple to stay hashable"
    assert hash(cfg) is not None, "OptimizerConfig must remain hashable"
