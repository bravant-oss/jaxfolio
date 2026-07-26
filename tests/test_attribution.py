"""Tests for constraint attribution — the shadow prices and the refusals.

Three families, in increasing strength:

*Identities* — relations the recovered multipliers must satisfy at any solution,
checkable without a reference solver: the pricing decomposition
``rc = intrinsic + imposed``, complementary slackness, the sign of a reduced cost
at each kind of active bound, and scale equivariance.

*Sensitivity* — the claim a user actually reads. ``shadow_price`` must equal
``d(objective)/d(limit)``, verified by central finite differences on re-solves.
This is the family that would catch a sign error, a factor of two, or a
mis-scaled probe step, and it needs no external solver. All three sign regimes are
covered, because each fails differently: minimize+cap, maximize+cap, and
minimize+floor (where taking an absolute value would silently invert the answer).

*Refusals* — assertions about what the report does **not** claim. A number that
cannot be stood behind must come back ``None``, not plausible-looking. Non-smooth
objectives, unconverged solves, scale-invariant objectives, degenerate vertices,
unidentified multipliers, and optimizers with no duals at all each get a test that
the refusal actually fires.
"""

from __future__ import annotations

import json
import pickle
import re

import numpy as np
import pytest

import jaxfolio as jf
from jaxfolio.attribution import explain

TECH_CAP = 0.30


@pytest.fixture(scope="module")
def panel():
    return jf.generate_returns(n_assets=8, n_days=500, seed=11)


@pytest.fixture(scope="module")
def names(panel):
    return list(panel.columns)


@pytest.fixture(scope="module")
def cov(panel):
    _mu, cov, _n, _m = jf.toolkit.moments(panel)
    return np.asarray(cov)


def _capped(panel, names, cap=TECH_CAP, **kw):
    """Minimum variance with a binding tech cap, converged tightly."""
    return jf.minimum_variance(
        panel,
        jf.OptimizerConfig(
            tol=kw.pop("tol", 1e-9),
            constraints=[jf.GroupCap("tech", names[:4], max=cap)],
            **kw,
        ),
    )


# --------------------------------------------------------------------------- #
# Identities
# --------------------------------------------------------------------------- #
def test_pricing_decomposition_is_exact(panel, names):
    """``rc_i == intrinsic_i + sum(imposed_i)`` for every asset on a bound.

    The identity the whole attribution rests on: a reduced cost is the asset's own
    unattractiveness plus what each constraint charges it. If this drifts, the
    ranking that picks a "primary" cause is ranking meaningless numbers.
    """
    report = explain(_capped(panel, names))
    checked = 0
    for a in report.assets_detail:
        if a.intrinsic_cost is None:
            continue
        total = a.intrinsic_cost + sum(v for _, v in a.imposed)
        assert np.isclose(a.reduced_cost, total, rtol=1e-5, atol=1e-12), (
            f"{a.asset}: rc={a.reduced_cost:.8g} != intrinsic+imposed={total:.8g}"
        )
        checked += 1
    assert checked > 0, "no asset sat on a bound; the test proved nothing"


def test_reduced_cost_signs_match_the_active_bound(panel, names):
    """At a lower bound ``rc >= 0``; at an upper bound ``rc <= 0``; interior ``rc == 0``.

    This is dual feasibility written per asset, and it is the sign convention every
    downstream reading depends on.
    """
    report = explain(_capped(panel, names))
    scale = max(abs(report.assets_detail[0].reduced_cost or 0.0), 1e-9)
    tol = 1e-4 * max(scale, 1e-9)
    seen = set()
    for a in report.assets_detail:
        if a.reduced_cost is None:
            continue
        seen.add(a.status)
        if a.status == "at_lower":
            assert a.reduced_cost >= -tol, f"{a.asset} at lower has rc={a.reduced_cost:.3e}"
        elif a.status == "at_upper":
            assert a.reduced_cost <= tol, f"{a.asset} at upper has rc={a.reduced_cost:.3e}"
    assert "at_lower" in seen, f"expected some asset at its lower bound, saw {seen}"


def test_complementary_slackness(panel, names):
    """A slack constraint carries a zero multiplier — exactly, not approximately."""
    report = explain(
        jf.minimum_variance(
            panel,
            jf.OptimizerConfig(
                tol=1e-9,
                constraints=[
                    jf.GroupCap("tech", names[:4], max=TECH_CAP),  # binds
                    jf.GroupCap("rest", names[4:], max=0.99),  # cannot bind
                ],
            ),
        )
    )
    by_name = {c.name: c for c in report.constraints}
    assert by_name["tech"].status == "binding", by_name["tech"]
    assert by_name["rest"].status == "inactive", by_name["rest"]
    assert by_name["rest"].multiplier == 0.0
    for c in report.constraints:
        if c.multiplier is not None:
            assert abs(c.multiplier * c.slack) < 1e-6, (
                f"{c.name}: multiplier {c.multiplier:.3e} x slack {c.slack:.3e}"
            )


def test_binding_cap_and_floor_have_opposite_multiplier_signs(panel, names):
    """A cap pushes weight down, a floor pushes it up; the multipliers must differ."""
    cap = explain(_capped(panel, names)).constraints[0]
    floor = explain(
        jf.minimum_variance(
            panel,
            jf.OptimizerConfig(
                tol=1e-9, constraints=[jf.GroupFloor("energy", names[4:], min=0.70)]
            ),
        )
    ).constraints[0]
    assert cap.status == floor.status == "binding"
    assert cap.multiplier > 0, f"binding cap should have theta > 0, got {cap.multiplier}"
    assert floor.multiplier < 0, f"binding floor should have theta < 0, got {floor.multiplier}"


def test_multipliers_scale_with_the_objective(panel, names):
    """Scaling the objective scales every multiplier by the same factor.

    ``minimum_variance`` on ``100 * returns`` has a covariance 1e4 times larger, so
    the gradient and every multiplier scale by 1e4 while the weights do not. Catches
    an absolute tolerance standing in for a relative one anywhere in the chain.
    """
    a = explain(_capped(panel, names))
    b = explain(_capped(panel * 100.0, names))
    assert np.allclose(a.weights, b.weights, atol=1e-4), "weights should be scale-invariant"
    ratio = b.constraints[0].multiplier / a.constraints[0].multiplier
    assert np.isclose(ratio, 1e4, rtol=1e-3), f"multiplier scaled by {ratio:.6g}, expected 1e4"


# --------------------------------------------------------------------------- #
# Sensitivity — the claim the user reads
# --------------------------------------------------------------------------- #
def _fd(fn, x, h):
    return (fn(x + h) - fn(x - h)) / (2 * h)


@pytest.mark.parametrize("cap", [0.25, 0.30, 0.35], ids=["tight", "mid", "loose"])
def test_shadow_price_equals_the_variance_derivative(panel, names, cov, cap):
    """minimize + cap: ``shadow_price == d(variance)/d(cap)``, and is negative."""

    def variance(c):
        w = _capped(panel, names, cap=float(c)).weights
        return float(w @ cov @ w)

    report = explain(_capped(panel, names, cap=cap))
    c = report.constraints[0]
    assert c.status == "binding", f"cap {cap} did not bind: {c.status}"
    fd = _fd(variance, cap, 1e-3)
    assert c.shadow_price < 0, "relaxing a cap must lower a minimized variance"
    assert np.isclose(c.shadow_price, fd, rtol=5e-3), (
        f"shadow_price={c.shadow_price:.6e} vs finite difference {fd:.6e}"
    )


def test_shadow_price_equals_the_sharpe_derivative(panel, names, cov):
    """maximize + cap: the sign flips with ``sense``, magnitude still matches."""
    mu, _c, _n, _m = jf.toolkit.moments(panel)
    mu = np.asarray(mu)

    def sharpe(c):
        w = jf.maximum_sharpe(
            panel,
            jf.OptimizerConfig(
                tol=1e-9, constraints=[jf.GroupCap("tech", names[:4], max=float(c))]
            ),
        ).weights
        return float((w @ mu) / np.sqrt(w @ cov @ w))

    report = explain(
        jf.maximum_sharpe(
            panel,
            jf.OptimizerConfig(tol=1e-9, constraints=[jf.GroupCap("tech", names[:4], max=0.20)]),
        )
    )
    c = report.constraints[0]
    assert c.status == "binding", c.status
    fd = _fd(sharpe, 0.20, 1e-3)
    assert c.shadow_price > 0, "relaxing a cap must raise a maximized Sharpe ratio"
    assert np.isclose(c.shadow_price, fd, rtol=5e-3), (
        f"shadow_price={c.shadow_price:.6e} vs finite difference {fd:.6e}"
    )


def test_shadow_price_of_a_floor_has_the_right_sign(panel, names, cov):
    """minimize + floor: raising the limit *tightens*, so variance rises.

    The case an ``abs()`` in the sign handling would invert while every cap test
    kept passing.
    """

    def variance(lo):
        w = jf.minimum_variance(
            panel,
            jf.OptimizerConfig(
                tol=1e-9, constraints=[jf.GroupFloor("energy", names[4:], min=float(lo))]
            ),
        ).weights
        return float(w @ cov @ w)

    report = explain(
        jf.minimum_variance(
            panel,
            jf.OptimizerConfig(
                tol=1e-9, constraints=[jf.GroupFloor("energy", names[4:], min=0.70)]
            ),
        )
    )
    c = report.constraints[0]
    assert c.status == "binding" and c.kind == "group_floor", (c.status, c.kind)
    fd = _fd(variance, 0.70, 1e-3)
    assert c.shadow_price > 0, "raising a floor tightens the problem, so variance must rise"
    assert np.isclose(c.shadow_price, fd, rtol=5e-3), (
        f"shadow_price={c.shadow_price:.6e} vs finite difference {fd:.6e}"
    )


def test_budget_shadow_price_equals_the_budget_derivative(panel, names, cov):
    """The cost of capital, checked the same way."""

    def variance(b):
        w = jf.minimum_variance(
            panel, jf.OptimizerConfig(tol=1e-9, constraints=[jf.Budget(float(b))])
        ).weights
        return float(w @ cov @ w)

    report = explain(
        jf.minimum_variance(panel, jf.OptimizerConfig(tol=1e-9, constraints=[jf.Budget(1.0)]))
    )
    assert report.budget_shadow_price is not None, report.budget_note
    fd = _fd(variance, 1.0, 1e-3)
    assert np.isclose(report.budget_shadow_price, fd, rtol=5e-3), (
        f"budget shadow price {report.budget_shadow_price:.6e} vs {fd:.6e}"
    )


# --------------------------------------------------------------------------- #
# Attribution — naming a single cause
# --------------------------------------------------------------------------- #
def test_a_zero_weight_is_attributed_to_the_named_cap_not_to_the_box(panel, names):
    """The headline claim: "0% because of the tech cap", not "because of w >= 0".

    At a zero weight the box multiplier *equals the whole reduced cost*, so ranking
    multipliers instead of the reduced-cost terms would make the box win every time
    and this attribution could never appear.
    """
    report = explain(_capped(panel, names, cap=0.10))
    tech = set(names[:4])
    attributed = [
        a
        for a in report.assets_detail
        if a.asset in tech and a.status == "at_lower" and a.primary == "tech"
    ]
    assert attributed, (
        "expected at least one capped-sector asset attributed to 'tech'; got "
        + ", ".join(f"{a.asset}->{a.primary}" for a in report.assets_detail if a.asset in tech)
    )
    a = attributed[0]
    assert a.primary_kind == "group_cap"
    assert a.primary_shadow_price is not None
    assert a.confidence == "unique"


def test_an_unconstrained_zero_is_attributed_to_own_merits(panel, names):
    """With no groups, every exclusion is the asset's own fault — the truthful answer."""
    report = explain(jf.minimum_variance(panel, jf.OptimizerConfig(tol=1e-9)))
    zeros = [a for a in report.assets_detail if a.status == "at_lower"]
    assert zeros, "expected some asset at zero in an unconstrained long-only solve"
    for a in zeros:
        assert a.primary == "own", f"{a.asset} attributed to {a.primary!r}, expected 'own'"
        assert a.primary_kind == "own_merits"


def test_interior_assets_have_a_vanishing_reduced_cost(panel, names):
    report = explain(_capped(panel, names))
    interior = [a for a in report.assets_detail if a.status == "interior"]
    assert interior, "expected interior holdings"
    scale = max(abs(report.constraints[0].multiplier), 1e-12)
    for a in interior:
        assert abs(a.reduced_cost) <= 1e-3 * scale, (
            f"{a.asset} is interior but rc={a.reduced_cost:.3e}"
        )


# --------------------------------------------------------------------------- #
# Refusals — what the report must not claim
# --------------------------------------------------------------------------- #
def test_min_cvar_refuses_shadow_prices(panel, names):
    """Non-smooth at the optimum, so stationarity need not hold even when exact.

    ``tau`` sits on a sample loss generically, and ``jax.grad`` returns one
    arbitrary subgradient there. A positive test that the refusal fires.
    """
    res = jf.min_cvar(panel, constraints=[jf.GroupCap("tech", names[:4], max=TECH_CAP)])
    report = explain(res)
    assert report.quality == "unreliable"
    assert "non-smooth" in report.reason
    assert all(c.shadow_price is None for c in report.constraints)
    assert all(a.primary_shadow_price is None for a in report.assets_detail)
    assert report.aux_stationarity is not None, "tau's own stationarity should still be reported"
    assert len(res.attribution.weights) == len(names), "duals must cover w only, not the tau tail"


def test_an_unconverged_solve_refuses_shadow_prices(panel, names):
    """A weight-update norm is not a KKT measure; the report computes its own.

    ``sgd`` with five iterations stops nowhere near a solution, and its own
    ``residual`` cannot reveal that. The independently computed KKT residual can.
    """
    res = _capped(panel, names, tol=1e-7, solver="sgd", max_iter=5)
    report = explain(res)
    assert report.quality == "unreliable", f"got {report.quality}, kkt={report.kkt_residual:.3e}"
    assert "converge" in report.reason
    assert all(c.shadow_price is None for c in report.constraints)
    assert res.metadata["kkt_residual"] > 1e-5, "the independent KKT measure should be large"


@pytest.mark.parametrize(
    "method", ["maximum_sharpe", "maximum_diversification"], ids=["sharpe", "diversification"]
)
def test_scale_invariant_objectives_refuse_a_budget_shadow_price(panel, names, method):
    """Degree-0 homogeneous objectives have ``g'w == 0``, so the budget prices nothing.

    Detected numerically rather than from a list of method names, so a custom
    scale-invariant objective gets the same protection.
    """
    report = explain(
        getattr(jf, method)(
            panel,
            jf.OptimizerConfig(tol=1e-9, constraints=[jf.GroupCap("tech", names[:4], max=0.20)]),
        )
    )
    assert report.budget_shadow_price is None
    assert report.budget_note is not None and "scale-invariant" in report.budget_note
    # Row multipliers stay meaningful; only the budget is affected.
    assert any(c.shadow_price is not None for c in report.constraints)


def test_unidentified_row_refuses_a_single_cause(panel, names):
    """When every member of a binding row sits on a bound, the split is arbitrary.

    Constructed so the row and the per-asset boxes bind *simultaneously* with no
    interior member: take the two assets the unconstrained solve wants most, cap
    each at 0.10, and cap their group at exactly 0.20. Both saturate, so only the
    sum ``theta + beta`` is determined and neither the row nor the box can be named
    as the cause.
    """
    free = jf.minimum_variance(panel, jf.OptimizerConfig(tol=1e-9)).weights
    top2 = [names[i] for i in np.argsort(-free)[:2]]
    report = explain(
        jf.minimum_variance(
            panel,
            jf.OptimizerConfig(
                tol=1e-9,
                constraints=[
                    jf.GroupCap("tech", top2, max=0.20),
                    jf.Box(upper=0.10, assets=top2, name="tech_names"),
                ],
            ),
        )
    )
    tech = report.constraints[0]
    assert tech.status == "unidentified", f"expected a degenerate row, got {tech.status!r}"
    assert np.isclose(tech.activity, 0.20, atol=1e-5), f"row should sit on its cap: {tech.activity}"
    assert tech.shadow_price is None, "an unidentified multiplier must not be quoted"
    assert tech.multiplier is None
    detail = [a for a in report.assets_detail if a.asset in set(top2)]
    assert all(a.status == "at_upper" for a in detail), [a.status for a in detail]
    got = [(a.asset, a.primary, a.confidence) for a in detail]
    assert all(a.confidence == "ambiguous" and a.primary is None for a in detail), (
        f"expected ambiguous attribution, got {got}"
    )


@pytest.mark.parametrize(
    "method",
    [
        "equal_weight",
        "inverse_volatility",
        "risk_parity",
        "hierarchical_risk_parity",
        "hierarchical_equal_risk",
        "mst_centrality",
    ],
)
def test_optimizers_without_duals_return_a_wellformed_report(panel, method):
    """Feasible-by-construction is not optimal-subject-to, and the report says so."""
    report = explain(getattr(jf, method)(panel))
    assert report.available is False
    assert report.quality == "unavailable"
    assert report.reason, "the reason must be specific, not empty"
    assert report.constraints == ()
    assert len(report.assets_detail) == len(report.assets)
    assert all(a.primary_shadow_price is None for a in report.assets_detail)
    frame = report.to_frame()
    assert len(frame) == len(report.assets)
    assert frame["shadow_price"].isna().all(), "no shadow price may be invented"


def test_risk_parity_points_at_its_own_attribution_output(panel):
    """Its reason must be actionable, not merely a refusal."""
    report = explain(jf.risk_parity(panel))
    assert "risk_contributions" in report.reason
    assert "log-barrier" in report.reason


def test_l2_regularization_is_warned_about(panel, names):
    """The multipliers then price the penalized objective, and L2 softens corners."""
    report = explain(_capped(panel, names, l2_reg=1e-3))
    assert any("l2_reg" in w for w in report.warnings), report.warnings


def test_non_convex_objective_is_flagged_as_local_only(panel, names):
    report = explain(
        jf.maximum_sharpe(
            panel,
            jf.OptimizerConfig(tol=1e-9, constraints=[jf.GroupCap("tech", names[:4], max=0.20)]),
        )
    )
    assert any("not convex" in w for w in report.warnings), report.warnings


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #
def test_metadata_digest_keys_and_json_serializability(panel, names):
    """Consumers that only read ``metadata`` must still see the summary, as JSON."""
    res = _capped(panel, names)
    expected = {
        "iterations",
        "kkt_residual",
        "stationarity",
        "dual_quality",
        "n_binding_constraints",
        "binding_constraints",
    }
    assert set(res.metadata) == expected, f"got {sorted(res.metadata)}"
    json.dumps(res.metadata)  # must not raise
    assert res.metadata["binding_constraints"] == ["tech"]
    assert res.metadata["n_binding_constraints"] == 1


def test_result_with_attribution_pickles(panel, names):
    """The payload is plain numpy and tuples, so a result stays portable."""
    res = _capped(panel, names)
    back = pickle.loads(pickle.dumps(res))
    assert np.allclose(back.weights, res.weights)
    assert back.attribution is not None
    assert explain(back).constraints[0].status == "binding"


def test_attribution_can_be_switched_off(panel, names):
    """The escape hatch for tight backtest loops; ``explain`` must still behave."""
    res = jf.minimum_variance(
        panel,
        jf.OptimizerConfig(
            attribution=False, constraints=[jf.GroupCap("tech", names[:4], max=TECH_CAP)]
        ),
    )
    assert res.attribution is None
    assert "kkt_residual" not in res.metadata
    report = explain(res)
    assert report.available is False and report.quality == "unavailable"


def test_report_surface(panel, names):
    """``to_frame`` / ``constraints_frame`` / ``for_asset`` / ``binding`` / text."""
    report = explain(_capped(panel, names, cap=0.10))
    frame = report.to_frame()
    assert list(frame.index) == list(report.assets)
    assert {
        "weight",
        "status",
        "reduced_cost",
        "intrinsic_cost",
        "imposed_total",
        "binding",
        "binding_kind",
        "shadow_price",
        "confidence",
    } <= set(frame.columns)

    cframe = report.constraints_frame()
    assert list(cframe.index) == ["tech"]
    assert cframe.loc["tech", "status"] == "binding"

    assert len(report.binding()) == 1
    assert report.for_asset(names[0]).asset == names[0]
    with pytest.raises(KeyError, match="not in this portfolio"):
        report.for_asset("NOPE")

    text = report.explain_text(top_n=3)
    assert "Constraint attribution" in text
    assert "quality:" in text
    assert repr(report).startswith("ConstraintReport(")


# --------------------------------------------------------------------------- #
# Printable and serialized renderings
# --------------------------------------------------------------------------- #
def _sections(table):
    """Map each section heading to the data lines under it (header + rule skipped)."""
    lines = table.splitlines()
    out, current = {}, None
    for line in lines:
        if line in ("CONSTRAINTS", "ASSETS", "NOTES"):
            current = out.setdefault(line, [])
            continue
        if not line.strip():
            current = None
        elif current is not None:
            current.append(line)
    for name in ("CONSTRAINTS", "ASSETS"):
        if name in out:
            out[name] = out[name][2:]  # drop the column header and its rule
    return out


def test_to_table_prints_every_row_and_every_asset(panel, names):
    """The table rendering is the whole report, not a preview of it."""
    report = explain(_capped(panel, names))
    table = report.to_table()

    assert "CONSTRAINT ATTRIBUTION -- Minimum Variance" in table
    assert "binding rows" in table
    assert "<= 30.00%" in table, "the bound column should name the cap that binds"
    assert "not annualized" in table, "the units caveat travels with the numbers"

    sections = _sections(table)
    assert len(sections["CONSTRAINTS"]) == len(report.constraints)
    assert len(sections["ASSETS"]) == len(names)
    assert [row.split()[0] for row in sections["ASSETS"]] == [
        a.asset
        for a in sorted(
            report.assets_detail, key=lambda a: (a.status == "interior", -abs(a.weight))
        )
    ]


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda p, n: _capped(p, n), id="binding-cap"),
        # Each of these carries a long sentence — a budget note, a quality reason, a
        # non-convexity warning — that would run past 150 columns unwrapped.
        pytest.param(
            lambda p, n: jf.maximum_sharpe(p, constraints=[jf.GroupCap("t", n[:4], max=TECH_CAP)]),
            id="scale-invariant",
        ),
        pytest.param(
            lambda p, n: jf.min_cvar(p, constraints=[jf.GroupCap("t", n[:4], max=TECH_CAP)]),
            id="non-smooth",
        ),
        pytest.param(lambda p, _n: jf.risk_parity(p), id="no-duals"),
    ],
)
def test_to_table_is_ascii_and_stays_within_a_terminal(panel, names, build):
    """A printed report lands in log files and non-UTF-8 consoles; keep it plain."""
    table = explain(build(panel, names)).to_table()
    table.encode("ascii")  # raises if typography leaked in from the prose rendering
    widest = max(table.splitlines(), key=len)
    assert len(widest) <= 100, f"{len(widest)} columns: {widest!r}"


def test_to_table_truncation_is_announced(panel, names):
    """Dropping rows silently would let a report read as complete when it is not."""
    report = explain(_capped(panel, names))
    table = report.to_table(max_assets=2)
    assert len(_sections(table)["ASSETS"]) == 3  # two assets plus the notice
    assert f"({len(names) - 2} further asset(s) not shown)" in table


def test_to_table_of_an_unavailable_report_says_so(panel):
    """No duals at all still renders — with the reason, and without inventing rows."""
    table = explain(jf.equal_weight(panel)).to_table()
    assert "unavailable:" in table
    assert "CONSTRAINTS" not in table
    assert "n/a" in _sections(table)["ASSETS"][0], "no cause can be named without duals"


def test_to_table_shows_suppressed_shadow_prices_as_missing(panel, names):
    """A suppressed number prints as ``n/a``, never as a plausible-looking zero."""
    cvar = jf.min_cvar(panel, constraints=[jf.GroupCap("tech", names[:4], max=TECH_CAP)])
    table = explain(cvar).to_table()
    assert re.search(r"dual quality\s+unreliable", table)
    for row in _sections(table)["CONSTRAINTS"]:
        assert "n/a" in row


def _walk(value):
    """Yield every scalar in a nested dict/list structure."""
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)
    else:
        yield value


def test_to_dict_is_json_safe_and_complete(panel, names):
    report = explain(_capped(panel, names))
    payload = report.to_dict()

    assert payload["schema"] == "jaxfolio.constraint_report/1"
    assert payload["available"] is True
    assert payload["quality"] == "exact"
    assert [c["name"] for c in payload["constraints"]] == [c.name for c in report.constraints]
    assert [a["asset"] for a in payload["assets"]] == list(report.assets)
    assert payload["assets"][0]["imposed"] == [] or set(payload["assets"][0]["imposed"][0]) == {
        "constraint",
        "charge",
    }

    # No numpy scalars: a float32 weight would serialize on some versions and raise
    # on others, so the boundary is enforced here rather than discovered in a log.
    for scalar in _walk(payload):
        assert scalar is None or type(scalar) in (bool, int, float, str), type(scalar)

    assert json.loads(report.to_json()) == payload
    assert len(report.to_json(indent=None).splitlines()) == 1


def test_to_dict_keeps_a_declined_number_as_none(panel, names):
    """``None`` must survive serialization — 0.0 would read as "costs nothing"."""
    cvar = jf.min_cvar(panel, constraints=[jf.GroupCap("tech", names[:4], max=TECH_CAP)])
    payload = json.loads(explain(cvar).to_json())

    assert payload["quality"] == "unreliable"
    assert payload["reason"]
    assert all(c["shadow_price"] is None for c in payload["constraints"])
    assert all(c["activity"] is not None for c in payload["constraints"]), (
        "primal facts are still reported when the duals are not"
    )


def test_unavailable_report_serializes(panel):
    payload = json.loads(explain(jf.risk_parity(panel)).to_json())
    assert payload["available"] is False
    assert payload["constraints"] == []
    assert "risk_contributions" in payload["reason"]
    assert len(payload["assets"]) == len(panel.columns)


def test_weights_are_unchanged_by_capturing_attribution(panel, names):
    """Computing the diagnostics must not perturb the solution by a single bit."""
    cons = [jf.GroupCap("tech", names[:4], max=TECH_CAP)]
    on = jf.minimum_variance(panel, jf.OptimizerConfig(constraints=cons, attribution=True))
    off = jf.minimum_variance(panel, jf.OptimizerConfig(constraints=cons, attribution=False))
    assert np.array_equal(on.weights, off.weights), (
        f"max|delta|={np.abs(on.weights - off.weights).max():.3g}"
    )
    assert on.metadata["iterations"] == off.metadata["iterations"]
