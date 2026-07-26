"""Explain *why* an optimizer produced the weights it did.

A convex optimizer is normally a black box: parameters in, weights out. When an
asset comes back at 0% there is nothing in the result that says whether it was
excluded on its own merits or crowded out by a constraint somewhere else. This
module answers that question, using the Lagrange multipliers the projection
already computes.

The recovery, in one identity
-----------------------------
Every solver here minimizes ``f`` over a convex set ``C`` by projected gradient,
so at a solution ``w*`` with ``g = grad f(w*)``, stationarity says ``w*`` is a
fixed point: ``w* == P_C(w* - s g)`` for any small step ``s > 0``. Projecting
``v = w* - s g`` and keeping the projection's *own* multipliers therefore hands
back the multipliers of the portfolio problem, divided by ``s``::

    lambda = lambda_proj / s          on the budget
    theta_k = theta_proj[k] / s       on named row k
    rc_i    = -beta_proj[i] / s       reduced cost of asset i

No second solve, no least-squares fit, no factorization — one extra gradient and
two extra projections, all outside ``lax.while_loop`` and so fully jit- and
vmap-safe.

The pricing decomposition
-------------------------
The reduced cost splits into exactly the terms a portfolio manager would name::

    rc_i = intrinsic_i + imposed_i     where
    intrinsic_i = g_i + lambda         the asset on its own merits, net of the
                                       cost of capital
    imposed_i   = theta_{k(i)}         what its group's constraint charges it

At an active bound, ``rc_i != 0`` and the question "what put this asset here?" is
answered by whichever *term* is largest. **Ranking the terms, not the
multipliers, is the whole trick.** The naive rule — take the largest multiplier
among the constraints touching asset ``i`` — is wrong in a way that looks right:
at ``w_i = 0`` the box multiplier equals the entire reduced cost, so it dominates
every group term by construction and ``box_lower`` would win every single time.
"binding: tech" could never fire. This is the textbook LP column-pricing identity
("reduced cost = own cost + the shadow price of every resource consumed"), and it
degrades correctly: with no groups, every zero attributes to ``own``, which is
the truthful answer.

Honesty
-------
Multipliers are only meaningful at a converged, non-degenerate solution, and the
number of ways that can fail is not small. Every one of them is checked and every
report carries a ``quality`` grade; when a number cannot be stood behind, it is
``None`` rather than a plausible-looking guess. See :class:`ConstraintReport` for
the full list, and the guide for what the report will refuse to claim.

Units and signs
---------------
Multipliers are in the units of the objective, **per period** — they are *not*
annualized, unlike ``expected_return`` / ``volatility`` / ``sharpe`` on the
result. Annualizing would be wrong in different ways for different objectives
(linear in the covariance for variance, square-root for Sharpe), so the raw value
is reported and the field says so.

``shadow_price`` is signed for the objective as the *user* names it: positive
means relaxing the constraint improves that objective. Internally every
multiplier is stored non-negative for a binding cap, and only the presentation
flips on ``sense``. The invariant that must never break: relaxing a constraint
can only ever weakly improve the optimum.
"""

from __future__ import annotations

import json
import textwrap
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from jaxfolio.constraints.compile import CompiledConstraints
    from jaxfolio.types import PortfolioResult

__all__ = [
    "AssetAttribution",
    "ConstraintDual",
    "ConstraintReport",
    "SolverDuals",
    "explain",
    "solver_duals",
]

# Quality thresholds on the *relative* stationarity residual ||d|| / ||grad f||.
# Relative, so they survive rescaling the covariance: an absolute threshold would
# grade a variance objective (gradients ~1e-4) and a Sharpe objective
# (gradients ~1) on wildly different effective standards.
#
# Note these are a *floor* on the grade, not the whole story. A relative measure
# alone cannot award "exact": the solver stops at an absolute ``tol`` on the
# projected-gradient norm, so the best achievable relative residual is
# ``tol / ||grad f||`` — for min-variance (gradients ~1e-4 at tol=1e-7) that is
# ~1e-3, and no relative threshold tight enough to mean "exact" would ever be
# reachable. So a solve that met the tolerance it was *given* is graded exact, and
# these thresholds catch solves that did not.
_UNRELIABLE = 1e-2
_APPROXIMATE = 1e-4
#: How many multiples of the requested ``tol`` still count as "converged as asked".
_TOL_SLACK = 10.0

#: Displacement used to probe the projection, as a fraction of the largest weight.
#: Small enough to stay inside the region where the projection's active set — and
#: therefore ``d`` — is constant in the step (``C`` is polyhedral, so that region
#: is a genuine interval), large enough that dividing the multipliers by it does
#: not amplify float32 noise. Verified by ``step_check``, which recomputes at a
#: tenth of it and reports the disagreement.
_PROBE_FRAC = 1e-2


# --------------------------------------------------------------------------- #
# Solver-side payload
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SolverDuals:
    """First-order information captured at a solved portfolio.

    Attached to :attr:`jaxfolio.types.PortfolioResult.attribution`. Deliberately
    numeric and self-contained — plain numpy arrays and tuples, no JAX arrays, no
    closures, no reference to the returns panel — so a result stays lightweight
    and picklable, and so :func:`explain` is pure post-processing that never
    re-solves.

    Attributes
    ----------
    weights, gradient:
        The solution and the objective gradient there.
    lower, upper:
        Effective per-asset bounds in force.
    budget, row_names, row_lower, row_upper, group_of:
        The constraint set, as resolved by
        :func:`jaxfolio.constraints.compile_constraints`.
    budget_multiplier, row_multipliers, reduced_costs:
        The recovered multipliers, in objective units per period. Non-negative for
        a binding cap; see the module docstring for the sign convention.
    row_identified, budget_identified:
        Whether each multiplier is pinned by the data at all. See
        :class:`jaxfolio.constraints.structured.ProjectionDuals`.
    kkt_residual, gradient_norm, step, step_check:
        Convergence evidence. ``kkt_residual`` is ``||d||`` where ``d`` is the
        feasible-descent part of the Moreau split; it is a genuine KKT measure for
        *every* solver family, including the optax ones whose own ``residual`` is
        only a weight-update norm. ``step_check`` re-probes at a tenth of ``step``
        and reports the disagreement relative to ``||grad f||``; a large value means
        the probe left the region where the projection's active set is constant, so
        the multipliers were read off the wrong face.
    aux_stationarity:
        For packed ``[w, tau]`` problems (CVaR), ``|d f / d tau|`` at the solution
        — the auxiliary variable is unconstrained, so its whole gradient is
        "descent the constraints could not absorb".
    tol:
        The convergence tolerance the solve was asked for. The quality grade is
        judged against it: "did this solve do what it was told to?" is a fairer and
        more portable question than any fixed absolute threshold.
    objective, sense, smooth, convex:
        Provenance the report needs to know what it may claim. ``sense`` is
        ``"minimize"`` or ``"maximize"`` for the objective **as the user names
        it**, which sets the reported sign of every shadow price.
    """

    weights: np.ndarray
    gradient: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    budget: float
    row_names: tuple[str, ...]
    row_lower: tuple[float, ...]
    row_upper: tuple[float, ...]
    group_of: tuple[int, ...]
    budget_multiplier: float
    row_multipliers: tuple[float, ...]
    reduced_costs: np.ndarray
    row_identified: tuple[bool, ...]
    budget_identified: bool
    kkt_residual: float
    gradient_norm: float
    step: float
    step_check: float
    aux_stationarity: float | None = None
    objective: str = ""
    sense: str = "minimize"
    smooth: bool = True
    convex: bool = True
    l2_reg: float = 0.0
    solver_kind: str = "spg"
    tol: float = 1e-7

    @property
    def stationarity(self) -> float:
        """``||d|| / ||grad f||`` — the scale-free measure the quality gate uses."""
        return float(self.kkt_residual / (self.gradient_norm + 1e-30))

    def summary(self) -> dict[str, Any]:
        """A small JSON-serializable digest for ``PortfolioResult.metadata``."""
        binding = [
            name
            for name, mult, ident in zip(
                self.row_names, self.row_multipliers, self.row_identified, strict=True
            )
            if ident and mult != 0.0
        ]
        return {
            "kkt_residual": float(self.kkt_residual),
            "stationarity": self.stationarity,
            "dual_quality": _grade(self),
            "n_binding_constraints": len(binding),
            "binding_constraints": binding,
        }


def _grade(duals: SolverDuals) -> str:
    """Grade how far the reported multipliers can be trusted.

    Two independent ways to fail, so two tests. A **non-smooth** objective is
    hopeless regardless of how tightly it converged: at a kink ``jax.grad`` returns
    one arbitrary subgradient, so stationarity need not hold even at the exact
    optimum. Otherwise the question is convergence, judged against the tolerance
    the caller actually requested (an absolute test) with a scale-free relative
    residual as a backstop for solvers whose ``tol`` means something else.
    """
    if not duals.smooth:
        return "unreliable"
    if duals.stationarity >= _UNRELIABLE:
        return "unreliable"
    if duals.step_check > _UNRELIABLE:
        return "approximate"
    met_tolerance = duals.kkt_residual <= _TOL_SLACK * duals.tol
    return "exact" if (met_tolerance or duals.stationarity < _APPROXIMATE) else "approximate"


def solver_duals(
    objective,
    obj_params,
    compiled: CompiledConstraints,
    weights,
    *,
    objective_name: str,
    sense: str = "minimize",
    smooth: bool = True,
    convex: bool = True,
    l2_reg: float = 0.0,
    solver_kind: str = "spg",
    tol: float = 1e-7,
    aux_dim: int = 0,
) -> SolverDuals:
    """Recover the KKT multipliers of a solved portfolio problem.

    Runs *after* the solve, outside any ``lax`` control flow, at the cost of one
    gradient evaluation and two projections. Solver-agnostic by construction,
    which is what lets it retrofit a real stationarity measure onto the optax
    solvers (whose own ``info["residual"]`` is a weight-update norm, explicitly
    not a KKT test).

    ``objective(z, obj_params)`` is the module-level objective that was minimized;
    ``weights`` is the returned iterate. For a packed ``[w, tau]`` variable pass
    ``aux_dim=1`` and the full packed vector — the projection's normal cone
    factorizes as ``N_C(w) x {0}``, so the weight-block multipliers are unaffected
    and ``tau``'s stationarity comes back separately.
    """
    import jax
    import jax.numpy as jnp

    from jaxfolio.constraints.structured import (
        project_box_budget_duals,
        project_grouped_duals,
    )

    z = jnp.asarray(weights)
    grad = jax.grad(lambda x: objective(x, obj_params))(z)
    w = z[:-aux_dim] if aux_dim else z
    g = grad[:-aux_dim] if aux_dim else grad

    lower = jnp.asarray(compiled.lower)
    upper = jnp.asarray(compiled.upper)
    budget = jnp.asarray(compiled.budget)
    if compiled.kind == "grouped":
        gid = jnp.asarray(compiled.group_of, dtype=jnp.int32)
        g_lo = jnp.asarray((*compiled.row_lower, 0.0))
        g_hi = jnp.asarray((*compiled.row_upper, 0.0))

        def project(v):
            return project_grouped_duals(v, lower, upper, budget, gid, g_lo, g_hi)
    else:
        # Every non-grouped kind — including the simplex, whose feasible set is
        # exactly the box [0, budget] intersected with the budget hyperplane — is
        # served by the box kernel, so the caller sees one uniform payload.
        def project(v):
            return project_box_budget_duals(v, lower, upper, budget)

    # Probe at a displacement scaled to the problem, then again ten times smaller.
    # C is polyhedral, so d_s is constant for all small enough s; disagreement
    # between the two means the larger probe left that region and the multipliers
    # are being read off the wrong face.
    scale = jnp.maximum(jnp.max(jnp.abs(w)), 1e-8)
    gnorm_inf = jnp.maximum(jnp.max(jnp.abs(g)), 1e-30)
    step = _PROBE_FRAC * scale / gnorm_inf

    def descent(s):
        return (project(w - s * g).weights - w) / s

    d_main = descent(step)
    d_fine = descent(step * 0.1)
    # Normalize by the gradient, NOT by ||d_fine||. Both descents have the units of
    # g, and at a converged solution both are essentially zero — dividing by
    # ||d_fine|| there divides noise by noise and reports a meaningless 1.0. Against
    # ||g|| the measure stays interpretable at both ends: ~0 when the two probes
    # agree (including when both are at the float floor), and O(1) when the larger
    # step has left the region where the active set is constant.
    gnorm = jnp.linalg.norm(g)
    step_check = jnp.linalg.norm(d_main - d_fine) / (gnorm + 1e-30)

    duals = project(w - step * g)
    n_rows = compiled.n_rows
    rows = np.asarray(duals.rows)[:n_rows] / float(step) if n_rows else np.zeros(0)
    # rc_i = g_i + lambda + theta_{k(i)} = -beta_i / step. See the module docstring;
    # the sign flip is why `bounds` is negative at a lower bound while a reduced
    # cost is positive there.
    reduced = -np.asarray(duals.bounds) / float(step)

    return SolverDuals(
        weights=np.asarray(w, dtype=float),
        gradient=np.asarray(g, dtype=float),
        lower=np.asarray(compiled.lower, dtype=float),
        upper=np.asarray(compiled.upper, dtype=float),
        budget=float(compiled.budget),
        row_names=tuple(compiled.row_names),
        row_lower=tuple(compiled.row_lower),
        row_upper=tuple(compiled.row_upper),
        group_of=tuple(compiled.group_of),
        budget_multiplier=float(duals.budget) / float(step),
        row_multipliers=tuple(float(x) for x in rows),
        reduced_costs=reduced,
        row_identified=tuple(bool(x) for x in np.asarray(duals.identified)[:n_rows]),
        budget_identified=bool(duals.budget_identified),
        kkt_residual=float(jnp.linalg.norm(d_main)),
        gradient_norm=float(gnorm),
        step=float(step),
        step_check=float(step_check),
        aux_stationarity=float(jnp.abs(grad[-1])) if aux_dim else None,
        objective=objective_name,
        sense=sense,
        smooth=smooth,
        convex=convex,
        l2_reg=float(l2_reg),
        solver_kind=solver_kind,
        tol=float(tol),
    )


# --------------------------------------------------------------------------- #
# Rendering helpers
#
# Deliberately ASCII: a printed report ends up in a log file, a CI artifact or a
# terminal on any platform, and "<=" survives an encoding that "≤" does not.
# The prose rendering (:meth:`ConstraintReport.explain_text`) keeps its typography.
# --------------------------------------------------------------------------- #
#: What a table cell shows when the report declines to quote a number.
_MISSING = "n/a"

#: Typography this module's own prose uses, and its ASCII equivalent. The reasons and
#: warnings are written for the guide, so they carry em dashes and math symbols; the
#: table has to render them on a console that may be ASCII-only. Applied to strings
#: *jaxfolio itself* wrote — asset and constraint names are user data and pass through
#: verbatim, because mangling a ticker to fit a charset would be worse than a wide byte.
_ASCII_SUBS = {
    "—": "--",  # em dash
    "–": "-",  # en dash
    "·": "-",  # middle dot
    "≤": "<=",
    "≥": ">=",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "…": "...",
    "→": "->",
    "×": "x",
    "≈": "~",
    "≠": "!=",
    "∑": "sum",
    "∇": "grad ",
    "‖": "||",
    "λ": "lambda",
    "θ": "theta",
    "β": "beta",
    "Σ": "Sigma",
    "τ": "tau",
}


def _ascii(text: str) -> str:
    """Transliterate jaxfolio's own prose to ASCII, losing nothing a reader needs.

    Known typography maps to its obvious equivalent; anything unforeseen is decomposed
    (so an accent is dropped rather than the character lost) and only then replaced, so
    this can never raise — a report that fails to render is worse than one that spells
    a dash as two hyphens.
    """
    for char, replacement in _ASCII_SUBS.items():
        text = text.replace(char, replacement)
    if text.isascii():
        return text
    stripped = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return stripped if stripped.strip() else text.encode("ascii", "replace").decode()


def _json_float(value: object) -> float | None:
    """A float JSON can hold, or ``None`` — never a numpy scalar, NaN or infinity."""
    if value is None:
        return None
    out = float(value)  # type: ignore[arg-type]
    return out if np.isfinite(out) else None


def _cell_num(value: float | None, spec: str = "+.3e") -> str:
    if value is None:
        return _MISSING
    out = float(value)
    return format(0.0 if out == 0.0 else out, spec)  # never "-0.000e+00"


def _cell_pct(value: float | None, spec: str = ".2%") -> str:
    """A percentage cell. Values below the displayed precision print as a clean zero:
    a slack of -2.7e-08 is zero at two decimals, and "-0.00%" only reads as a bug."""
    if value is None:
        return _MISSING
    out = float(value)
    return format(0.0 if abs(out) < 5e-7 else out, spec)


def _cell_bound(lower: float, upper: float, budget: float = 1.0) -> str:
    """A row's own limits as one cell: a cap, a floor, or a band.

    A cap at or above the budget, and a floor at zero, cannot bind — they are not
    shown, so the column only ever names a limit that could do something.
    """
    floor = lower if lower > 0.0 else None
    cap = upper if upper < budget else None
    if floor is not None and cap is not None:
        return f"{floor:.2%} to {cap:.2%}"
    if cap is not None:
        return f"<= {cap:.2%}"
    if floor is not None:
        return f">= {floor:.2%}"
    return "unbounded"


def _render_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    aligns: str,
    *,
    indent: str = "",
    gap: str = "  ",
) -> list[str]:
    """Fixed-width table as a list of lines: header, hairline rule, then the rows.

    ``aligns`` is one character per column, ``"<"`` or ``">"``. Widths come from the
    content, so no column is ever truncated — a report that clips an asset name is
    worse than one that is a few characters wider than expected.
    """
    widths = [
        max(len(str(header)), *(len(str(row[i])) for row in rows)) if rows else len(str(header))
        for i, header in enumerate(headers)
    ]

    def line(cells: Sequence[str]) -> str:
        parts = (format(str(c), f"{a}{w}") for c, a, w in zip(cells, aligns, widths, strict=True))
        return (indent + gap.join(parts)).rstrip()

    return [
        line(headers),
        indent + gap.join("-" * w for w in widths),
        *(line(row) for row in rows),
    ]


def _wrap(text: str, *, bullet: str = "", width: int = 78) -> list[str]:
    """ASCII-fold a sentence and wrap it to ``width``, hanging-indented under its bullet.

    The tables set their own width from their content; the prose around them is what
    would otherwise run to 150 columns and make the report unreadable in a terminal.
    Folding happens before wrapping, since ``--`` is a character wider than ``—``.
    """
    return textwrap.wrap(
        _ascii(text),
        width=width,
        initial_indent=bullet,
        subsequent_indent=" " * len(bullet),
    ) or [bullet.rstrip()]


def _cause_cell(detail: AssetAttribution) -> str:
    """The "cause" column: the row responsible, the asset's own merits, or nothing."""
    if detail.primary is None:
        return "ambiguous" if detail.confidence == "ambiguous" else _MISSING
    if detail.primary == "own":
        return "own merits"
    return f"{detail.primary} ({detail.primary_kind})"


def _render_fields(pairs: Sequence[tuple[str, str]], *, indent: str = "") -> list[str]:
    """A key/value block with the values aligned into a column."""
    width = max((len(key) for key, _value in pairs), default=0)
    return [f"{indent}{key:<{width}}  {value}".rstrip() for key, value in pairs]


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConstraintDual:
    """One constraint, its activity, and what it costs.

    ``status`` is one of:

    ``binding``
        Active *and* carrying a non-zero multiplier: relaxing it will move the
        optimum.
    ``weakly_active``
        Active but with a zero multiplier — a degenerate vertex. Relaxing it will
        **not** move anything. Never reported as a cause.
    ``inactive``
        Slack.
    ``unidentified``
        Active, but the multiplier is not pinned by the data: every member sits on
        a bound, so any split of the shadow price between this row and those bounds
        is an equally valid KKT certificate. ``shadow_price`` is ``None``.
    """

    name: str
    kind: str
    activity: float
    lower: float
    upper: float
    slack: float
    multiplier: float | None
    shadow_price: float | None
    status: str
    members: tuple[str, ...] = ()


@dataclass(frozen=True)
class AssetAttribution:
    """Why one asset holds the weight it does.

    ``primary`` is ``"own"`` when the asset is there (or absent) on its own
    merits, a constraint name when a constraint put it there, or ``None`` when the
    answer is genuinely ambiguous — see ``confidence``.
    """

    asset: str
    weight: float
    status: str
    reduced_cost: float | None
    intrinsic_cost: float | None
    imposed: tuple[tuple[str, float], ...]
    primary: str | None
    primary_kind: str | None
    primary_shadow_price: float | None
    confidence: str
    reason: str | None = None


@dataclass(frozen=True)
class ConstraintReport:
    """The result of :func:`explain` — per-constraint and per-asset attribution.

    ``quality`` grades everything else in the report:

    ``exact``
        Converged to machine precision; multipliers are trustworthy.
    ``approximate``
        Converged loosely, or the step probe disagreed. Numbers are shown with a
        warning attached.
    ``unreliable``
        Not converged, or the objective is non-smooth at the optimum. **Shadow
        prices are suppressed entirely** — only primal facts are reported.
    ``unavailable``
        This optimizer has no multipliers at all; ``reason`` says why.

    Four renderings, for four audiences: :meth:`explain_text` reads the report as
    prose, :meth:`to_table` lays the same facts out as fixed-width tables for a
    terminal or a log, :meth:`to_dict` / :meth:`to_json` serialize it whole for a
    machine, and :meth:`to_frame` / :meth:`constraints_frame` hand it to pandas.
    All five are pure views over the same fields — none of them recompute anything.
    """

    method: str
    objective: str
    sense: str
    assets: tuple[str, ...]
    weights: np.ndarray
    available: bool
    quality: str
    reason: str | None
    stationarity: float | None
    kkt_residual: float | None
    step_check: float | None
    budget_multiplier: float | None
    budget_shadow_price: float | None
    budget_note: str | None
    constraints: tuple[ConstraintDual, ...]
    assets_detail: tuple[AssetAttribution, ...]
    warnings: tuple[str, ...] = ()
    aux_stationarity: float | None = None
    l2_reg: float = 0.0
    _index: dict[str, int] = field(default_factory=dict, repr=False, compare=False)

    def binding(self) -> tuple[ConstraintDual, ...]:
        """Constraints that are active *and* moving the optimum."""
        return tuple(c for c in self.constraints if c.status == "binding")

    def for_asset(self, asset: str) -> AssetAttribution:
        """Attribution for one asset by name."""
        try:
            return self.assets_detail[self._index[asset]]
        except KeyError:
            raise KeyError(f"{asset!r} is not in this portfolio") from None

    def to_frame(self) -> pd.DataFrame:
        """Per-asset attribution as a pandas frame, indexed by asset."""
        import pandas as pd

        rows = []
        for a in self.assets_detail:
            rows.append(
                {
                    "asset": a.asset,
                    "weight": a.weight,
                    "status": a.status,
                    "reduced_cost": np.nan if a.reduced_cost is None else a.reduced_cost,
                    "intrinsic_cost": np.nan if a.intrinsic_cost is None else a.intrinsic_cost,
                    "imposed_total": float(sum(v for _, v in a.imposed)),
                    "binding": a.primary,
                    "binding_kind": a.primary_kind,
                    "shadow_price": (
                        np.nan if a.primary_shadow_price is None else a.primary_shadow_price
                    ),
                    "confidence": a.confidence,
                }
            )
        return pd.DataFrame(rows).set_index("asset")

    def constraints_frame(self) -> pd.DataFrame:
        """Per-constraint activity and shadow price, indexed by name.

        A different question from :meth:`to_frame`: which *constraint* is costing
        the portfolio the most, rather than what put each asset where it is.
        """
        import pandas as pd

        rows = [
            {
                "name": c.name,
                "kind": c.kind,
                "activity": c.activity,
                "lower": c.lower,
                "upper": c.upper,
                "slack": c.slack,
                "multiplier": np.nan if c.multiplier is None else c.multiplier,
                "shadow_price": np.nan if c.shadow_price is None else c.shadow_price,
                "status": c.status,
                "n_members": len(c.members),
            }
            for c in self.constraints
        ]
        return pd.DataFrame(rows).set_index("name")

    def explain_text(self, *, top_n: int = 10) -> str:
        """A readable rendering of the report."""
        out = [f"Constraint attribution — {self.method} ({len(self.assets)} assets)"]
        if not self.available:
            out.append(f"  unavailable: {self.reason}")
            return "\n".join(out)
        line = f"  quality: {self.quality}"
        if self.stationarity is not None:
            line += f"  ·  stationarity {self.stationarity:.2e}"
        out.append(line)
        if self.budget_shadow_price is not None:
            out.append(f"  cost of capital (budget): {self.budget_shadow_price:+.5f} per unit")
        elif self.budget_note:
            out.append(f"  cost of capital (budget): n/a — {self.budget_note}")
        for w in self.warnings:
            out.append(f"  ! {w}")

        ranked = sorted(self.assets_detail, key=lambda a: (a.status == "interior", -abs(a.weight)))
        for a in ranked[:top_n]:
            out.append("")
            tag = "" if a.status == "interior" else f"   [{a.status.replace('_', ' ')}]"
            out.append(f"Asset {a.asset}: {a.weight:.2%}{tag}")
            if a.reason:
                out.append(f"  {a.reason}")
                continue
            if a.primary is None:
                out.append("  cause: ambiguous — several constraints are tied")
            elif a.primary == "own":
                out.append(f"  cause: own merits ({a.primary_kind})")
            else:
                sp = a.primary_shadow_price
                shown = "n/a" if sp is None else f"{sp:+.5f}"
                out.append(f"  binding: {a.primary} ({a.primary_kind})   shadow price {shown}")
            if a.intrinsic_cost is not None and a.imposed:
                charged = "  ·  ".join(f"charged by {n} {v:+.5f}" for n, v in a.imposed)
                out.append(
                    f"  own merit {a.intrinsic_cost:+.5f}  ·  {charged}"
                    f"  ·  net {a.reduced_cost:+.5f}"
                )
        return "\n".join(out)

    # ----------------------------------------------------------------------- #
    # Printable report
    # ----------------------------------------------------------------------- #
    def to_table(self, *, max_assets: int | None = None, notes: bool = True) -> str:
        """The report as fixed-width tables — for a terminal, a log, or a print-out.

        Same facts as :meth:`explain_text`, laid out as columns instead of prose: a
        header block, one row per named constraint, one row per asset, and the notes
        that qualify them. Plain text, and every string jaxfolio itself wrote is folded
        to ASCII, so the report survives a log file, a CI artifact and a non-UTF-8
        console. Asset and constraint *names* are yours and pass through verbatim — a
        report that mangles a ticker to fit a charset would be the worse trade.

        Parameters
        ----------
        max_assets:
            Show only the first ``max_assets`` asset rows, ordered so that the assets
            sitting on a bound — the ones a constraint may be responsible for — come
            first. ``None`` (the default) prints every asset: a report that silently
            drops rows is worse than a long one.
        notes:
            Include the trailing notes block (units caveat, quality reason, warnings).

        Examples
        --------
        >>> print(jf.explain(result).to_table())  # doctest: +SKIP
        """
        rule = "=" * 78
        out = [rule, f"CONSTRAINT ATTRIBUTION -- {_ascii(self.method)}", rule]

        fields = [("assets", str(len(self.assets)))]
        if self.objective:
            fields.append(("objective", f"{_ascii(self.objective)} ({self.sense})"))
        fields.append(("dual quality", self.quality))
        if self.stationarity is not None:
            fields.append(("stationarity", f"{self.stationarity:.2e}"))
        if self.kkt_residual is not None:
            fields.append(("KKT residual", f"{self.kkt_residual:.2e}"))
        if self.aux_stationarity is not None:
            fields.append(("aux stationarity", f"{self.aux_stationarity:.2e}"))
        extra_notes = []
        if self.budget_shadow_price is not None:
            fields.append(("cost of capital", f"{self.budget_shadow_price:+.3e} per unit"))
        elif self.budget_note:
            # The note is a sentence, not a value — it goes to NOTES where it can wrap,
            # rather than pushing the header block out to 150 columns.
            fields.append(("cost of capital", f"{_MISSING} (see notes)"))
            extra_notes.append(f"cost of capital: {self.budget_note}")
        if self.available:
            binding = [c.name for c in self.binding()]
            summary = f"{len(binding)} of {len(self.constraints)}"
            fields.append(
                ("binding rows", f"{summary}: {', '.join(binding)}" if binding else summary)
            )
        out += _render_fields(fields)

        if not self.available:
            out += ["", *_wrap(f"unavailable: {self.reason}")]

        if self.constraints:
            out += ["", "CONSTRAINTS"]
            out += _render_table(
                [
                    "name",
                    "kind",
                    "activity",
                    "bound",
                    "slack",
                    "multiplier",
                    "shadow price",
                    "status",
                ],
                [
                    [
                        c.name,
                        c.kind,
                        _cell_pct(c.activity),
                        _cell_bound(c.lower, c.upper),
                        _cell_pct(c.slack),
                        _cell_num(c.multiplier),
                        _cell_num(c.shadow_price),
                        c.status,
                    ]
                    for c in self.constraints
                ],
                "<<>><>><",
            )

        ranked = sorted(self.assets_detail, key=lambda a: (a.status == "interior", -abs(a.weight)))
        shown = ranked if max_assets is None else ranked[:max_assets]
        if shown:
            out += ["", "ASSETS"]
            out += _render_table(
                ["asset", "weight", "status", "own merit", "imposed", "net", "cause", "confidence"],
                [
                    [
                        a.asset,
                        _cell_pct(a.weight),
                        a.status.replace("_", " "),
                        _cell_num(a.intrinsic_cost),
                        _cell_num(sum(v for _name, v in a.imposed)) if a.imposed else _MISSING,
                        _cell_num(a.reduced_cost),
                        _cause_cell(a),
                        a.confidence,
                    ]
                    for a in shown
                ],
                "<><>>><<",
            )
            if len(shown) < len(ranked):
                out.append(f"({len(ranked) - len(shown)} further asset(s) not shown)")

        if notes:
            lines = ["multipliers are per period, in the objective's own units (not annualized)"]
            lines += extra_notes
            if self.reason and self.available:  # an unavailable reason is already printed above
                lines.append(self.reason)
            # No l2_reg note of our own: `warnings` already carries that one, in more
            # detail. Two phrasings of one caveat read as two caveats.
            lines += list(self.warnings)
            out += ["", "NOTES"]
            for line in lines:
                out += _wrap(line, bullet="- ")

        return "\n".join(out)

    # ----------------------------------------------------------------------- #
    # Serialized report
    # ----------------------------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        """The whole report as a JSON-serializable dict.

        Nothing is dropped and nothing is rounded: every constraint, every asset and
        every diagnostic, with plain Python scalars throughout (no numpy types). A
        number the report declines to stand behind stays ``None`` rather than
        becoming ``0.0`` — the distinction is the point of the report, so it must
        survive serialization. ``schema`` is there so a consumer can tell versions
        apart if this grows.

        The digest on ``result.metadata`` is the small, always-present cousin of
        this: five keys for a backtest log, where this is the whole picture.
        """
        return {
            "schema": "jaxfolio.constraint_report/1",
            "method": self.method,
            "objective": self.objective or None,
            "sense": self.sense,
            "available": self.available,
            "quality": self.quality,
            "reason": self.reason,
            "diagnostics": {
                "kkt_residual": _json_float(self.kkt_residual),
                "stationarity": _json_float(self.stationarity),
                "step_check": _json_float(self.step_check),
                "aux_stationarity": _json_float(self.aux_stationarity),
                "l2_reg": _json_float(self.l2_reg),
            },
            "budget": {
                "multiplier": _json_float(self.budget_multiplier),
                "shadow_price": _json_float(self.budget_shadow_price),
                "note": self.budget_note,
            },
            "constraints": [
                {
                    "name": c.name,
                    "kind": c.kind,
                    "activity": _json_float(c.activity),
                    "lower": _json_float(c.lower),
                    "upper": _json_float(c.upper),
                    "slack": _json_float(c.slack),
                    "multiplier": _json_float(c.multiplier),
                    "shadow_price": _json_float(c.shadow_price),
                    "status": c.status,
                    "members": list(c.members),
                }
                for c in self.constraints
            ],
            "assets": [
                {
                    "asset": a.asset,
                    "weight": _json_float(a.weight),
                    "status": a.status,
                    "reduced_cost": _json_float(a.reduced_cost),
                    "intrinsic_cost": _json_float(a.intrinsic_cost),
                    "imposed": [
                        {"constraint": name, "charge": _json_float(value)}
                        for name, value in a.imposed
                    ],
                    "cause": a.primary,
                    "cause_kind": a.primary_kind,
                    "cause_shadow_price": _json_float(a.primary_shadow_price),
                    "confidence": a.confidence,
                    "reason": a.reason,
                }
                for a in self.assets_detail
            ],
            "warnings": list(self.warnings),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """The report as a JSON string. ``indent=None`` for a single compact line."""
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        n = len(self.binding()) if self.available else 0
        return (
            f"ConstraintReport(method={self.method!r}, quality={self.quality!r}, "
            f"binding={n}, assets={len(self.assets)})"
        )


_NO_DUALS_REASONS = {
    "Equal Weight": (
        "equal_weight assigns 1/N analytically; no optimization program is solved, so "
        "no constraint is ever active"
    ),
    "Inverse Volatility": (
        "inverse_volatility is an analytic weighting; no constraints were optimized against"
    ),
    "Risk Parity (ERC)": (
        "risk_parity solves a log-barrier program (min 0.5 x'Sigma x - (1/N) sum log x); the "
        "barrier keeps every x strictly positive so no bound is ever active, and the budget is "
        "imposed by normalizing the fixed point rather than as a constraint. For per-asset "
        'attribution on this method see metadata["risk_contributions"]'
    ),
}


def explain(
    result: PortfolioResult,
    *,
    prim_tol: float | None = None,
    dual_tol: float | None = None,
) -> ConstraintReport:
    """Explain a solved portfolio: which constraints bind, and what they cost.

    Pure post-processing — reads only the payload captured at solve time and never
    re-solves. Always returns a well-formed report: optimizers that have no
    multipliers (closed-form weightings, the graph methods, the learning policies)
    come back with ``available=False`` and a specific ``reason``, with the primal
    facts still filled in.

    Parameters
    ----------
    prim_tol:
        How close to a bound counts as *on* it. Defaults to ``1e-6`` scaled by the
        bound magnitudes.
    dual_tol:
        How large a multiplier counts as non-zero. Defaults to ``1e-6`` scaled by
        ``||grad f||``, so it survives rescaling the objective.
    """
    duals: SolverDuals | None = getattr(result, "attribution", None)
    assets = tuple(result.assets)
    weights = np.asarray(result.weights, dtype=float)
    index = {a: i for i, a in enumerate(assets)}

    if duals is None:
        reason = _NO_DUALS_REASONS.get(
            result.method,
            f"{result.method} does not emit solver diagnostics — it is feasible by "
            "construction rather than optimal subject to constraints, so a zero weight "
            "is caused by the construction, not by a constraint",
        )
        return ConstraintReport(
            method=result.method,
            objective="",
            sense="minimize",
            assets=assets,
            weights=weights,
            available=False,
            quality="unavailable",
            reason=reason,
            stationarity=None,
            kkt_residual=None,
            step_check=None,
            budget_multiplier=None,
            budget_shadow_price=None,
            budget_note=None,
            constraints=(),
            assets_detail=tuple(
                AssetAttribution(
                    asset=a,
                    weight=float(weights[i]),
                    status="unknown",
                    reduced_cost=None,
                    intrinsic_cost=None,
                    imposed=(),
                    primary=None,
                    primary_kind=None,
                    primary_shadow_price=None,
                    confidence="unavailable",
                    reason=reason,
                )
                for i, a in enumerate(assets)
            ),
            _index=index,
        )

    quality = _grade(duals)
    sigma = 1.0 if duals.sense == "maximize" else -1.0
    trust = quality in ("exact", "approximate")

    bound_scale = max(float(np.max(np.abs(duals.upper))), float(np.max(np.abs(duals.lower))), 1e-12)
    ptol = prim_tol if prim_tol is not None else 1e-6 * bound_scale
    dtol = dual_tol if dual_tol is not None else 1e-6 * max(duals.gradient_norm, 1e-12)

    warnings = _collect_warnings(duals, quality)
    constraints = _row_duals(duals, assets, sigma, ptol, dtol, trust)
    detail = _asset_attributions(duals, assets, constraints, ptol, dtol, trust)

    budget_note = None
    budget_sp: float | None = None
    budget_mult: float | None = None
    if not trust:
        budget_note = "suppressed: " + quality
    elif not duals.budget_identified:
        budget_note = (
            "not identified — every interior asset lies inside a binding row, so "
            "stationarity pins only the sum of the budget and row multipliers"
        )
    elif _is_scale_invariant(duals):
        budget_note = (
            "not meaningful: the objective is scale-invariant (degree-0 homogeneous), "
            "so the optimum is unchanged by a change of budget"
        )
    else:
        budget_mult = duals.budget_multiplier
        budget_sp = sigma * duals.budget_multiplier

    return ConstraintReport(
        method=result.method,
        objective=duals.objective,
        sense=duals.sense,
        assets=assets,
        weights=weights,
        available=True,
        quality=quality,
        reason=None if trust else _quality_reason(duals, quality),
        stationarity=duals.stationarity,
        kkt_residual=duals.kkt_residual,
        step_check=duals.step_check,
        budget_multiplier=budget_mult,
        budget_shadow_price=budget_sp,
        budget_note=budget_note,
        constraints=constraints,
        assets_detail=detail,
        warnings=warnings,
        aux_stationarity=duals.aux_stationarity,
        l2_reg=duals.l2_reg,
        _index=index,
    )


def _is_scale_invariant(duals: SolverDuals) -> bool:
    """Detect a degree-0 homogeneous objective numerically, not by a name list.

    By Euler's identity such an objective has ``g'w == 0``, so a budget shadow
    price is meaningless — the optimum is unchanged up to scaling. Maximum-Sharpe
    and maximum-diversification are the cases in practice, but testing the
    property rather than the method name means a custom objective gets the same
    protection.
    """
    g, w = duals.gradient, duals.weights
    denom = float(np.linalg.norm(g) * np.linalg.norm(w)) + 1e-30
    return abs(float(g @ w)) / denom < 1e-4


def _quality_reason(duals: SolverDuals, quality: str) -> str:
    if not duals.smooth:
        return (
            f"the {duals.objective} objective is non-smooth at its optimum, so the gradient "
            "is one arbitrary subgradient and stationarity need not hold even at the exact "
            "solution; shadow prices are suppressed"
        )
    return (
        f"the solve did not converge tightly enough (KKT residual "
        f"{duals.kkt_residual:.2e} against a requested tol of {duals.tol:.1e}; relative "
        f"{duals.stationarity:.2e}); shadow prices are suppressed"
    )


def _collect_warnings(duals: SolverDuals, quality: str) -> tuple[str, ...]:
    out: list[str] = []
    if duals.l2_reg > 0:
        out.append(
            f"l2_reg={duals.l2_reg:g} is active, so multipliers price the penalized "
            f"objective ({duals.objective} + {duals.l2_reg:g}*||w||^2), and the penalty "
            "softens corners — a constraint doing economic work may read as inactive"
        )
    if not duals.convex:
        out.append(
            f"the {duals.objective} objective is not convex, so these multipliers are valid "
            "only in the neighborhood of this solution; no global claim is made"
        )
    if duals.solver_kind != "spg" and quality != "exact":
        out.append(
            f"solver {duals.solver_kind!r} reports a weight-update norm rather than a KKT "
            "measure; the stationarity shown here was computed independently"
        )
    if quality == "approximate":
        out.append("multipliers are approximate — treat magnitudes as indicative, not exact")
    return tuple(out)


def _row_duals(
    duals: SolverDuals,
    assets: tuple[str, ...],
    sigma: float,
    ptol: float,
    dtol: float,
    trust: bool,
) -> tuple[ConstraintDual, ...]:
    """Build the per-constraint view of the named rows."""
    group_of = np.asarray(duals.group_of)
    out: list[ConstraintDual] = []
    for k, name in enumerate(duals.row_names):
        members = np.nonzero(group_of == k)[0]
        activity = float(duals.weights[members].sum())
        low, high = duals.row_lower[k], duals.row_upper[k]
        mult = duals.row_multipliers[k]
        cap_active = (high - activity) <= ptol
        floor_active = (activity - low) <= ptol
        at_bound = cap_active or floor_active
        if not duals.row_identified[k] and at_bound:
            status, shown_mult, shown_sp = "unidentified", None, None
        elif not at_bound:
            status, shown_mult, shown_sp = "inactive", 0.0, 0.0
        elif abs(mult) <= dtol:
            status, shown_mult, shown_sp = "weakly_active", 0.0, 0.0
        else:
            # shadow_price = d(named objective) / d(active limit). Keep the sign of
            # the multiplier: for a binding *floor* theta is negative, and raising
            # the floor tightens the problem — abs() here would report the
            # improvement as if the floor were a cap.
            status, shown_mult, shown_sp = "binding", mult, sigma * mult
        if not trust:
            shown_mult = shown_sp = None
        out.append(
            ConstraintDual(
                name=name,
                kind="group_cap" if cap_active else "group_floor" if floor_active else "group_row",
                activity=activity,
                lower=float(low),
                upper=float(high),
                slack=float(min(high - activity, activity - low)),
                multiplier=shown_mult,
                shadow_price=shown_sp,
                status=status,
                members=tuple(assets[i] for i in members),
            )
        )
    return tuple(out)


def _asset_attributions(
    duals: SolverDuals,
    assets: tuple[str, ...],
    constraints: tuple[ConstraintDual, ...],
    ptol: float,
    dtol: float,
    trust: bool,
) -> tuple[AssetAttribution, ...]:
    group_of = np.asarray(duals.group_of)
    out: list[AssetAttribution] = []
    for i, name in enumerate(assets):
        w = float(duals.weights[i])
        at_lower = (w - duals.lower[i]) <= ptol
        at_upper = (duals.upper[i] - w) <= ptol
        if at_lower and at_upper:
            status = "pinned"  # lower == upper: no freedom at all
        elif at_lower:
            status = "at_lower"
        elif at_upper:
            status = "at_upper"
        else:
            status = "interior"

        rc = float(duals.reduced_costs[i])
        if not trust or status == "interior":
            out.append(
                AssetAttribution(
                    asset=name,
                    weight=w,
                    status=status,
                    reduced_cost=None if not trust else rc,
                    intrinsic_cost=None,
                    imposed=(),
                    primary=None if status != "interior" else "own",
                    primary_kind=None if status != "interior" else "interior",
                    primary_shadow_price=None,
                    confidence="unavailable" if not trust else "unique",
                    reason=(
                        None
                        if trust
                        else "shadow prices suppressed — see the report's quality grade"
                    ),
                )
            )
            continue

        k = int(group_of[i])
        row = constraints[k] if k < len(constraints) else None
        imposed = duals.row_multipliers[k] if k < len(duals.row_multipliers) else 0.0
        intrinsic = rc - imposed

        # Rank the *terms* of the reduced cost, not the multipliers. At a lower
        # bound the terms pushing the asset there are positive; at an upper bound
        # they are negative. See the module docstring for why ranking multipliers
        # would make the box win every time.
        orient = 1.0 if status == "at_lower" else -1.0
        terms: list[tuple[str, str, float]] = [("own", "own_merits", intrinsic)]
        if row is not None and row.status == "binding":
            terms.append((row.name, row.kind, imposed))

        best = max(terms, key=lambda t: orient * t[2])
        contenders = [t for t in terms if orient * t[2] > dtol]
        # Order matters: an unidentified row poisons this asset's attribution
        # regardless of how the terms rank, because the split of the reduced cost
        # between the row and the box is arbitrary. Checking the term ranking first
        # would confidently report "own" using an `imposed` term that is only zero
        # by the minimum-norm convention, not by fact.
        if row is not None and row.status == "unidentified":
            primary, primary_kind, confidence = None, None, "ambiguous"
        elif not contenders:
            primary, primary_kind, confidence = "own", "own_merits", "unique"
        else:
            top = orient * best[2]
            near = 1e-6 * max(abs(top), 1e-12)
            tied = [t for t in contenders if abs(orient * t[2] - top) <= near]
            if len(tied) > 1:
                primary, primary_kind, confidence = None, None, "ambiguous"
            else:
                primary, primary_kind, confidence = best[0], best[1], "unique"

        sp: float | None = None
        if primary == "own":
            sp = None
        elif primary is not None and row is not None:
            sp = row.shadow_price

        out.append(
            AssetAttribution(
                asset=name,
                weight=w,
                status=status,
                reduced_cost=rc,
                intrinsic_cost=intrinsic,
                imposed=((row.name, imposed),) if row is not None else (),
                primary=primary,
                primary_kind=primary_kind if primary else None,
                primary_shadow_price=sp,
                confidence=confidence,
                reason=None,
            )
        )
    return tuple(out)
