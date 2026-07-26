"""Compile named constraint specifications into a solver-ready projection.

:mod:`jaxfolio.constraints.spec` describes constraints in the user's vocabulary —
asset *names*, sector caps, per-name overrides. The solver needs index arrays and
a projection function. This module bridges the two, and it is also where every
error a user can actually make gets caught, with the constraint named and the gap
quantified.

Resolution happens **host-side, in numpy, once per solve** — right after
:func:`jaxfolio.results.moments` yields the asset names. Nothing here runs under
``jit``.

The jit-cache contract
----------------------
:func:`~jaxfolio.optimizers.base.solve_constrained` keys its cache on the
*identity* of the projection function plus the shapes of its traced arguments. So
the split matters:

* **Static** — the projection kind (which selects the function object) and the
  shapes ``n_assets`` and ``n_rows``.
* **Traced** — every constraint *value*: bounds, budget, row limits, and even the
  group-id vector.

Consequence: changing a cap from 30% to 32%, tightening a per-asset bound, or
reassigning which tickers belong to a sector costs **zero recompiles**. Only
changing the *number* of rows, or the asset count, adds a cache entry. That is
what makes a what-if re-solve cheap enough to be interactive.

Backwards compatibility is exact, not approximate: with no named constraints and
scalar bounds, :meth:`CompiledConstraints.projection` returns the *same function
objects and the same Python-float tuples* that
:func:`~jaxfolio.optimizers.base.select_projection` returned before this module
existed. Not "equivalent" — identical, so no existing solve gains a cache entry,
changes dtype, or moves a single bit.
"""

from __future__ import annotations

import difflib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from jaxfolio.constraints.spec import Box, Budget, Constraint, RowConstraint

__all__ = [
    "CompiledConstraints",
    "InfeasibleConstraints",
    "check_feasible",
    "compile_constraints",
]

# Projection kinds, cheapest sufficient kernel first. The kind selects a
# module-level projection function in ``jaxfolio.optimizers.base``, so it is part
# of the jit cache key.
KIND_SIMPLEX = "simplex"
KIND_BOX = "box"
KIND_BOXVEC = "boxvec"
KIND_GROUPED = "grouped"


class InfeasibleConstraints(ValueError):
    """The requested constraint set admits no feasible portfolio.

    Raised at compile time, before any moments are estimated or any solve is
    attempted. Because the supported rows have disjoint supports, the feasibility
    test is *exact* — necessary and sufficient — so this is never a false alarm,
    and conversely a set that compiles can always be satisfied.
    """


def _resolve_one(asset: str | int, index: dict[str, int], n: int, where: str) -> int:
    """Map an asset name or positional index to a column index."""
    if isinstance(asset, int):
        if not -n <= asset < n:
            raise IndexError(f"{where}: asset index {asset} is out of range for {n} assets")
        return asset % n
    try:
        return index[asset]
    except KeyError:
        suggestion = difflib.get_close_matches(asset, index, n=1)
        hint = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
        raise KeyError(
            f"{where}: unknown asset {asset!r} — it is not in the returns panel.{hint}"
        ) from None


def _broadcast(bound: float | tuple[float, ...], width: int, where: str) -> np.ndarray:
    """Expand a scalar-or-sequence bound to a float array of length ``width``."""
    if isinstance(bound, tuple):
        if len(bound) != width:
            raise ValueError(
                f"{where}: per-asset bounds have length {len(bound)} but apply to {width} assets"
            )
        return np.asarray(bound, dtype=float)
    return np.full(width, float(bound))


@dataclass(frozen=True)
class CompiledConstraints:
    """A resolved, feasible constraint set ready to hand to the solver.

    Everything is stored host-side as tuples so the object stays hashable and
    picklable; the traced JAX arrays are built on demand by :meth:`projection`.
    That also makes :meth:`perturb` a plain field replacement, which is what lets
    a what-if re-solve reuse the compiled kernel.

    Attributes
    ----------
    kind:
        Which projection kernel is needed — see the module constants. Part of the
        jit cache key.
    assets:
        Asset names, in panel order.
    budget:
        Required ``sum(w)``.
    lower, upper:
        Effective per-asset bounds, length ``n_assets``, after merging the
        implicit bounds with every :class:`~jaxfolio.constraints.spec.Box`.
    row_names:
        Names of the constraint rows, index-aligned with ``row_lower`` /
        ``row_upper`` and with the group ids in ``group_of``.
    row_lower, row_upper:
        Effective row limits, already clamped to what the box alone permits
        (``sum`` of member bounds). Clamping keeps every value finite — no
        infinities enter the kernel — and makes "this row is slack" an exact test
        rather than a tolerance.
    group_of:
        Length ``n_assets``; ``group_of[i]`` is the row index constraining asset
        ``i``, or ``len(row_names)`` for the *ungrouped sentinel*.
    specs:
        The original specifications, for reporting and :meth:`perturb`.
    long_only, weight_bounds:
        The implicit bounds this set was compiled against. Retained so
        :meth:`perturb` can recompile faithfully — bounds that came from an
        ``OptimizerConfig`` rather than from a ``Box`` spec would otherwise be
        silently reset to the defaults.
    """

    kind: str
    assets: tuple[str, ...]
    budget: float
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    row_names: tuple[str, ...]
    row_lower: tuple[float, ...]
    row_upper: tuple[float, ...]
    group_of: tuple[int, ...]
    specs: tuple[Constraint, ...]
    long_only: bool = True
    weight_bounds: tuple[float, float] = (0.0, 1.0)

    @property
    def n_assets(self) -> int:
        return len(self.assets)

    @property
    def n_rows(self) -> int:
        return len(self.row_names)

    def members(self, name: str) -> tuple[str, ...]:
        """Asset names constrained by row ``name``."""
        k = self.row_names.index(name)
        return tuple(a for a, g in zip(self.assets, self.group_of, strict=True) if g == k)

    def projection(self, *, aux: bool = False, path: bool = False) -> tuple[Any, tuple]:
        """Return ``(projection_fn, pparams)`` for the cached solver kernel.

        ``aux=True`` selects the variant that projects only the weight block of a
        packed ``[w, tau]`` vector (the CVaR optimizer); ``path=True`` the
        row-wise variant for a ``(T, N)`` weight path. Imported lazily because
        :mod:`jaxfolio.optimizers.base` imports this package.
        """
        from jaxfolio.optimizers.base import projection_for

        return projection_for(self, aux=aux, path=path)

    def pparams(self) -> tuple:
        """The traced projection parameters for this constraint set.

        Bounds and limits are returned as ``jnp`` arrays for the vector kinds and
        as plain Python floats for the two scalar kinds — the latter reproducing
        byte-for-byte what the pre-existing ``select_projection`` produced.
        """
        import jax.numpy as jnp

        if self.kind == KIND_SIMPLEX:
            return (self.budget,)
        if self.kind == KIND_BOX:
            return (self.lower[0], self.upper[0], self.budget)
        lower = jnp.asarray(self.lower)
        upper = jnp.asarray(self.upper)
        if self.kind == KIND_BOXVEC:
            return (lower, upper, jnp.asarray(self.budget))
        return (
            lower,
            upper,
            jnp.asarray(self.budget),
            jnp.asarray(self.group_of, dtype=jnp.int32),
            # The sentinel's limits are never read (its multiplier is pinned to
            # zero), but the arrays must be n_rows + 1 long to size the segments.
            jnp.asarray((*self.row_lower, 0.0)),
            jnp.asarray((*self.row_upper, 0.0)),
        )

    def perturb(self, name: str, **changes: float) -> CompiledConstraints:
        """Return a copy with one constraint's limits changed.

        Only *values* move, so the perturbed set keeps the same ``kind`` and row
        count and therefore the same compiled kernel — the basis of a cheap
        what-if re-solve. Recompiles the specs so the new limits are re-validated
        and re-clamped, and so an infeasible relaxation is still caught.
        """
        if name not in self.row_names and name != "budget":
            known = ", ".join(self.row_names) or "(none)"
            raise KeyError(f"no constraint named {name!r} in this problem; rows are: {known}")
        specs = [replace(s, **changes) if s.name == name else s for s in self.specs]
        out = compile_constraints(
            specs,
            self.assets,
            long_only=self.long_only,
            weight_bounds=self.weight_bounds,
        )
        if out.kind != self.kind or out.n_rows != self.n_rows:
            raise ValueError(
                f"perturbing {name!r} changed the projection kind "
                f"({self.kind} -> {out.kind}); that is a structural change, not a relaxation"
            )
        return out


def compile_constraints(
    constraints: Sequence[Constraint] | None,
    assets: Sequence[str],
    *,
    long_only: bool = True,
    weight_bounds: tuple[float, float] = (0.0, 1.0),
) -> CompiledConstraints:
    """Resolve constraint specifications against an asset universe.

    Parameters
    ----------
    constraints:
        Specifications from :mod:`jaxfolio.constraints.spec`. Empty or ``None``
        reproduces the pre-existing behaviour exactly (see the module docstring).
    assets:
        Asset names in panel order, as returned by
        :func:`jaxfolio.results.moments`.
    long_only, weight_bounds:
        The implicit bounds, normally ``config.long_only`` and
        ``config.bounds()``. An unscoped :class:`~jaxfolio.constraints.spec.Box`
        replaces them; a scoped one intersects with them.

    Raises
    ------
    InfeasibleConstraints
        The set admits no feasible portfolio. Exact, not heuristic.
    ValueError
        Two rows share an asset (see :mod:`jaxfolio.constraints.structured`), two
        constraints share a name, or more than one budget is given.
    KeyError
        A constraint names an asset that is not in the panel.
    """
    names = tuple(str(a) for a in assets)
    n = len(names)
    if n == 0:
        raise ValueError("compile_constraints: the asset universe is empty")
    index = {a: i for i, a in enumerate(names)}
    specs = tuple(constraints or ())

    for spec in specs:
        if not isinstance(spec, Constraint):
            raise TypeError(
                "constraints must be jaxfolio.constraints specifications, got "
                f"{type(spec).__name__}"
            )
    seen: dict[str, Constraint] = {}
    for spec in specs:
        if spec.name in seen:
            raise ValueError(
                f"two constraints are both named {spec.name!r}; names must be unique so "
                "attribution can refer to them unambiguously"
            )
        seen[spec.name] = spec

    budgets = [s for s in specs if isinstance(s, Budget)]
    if len(budgets) > 1:
        raise ValueError(
            f"{len(budgets)} Budget constraints given ({', '.join(b.name for b in budgets)}); "
            "a portfolio has exactly one budget"
        )
    budget = float(budgets[0].total) if budgets else 1.0

    lower, upper = _merge_boxes(specs, names, index, n, long_only, weight_bounds)
    rows = [s for s in specs if isinstance(s, RowConstraint)]
    row_names, row_lower, row_upper, group_of = _build_rows(rows, index, n, lower, upper)

    _check_feasible_arrays(
        lower, upper, budget, row_names, row_lower, row_upper, group_of, names, rows
    )

    return CompiledConstraints(
        kind=_select_kind(lower, upper, row_names, long_only, budget),
        assets=names,
        budget=budget,
        lower=tuple(float(x) for x in lower),
        upper=tuple(float(x) for x in upper),
        row_names=tuple(row_names),
        row_lower=tuple(float(x) for x in row_lower),
        row_upper=tuple(float(x) for x in row_upper),
        group_of=tuple(int(g) for g in group_of),
        specs=specs,
        long_only=bool(long_only),
        weight_bounds=(float(weight_bounds[0]), float(weight_bounds[1])),
    )


def _merge_boxes(
    specs: tuple[Constraint, ...],
    names: tuple[str, ...],
    index: dict[str, int],
    n: int,
    long_only: bool,
    weight_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve the implicit bounds and every Box spec into ``(lower, upper)``."""
    boxes = [s for s in specs if isinstance(s, Box)]
    unscoped = [b for b in boxes if b.assets is None]
    if unscoped:
        # Explicit universe-wide bounds replace the config defaults; several of
        # them intersect with each other.
        lower = np.full(n, -np.inf)
        upper = np.full(n, np.inf)
        for box in unscoped:
            where = f"Box({box.name!r})"
            lower = np.maximum(lower, _broadcast(box.lower, n, where))
            upper = np.minimum(upper, _broadcast(box.upper, n, where))
    else:
        lo, hi = weight_bounds
        lower = np.full(n, float(lo))
        upper = np.full(n, float(hi))

    for box in boxes:
        if box.assets is None:
            continue
        where = f"Box({box.name!r})"
        cols = np.array([_resolve_one(a, index, n, where) for a in box.assets], dtype=int)
        lower[cols] = np.maximum(lower[cols], _broadcast(box.lower, len(cols), where))
        upper[cols] = np.minimum(upper[cols], _broadcast(box.upper, len(cols), where))

    if long_only:
        # ``long_only`` is a floor on the floor: it can only ever tighten, never
        # licence a short that an explicit Box forbade.
        lower = np.maximum(lower, 0.0)

    bad = np.nonzero(lower > upper)[0]
    if bad.size:
        i = int(bad[0])
        raise InfeasibleConstraints(
            f"asset {names[i]!r} has no admissible weight: the bounds in force require "
            f"{lower[i]:.4g} <= w <= {upper[i]:.4g}. Check for two Box constraints whose "
            "intersection is empty."
        )
    return lower, upper


def _build_rows(
    rows: list[RowConstraint],
    index: dict[str, int],
    n: int,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[list[str], list[float], list[float], np.ndarray]:
    """Assign group ids and clamp each row's limits to what the box permits."""
    group_of = np.full(n, len(rows), dtype=np.int32)  # default: ungrouped sentinel
    owner: dict[int, str] = {}
    row_names: list[str] = []
    row_lower: list[float] = []
    row_upper: list[float] = []

    for k, row in enumerate(rows):
        where = f"{type(row).__name__}({row.name!r})"
        cols = []
        for asset in row.members():
            col = _resolve_one(asset, index, n, where)
            if col in owner:
                raise ValueError(
                    f"{where} and {owner[col]!r} both constrain asset index {col}. "
                    "Group constraints must partition the universe — each asset may belong "
                    "to at most one group. That disjointness is what makes the projection "
                    "exact and the shadow prices trustworthy; overlapping groups would need "
                    "an alternating projection that is inexact at any finite iteration count. "
                    "Workaround: express one dimension as a partition and price the other "
                    "with l2_reg or a custom objective penalty."
                )
            owner[col] = row.name
            cols.append(col)
        idx = np.asarray(cols, dtype=int)
        group_of[idx] = k
        # Clamp to the box-implied reachable range. This keeps every limit finite
        # (an absent min/max becomes the reachable extreme rather than an
        # infinity) and makes ``target == freesum`` an exact slack test in the
        # kernel.
        reach_lo = float(lower[idx].sum())
        reach_hi = float(upper[idx].sum())
        lo = reach_lo if row.min is None else max(float(row.min), reach_lo)
        hi = reach_hi if row.max is None else min(float(row.max), reach_hi)
        row_names.append(row.name)
        row_lower.append(lo)
        row_upper.append(hi)
    return row_names, row_lower, row_upper, group_of


def _check_feasible_arrays(
    lower: np.ndarray,
    upper: np.ndarray,
    budget: float,
    row_names: list[str],
    row_lower: list[float],
    row_upper: list[float],
    group_of: np.ndarray,
    names: tuple[str, ...],
    rows: list[RowConstraint],
) -> None:
    """Exact feasibility test for box + budget + disjoint rows.

    Because the rows are disjoint, the reachable total is exactly the sum of each
    row's reachable interval plus the ungrouped assets' box — so these bounds are
    necessary *and* sufficient, and a set that passes can always be satisfied.
    """
    for k, name in enumerate(row_names):
        if row_lower[k] > row_upper[k] + 1e-12:
            spec = rows[k]
            members = [names[i] for i in np.nonzero(group_of == k)[0]]
            asked = []
            if spec.min is not None:
                asked.append(f"min={spec.min:.4g}")
            if spec.max is not None:
                asked.append(f"max={spec.max:.4g}")
            raise InfeasibleConstraints(
                f"constraint {name!r} ({', '.join(asked)}) cannot be met: its "
                f"{len(members)} members admit a summed weight only in "
                f"[{sum(lower[i] for i in np.nonzero(group_of == k)[0]):.4g}, "
                f"{sum(upper[i] for i in np.nonzero(group_of == k)[0]):.4g}] "
                f"given their per-asset bounds. Members: {', '.join(members[:8])}"
                f"{'...' if len(members) > 8 else ''}"
            )

    ungrouped = group_of == len(row_names)
    total_lo = float(sum(row_lower)) + float(lower[ungrouped].sum())
    total_hi = float(sum(row_upper)) + float(upper[ungrouped].sum())
    if not (total_lo - 1e-9 <= budget <= total_hi + 1e-9):
        detail = ", ".join(
            f"{name!r} in [{lo:.4g}, {hi:.4g}]"
            for name, lo, hi in zip(row_names, row_lower, row_upper, strict=True)
        )
        n_ung = int(ungrouped.sum())
        ung_note = (
            f" plus {n_ung} ungrouped asset(s) contributing "
            f"[{float(lower[ungrouped].sum()):.4g}, {float(upper[ungrouped].sum()):.4g}]"
            if n_ung
            else " and every asset belongs to a group"
        )
        side = "above" if budget > total_hi else "below"
        gap = budget - total_hi if budget > total_hi else total_lo - budget
        raise InfeasibleConstraints(
            f"the constraints admit a total weight only in [{total_lo:.4g}, {total_hi:.4g}], "
            f"but the budget requires {budget:.4g} — {side} the reachable range by {gap:.4g}. "
            f"Rows: {detail or '(none)'}{ung_note}."
        )


def _select_kind(
    lower: np.ndarray,
    upper: np.ndarray,
    row_names: list[str],
    long_only: bool,
    budget: float,
) -> str:
    """Pick the cheapest kernel that can represent this set.

    The simplex condition is exactly the one the pre-existing ``select_projection``
    used (``long_only and lo <= 0 and hi >= 1``), plus ``budget == 1`` — which was
    hardcoded there. So a call with no named constraints lands on the identical
    kernel, and hence the identical cache entry, it always did.
    """
    if row_names:
        return KIND_GROUPED
    if not (np.all(lower == lower[0]) and np.all(upper == upper[0])):
        return KIND_BOXVEC
    lo, hi = float(lower[0]), float(upper[0])
    if long_only and lo <= 0.0 and hi >= 1.0 and budget == 1.0:
        return KIND_SIMPLEX
    return KIND_BOX


def check_feasible(
    constraints: Sequence[Constraint] | None,
    assets: Sequence[str],
    *,
    long_only: bool = True,
    weight_bounds: tuple[float, float] = (0.0, 1.0),
) -> None:
    """Validate a constraint set without solving anything.

    Raises the same errors :func:`compile_constraints` would. Useful for
    validating user input up front — e.g. in a UI — since the feasibility test is
    exact and costs nothing.
    """
    compile_constraints(constraints, assets, long_only=long_only, weight_bounds=weight_bounds)
