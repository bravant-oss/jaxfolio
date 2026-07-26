"""Named constraint specifications.

Where :mod:`jaxfolio.constraints.projections` provides *anonymous* projections
(``w -> w`` maps onto a feasible set), this module lets a caller **name** the
constraints that shape a portfolio::

    cons = [
        jf.GroupCap("tech", ["AAPL", "MSFT", "NVDA"], max=0.30),
        jf.Box(lower=0.0, upper=0.10),
    ]
    res = jf.minimum_variance(returns, constraints=cons)

Naming is what makes a solution explainable: once a cap has an identity, the
solver's Lagrange multiplier for it becomes a *shadow price* attributable to
``"tech"`` rather than an anonymous number.

These are pure specifications — frozen, hashable, and free of any JAX or solver
dependency (they import stdlib only). They validate their own *structure* at
construction: types, tuple coercion, finiteness, ``min <= max``. They can not
validate *feasibility*, which additionally needs the asset universe and the
budget; :func:`jaxfolio.constraints.compile.compile_constraints` does that. The
two-stage split mirrors :class:`~jaxfolio.types.OptimizerConfig`, which likewise
resolves what it can at construction and defers the rest.

Hashability is load-bearing, not incidental: these objects travel on
:class:`~jaxfolio.types.OptimizerConfig`, which is a frozen dataclass used as a
``functools.partial`` payload and a dict key throughout the backtester. Every
sequence field is therefore coerced to a tuple.

The supported feasible set is
``{w : l <= w <= u, 1'w = b, L_k <= a_k'w <= U_k}`` where the rows ``a_k`` must
have **pairwise disjoint supports** — each asset may be touched by at most one
named row. That restriction is not an oversight; see
:mod:`jaxfolio.constraints.structured` for why it is exactly what makes the
projection exact and the shadow prices trustworthy.

Not expressible here: rows with coefficients other than 1 (a general
``LinearConstraint`` for beta/factor-exposure limits — the inner bisection
generalizes readily, but the *outer* budget bisection is only provably monotone
for all-ones rows, so it is tracked rather than guessed at), overlapping or
nested groups (structural, per above),
cardinality and minimum-position-size limits (non-convex — no Euclidean
projection exists for a projected-gradient method to converge through), and
gross-exposure/leverage caps (``||w||_1 <= G``, which needs a third bisection
level and is only meaningful for long-short books — tracked, not yet
implemented).
"""

from __future__ import annotations

import abc
import math
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "Box",
    "Budget",
    "Constraint",
    "GroupCap",
    "GroupFloor",
    "RowConstraint",
]

# Sequence types that are *not* valid asset collections, so a bare string does
# not silently become a tuple of characters.
_Scalar = (int, float)


def _check_name(name: object) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"constraint name must be a non-empty string, got {name!r}")
    return name


def _check_finite(value: object, what: str) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{what} must be a real number, got {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{what} must be finite, got {out!r}")
    return out


def _coerce_bound(value: object, what: str) -> float | tuple[float, ...]:
    """Coerce a scalar-or-per-asset bound to a float or a tuple of floats."""
    if isinstance(value, _Scalar) and not isinstance(value, bool):
        return _check_finite(value, what)
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{what} must be a number or a sequence of numbers, got {value!r}")
    if not value:
        raise ValueError(f"{what} must be non-empty when given as a sequence")
    return tuple(_check_finite(v, f"{what}[{i}]") for i, v in enumerate(value))


def _coerce_members(assets: object, what: str) -> tuple[str, ...] | tuple[int, ...]:
    """Coerce an asset collection to a tuple of all-names or all-indices."""
    if isinstance(assets, str) or not isinstance(assets, Sequence):
        raise TypeError(f"{what} must be a sequence of asset names or indices, got {assets!r}")
    items = tuple(assets)
    if not items:
        raise ValueError(f"{what} must name at least one asset")
    all_str = all(isinstance(a, str) for a in items)
    # bool is an int subclass; reject it explicitly so `[True, False]` is not
    # mistaken for the index pair (1, 0).
    all_int = all(isinstance(a, int) and not isinstance(a, bool) for a in items)
    if not (all_str or all_int):
        raise TypeError(
            f"{what} must be either all asset names (str) or all positional indices (int), "
            f"not a mix: {items!r}"
        )
    seen: set[object] = set()
    for a in items:
        if a in seen:
            raise ValueError(f"{what} lists {a!r} more than once")
        seen.add(a)
    return items  # type: ignore[return-value]


def _check_interval(low: float | None, high: float | None, name: str) -> None:
    if low is None and high is None:
        raise ValueError(
            f"constraint {name!r} sets neither min nor max — it would constrain nothing. "
            "Pass max=... for a cap, min=... for a floor, or both for a band."
        )
    if low is not None and high is not None and low > high:
        raise ValueError(f"constraint {name!r} has min={low} above max={high}")


class Constraint(abc.ABC):
    """Base class for every named constraint.

    Deliberately *not* a dataclass: it declares no fields, so subclasses stay
    free to order their own (a dataclass base would force every subclass field
    to come after the base's and to carry a default).
    """

    name: str

    @abc.abstractmethod
    def _validate(self) -> None:
        """Raise if the specification is structurally invalid."""


class RowConstraint(Constraint):
    """A named scalar row ``L <= a'w <= U``.

    Group caps and general linear constraints differ only in their coefficients,
    so the compiler and the dual extractor treat both through this interface.
    """

    min: float | None
    max: float | None

    @abc.abstractmethod
    def terms(self) -> tuple[tuple[str | int, float], ...]:
        """Return the row as ``((asset, coefficient), ...)``."""

    def members(self) -> tuple[str | int, ...]:
        """The assets this row touches — its support."""
        return tuple(asset for asset, _ in self.terms())


@dataclass(frozen=True)
class Budget(Constraint):
    """Total invested weight: ``sum(w) == total``.

    Always present. Instantiate one explicitly only to override the default
    fully-invested ``1.0`` — e.g. ``Budget(0.0)`` for a dollar-neutral book.
    """

    total: float = 1.0
    name: str = "budget"

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        _check_name(self.name)
        object.__setattr__(self, "total", _check_finite(self.total, "Budget.total"))


@dataclass(frozen=True)
class Box(Constraint):
    """Per-asset weight bounds ``lower <= w_i <= upper``.

    ``lower`` and ``upper`` are each either a scalar applied to every asset in
    scope, or a per-asset sequence. ``assets=None`` (the default) scopes the box
    to the whole universe and *replaces* the bounds implied by
    ``OptimizerConfig.long_only`` / ``weight_bounds``; an ``assets``-scoped box
    *intersects* with whatever is already in force, so per-name overrides
    compose::

        Box(lower=0.0, upper=0.10)                    # every asset capped at 10%
        Box(upper=0.02, assets=["ILLIQ"], name="illiquid")   # tighter on one name

    A per-asset sequence must match the length of ``assets``, or of the whole
    universe when ``assets is None``.
    """

    lower: float | tuple[float, ...] = 0.0
    upper: float | tuple[float, ...] = 1.0
    assets: tuple[str, ...] | tuple[int, ...] | None = None
    name: str = "box"

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        _check_name(self.name)
        object.__setattr__(self, "lower", _coerce_bound(self.lower, f"Box({self.name!r}).lower"))
        object.__setattr__(self, "upper", _coerce_bound(self.upper, f"Box({self.name!r}).upper"))
        if self.assets is not None:
            object.__setattr__(
                self, "assets", _coerce_members(self.assets, f"Box({self.name!r}).assets")
            )
        widths = {len(b) for b in (self.lower, self.upper) if isinstance(b, tuple)}
        if len(widths) > 1:
            raise ValueError(
                f"Box({self.name!r}): lower and upper are per-asset sequences of "
                f"different lengths {sorted(widths)}"
            )
        if widths and self.assets is not None and len(self.assets) not in widths:
            raise ValueError(
                f"Box({self.name!r}): per-asset bounds have length {widths.pop()} but "
                f"assets names {len(self.assets)} assets"
            )
        lows = self.lower if isinstance(self.lower, tuple) else (self.lower,)
        highs = self.upper if isinstance(self.upper, tuple) else (self.upper,)
        if len(lows) == len(highs):
            pairs = zip(lows, highs, strict=True)
        else:  # one scalar, one sequence — broadcast the scalar
            n = max(len(lows), len(highs))
            pairs = zip(
                lows * n if len(lows) == 1 else lows,
                highs * n if len(highs) == 1 else highs,
                strict=True,
            )
        for i, (lo, hi) in enumerate(pairs):
            if lo > hi:
                raise ValueError(
                    f"Box({self.name!r}): lower bound {lo} exceeds upper bound {hi} at position {i}"
                )

    def width(self) -> int | None:
        """Length of the per-asset bounds, or ``None`` if both are scalars."""
        for b in (self.lower, self.upper):
            if isinstance(b, tuple):
                return len(b)
        return None


@dataclass(frozen=True)
class GroupCap(RowConstraint):
    """A cap and/or floor on the summed weight of a named group of assets.

    The canonical sector constraint::

        GroupCap("tech", ["AAPL", "MSFT", "NVDA"], max=0.30)   # tech <= 30%
        GroupCap("energy", ["XOM", "CVX"], min=0.05)           # energy >= 5%
        GroupCap("bonds", ["TLT", "IEF"], min=0.20, max=0.40)  # a band

    Groups must be **disjoint** — no asset may appear in two of them. See
    :func:`jaxfolio.constraints.compile.compile_constraints`, which enforces it,
    and :mod:`jaxfolio.constraints.structured` for why.
    """

    name: str
    assets: tuple[str, ...] | tuple[int, ...]
    max: float | None = None
    min: float | None = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        _check_name(self.name)
        object.__setattr__(
            self, "assets", _coerce_members(self.assets, f"GroupCap({self.name!r}).assets")
        )
        for attr in ("min", "max"):
            value = getattr(self, attr)
            if value is not None:
                object.__setattr__(
                    self, attr, _check_finite(value, f"GroupCap({self.name!r}).{attr}")
                )
        _check_interval(self.min, self.max, self.name)

    def terms(self) -> tuple[tuple[str | int, float], ...]:
        return tuple((asset, 1.0) for asset in self.assets)

    def members(self) -> tuple[str | int, ...]:
        return tuple(self.assets)


def GroupFloor(  # noqa: N802 - a spec constructor, named like the class it returns
    name: str,
    assets: Sequence[str] | Sequence[int],
    min: float,  # noqa: A002 - mirrors GroupCap's public field name
) -> GroupCap:
    """A named floor on a group's summed weight — sugar for ``GroupCap(min=...)``.

    Returns a :class:`GroupCap` rather than a distinct class so that a group can
    never carry a floor and a cap that disagree, and so the compiler and the
    attribution report have exactly one row type to reason about.
    """
    return GroupCap(name, tuple(assets), min=min)
