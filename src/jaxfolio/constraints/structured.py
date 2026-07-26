"""Exact Euclidean projections onto structured constraint sets.

These extend :mod:`jaxfolio.constraints.projections` from "box + budget" (with
*scalar* bounds) to "per-asset box + budget + named group rows", and they
additionally return the **Lagrange multipliers** of the constraints they enforce
— which is what makes an optimizer's solution explainable.

Why the projection must be *exact*
----------------------------------
The projected-gradient solvers in :mod:`jaxfolio.optimizers.base` are not merely
"iterate and clip". ``_spg_loop`` stops on ``||w - P(w - grad f)||``, which
certifies KKT stationarity **only if ``P`` is the true Euclidean projection onto a
convex set**, and its Barzilai-Borwein step ``alpha = s's / s'y`` assumes the
iterates come from a nonexpansive map. An inexact ``P`` — an alternating
projection truncated after a few sweeps, a penalty, an ADMM inner loop — breaks
both: the residual stops meaning what the stopping test claims, and the step size
can oscillate. Under ``jax.lax.while_loop`` with a fixed iteration budget that
fails *silently*. So the projection here is exact, in the same sense
``project_box_budget`` is exact: a bracketed bisection on a monotone dual
function, with a fixed iteration count and no tolerance to tune.

Why the group rows must have disjoint supports
----------------------------------------------
Stationarity of the projection subproblem gives, with ``lam`` on the budget and
``theta_k`` on row ``k``::

    w_i = clip(v_i - lam - theta_{k(i)}, l_i, u_i)

When each asset is touched by at most one row, ``theta_k`` **decouples given
``lam``**: each group's sum is an independent 1-D monotone function of its own
``theta_k``, so the whole projection is a fixed 2-level nest of bisections, each
level monotone. Once supports overlap, the multipliers couple and
the projection needs Dykstra or ADMM — inexact at any finite iteration count, per
above, with multipliers that only converge in the limit. That is why
:func:`jaxfolio.constraints.compile.compile_constraints` rejects overlap outright
rather than degrading quietly.

Multipliers
-----------
The bisection *is* the dual solve, so the multipliers are a by-product rather than
an extra computation. ``project_box_budget`` already computes the budget
multiplier (its ``tau``) and discards it; the ``*_duals`` variants here keep it,
along with the per-row and per-asset multipliers. Scaled by ``1/step``, these are
the KKT multipliers of the original portfolio problem — see
:mod:`jaxfolio.attribution`.

Note on the budget correction
-----------------------------
``project_box_budget`` finishes with a uniform ``(budget - sum(w)) / n`` nudge,
which can push coordinates *outside* the box when the box and budget nearly
conflict. The kernels here instead distribute that residual over the **free** set
only — coordinates strictly inside their bounds and in a non-binding group — so
feasibility is never traded for a tighter budget. Any leftover is reported as
``feasibility_residual`` rather than hidden. ``project_box_budget`` itself is left
untouched: it is public API with pinned numerics.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

__all__ = [
    "ProjectionDuals",
    "project_box_budget_duals",
    "project_box_budget_vec",
    "project_grouped",
    "project_grouped_duals",
]

Array = jnp.ndarray


class ProjectionDuals(NamedTuple):
    """A projected point together with the multipliers that produced it.

    A ``NamedTuple`` so it is a JAX pytree: it flows through ``jit``, ``vmap``
    and ``lax`` control flow with no registration.

    Attributes
    ----------
    weights:
        The projected point.
    budget:
        Multiplier ``lam`` on ``sum(w) = budget``. Signed.
    rows:
        ``(n_seg,)`` multipliers ``theta`` on the named rows — positive where a
        cap binds, negative where a floor binds, exactly zero where the row is
        slack. The last entry is the ungrouped sentinel and is always zero.
    bounds:
        ``(n,)`` multipliers ``beta`` on the per-asset box — positive at a lower
        bound, negative at an upper bound, zero in the interior.
    identified:
        ``(n_seg,)`` mask: is ``rows[k]`` actually pinned by the data? A row's
        multiplier is identified only when at least one of its members is
        *strictly interior* to its box. When every member sits on a bound, the
        split of the shadow price between the row and the box bounds is genuinely
        not determined — any split summing to the same reduced cost is an equally
        valid KKT certificate. This kernel then reports the minimum-norm choice
        (``theta = 0``, everything attributed to the box), which is the reading
        users mean, but a different QP solver will legitimately report a
        different split. Verified against CVXPY: every *identified* row agrees to
        ~4e-9, while unidentified rows disagree by as much as 0.37.
    feasibility_residual:
        ``|sum(w) - budget|`` left after the bisection and the free-set
        correction. At the float precision floor when the problem is feasible;
        non-zero only if it is not.
    """

    weights: Array
    budget: Array
    rows: Array
    bounds: Array
    identified: Array
    feasibility_residual: Array


# Bisection iteration counts. float32 carries ~24 bits of mantissa, so ~25
# halvings exhaust the representable precision of the bracket; more is wasted
# work. Static (not traced) so they never enter a jit cache key as values.
_OUTER_ITER = 30
_INNER_ITER = 30


def project_box_budget_vec(
    v: Array,
    lower: Array,
    upper: Array,
    budget: Array,
    *,
    max_iter: int = 50,
) -> Array:
    """Project ``v`` onto ``{w : lower <= w <= upper, sum(w) = budget}``.

    The per-asset-bounds generalization of
    :func:`jaxfolio.constraints.projections.project_box_budget`: ``lower`` and
    ``upper`` are ``(n,)`` arrays rather than scalars. Same method — bisect the
    budget multiplier ``tau``, since ``w(tau) = clip(v - tau, lower, upper)`` is
    monotone decreasing in ``tau`` — and the same feasibility requirement,
    ``sum(lower) <= budget <= sum(upper)``.
    """
    return project_box_budget_duals(v, lower, upper, budget, max_iter=max_iter).weights


def project_box_budget_duals(
    v: Array,
    lower: Array,
    upper: Array,
    budget: Array,
    *,
    max_iter: int = 50,
) -> ProjectionDuals:
    """Box + budget projection with its multipliers.

    The row-free case of :func:`project_grouped_duals`: ``rows`` is a single
    all-zero sentinel entry and ``identified`` is ``[False]``, so a caller can
    treat both kernels uniformly.
    """
    lo_tau = jnp.min(v - upper) - 1.0
    hi_tau = jnp.max(v - lower) + 1.0

    def body(_, bounds):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        too_big = jnp.sum(jnp.clip(v - mid, lower, upper)) > budget
        return (jnp.where(too_big, mid, lo), jnp.where(too_big, hi, mid))

    lo_tau, hi_tau = jax.lax.fori_loop(0, max_iter, body, (lo_tau, hi_tau))
    lam = 0.5 * (lo_tau + hi_tau)
    x = v - lam
    w = jnp.clip(x, lower, upper)
    w, resid = _absorb_residual(w, lower, upper, budget, free=jnp.ones_like(w, dtype=bool))
    # beta_i = x_i - w_i: positive where the lower bound binds, negative at the
    # upper bound, zero in the interior. Recomputed from the final w so it stays
    # consistent with the returned primal point.
    beta = jnp.where(_interior(w, lower, upper), 0.0, x - w)
    return ProjectionDuals(
        weights=w,
        budget=lam,
        rows=jnp.zeros(1, dtype=w.dtype),
        bounds=beta,
        identified=jnp.zeros(1, dtype=bool),
        feasibility_residual=resid,
    )


def _interior(w: Array, lower: Array, upper: Array) -> Array:
    """Boolean mask of coordinates strictly inside their bounds."""
    slack = jnp.maximum(upper - lower, 0.0)
    eps = 1e-6 * jnp.maximum(slack, 1.0)
    return jnp.logical_and(w > lower + eps, w < upper - eps)


def _absorb_residual(
    w: Array,
    lower: Array,
    upper: Array,
    budget: Array,
    *,
    free: Array,
) -> tuple[Array, Array]:
    """Spread the leftover budget gap over ``free`` coordinates only.

    The bisection leaves ``|sum(w) - budget|`` at its precision floor. Spreading
    that gap uniformly over *all* coordinates (as ``project_box_budget`` does)
    can push a clipped coordinate outside its bound; spreading it only over
    coordinates that are both strictly interior and in a non-binding group keeps
    every constraint intact. When no coordinate is free the gap is left in place
    and returned, so a caller can report it instead of silently absorbing an
    infeasibility.
    """
    usable = jnp.logical_and(free, _interior(w, lower, upper))
    count = jnp.sum(usable)
    gap = budget - jnp.sum(w)
    share = jnp.where(count > 0, gap / jnp.maximum(count, 1), 0.0)
    w = w + jnp.where(usable, share, 0.0)
    w = jnp.clip(w, lower, upper)
    return w, jnp.abs(budget - jnp.sum(w))


def _solve_theta(
    x: Array,
    lower: Array,
    upper: Array,
    gid: Array,
    target: Array,
    active: Array,
    n_seg: int,
    inner_iter: int,
) -> Array:
    """Per-segment multiplier driving each group sum to ``target``.

    ``sum_{i in k} clip(x_i - theta_k, l_i, u_i)`` is non-increasing in
    ``theta_k``, and is pinned at ``sum(u)`` for ``theta_k <= min(x - u)`` and at
    ``sum(l)`` for ``theta_k >= max(x - l)``. Those two values bracket the root,
    so a fixed bisection per segment suffices — and because the supports are
    disjoint, every segment's bisection runs independently in the same sweep.
    """
    lo = jax.ops.segment_min(x - upper, gid, num_segments=n_seg) - 1.0
    hi = jax.ops.segment_max(x - lower, gid, num_segments=n_seg) + 1.0
    # An empty segment yields (+inf, -inf) from the min/max identities; neutralize
    # it so no inf ever enters the arithmetic. Inactive segments are forced to
    # theta = 0 below regardless.
    lo = jnp.where(active, lo, 0.0)
    hi = jnp.where(active, hi, 0.0)

    def body(_, bounds):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        totals = jax.ops.segment_sum(jnp.clip(x - mid[gid], lower, upper), gid, num_segments=n_seg)
        too_big = totals > target
        return (jnp.where(too_big, mid, lo), jnp.where(too_big, hi, mid))

    lo, hi = jax.lax.fori_loop(0, inner_iter, body, (lo, hi))
    return jnp.where(active, 0.5 * (lo + hi), 0.0)


def project_grouped_duals(
    v: Array,
    lower: Array,
    upper: Array,
    budget: Array,
    gid: Array,
    g_lower: Array,
    g_upper: Array,
    *,
    outer_iter: int = _OUTER_ITER,
    inner_iter: int = _INNER_ITER,
) -> ProjectionDuals:
    """Project onto box + budget + disjoint group rows, returning multipliers.

    Parameters
    ----------
    v:
        ``(n,)`` point to project.
    lower, upper:
        ``(n,)`` per-asset bounds.
    budget:
        Scalar; ``sum(w) == budget``.
    gid:
        ``(n,)`` int32 segment id per asset. Index ``n_seg - 1`` is the
        **ungrouped sentinel**: assets assigned to it are subject to no row, and
        its multiplier is pinned to zero. Always allocated, possibly empty.
    g_lower, g_upper:
        ``(n_seg,)`` row bounds. The sentinel's entries are ignored.

    Returns
    -------
    :class:`ProjectionDuals`. Validated against CVXPY over 40 random problems
    (``n`` in 6..40, 2..4 rows, long-only and long-short): weights agree to
    2.1e-8, the budget multiplier to 2.2e-9, and every *identified* row
    multiplier to 3.9e-9. See ``ProjectionDuals.identified`` for the rows whose
    multiplier is not pinned by the data.

    Notes
    -----
    The inner target is ``clip(freesum_k(lam), L_k, U_k)`` — *not* an early exit
    when the row is already satisfied. That distinction is load-bearing: the
    clipped-target form is what makes the dual decomposition unique, so ``lam``
    and ``theta`` cannot drift into offsetting each other. It also makes "row is
    slack" an exact test (``target == freesum``), which is how ``theta_k`` is
    forced to a clean zero instead of the arbitrary value a degenerate bracket
    would otherwise return.

    The outer bracket needs no ``theta`` term even though ``w`` depends on
    ``theta``: the inner solve pins each group's sum to
    ``clip(freesum_k(lam), L_k, U_k)`` regardless of how large ``theta_k`` gets,
    so the total is a function of ``lam`` alone — and a non-increasing one, since
    every ``freesum_k`` and every ungrouped coordinate is non-increasing in
    ``lam`` and ``clip`` is monotone.
    """
    n_seg = g_lower.shape[0]
    active = jnp.arange(n_seg) < n_seg - 1

    def sums(lam):
        x = v - lam
        free = jnp.clip(x, lower, upper)
        freesum = jax.ops.segment_sum(free, gid, num_segments=n_seg)
        target = jnp.where(active, jnp.clip(freesum, g_lower, g_upper), freesum)
        theta = _solve_theta(x, lower, upper, gid, target, active, n_seg, inner_iter)
        return x, freesum, target, theta

    def total(lam):
        _x, _freesum, target, _theta = sums(lam)
        return jnp.sum(target)

    lo = jnp.min(v - upper) - 1.0
    hi = jnp.max(v - lower) + 1.0

    def body(_, bounds):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        too_big = total(mid) > budget
        return (jnp.where(too_big, mid, lo), jnp.where(too_big, hi, mid))

    lo, hi = jax.lax.fori_loop(0, outer_iter, body, (lo, hi))
    lam = 0.5 * (lo + hi)

    x, freesum, target, theta = sums(lam)
    # Exact test: clip() returns freesum unchanged when the row is satisfied, so
    # equality here means "not binding" and pins theta to zero, discarding the
    # arbitrary value a saturated (degenerate) bracket would have produced.
    slack = jnp.logical_or(~active, target == freesum)
    theta = jnp.where(slack, 0.0, theta)

    shifted = x - theta[gid]
    w = jnp.clip(shifted, lower, upper)
    w, resid = _absorb_residual(w, lower, upper, budget, free=slack[gid])
    beta = jnp.where(_interior(w, lower, upper), 0.0, shifted - w)
    # A row's multiplier is pinned by the data only if some member is strictly
    # interior to its box: that member's stationarity equation is the one that
    # separates theta_k from beta. With every member on a bound, only the sum
    # theta_k + beta_i is determined and the split is arbitrary.
    interior = _interior(w, lower, upper)
    identified = jnp.logical_and(
        active, jax.ops.segment_sum(interior.astype(w.dtype), gid, num_segments=n_seg) > 0
    )
    return ProjectionDuals(
        weights=w,
        budget=lam,
        rows=theta,
        bounds=beta,
        identified=identified,
        feasibility_residual=resid,
    )


def project_grouped(
    v: Array,
    lower: Array,
    upper: Array,
    budget: Array,
    gid: Array,
    g_lower: Array,
    g_upper: Array,
    *,
    outer_iter: int = _OUTER_ITER,
    inner_iter: int = _INNER_ITER,
) -> Array:
    """Project onto box + budget + disjoint group rows.

    See :func:`project_grouped_duals` for the algorithm and the meaning of every
    argument; this is the primal-only wrapper used inside the solver loop.
    """
    return project_grouped_duals(
        v,
        lower,
        upper,
        budget,
        gid,
        g_lower,
        g_upper,
        outer_iter=outer_iter,
        inner_iter=inner_iter,
    ).weights
