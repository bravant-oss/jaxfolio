"""Shared data structures for jaxfolio.

These are lightweight, framework-agnostic containers. Optimizers are configured
by an :class:`OptimizerConfig` (plus a :class:`TradingCosts` spec for the
cost-aware multi-period optimizer) and return a :class:`PortfolioResult`; the
backtester and plotting utilities consume it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from jaxfolio.solvers import SolverSpec


@dataclass(frozen=True)
class OptimizerConfig:
    """Configuration shared by the projected-gradient classical optimizers.

    Attributes
    ----------
    risk_free_rate:
        Per-period risk-free rate used by Sharpe-style objectives.
    long_only:
        If ``True``, weights are projected onto the simplex (no shorting).
    weight_bounds:
        Explicit ``(lower, upper)`` per-asset bounds applied alongside the budget
        constraint. When left as ``None`` the effective bounds follow
        ``long_only``: ``(0, 1)`` when long-only, ``(-1, 1)`` when shorting is
        allowed — so ``OptimizerConfig(long_only=False)`` permits shorting
        without also having to set bounds. Set this to override those defaults.
    max_iter:
        Maximum projected-gradient iterations.
    solver:
        Which projected-gradient solver to use. Three spellings are accepted:

        * ``"spg"`` (default) — a spectral projected-gradient method with
          Barzilai-Borwein step sizes: it needs no learning-rate tuning and
          converges to the constrained optimum (matching a dedicated QP solver)
          in far fewer iterations.
        * the name of **any optax optimizer** — ``"adam"``, ``"adamw"``,
          ``"sgd"``, ``"rmsprop"``, ``"lion"``, ... resolved against ``optax``
          and ``optax.contrib``. See :func:`jaxfolio.solvers.available_solvers`.
        * an optax **factory callable** — ``optax.adamw``, or a module-level
          function of your own returning a ``GradientTransformation`` (the way
          to use an ``optax.chain(...)`` composition).

        The optax solvers keep smooth fixed-step dynamics, which are preferable
        when differentiating *through* the optimizer to train an allocation
        policy. Pass the factory, not a pre-built ``optax.adam(1e-2)``: a
        pre-built transformation is a new object on every call and would
        recompile the solver kernel each solve.
    solver_options:
        Extra keyword arguments forwarded to the optax factory, e.g.
        ``{"weight_decay": 1e-3}`` or ``{"momentum": 0.9, "nesterov": True}``.
        Only valid for optax solvers — ``"spg"`` is tuned via ``learning_rate``
        and ``tol``. Values must be hashable: they become part of the solver's
        jit cache key, so each distinct combination costs one compile.
    learning_rate:
        Step size for the solver. ``None`` (default) lets the solver pick it
        automatically — a curvature-based (``1/L``) initial step for ``"spg"``,
        or ``1e-2`` for any optax optimizer. Set a float to override.
    tol:
        Convergence tolerance. For ``"spg"`` this is the projected-gradient
        (KKT stationarity) norm — zero exactly at a KKT point. For the optax
        solvers it is the weight-update norm, which is only a proxy for
        optimality: optimizers whose step size does not shrink near the optimum
        (``"sign_sgd"``, ``"lion"``, plain ``"sgd"``) may either run to
        ``max_iter`` or stop early when the budget projection cancels a uniform
        step. Prefer ``"spg"`` when you want the exact constrained optimum.
    l2_reg:
        Optional L2 penalty on weights (encourages diversification).
    """

    risk_free_rate: float = 0.0
    long_only: bool = True
    weight_bounds: tuple[float, float] | None = None
    max_iter: int = 2000
    solver: str | Callable[..., Any] = "spg"
    # ``hash=False`` keeps the frozen dataclass hashable despite the mapping;
    # equality still compares it. The hashable form used as a jit cache key is
    # produced by ``solver_spec()``.
    solver_options: Mapping[str, Any] | None = field(default=None, hash=False)
    learning_rate: float | None = None
    tol: float = 1e-7
    l2_reg: float = 0.0

    def __post_init__(self) -> None:
        # Fail fast, at construction, with a suggestion — rather than deep inside
        # the solver after the moments have already been estimated.
        self.solver_spec()

    def solver_spec(self) -> SolverSpec:
        """Resolve ``solver`` / ``solver_options`` into a hashable solver key."""
        from jaxfolio.solvers import resolve_solver  # lazy: avoids an import cycle

        return resolve_solver(self.solver, self.solver_options)

    def bounds(self) -> tuple[float, float]:
        """Resolve the effective per-asset weight bounds.

        Honors an explicit ``weight_bounds``; otherwise defaults to ``(0, 1)``
        for a long-only portfolio and ``(-1, 1)`` when shorting is allowed.
        """
        if self.weight_bounds is not None:
            return self.weight_bounds
        return (0.0, 1.0) if self.long_only else (-1.0, 1.0)


def _coerce_coefficient(value, field_name: str) -> float | tuple[float, ...]:
    """Normalize a cost coefficient to a ``float`` or a ``tuple[float, ...]``.

    Coefficients must never be stored as arrays: :class:`TradingCosts` is a
    frozen dataclass, so its autogenerated ``__hash__`` reads every field, and an
    ndarray field would make the class unhashable. Tuples keep it hashable,
    comparable, and printable while still accepting per-asset input.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        out = float(value)
        if not math.isfinite(out):
            raise ValueError(f"TradingCosts.{field_name} must be finite, got {out}")
        if out < 0.0:
            raise ValueError(f"TradingCosts.{field_name} must be non-negative, got {out}")
        return out

    if isinstance(value, (Sequence, np.ndarray)) and not isinstance(value, (str, bytes)):
        arr = np.asarray(value, dtype=float)
        if arr.ndim != 1:
            raise ValueError(
                f"TradingCosts.{field_name} must be a scalar or a 1-D per-asset "
                f"sequence, got shape {arr.shape}"
            )
        if not np.isfinite(arr).all():
            bad = int(np.argmin(np.isfinite(arr)))
            raise ValueError(f"TradingCosts.{field_name}[{bad}] must be finite, got {arr[bad]}")
        if (arr < 0.0).any():
            bad = int(np.argmin(arr))
            raise ValueError(
                f"TradingCosts.{field_name}[{bad}] must be non-negative, got {arr[bad]}"
            )
        return tuple(float(x) for x in arr)

    raise TypeError(
        f"TradingCosts.{field_name} must be a number or a per-asset sequence, "
        f"got {type(value).__name__}"
    )


@dataclass(frozen=True)
class TradingCosts:
    """Per-period trading frictions for the multi-period optimizer.

    The three cost coefficients are quoted in **basis points of traded notional
    per unit of** ``|dw|``, **one-way, per period** — the units a trader states
    them in. Each accepts a scalar (applied to every asset) or a per-asset
    sequence of length ``n``; sequences are stored as tuples so the dataclass
    stays hashable.

    Costs are *per period* and never annualized: the objective trades them off
    against per-period ``mu`` and ``Sigma``, so annualizing would double-count.

    Attributes
    ----------
    spread_bps:
        Bid-ask **half**-spread, in bps, charged linearly on ``|dw|``. A round
        trip therefore costs ``2 * spread_bps``.
    commission_bps:
        Proportional commission / fee, in bps, also linear on ``|dw|``. Kept
        separate from ``spread_bps`` only for reporting clarity — the objective
        sums the two.
    impact_bps:
        Quadratic market-impact coefficient, in bps per unit of ``dw**2``. Models
        the convex part of execution cost (temporary impact), which is what makes
        splitting a large trade across periods strictly cheaper than trading it
        all at once.
    turnover_penalty:
        An extra L1 penalty on ``|dw|`` in **decimal** units (not bps), for soft
        turnover control that is not a real cash cost. Reported separately from
        ``total_cost`` so P&L attribution stays honest.
    smoothing:
        Starting (widest) relative Huber half-width used to make the L1 cost
        differentiable. The absolute width is ``smoothing * budget / n``, i.e. a
        fraction of the natural trade unit, which makes the resulting bias
        independent of ``n``. Must be strictly positive: at ``0`` the gradient is
        undefined at ``dw == 0``, which is precisely where the optimum sits when
        costs bind.

        The optimizer walks a geometrically shrinking ladder from this width and
        keeps whichever stage is best under the *exact* objective, so this is a
        starting point for a search rather than a value the answer hinges on —
        which is why it is a tuning knob and not part of the API contract.

    Notes
    -----
    Unit bridge to the backtester: ``TradingCosts(spread_bps=10.0)`` prices the
    same linear cost that ``backtest(transaction_cost=0.0010)`` charges. The
    engine keyword is decimal for historical reasons; this class is bps.

    Turnover control here is **soft** — costs are priced, not capped. See the
    multi-period guide for why a hard turnover budget is not expressible in this
    solver, and what to do instead.

    Examples
    --------
    >>> TradingCosts(spread_bps=5.0, commission_bps=1.0).linear_decimal(3)
    array([0.0006, 0.0006, 0.0006])
    """

    spread_bps: float | tuple[float, ...] = 0.0
    commission_bps: float | tuple[float, ...] = 0.0
    impact_bps: float | tuple[float, ...] = 0.0
    turnover_penalty: float = 0.0
    smoothing: float = 1e-1

    def __post_init__(self) -> None:
        # ``frozen=True`` blocks plain attribute assignment; coercion has to go
        # through object.__setattr__. Doing it here (rather than in as_params)
        # means an invalid cost fails at construction, next to the caller's typo.
        for name in ("spread_bps", "commission_bps", "impact_bps"):
            object.__setattr__(self, name, _coerce_coefficient(getattr(self, name), name))

        penalty = float(self.turnover_penalty)
        if not math.isfinite(penalty) or penalty < 0.0:
            raise ValueError(
                f"TradingCosts.turnover_penalty must be finite and non-negative, got {penalty}"
            )
        object.__setattr__(self, "turnover_penalty", penalty)

        smoothing = float(self.smoothing)
        if not math.isfinite(smoothing) or smoothing <= 0.0:
            raise ValueError(
                "TradingCosts.smoothing must be strictly positive (the L1 cost is "
                f"non-differentiable at zero trade without it), got {smoothing}"
            )
        object.__setattr__(self, "smoothing", smoothing)

    def is_active(self) -> bool:
        """True if any cost coefficient is non-zero (reporting/metadata only)."""
        return (
            any(
                np.any(np.asarray(getattr(self, name), dtype=float) != 0.0)
                for name in ("spread_bps", "commission_bps", "impact_bps")
            )
            or self.turnover_penalty != 0.0
        )

    def _broadcast(self, name: str, n: int) -> np.ndarray:
        """Broadcast one coefficient to length ``n``, naming it if it cannot be."""
        arr = np.atleast_1d(np.asarray(getattr(self, name), dtype=float))
        if arr.size == 1:
            return np.full(n, float(arr[0]))
        if arr.size != n:
            raise ValueError(f"TradingCosts.{name} has length {arr.size}, expected 1 or {n}")
        return arr

    def linear_decimal(self, n: int) -> np.ndarray:
        """Total linear cost ``(spread_bps + commission_bps) / 1e4``, length ``n``."""
        return (self._broadcast("spread_bps", n) + self._broadcast("commission_bps", n)) / 1e4

    def quadratic_decimal(self, n: int) -> np.ndarray:
        """Quadratic impact coefficient ``impact_bps / 1e4``, length ``n``."""
        return self._broadcast("impact_bps", n) / 1e4

    def as_params(self, n: int) -> dict:
        """Traced objective parameters for ``n`` assets.

        Returns ``{"c_lin": (n,), "c_quad": (n,)}`` as JAX arrays. Both are
        *traced* rather than static, so changing a cost never recompiles the
        solver kernel. ``turnover_penalty`` is folded into ``c_lin`` because it
        enters the objective identically; the split is preserved for reporting by
        :meth:`linear_decimal`.
        """
        import jax.numpy as jnp  # lazy: keeps this module importable without jax

        c_lin = self.linear_decimal(n) + self.turnover_penalty
        return {
            "c_lin": jnp.asarray(c_lin, dtype=float),
            "c_quad": jnp.asarray(self.quadratic_decimal(n), dtype=float),
        }


@dataclass
class PortfolioResult:
    """The output of an optimizer: weights plus diagnostics.

    Attributes
    ----------
    weights:
        Portfolio weights as a 1-D numpy array (sums to 1).
    assets:
        Asset names aligned with ``weights``.
    method:
        Human-readable optimizer name.
    expected_return:
        Annualized (or per-period, if unscaled) expected portfolio return.
    volatility:
        Portfolio volatility on the same scale as ``expected_return``.
    sharpe:
        Sharpe ratio implied by ``expected_return`` / ``volatility``.
    metadata:
        Free-form diagnostics (iterations, converged flag, cluster labels, ...).
        Values are kept JSON-serializable by convention; bulk arrays belong in a
        dedicated field like ``trajectory``.
    trajectory:
        Optional ``(T, n_assets)`` planned weight *path* for multi-period
        optimizers, where row ``t`` is the portfolio to hold in period ``t`` and
        row 0 equals ``weights``. ``None`` for every single-period optimizer.
        :func:`jaxfolio.backtest.backtest` executes this path step by step when
        it is present.
    """

    weights: np.ndarray
    assets: list[str]
    method: str
    expected_return: float | None = None
    volatility: float | None = None
    sharpe: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    trajectory: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=float).reshape(-1)
        if len(self.assets) != self.weights.shape[0]:
            n_assets, n_weights = len(self.assets), self.weights.shape[0]
            # A common slip is handing the (T, n) path to ``weights``, which the
            # reshape above silently flattens — say so rather than just the counts.
            hint = ""
            if n_assets > 1 and n_weights > n_assets and n_weights % n_assets == 0:
                hint = " (looks like a flattened weight path — did you mean trajectory=?)"
            raise ValueError(f"assets ({n_assets}) and weights ({n_weights}) length mismatch{hint}")
        if self.trajectory is not None:
            traj = np.asarray(self.trajectory, dtype=float)
            if traj.ndim != 2 or traj.shape[1] != self.weights.shape[0]:
                raise ValueError(
                    f"trajectory shape {traj.shape} is incompatible with "
                    f"{self.weights.shape[0]} assets; expected (T, {self.weights.shape[0]})"
                )
            self.trajectory = traj

    def as_dict(self) -> dict[str, float]:
        """Return ``{asset: weight}`` sorted by descending absolute weight."""
        pairs = zip(self.assets, self.weights.tolist(), strict=True)
        return dict(sorted(pairs, key=lambda kv: abs(kv[1]), reverse=True))

    def top(self, n: int = 10) -> dict[str, float]:
        """Return the ``n`` largest holdings by absolute weight."""
        return dict(list(self.as_dict().items())[:n])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        head = ", ".join(f"{a}={w:+.3f}" for a, w in list(self.top(5).items()))
        sr = f"{self.sharpe:.2f}" if self.sharpe is not None else "n/a"
        # Conditional so single-period reprs stay byte-identical to before.
        steps = f", steps={self.trajectory.shape[0]}" if self.trajectory is not None else ""
        return f"PortfolioResult(method={self.method!r}, sharpe={sr}{steps}, top=[{head}])"
