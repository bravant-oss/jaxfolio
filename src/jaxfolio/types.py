"""Shared data structures for jaxfolio.

These are lightweight, framework-agnostic containers. Optimizers return a
:class:`PortfolioResult`; the backtester and plotting utilities consume it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
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
    """

    weights: np.ndarray
    assets: list[str]
    method: str
    expected_return: float | None = None
    volatility: float | None = None
    sharpe: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=float).reshape(-1)
        if len(self.assets) != self.weights.shape[0]:
            raise ValueError(
                f"assets ({len(self.assets)}) and weights ({self.weights.shape[0]}) length mismatch"
            )

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
        return f"PortfolioResult(method={self.method!r}, sharpe={sr}, top=[{head}])"
