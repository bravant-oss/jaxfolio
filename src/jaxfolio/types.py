"""Shared data structures for jaxfolio.

These are lightweight, framework-agnostic containers. Optimizers return a
:class:`PortfolioResult`; the backtester and plotting utilities consume it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


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
        Which projected-gradient solver to use. ``"spg"`` (default) is a
        spectral projected-gradient method with Barzilai-Borwein step sizes: it
        needs no learning-rate tuning and converges to the constrained optimum
        (matching a dedicated QP solver) in far fewer iterations. ``"adam"``
        keeps the smooth optax-Adam dynamics, which are preferable when
        differentiating *through* the optimizer to train an allocation policy.
    learning_rate:
        Step size for the solver. ``None`` (default) lets the solver pick it
        automatically — a curvature-based (``1/L``) initial step for ``"spg"``,
        or ``1e-2`` for ``"adam"``. Set a float to override.
    tol:
        Convergence tolerance. For ``"spg"`` this is the projected-gradient
        (KKT stationarity) norm; for ``"adam"`` it is the weight-update norm.
    l2_reg:
        Optional L2 penalty on weights (encourages diversification).
    """

    risk_free_rate: float = 0.0
    long_only: bool = True
    weight_bounds: tuple[float, float] | None = None
    max_iter: int = 2000
    solver: str = "spg"
    learning_rate: float | None = None
    tol: float = 1e-7
    l2_reg: float = 0.0

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
