"""Author your own portfolio strategy with minimal boilerplate.

Two entry points, both producing a proper :class:`PortfolioResult` that works
everywhere a built-in optimizer does (backtester, ``compare``, plots):

* :func:`custom_strategy` / :meth:`CustomStrategy.from_weights` — wrap a function
  that maps returns to a weight vector (array) or a ``{asset: weight}`` mapping.
  Weights are validated, optionally renormalized, and annualized diagnostics are
  attached automatically.
* :meth:`CustomStrategy.from_objective` — provide a JAX objective ``f(w, ctx)``
  and let the shared projected-gradient solver optimize it under the configured
  constraints (this is exactly how the built-in classical optimizers are built).

Both optionally register the resulting strategy by name via :mod:`jaxfolio.registry`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from jaxfolio import toolkit as tk
from jaxfolio.registry import register_strategy
from jaxfolio.types import OptimizerConfig, PortfolioResult

Array = jnp.ndarray


@dataclass
class _Context:
    """The moment context handed to an objective-based custom strategy.

    Attributes: ``mu`` (mean returns), ``cov`` (covariance), ``returns`` (the
    raw ``T x N`` matrix), ``assets`` (names), ``n`` (number of assets).
    """

    mu: Array
    cov: Array
    returns: Array
    assets: list[str]

    @property
    def n(self) -> int:
        return len(self.assets)


def _coerce_weights(raw, assets: list[str]) -> np.ndarray:
    """Turn a user weight output (array or ``{asset: w}``) into an aligned array."""
    if isinstance(raw, Mapping):
        return np.array([float(raw.get(a, 0.0)) for a in assets], dtype=float)
    arr = np.asarray(raw, dtype=float).reshape(-1)
    if arr.shape[0] != len(assets):
        raise ValueError(
            f"weight function returned {arr.shape[0]} weights for {len(assets)} assets"
        )
    return arr


class CustomStrategy:
    """A user-defined strategy exposed with the standard optimizer interface.

    Instances are callable as ``strategy(returns) -> PortfolioResult``, so they
    drop straight into :func:`jaxfolio.backtest.compare` and the plotting layer.
    """

    def __init__(self, name: str, fn: Callable[[object], PortfolioResult]):
        self.name = name
        self._fn = fn

    def __call__(self, returns) -> PortfolioResult:
        return self._fn(returns)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"CustomStrategy({self.name!r})"

    # --- constructors ---------------------------------------------------- #
    @classmethod
    def from_weights(
        cls,
        name: str,
        weight_fn: Callable[[object], object],
        *,
        renormalize: bool = True,
        risk_free: float = 0.0,
        register: bool = False,
        description: str = "",
    ) -> CustomStrategy:
        """Wrap a ``returns -> weights`` function into a full strategy.

        ``weight_fn`` may return a numpy/JAX array aligned to the asset columns or
        a ``{asset: weight}`` mapping. If ``renormalize`` is ``True`` (default)
        and the weights sum to a non-zero value, they are scaled to sum to 1.
        """

        def run(returns) -> PortfolioResult:
            mu, cov, names, _mat = tk.moments(returns)
            w = _coerce_weights(weight_fn(returns), names)
            # Only rescale to a fully-invested book when the net exposure is
            # positive. Dividing by a negative sum would flip every sign and
            # invert a net-short / dollar-neutral stance, so those are left as-is.
            if renormalize and w.sum() > 1e-12:
                w = w / w.sum()
            return tk.finalize_result(
                w,
                names,
                name,
                mu=mu,
                cov=cov,
                risk_free=risk_free,
                metadata={"custom": True, "kind": "weights"},
            )

        strat = cls(name, run)
        if register:
            register_strategy(name, description=description, overwrite=True)(strat)
        return strat

    @classmethod
    def from_objective(
        cls,
        name: str,
        objective_fn: Callable[[Array, _Context], Array],
        *,
        config: OptimizerConfig | None = None,
        register: bool = False,
        description: str = "",
    ) -> CustomStrategy:
        """Build a strategy by minimizing a JAX objective under constraints.

        ``objective_fn(w, ctx)`` must be JAX-differentiable and return a scalar to
        *minimize*; ``ctx`` is a :class:`_Context` exposing ``mu``, ``cov``,
        ``returns``, ``assets``, ``n``. The shared projected-gradient solver and
        the configured constraint set (long-only / bounds) are reused verbatim.
        """
        cfg = config or OptimizerConfig()

        def run(returns) -> PortfolioResult:
            mu, cov, names, mat = tk.moments(returns)
            ctx = _Context(mu=mu, cov=cov, returns=mat, assets=names)
            projection = tk.make_projection(cfg.long_only, cfg.bounds())
            w, info = tk.solve_projected_gradient(
                lambda w: objective_fn(w, ctx),
                tk.equal_start(len(names)),
                projection,
                learning_rate=cfg.learning_rate,
                max_iter=cfg.max_iter,
                tol=cfg.tol,
            )
            return tk.finalize_result(
                w,
                names,
                name,
                mu=mu,
                cov=cov,
                risk_free=cfg.risk_free_rate,
                metadata={
                    "custom": True,
                    "kind": "objective",
                    "iterations": int(info["iterations"]),
                },
            )

        strat = cls(name, run)
        if register:
            register_strategy(name, description=description, overwrite=True)(strat)
        return strat


def custom_strategy(
    name: str,
    weight_fn: Callable[[object], object],
    *,
    renormalize: bool = True,
    risk_free: float = 0.0,
    register: bool = False,
    description: str = "",
) -> CustomStrategy:
    """Shorthand for :meth:`CustomStrategy.from_weights` (the common case)."""
    return CustomStrategy.from_weights(
        name,
        weight_fn,
        renormalize=renormalize,
        risk_free=risk_free,
        register=register,
        description=description,
    )
