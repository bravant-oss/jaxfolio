"""A lightweight registry for portfolio strategies.

Any callable with the optimizer shape ``method(returns, ...) -> PortfolioResult``
can be registered by name and then looked up, listed, and mixed with the
built-ins in the backtester. Built-in optimizers register themselves at import
time (see :func:`_register_builtins`), so :func:`list_strategies` always reflects
the full catalog — including the user's own additions.

Usage
-----
>>> from jaxfolio import register_strategy, get_strategy, list_strategies
>>> @register_strategy("my_momentum", description="12-1 momentum tilt")
... def my_momentum(returns):
...     ...
>>> get_strategy("my_momentum") is my_momentum
True
>>> "my_momentum" in list_strategies()
True
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

Strategy = Callable[..., object]


@dataclass(frozen=True)
class StrategyInfo:
    """A registered strategy: its callable, canonical name, and description."""

    name: str
    func: Strategy
    description: str = ""
    builtin: bool = False


_REGISTRY: dict[str, StrategyInfo] = {}


def register_strategy(
    name: str | None = None,
    *,
    description: str = "",
    builtin: bool = False,
    overwrite: bool = False,
) -> Callable[[Strategy], Strategy]:
    """Register a strategy under ``name`` (defaults to the function's name).

    Usable as a decorator (``@register_strategy("x")``) or directly
    (``register_strategy("x")(fn)``). The wrapped function is returned unchanged,
    so decorating never alters behavior. Re-registering an existing name raises
    unless ``overwrite=True``.
    """

    def decorator(func: Strategy) -> Strategy:
        key = name or getattr(func, "__name__", None)
        if not key:
            raise ValueError("register_strategy needs a name (no __name__ on the callable)")
        if key in _REGISTRY and not overwrite:
            raise ValueError(
                f"strategy {key!r} is already registered; pass overwrite=True to replace it"
            )
        _REGISTRY[key] = StrategyInfo(name=key, func=func, description=description, builtin=builtin)
        return func

    return decorator


def get_strategy(name: str) -> Strategy:
    """Return the registered strategy callable for ``name``."""
    try:
        return _REGISTRY[name].func
    except KeyError:
        raise KeyError(f"unknown strategy {name!r}; registered: {sorted(_REGISTRY)}") from None


def strategy_info(name: str) -> StrategyInfo:
    """Return the full :class:`StrategyInfo` record for ``name``."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown strategy {name!r}") from None


def list_strategies(*, builtin_only: bool = False, custom_only: bool = False) -> list[str]:
    """List registered strategy names (optionally filtered by origin)."""
    names = sorted(_REGISTRY)
    if builtin_only:
        return [n for n in names if _REGISTRY[n].builtin]
    if custom_only:
        return [n for n in names if not _REGISTRY[n].builtin]
    return names


def unregister(name: str) -> None:
    """Remove a strategy from the registry (no error if absent)."""
    _REGISTRY.pop(name, None)


def registry() -> dict[str, StrategyInfo]:
    """Return a shallow copy of the registry mapping (for inspection)."""
    return dict(_REGISTRY)


def _register_builtins() -> None:
    """Register the built-in optimizers. Called once at package import."""
    from jaxfolio import optimizers as _opt

    catalog = {
        "equal_weight": "Equal-weight (1/N) baseline",
        "inverse_volatility": "Inverse-volatility (naive risk parity)",
        "minimum_variance": "Global minimum-variance portfolio",
        "mean_variance": "Markowitz mean-variance",
        "maximum_sharpe": "Maximum-Sharpe (tangency) portfolio",
        "maximum_diversification": "Maximum diversification ratio",
        "risk_parity": "Equal-risk-contribution (ERC)",
        "kelly": "Kelly / log-growth optimal",
        "min_cvar": "Minimum conditional value-at-risk",
        "black_litterman": "Black-Litterman equilibrium + views",
        "deep_sharpe": "Differentiable MLP Sharpe policy",
        "online_gradient": "Online exponentiated-gradient portfolio",
        "hierarchical_risk_parity": "Hierarchical risk parity (HRP)",
        "hierarchical_equal_risk": "Hierarchical equal risk contribution (HERC)",
        "mst_centrality": "Minimum-spanning-tree centrality",
    }
    for key, desc in catalog.items():
        func = getattr(_opt, key, None)
        if func is not None and key not in _REGISTRY:
            _REGISTRY[key] = StrategyInfo(name=key, func=func, description=desc, builtin=True)
