"""jaxfolio — portfolio optimization and options strategies in JAX.

A modern, JAX-powered library covering traditional, learning-based, and
graph-based portfolio optimizers, a differentiable options toolkit, a
walk-forward backtester, and dark-themed visualizations.

Quickstart
----------
>>> import jaxfolio as jf
>>> returns = jf.generate_returns(n_assets=8, seed=0)
>>> result = jf.maximum_sharpe(returns)
>>> result.top(3)
"""

from __future__ import annotations

from jaxfolio.custom import CustomStrategy, custom_strategy
from jaxfolio.data import (
    generate_prices,
    generate_returns,
    load_csv,
    load_option_chain,
    load_parquet,
    load_yfinance,
    to_returns,
    train_test_split,
)
from jaxfolio.llm import (
    llm_agent_portfolio,
    llm_black_litterman,
    llm_sentiment_portfolio,
)
from jaxfolio.optimizers import (
    black_litterman,
    deep_sharpe,
    equal_weight,
    hierarchical_equal_risk,
    hierarchical_risk_parity,
    inverse_volatility,
    kelly,
    maximum_diversification,
    maximum_sharpe,
    mean_variance,
    min_cvar,
    minimum_variance,
    mst_centrality,
    online_gradient,
    risk_parity,
)
from jaxfolio.registry import (
    _register_builtins,
    get_strategy,
    list_strategies,
    register_strategy,
    strategy_info,
    unregister,
)
from jaxfolio.types import OptimizerConfig, PortfolioResult

# Register the built-in optimizers so they show up in the strategy registry
# alongside any user-defined strategies.
_register_builtins()

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # types
    "PortfolioResult",
    "OptimizerConfig",
    # registry & custom strategies
    "register_strategy",
    "get_strategy",
    "list_strategies",
    "strategy_info",
    "unregister",
    "custom_strategy",
    "CustomStrategy",
    # data
    "generate_prices",
    "generate_returns",
    "load_csv",
    "load_parquet",
    "load_yfinance",
    "load_option_chain",
    "to_returns",
    "train_test_split",
    # classical optimizers
    "equal_weight",
    "inverse_volatility",
    "minimum_variance",
    "mean_variance",
    "maximum_sharpe",
    "maximum_diversification",
    "risk_parity",
    "kelly",
    "min_cvar",
    "black_litterman",
    # learning optimizers
    "deep_sharpe",
    "online_gradient",
    # graph optimizers
    "hierarchical_risk_parity",
    "hierarchical_equal_risk",
    "mst_centrality",
    # LLM (local-model) strategies
    "llm_black_litterman",
    "llm_sentiment_portfolio",
    "llm_agent_portfolio",
]
