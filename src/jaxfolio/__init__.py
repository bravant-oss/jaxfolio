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

from jaxfolio.attribution import ConstraintReport, explain
from jaxfolio.constraints import Box, Budget, GroupCap, GroupFloor, InfeasibleConstraints
from jaxfolio.custom import CustomStrategy, custom_strategy
from jaxfolio.data import (
    asset_columns,
    date_column,
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
    multi_period_mean_variance,
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
from jaxfolio.solvers import available_solvers
from jaxfolio.types import OptimizerConfig, PortfolioResult, TradingCosts

# Register the built-in optimizers so they show up in the strategy registry
# alongside any user-defined strategies.
_register_builtins()

__version__ = "0.1.1"

__all__ = [
    "__version__",
    # types
    "PortfolioResult",
    "OptimizerConfig",
    # named constraints
    "Box",
    "Budget",
    "GroupCap",
    "GroupFloor",
    "InfeasibleConstraints",
    # explainability
    "explain",
    "ConstraintReport",
    "TradingCosts",
    "available_solvers",
    # registry & custom strategies
    "register_strategy",
    "get_strategy",
    "list_strategies",
    "strategy_info",
    "unregister",
    "custom_strategy",
    "CustomStrategy",
    # data
    "asset_columns",
    "date_column",
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
    # multi-period optimizers
    "multi_period_mean_variance",
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
