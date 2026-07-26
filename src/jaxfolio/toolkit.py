"""Public building blocks for authoring custom portfolio strategies.

This module is the *stable, documented surface* for writing your own optimizer.
Everything a built-in optimizer uses internally is re-exported here so a custom
strategy can be written the same idiomatic way:

>>> from jaxfolio import toolkit as tk
>>> def my_strategy(returns):
...     mu, cov, names, _ = tk.moments(returns)
...     projection = tk.make_projection(long_only=True, weight_bounds=(0.0, 1.0))
...     def objective(w):
...         return w @ cov @ w - 0.1 * (w @ mu)   # min-variance with a return tilt
...     w, _info = tk.solve_projected_gradient(objective, tk.equal_start(len(names)), projection)
...     return tk.finalize_result(w, names, "My Strategy", mu=mu, cov=cov)

The heavy lifting (annualized diagnostics, weight cleanup) lives in
:func:`finalize_result`, which the built-in classical and graph optimizers also
use, so custom results are indistinguishable from built-in ones downstream
(backtester, ``compare``, plots).
"""

from __future__ import annotations

import jax.numpy as jnp

from jaxfolio.attribution import ConstraintReport, SolverDuals, explain, solver_duals
from jaxfolio.constraints.compile import (
    CompiledConstraints,
    InfeasibleConstraints,
    check_feasible,
    compile_constraints,
)
from jaxfolio.constraints.projections import (
    normalize_weights,
    project_box_budget,
    project_simplex,
    softmax_weights,
)
from jaxfolio.constraints.spec import Box, Budget, Constraint, GroupCap, GroupFloor
from jaxfolio.constraints.structured import (
    ProjectionDuals,
    project_box_budget_duals,
    project_box_budget_vec,
    project_grouped,
    project_grouped_duals,
)
from jaxfolio.moments.estimators import (
    as_matrix,
    correlation_from_covariance,
    ewma_covariance,
    ledoit_wolf_covariance,
    mean_returns,
    sample_covariance,
)
from jaxfolio.optimizers.base import (
    make_projection,
    portfolio_return,
    portfolio_variance,
    portfolio_volatility,
    select_projection,
    sharpe_ratio,
    solve_constrained,
    solve_projected_gradient,
)
from jaxfolio.optimizers.multiperiod import solve_weight_path
from jaxfolio.results import PERIODS_PER_YEAR, equal_start, finalize_result, moments
from jaxfolio.solvers import SolverSpec, available_solvers, resolve_solver
from jaxfolio.types import OptimizerConfig, PortfolioResult, TradingCosts

Array = jnp.ndarray

__all__ = [
    # moment estimation
    "as_matrix",
    "moments",
    "mean_returns",
    "sample_covariance",
    "ewma_covariance",
    "ledoit_wolf_covariance",
    "correlation_from_covariance",
    # constraints / projections
    "make_projection",
    "project_simplex",
    "project_box_budget",
    "project_box_budget_vec",
    "project_grouped",
    # named constraints
    "Box",
    "Budget",
    "Constraint",
    "GroupCap",
    "GroupFloor",
    "CompiledConstraints",
    "InfeasibleConstraints",
    "compile_constraints",
    "check_feasible",
    # explainability
    "explain",
    "solver_duals",
    "ConstraintReport",
    "SolverDuals",
    "ProjectionDuals",
    "project_box_budget_duals",
    "project_grouped_duals",
    "normalize_weights",
    "softmax_weights",
    # solver core
    "solve_projected_gradient",
    "solve_constrained",
    "select_projection",
    "equal_start",
    # multi-period weight paths
    "solve_weight_path",
    # solver selection (any optax optimizer)
    "available_solvers",
    "resolve_solver",
    "SolverSpec",
    # portfolio math
    "portfolio_return",
    "portfolio_variance",
    "portfolio_volatility",
    "sharpe_ratio",
    # result assembly
    "finalize_result",
    "PERIODS_PER_YEAR",
    # re-exported types for convenience
    "PortfolioResult",
    "OptimizerConfig",
    "TradingCosts",
]

# ``moments``, ``equal_start`` and ``finalize_result`` live in
# :mod:`jaxfolio.results` (a dependency-light module) and are re-exported here so
# the toolkit is the single import a strategy author needs.
