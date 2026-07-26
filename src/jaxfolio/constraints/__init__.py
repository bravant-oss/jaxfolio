"""Constraints on portfolio weights — as projections, and as named specifications.

Two layers live here:

* **Projections** (:mod:`~jaxfolio.constraints.projections`,
  :mod:`~jaxfolio.constraints.structured`) — pure ``w -> w`` maps onto a feasible
  set, which is how the projected-gradient solvers enforce constraints.
* **Specifications** (:mod:`~jaxfolio.constraints.spec`) — *named* constraints
  such as ``GroupCap("tech", [...], max=0.30)``, compiled to a projection by
  :func:`~jaxfolio.constraints.compile.compile_constraints`.

The naming is the point: a projection can only say "here is a feasible weight
vector", whereas a named constraint carries an identity, so its Lagrange
multiplier can be reported as *the shadow price of ``"tech"``* and a zero weight
can be attributed to the constraint that caused it.
"""

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
from jaxfolio.constraints.spec import (
    Box,
    Budget,
    Constraint,
    GroupCap,
    GroupFloor,
    RowConstraint,
)
from jaxfolio.constraints.structured import (
    ProjectionDuals,
    project_box_budget_duals,
    project_box_budget_vec,
    project_grouped,
    project_grouped_duals,
)

__all__ = [
    # specifications
    "Box",
    "Budget",
    "Constraint",
    "GroupCap",
    "GroupFloor",
    "RowConstraint",
    # compilation
    "CompiledConstraints",
    "InfeasibleConstraints",
    "check_feasible",
    "compile_constraints",
    # projections
    "project_simplex",
    "project_box_budget",
    "project_box_budget_vec",
    "project_grouped",
    "normalize_weights",
    "softmax_weights",
    # projections with multipliers
    "ProjectionDuals",
    "project_box_budget_duals",
    "project_grouped_duals",
]
