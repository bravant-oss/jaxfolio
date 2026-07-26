# Optimizers

All optimizers share the `method(returns, ...) → PortfolioResult` interface. See
the [Optimizers guide](../guide/optimizers.md) for the mathematics and when to
use each.

## Classical

::: jaxfolio.optimizers.classical
    options:
      heading_level: 3

## Multi-period

Path-aware, cost-aware optimization over a horizon. See the
[multi-period guide](../guide/multiperiod.md) for the cost model and its limits.

::: jaxfolio.optimizers.multiperiod
    options:
      heading_level: 3

## Learning-based

::: jaxfolio.optimizers.learning
    options:
      heading_level: 3

## Graph-based

::: jaxfolio.optimizers.graph
    options:
      heading_level: 3

## Solver core

The shared projected-gradient solver and portfolio-math primitives underlying the
constrained classical methods.

::: jaxfolio.optimizers.base
    options:
      heading_level: 3

## Constraints

Named constraint specifications and the projections that enforce them. See the
[constraints guide](../guide/constraints.md) for the partition requirement, the
feasibility guarantees, and what is not expressible.

::: jaxfolio.constraints.spec
    options:
      heading_level: 3

::: jaxfolio.constraints.compile
    options:
      heading_level: 3

::: jaxfolio.constraints.structured
    options:
      heading_level: 3
