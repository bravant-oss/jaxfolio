# Optimizers

All optimizers share the `method(returns, ...) → PortfolioResult` interface. See
the [Optimizers guide](../guide/optimizers.md) for the mathematics and when to
use each.

## Classical

::: jaxfolio.optimizers.classical
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
