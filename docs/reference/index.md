# API reference

This reference is generated directly from the source docstrings, so it always
matches the installed version. It is organized by subsystem; the top-level
`jaxfolio` namespace re-exports the most common entry points.

<div class="jf-grid" markdown>

<div class="jf-card" markdown>
### [Optimizers](optimizers.md)
Classical, learning, and graph methods, plus the shared solver.
</div>

<div class="jf-card" markdown>
### [Backtest](backtest.md)
The walk-forward engine, `compare`, and the metric suite.
</div>

<div class="jf-card" markdown>
### [Options](options.md)
Pricing, autodiff Greeks, multi-leg strategies, and overlays.
</div>

<div class="jf-card" markdown>
### [LLM](llm.md)
Local-model clients and the view-generation strategies.
</div>

<div class="jf-card" markdown>
### [Toolkit &amp; custom](toolkit.md)
Building blocks for authoring your own strategies.
</div>

<div class="jf-card" markdown>
### [Data &amp; moments](data.md)
Synthetic data, loaders, and covariance estimators.
</div>

<div class="jf-card" markdown>
### [Visualization](viz.md)
Plotting functions and the dark theme.
</div>

<div class="jf-card" markdown>
### [Types &amp; registry](types.md)
`PortfolioResult`, `OptimizerConfig`, and the registry.
</div>

</div>

## Top-level namespace

The following are importable directly as `jaxfolio.<name>` (aliased `jf`):

```python
import jaxfolio as jf
```

::: jaxfolio
    options:
      members: false
      show_root_heading: false
      show_source: false
