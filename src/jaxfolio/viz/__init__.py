"""Dark-themed visualizations."""

from jaxfolio.viz import theme
from jaxfolio.viz.plots import (
    dashboard,
    plot_correlation_heatmap,
    plot_correlation_network,
    plot_dendrogram,
    plot_drawdown,
    plot_efficient_frontier,
    plot_equity_curves,
    plot_greeks_profile,
    plot_metrics_table,
    plot_payoff,
    plot_risk_contributions,
    plot_vol_surface,
    plot_weight_evolution,
    plot_weights,
    save,
)

__all__ = [
    "theme",
    "plot_weights",
    "plot_efficient_frontier",
    "plot_equity_curves",
    "plot_drawdown",
    "plot_weight_evolution",
    "plot_risk_contributions",
    "plot_correlation_network",
    "plot_correlation_heatmap",
    "plot_dendrogram",
    "plot_metrics_table",
    "plot_payoff",
    "plot_greeks_profile",
    "plot_vol_surface",
    "dashboard",
    "save",
]
