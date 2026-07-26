"""Dark-themed visualizations."""

from jaxfolio.viz import theme
from jaxfolio.viz.plots import (
    dashboard,
    multiperiod_dashboard,
    plot_correlation_heatmap,
    plot_correlation_network,
    plot_cost_comparison,
    plot_dendrogram,
    plot_drawdown,
    plot_efficient_frontier,
    plot_equity_curves,
    plot_greeks_profile,
    plot_metrics_table,
    plot_path_convergence,
    plot_payoff,
    plot_risk_contributions,
    plot_turnover_schedule,
    plot_vol_surface,
    plot_weight_evolution,
    plot_weight_path,
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
    # multi-period
    "plot_weight_path",
    "plot_turnover_schedule",
    "plot_path_convergence",
    "plot_cost_comparison",
    "multiperiod_dashboard",
    # options
    "plot_payoff",
    "plot_greeks_profile",
    "plot_vol_surface",
    "dashboard",
    "save",
]
