"""Dark-themed, publication-quality plots for jaxfolio.

Every function returns a matplotlib ``Figure`` so callers can further customize
or save. The functions cover the portfolio workflow (weights, efficient frontier,
equity curves, drawdowns, risk contributions, correlation network, HRP
dendrogram) and the options workflow (payoff diagrams, Greeks profiles, vol
surface). Colors are drawn from the validated dark palette in :mod:`.theme`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle

from jaxfolio.viz import theme

theme.use_dark_theme()


def _fig(ax_w: float = 9.0, ax_h: float = 5.5) -> tuple[Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(ax_w, ax_h))
    return fig, ax


def _card_layer(fig: Figure) -> plt.Axes:
    """Return (creating once) a full-figure background axes that holds the cards.

    Cards are drawn here, at a low zorder, so the real plot axes added afterwards
    render on top of them (figure-level artists would instead cover the plots).
    """
    bg = getattr(fig, "_jf_card_layer", None)
    if bg is None:
        bg = fig.add_axes([0, 0, 1, 1])
        bg.set_axis_off()
        bg.set_xlim(0, 1)
        bg.set_ylim(0, 1)
        bg.set_zorder(-10)
        bg.patch.set_alpha(0.0)
        fig._jf_card_layer = bg
    return bg


def _card_rect(fig: Figure, spec, *, fill: str | None = None) -> object:
    """Draw a rounded card behind a gridspec cell and return its figure-space bbox."""
    bb = spec.get_position(fig)
    bg = _card_layer(fig)
    bg.add_patch(
        FancyBboxPatch(
            (bb.x0, bb.y0),
            bb.width,
            bb.height,
            boxstyle="round,pad=0,rounding_size=0.014",
            facecolor=fill or theme.CARD,
            edgecolor=theme.BORDER,
            linewidth=1.3,
            mutation_aspect=fig.get_figheight() / fig.get_figwidth(),
        )
    )
    return bb


def _card_axes(
    fig: Figure,
    spec,
    *,
    title: str | None = None,
    subtitle: str | None = None,
    legend: list[tuple[str, str]] | None = None,
    pads=(0.028, 0.020, 0.058, 0.075),
) -> plt.Axes:
    """A plot axes inset inside a titled card.

    ``pads`` is ``(left, right, bottom, top)`` in figure fractions; the top pad
    leaves room for the card header (title + optional legend dots).
    """
    bb = _card_rect(fig, spec)
    lp, rp, bp, tp = pads
    # Header text, aligned to the card's top-left.
    if title:
        fig.text(
            bb.x0 + 0.013,
            bb.y1 - 0.020,
            title,
            fontsize=12,
            fontweight="bold",
            color=theme.INK_PRIMARY,
            ha="left",
            va="top",
            zorder=3,
        )
    if subtitle:
        fig.text(
            bb.x0 + 0.013,
            bb.y1 - 0.040,
            subtitle,
            fontsize=8.5,
            color=theme.INK_MUTED,
            ha="left",
            va="top",
            zorder=3,
        )
    if legend:
        x = bb.x0 + 0.013
        y = bb.y1 - (0.052 if subtitle else 0.040)
        for lbl, col in legend:
            fig.text(x, y, "●", fontsize=8, color=col, ha="left", va="top", zorder=3)
            fig.text(
                x + 0.010,
                y,
                lbl,
                fontsize=8,
                color=theme.INK_SECONDARY,
                ha="left",
                va="top",
                zorder=3,
            )
            x += 0.012 + 0.0075 * len(lbl)
    ax = fig.add_axes([bb.x0 + lp, bb.y0 + bp, bb.width - lp - rp, bb.height - bp - tp])
    ax.set_facecolor(theme.CARD)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def _stat_card(
    fig: Figure, spec, label: str, value: str, accent: str, delta: str | None, good: bool
) -> None:
    """A KPI card: metric label, big value, accent bar, and a 'vs baseline' delta."""
    bb = _card_rect(fig, spec)
    fig.add_artist(
        Rectangle(
            (bb.x0 + 0.014, bb.y1 - 0.026),
            0.040,
            0.005,
            transform=fig.transFigure,
            facecolor=accent,
            edgecolor="none",
            zorder=3,
        )
    )
    fig.text(
        bb.x0 + 0.014,
        bb.y1 - 0.050,
        label.upper(),
        fontsize=8.5,
        fontweight="bold",
        color=theme.INK_MUTED,
        ha="left",
        va="top",
        zorder=3,
    )
    fig.text(
        bb.x0 + 0.014,
        bb.y0 + bb.height * 0.44,
        value,
        fontsize=25,
        fontweight="bold",
        color=theme.INK_PRIMARY,
        ha="left",
        va="center",
        zorder=3,
    )
    if delta:
        arrow = "▲" if good else "▼"
        col = theme.GOOD if good else theme.CRITICAL
        fig.text(
            bb.x0 + 0.014,
            bb.y0 + 0.026,
            f"{arrow} {delta}",
            fontsize=9,
            color=col,
            ha="left",
            va="center",
            zorder=3,
            family="DejaVu Sans",
        )


def plot_weights(result, *, top_n: int = 15, title: str | None = None) -> Figure:
    """Horizontal bar chart of portfolio weights (largest holdings on top)."""
    holdings = result.top(top_n)
    names = list(holdings.keys())[::-1]
    vals = list(holdings.values())[::-1]
    colors = [theme.GOOD if v >= 0 else theme.CRITICAL for v in vals]

    fig, ax = _fig(8, max(3.5, 0.4 * len(names) + 1.2))
    bars = ax.barh(names, vals, color=colors, height=0.68)
    ax.axvline(0, color=theme.BASELINE, linewidth=1.0)
    ax.set_title(title or f"{result.method} — Allocation", pad=12)
    ax.set_xlabel("Weight")
    for bar, v in zip(bars, vals, strict=True):
        ax.text(
            v + (0.005 if v >= 0 else -0.005),
            bar.get_y() + bar.get_height() / 2,
            f"{v:.1%}",
            va="center",
            ha="left" if v >= 0 else "right",
            color=theme.INK_SECONDARY,
            fontsize=9,
        )
    ax.margins(x=0.15)
    fig.tight_layout()
    return fig


def plot_efficient_frontier(
    returns: pd.DataFrame,
    *,
    n_portfolios: int = 4000,
    highlight: dict[str, object] | None = None,
    seed: int = 0,
) -> Figure:
    """Monte-Carlo efficient frontier colored by Sharpe ratio.

    Simulates random long-only portfolios and plots them in risk/return space,
    colored by Sharpe. Optionally overlays named :class:`PortfolioResult` markers
    passed via ``highlight`` (``{label: result}``).
    """
    from jaxfolio.moments.estimators import as_matrix, mean_returns, sample_covariance

    mat, _ = as_matrix(returns)
    mu = np.asarray(mean_returns(mat)) * 252
    cov = np.asarray(sample_covariance(mat)) * 252
    n = mu.shape[0]

    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.ones(n), size=n_portfolios)
    rets = w @ mu
    vols = np.sqrt(np.einsum("ij,jk,ik->i", w, cov, w))
    sharpes = rets / vols

    fig, ax = _fig(9, 6)
    sc = ax.scatter(
        vols, rets, c=sharpes, cmap=theme.SEQUENTIAL_CMAP, s=8, alpha=0.65, edgecolors="none"
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Sharpe ratio", color=theme.INK_SECONDARY)
    cbar.ax.yaxis.set_tick_params(color=theme.INK_MUTED)

    if highlight:
        for i, (label, res) in enumerate(highlight.items()):
            if res.volatility is None or res.expected_return is None:
                continue
            ax.scatter(
                res.volatility,
                res.expected_return,
                marker="*",
                s=340,
                color=theme.color(i),
                edgecolors=theme.INK_PRIMARY,
                linewidths=1.0,
                zorder=5,
                label=label,
            )
        ax.legend(loc="lower right")

    ax.set_title("Efficient Frontier — Random Portfolios", pad=12)
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized return")
    fig.tight_layout()
    return fig


def plot_equity_curves(results: dict, *, log_scale: bool = False) -> Figure:
    """Overlay cumulative equity curves for several backtest results.

    ``results`` maps ``name -> BacktestResult``. Each curve is direct-labeled at
    its right end so identity never relies on color alone.
    """
    fig, ax = _fig(10, 5.5)
    for i, (name, res) in enumerate(results.items()):
        curve = res.equity_curve
        c = theme.color(i)
        ax.plot(curve.index, curve.values, color=c, label=name)
        ax.text(
            curve.index[-1],
            curve.values[-1],
            f"  {name}",
            color=c,
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    if log_scale:
        ax.set_yscale("log")
    ax.set_title("Cumulative Growth of $1", pad=12)
    ax.set_ylabel("Portfolio value")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_drawdown(results: dict) -> Figure:
    """Underwater (drawdown) plot for one or more backtest results."""
    fig, ax = _fig(10, 4.0)
    for i, (name, res) in enumerate(results.items()):
        dd = res.drawdown
        ax.fill_between(dd.index, dd.values, 0.0, color=theme.color(i), alpha=0.25)
        ax.plot(dd.index, dd.values, color=theme.color(i), label=name, linewidth=1.5)
    ax.set_title("Drawdown", pad=12)
    ax.set_ylabel("Drawdown")
    ax.legend(loc="lower left")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_weight_evolution(result_or_weights) -> Figure:
    """Stacked-area chart of weight evolution over a backtest.

    Accepts a :class:`BacktestResult` or a weights DataFrame (dates x assets).
    """
    weights = getattr(result_or_weights, "weights", result_or_weights)
    weights = weights.clip(lower=0.0)  # stacked area needs non-negative bands

    fig, ax = _fig(10, 5.5)
    cols = weights.columns
    colors = [theme.color(i) for i in range(len(cols))]
    ax.stackplot(
        weights.index,
        *[weights[c].to_numpy() for c in cols],
        labels=list(cols),
        colors=colors,
        edgecolor=theme.SURFACE,
        linewidth=0.3,
    )
    ax.set_title("Weight Evolution", pad=12)
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_risk_contributions(result) -> Figure:
    """Bar chart of each asset's risk contribution (needs ERC metadata or recompute)."""
    rc = result.metadata.get("risk_contributions")
    if rc is None:
        # Fall back to weight * marginal not available; show weights instead.
        rc = result.weights.tolist()
        ylabel = "Weight (risk contribution unavailable)"
    else:
        ylabel = "Risk contribution"

    order = np.argsort(rc)[::-1]
    names = [result.assets[i] for i in order]
    vals = [rc[i] for i in order]

    fig, ax = _fig(9, 4.5)
    ax.bar(names, vals, color=theme.CATEGORICAL[0], width=0.7)
    ax.axhline(
        np.mean(vals), color=theme.WARNING, linestyle="--", linewidth=1.2, label="equal target"
    )
    ax.set_title(f"{result.method} — Risk Contributions", pad=12)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_correlation_network(returns: pd.DataFrame, *, threshold: float = 0.4) -> Figure:
    """Force-directed-style correlation network via the minimum spanning tree.

    Nodes are assets placed on a circle; edges are the MST of the correlation
    distance, plus any strong pairwise correlations above ``threshold``. Node
    size encodes degree centrality.
    """
    from scipy.sparse.csgraph import minimum_spanning_tree

    from jaxfolio.moments.estimators import (
        as_matrix,
        correlation_from_covariance,
        sample_covariance,
    )

    mat, names = as_matrix(returns)
    corr = np.asarray(correlation_from_covariance(sample_covariance(mat)))
    n = len(names)
    dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, 1))
    np.fill_diagonal(dist, 0)
    mst = minimum_spanning_tree(dist).toarray()
    adj = (mst + mst.T) > 0

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = np.column_stack([np.cos(angles), np.sin(angles)])
    degree = adj.sum(axis=1)

    fig, ax = _fig(7.5, 7.5)
    # MST edges.
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                ax.plot(
                    [pos[i, 0], pos[j, 0]],
                    [pos[i, 1], pos[j, 1]],
                    color=theme.INK_MUTED,
                    linewidth=1.2,
                    alpha=0.7,
                    zorder=1,
                )
    # Strong extra correlations (dashed accent).
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i, j] and corr[i, j] > threshold:
                ax.plot(
                    [pos[i, 0], pos[j, 0]],
                    [pos[i, 1], pos[j, 1]],
                    color=theme.CATEGORICAL[0],
                    linewidth=0.8,
                    alpha=0.35,
                    linestyle="--",
                    zorder=1,
                )

    sizes = 200 + 500 * (degree / max(degree.max(), 1))
    ax.scatter(
        pos[:, 0],
        pos[:, 1],
        s=sizes,
        c=degree,
        cmap=theme.SEQUENTIAL_CMAP,
        edgecolors=theme.INK_PRIMARY,
        linewidths=1.0,
        zorder=2,
    )
    for i, name in enumerate(names):
        ax.text(
            pos[i, 0] * 1.15,
            pos[i, 1] * 1.15,
            name,
            ha="center",
            va="center",
            fontsize=8,
            color=theme.INK_SECONDARY,
        )
    ax.set_title("Correlation Network (MST)", pad=12)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_dendrogram(result) -> Figure:
    """Plot the HRP linkage dendrogram from a hierarchical_risk_parity result."""
    from scipy.cluster.hierarchy import dendrogram

    link = result.metadata.get("linkage")
    if link is None:
        raise ValueError("result has no 'linkage' metadata; use hierarchical_risk_parity")
    link = np.asarray(link)

    fig, ax = _fig(9, 5)
    with plt.rc_context({"lines.linewidth": 1.5}):
        dendrogram(
            link,
            labels=result.assets,
            ax=ax,
            color_threshold=0,
            above_threshold_color=theme.CATEGORICAL[0],
        )
    ax.set_title("HRP Cluster Dendrogram", pad=12)
    ax.set_ylabel("Correlation distance")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(returns: pd.DataFrame, *, order: list[int] | None = None) -> Figure:
    """Clustered correlation heatmap (diverging blue↔red), optionally reordered."""
    from jaxfolio.moments.estimators import (
        as_matrix,
        correlation_from_covariance,
        sample_covariance,
    )

    mat, names = as_matrix(returns)
    corr = np.asarray(correlation_from_covariance(sample_covariance(mat)))
    if order is not None:
        corr = corr[np.ix_(order, order)]
        names = [names[i] for i in order]

    fig, ax = _fig(7.5, 6.5)
    im = ax.imshow(corr, cmap=theme.DIVERGING_CMAP, vmin=-1, vmax=1, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Correlation", color=theme.INK_SECONDARY)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title("Asset Correlation Matrix", pad=12)
    fig.tight_layout()
    return fig


def plot_metrics_table(metrics_df: pd.DataFrame) -> Figure:
    """Render a metrics comparison table as a styled dark figure."""
    fmt = metrics_df.copy()
    pct_cols = [c for c in fmt.columns if c not in ("sharpe", "sortino", "calmar")]
    for c in fmt.columns:
        if c in pct_cols:
            fmt[c] = fmt[c].map(lambda v: f"{v:.1%}")
        else:
            fmt[c] = fmt[c].map(lambda v: f"{v:.2f}")

    fig, ax = _fig(min(2 + 1.4 * len(fmt.columns), 16), 1.0 + 0.5 * len(fmt))
    ax.axis("off")
    tbl = ax.table(
        cellText=fmt.values,
        rowLabels=fmt.index,
        colLabels=fmt.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.5)
    for (row, _col), cell in tbl.get_celld().items():
        cell.set_edgecolor(theme.GRID)
        cell.set_facecolor(theme.SURFACE if row > 0 else theme.PLANE)
        cell.set_text_props(color=theme.INK_SECONDARY if row > 0 else theme.INK_PRIMARY)
    ax.set_title("Backtest Metrics", pad=16, color=theme.INK_PRIMARY)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Options plots
# --------------------------------------------------------------------------- #
def plot_payoff(strategy, *, spot: float | None = None, spread: float = 0.4) -> Figure:
    """Payoff-at-expiry diagram for an :class:`OptionStrategy`.

    Profit region is shaded green, loss region red; break-even points and the
    reference spot are annotated.
    """
    strikes = [leg.strike for leg in strategy.legs]
    center = spot or float(np.mean(strikes))
    grid = np.linspace(center * (1 - spread), center * (1 + spread), 500)
    pnl = strategy.payoff_at_expiry(grid)

    fig, ax = _fig(9, 5)
    ax.axhline(0, color=theme.BASELINE, linewidth=1.0)
    ax.plot(grid, pnl, color=theme.INK_PRIMARY, linewidth=2.2, zorder=3)
    ax.fill_between(grid, pnl, 0, where=pnl >= 0, color=theme.GOOD, alpha=0.25)
    ax.fill_between(grid, pnl, 0, where=pnl < 0, color=theme.CRITICAL, alpha=0.25)

    for be in strategy.break_evens(grid):
        ax.axvline(be, color=theme.WARNING, linestyle="--", linewidth=1.0, alpha=0.8)
        ax.text(
            be,
            ax.get_ylim()[1] * 0.92,
            f"BE {be:.1f}",
            color=theme.WARNING,
            fontsize=8,
            ha="center",
        )
    if spot:
        ax.axvline(spot, color=theme.CATEGORICAL[0], linewidth=1.2, alpha=0.7)
        ax.text(
            spot,
            ax.get_ylim()[0] * 0.9,
            f"spot {spot:.0f}",
            color=theme.CATEGORICAL[0],
            fontsize=8,
            ha="center",
        )

    ax.set_title(f"{strategy.name} — Payoff at Expiry", pad=12)
    ax.set_xlabel("Underlying price at expiry")
    ax.set_ylabel("Profit / Loss")
    fig.tight_layout()
    return fig


def plot_greeks_profile(
    strategy,
    *,
    spot: float,
    vol: float = 0.25,
    rate: float = 0.0,
    spread: float = 0.4,
    greeks: tuple[str, ...] = ("delta", "gamma", "vega", "theta"),
) -> Figure:
    """Plot net position Greeks as a function of the underlying spot."""
    grid = np.linspace(spot * (1 - spread), spot * (1 + spread), 120)
    curves = {g: [] for g in greeks}
    for s in grid:
        agg = strategy.greeks(float(s), vol, rate)
        for g in greeks:
            curves[g].append(agg[g])

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.ravel()
    for i, g in enumerate(greeks):
        ax = axes[i]
        ax.axhline(0, color=theme.BASELINE, linewidth=0.8)
        ax.axvline(spot, color=theme.INK_MUTED, linewidth=0.8, linestyle=":")
        ax.plot(grid, curves[g], color=theme.color(i), linewidth=2.0)
        ax.set_title(g.capitalize(), color=theme.INK_PRIMARY, fontsize=11)
        ax.set_xlabel("Spot")
    fig.suptitle(f"{strategy.name} — Greeks Profile", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_vol_surface(
    spot: float,
    strikes: np.ndarray,
    expiries: np.ndarray,
    ivs: np.ndarray,
) -> Figure:
    """Implied-volatility surface as a filled contour (strike x expiry -> IV).

    ``ivs`` is a 2-D grid indexed ``[expiry, strike]``.
    """
    fig, ax = _fig(9, 6)
    K, T = np.meshgrid(strikes, expiries)
    cf = ax.contourf(K, T, ivs, levels=18, cmap=theme.SEQUENTIAL_CMAP)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Implied volatility", color=theme.INK_SECONDARY)
    ax.axvline(spot, color=theme.WARNING, linestyle="--", linewidth=1.2, label="spot")
    ax.set_title("Implied Volatility Surface", pad=12)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Time to expiry (years)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def dashboard(results: dict, returns: pd.DataFrame, *, highlight: dict | None = None) -> Figure:
    """Composite strategy-comparison report: KPI strip, equity, drawdown, frontier, table.

    ``results`` maps ``name -> BacktestResult``; ``highlight`` maps
    ``name -> PortfolioResult`` for the frontier overlay. The layout is a clean
    reporting grid — a headline KPI row for the best strategy (by Sharpe), the
    equity and drawdown panels, and the efficient frontier beside a ranked
    risk-adjusted (Sharpe) bar chart.
    """
    from jaxfolio.moments.estimators import as_matrix, mean_returns, sample_covariance

    names = list(results.keys())
    color_of = {name: theme.color(i) for i, name in enumerate(names)}
    best = max(names, key=lambda n: results[n].metrics.get("sharpe", float("-inf")))
    bm = results[best].metrics

    # Baseline for the KPI deltas: a 1/N-style equal-weight strategy if present,
    # otherwise the cross-strategy median of each metric.
    base_name = next((n for n in names if n in ("1/N", "Equal Weight", "Equal-Weight")), None)

    def _baseline(metric: str) -> float:
        if base_name:
            return results[base_name].metrics[metric]
        return float(np.median([results[n].metrics[metric] for n in names]))

    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor(theme.PLANE)
    gs = fig.add_gridspec(
        4,
        4,
        height_ratios=[0.30, 0.60, 1.0, 1.0],
        hspace=0.40,
        wspace=0.16,
        left=0.028,
        right=0.985,
        top=0.965,
        bottom=0.045,
    )

    # --- Header card --------------------------------------------------------- #
    hb = _card_rect(fig, gs[0, :])
    fig.text(
        hb.x0 + 0.016,
        hb.y0 + hb.height * 0.62,
        "Strategy Comparison",
        fontsize=20,
        fontweight="bold",
        color=theme.INK_PRIMARY,
        ha="left",
        va="center",
        zorder=3,
    )
    fig.text(
        hb.x0 + 0.016,
        hb.y0 + hb.height * 0.26,
        f"Walk-forward backtest · net of transaction costs · headline = {best} (best Sharpe)",
        fontsize=10.5,
        color=theme.INK_MUTED,
        ha="left",
        va="center",
        zorder=3,
    )
    fig.text(
        hb.x1 - 0.016,
        hb.y0 + hb.height * 0.5,
        "jaxfolio",
        fontsize=14,
        fontweight="bold",
        color=theme.INK_SECONDARY,
        ha="right",
        va="center",
        zorder=3,
    )

    # --- KPI cards (best strategy, with deltas vs baseline) ------------------ #
    d_ret = bm["annual_return"] - _baseline("annual_return")
    d_vol = bm["annual_volatility"] - _baseline("annual_volatility")
    d_shp = bm["sharpe"] - _baseline("sharpe")
    d_dd = bm["max_drawdown"] - _baseline("max_drawdown")
    vs = f"vs {base_name}" if base_name else "vs median"
    ret_accent = theme.GOOD if bm["annual_return"] >= 0 else theme.CRITICAL
    cards = [
        ("Ann. return", f"{bm['annual_return']:.1%}", ret_accent, f"{d_ret:+.1%} {vs}", d_ret >= 0),
        (
            "Ann. volatility",
            f"{bm['annual_volatility']:.1%}",
            theme.CATEGORICAL[0],
            f"{d_vol:+.1%} {vs}",
            d_vol <= 0,
        ),
        ("Sharpe", f"{bm['sharpe']:.2f}", theme.CATEGORICAL[6], f"{d_shp:+.2f} {vs}", d_shp >= 0),
        (
            "Max drawdown",
            f"{bm['max_drawdown']:.1%}",
            theme.WARNING,
            f"{d_dd:+.1%} {vs}",
            d_dd >= 0,
        ),
    ]
    for col, (label, value, accent, delta, good) in enumerate(cards):
        _stat_card(fig, gs[1, col], label, value, accent, delta, good)

    # --- Equity curves card (direct-labeled, de-collided) -------------------- #
    ax1 = _card_axes(
        fig,
        gs[2, 0:2],
        title="Cumulative growth of $1",
        subtitle="Value of $1 invested at inception",
        pads=(0.028, 0.090, 0.058, 0.075),  # extra right pad reserves room for end-labels
    )
    ends = []
    for name in names:
        curve = results[name].equity_curve
        ax1.plot(curve.index, curve.values, color=color_of[name], linewidth=2.0)
        ends.append((float(curve.values[-1]), name, curve.index[-1]))
    ax1.axhline(1.0, color=theme.BASELINE, linewidth=1.0, zorder=0)
    ax1.margins(x=0.03)
    ax1.set_ylabel("Portfolio value")
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    # Spread the end-labels vertically so overlapping endpoints stay legible.
    y0, y1 = ax1.get_ylim()
    label_x = ax1.get_xlim()[1]
    gap = 0.052 * (y1 - y0)
    ends.sort(key=lambda e: e[0])
    ys = [e[0] for e in ends]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < gap:
            ys[i] = ys[i - 1] + gap
    for (_orig, name, _x), y in zip(ends, ys, strict=True):
        ax1.text(
            label_x,
            y,
            f" {name}",
            color=color_of[name],
            va="center",
            fontsize=8.5,
            fontweight="bold",
            clip_on=False,
        )

    # --- Drawdown card ------------------------------------------------------- #
    ax2 = _card_axes(
        fig, gs[2, 2:4], title="Drawdown", subtitle=f"Underwater curve · {best} emphasized"
    )
    for name in names:
        dd = results[name].drawdown
        emph = name == best
        ax2.plot(
            dd.index,
            dd.values,
            color=color_of[name],
            linewidth=2.0 if emph else 1.1,
            alpha=1.0 if emph else 0.6,
            zorder=3 if emph else 2,
        )
    ax2.fill_between(
        results[best].drawdown.index,
        results[best].drawdown.values,
        0,
        color=color_of[best],
        alpha=0.18,
        zorder=1,
    )
    ax2.axhline(0.0, color=theme.BASELINE, linewidth=1.0)
    ax2.set_ylabel("Drawdown")

    # --- Efficient frontier card --------------------------------------------- #
    ax3 = _card_axes(
        fig,
        gs[3, 0:2],
        title="Efficient frontier",
        subtitle="Random long-only portfolios, colored by Sharpe",
        pads=(0.028, 0.052, 0.058, 0.075),
    )
    mat, _ = as_matrix(returns)
    mu = np.asarray(mean_returns(mat)) * 252
    cov = np.asarray(sample_covariance(mat)) * 252
    rng = np.random.default_rng(0)
    w = rng.dirichlet(np.ones(mu.shape[0]), size=3000)
    rr = w @ mu
    vv = np.sqrt(np.einsum("ij,jk,ik->i", w, cov, w))
    sc = ax3.scatter(
        vv, rr, c=rr / vv, cmap=theme.SEQUENTIAL_CMAP, s=7, alpha=0.55, edgecolors="none"
    )
    if highlight:
        for i, (label, res) in enumerate(highlight.items()):
            if res.volatility and res.expected_return:
                ax3.scatter(
                    res.volatility,
                    res.expected_return,
                    marker="*",
                    s=300,
                    color=theme.color(i),
                    edgecolors=theme.INK_PRIMARY,
                    linewidths=1.0,
                    zorder=5,
                )
                ax3.annotate(
                    f" {label}",
                    (res.volatility, res.expected_return),
                    color=theme.INK_SECONDARY,
                    fontsize=8.5,
                    fontweight="bold",
                    xytext=(7, 0),
                    textcoords="offset points",
                    va="center",
                )
    cbar = fig.colorbar(sc, ax=ax3, pad=0.02, fraction=0.045)
    cbar.set_label("Sharpe", color=theme.INK_SECONDARY, fontsize=9)
    cbar.outline.set_edgecolor(theme.BORDER)
    cbar.ax.tick_params(labelsize=8)
    cbar.locator = plt.MaxNLocator(5)
    cbar.update_ticks()
    ax3.set_xlabel("Annualized volatility")
    ax3.set_ylabel("Annualized return")

    # --- Risk-adjusted ranking card (horizontal Sharpe bars) ----------------- #
    ax4 = _card_axes(
        fig,
        gs[3, 2:4],
        title="Risk-adjusted ranking",
        subtitle="Sharpe ratio by strategy · annualized return labelled",
        pads=(0.030, 0.028, 0.058, 0.090),
    )
    ranked = sorted(names, key=lambda n: results[n].metrics["sharpe"])  # best ends on top
    sharpes = [results[n].metrics["sharpe"] for n in ranked]
    rets = [results[n].metrics["annual_return"] for n in ranked]
    smax = max(sharpes + [0.0])
    smin = min(sharpes + [0.0])
    span = max(smax - smin, 1e-6)
    # Reserve a left band (left of the zero axis) for the strategy name labels.
    name_band = span * 1.05
    x_left = smin - name_band
    ax4.set_xlim(x_left, smax + span * 0.16)
    ax4.set_ylim(-0.7, len(ranked) - 0.3)

    for i, name in enumerate(ranked):
        emph = name == best
        ax4.barh(
            i,
            sharpes[i],
            height=0.62,
            color=color_of[name],
            alpha=1.0 if emph else 0.82,
            zorder=3,
            edgecolor=theme.INK_PRIMARY if emph else "none",
            linewidth=1.2 if emph else 0.0,
        )
        ax4.text(
            x_left + name_band * 0.03,
            i,
            name,
            color=color_of[name],
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
            zorder=4,
        )
        off = span * 0.02
        ax4.text(
            sharpes[i] + (off if sharpes[i] >= 0 else -off),
            i,
            f"{sharpes[i]:.2f}  ·  {rets[i]:+.1%}",
            va="center",
            ha="left" if sharpes[i] >= 0 else "right",
            color=theme.INK_SECONDARY,
            fontsize=8.5,
            zorder=4,
        )

    ax4.axvline(0, color=theme.BASELINE, linewidth=1.2, zorder=2)
    ax4.set_yticks([])
    ax4.grid(axis="x", color=theme.GRID, linewidth=0.7, alpha=0.7)
    ax4.set_axisbelow(True)
    ax4.tick_params(axis="x", labelsize=8)
    ax4.set_xlabel("Sharpe ratio")
    # Only show gridlines/ticks in the bar region (>= 0), not the name band.
    ax4.set_xticks([t for t in ax4.get_xticks() if t >= 0 and t <= smax + span * 0.16])

    return fig


def save(fig: Figure, path: str, *, dpi: int = 150) -> None:
    """Save a figure with the dark background preserved."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=theme.PLANE)
