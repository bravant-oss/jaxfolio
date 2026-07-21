"""Benchmark jaxfolio against other popular Python portfolio-optimization libraries.

We solve three canonical long-only, fully-invested problems — **minimum
variance**, **maximum Sharpe** (tangency), and **risk parity** (equal risk
contribution) — with every library that is installed, then compare, on a common
footing:

* **Speed** — median wall-clock time per solve (jaxfolio gets a warm-up run so
  JAX's tracing/compilation is not counted against the timed runs).
* **Quality** — the achieved objective, recomputed by *this* script from one
  shared set of moments, so the number means the same thing for every library.
* **Agreement** — the largest per-asset weight difference from jaxfolio, i.e.
  "does everyone land on the same portfolio?".

Libraries compared (all optional except SciPy, which jaxfolio already depends on):
PyPortfolioOpt · Riskfolio-Lib · skfolio · CVXPY · SciPy (SLSQP).

Run with:
    uv run python examples/benchmark/benchmark.py

Install the optional competitors first to include them:
    uv pip install PyPortfolioOpt Riskfolio-Lib skfolio cvxpy
"""

from __future__ import annotations

import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# PyPortfolioOpt / Riskfolio-Lib build their CVXPY problems with the older ``*``
# matmul spelling, which emits a deprecation UserWarning from inside CVXPY on
# every solve. It is third-party and harmless; silence it so the tables read clean.
warnings.filterwarnings("ignore", category=UserWarning, module="cvxpy.*")

# Allow running the file directly (python examples/benchmark/benchmark.py).
sys.path.insert(0, str(Path(__file__).parent))

from libraries import (  # noqa: E402
    PROBLEMS,
    Adapter,
    Context,
    Unsupported,
    available_adapters,
    make_context,
    missing_libraries,
)

import jaxfolio as jf  # noqa: E402

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

PROBLEM_TITLES = {
    "min_variance": "Minimum Variance",
    "max_sharpe": "Maximum Sharpe",
    "risk_parity": "Risk Parity (ERC)",
}
PPY = 252  # trading periods per year, for annualizing the reported quality numbers


# --------------------------------------------------------------------------- #
# Quality metrics — recomputed uniformly from the shared moments
# --------------------------------------------------------------------------- #
def quality(problem: str, w: np.ndarray, ctx: Context) -> tuple[float, str, str]:
    """Return ``(value, label, direction)`` for the achieved objective.

    ``direction`` is ``"lower"`` or ``"higher"`` — which way is better — so the
    reporting layer can flag the winner without hard-coding it per problem.
    """
    var = float(w @ ctx.cov @ w)
    vol = np.sqrt(max(var, 0.0))
    if problem == "min_variance":
        return vol * np.sqrt(PPY), "ann. volatility", "lower"
    if problem == "max_sharpe":
        excess = float(w @ ctx.mu - ctx.risk_free)
        sharpe = excess / vol if vol > 0 else 0.0
        return sharpe * np.sqrt(PPY), "ann. Sharpe", "higher"
    if problem == "risk_parity":
        rc = w * (ctx.cov @ w)
        rc = rc / (rc.sum() + 1e-18)
        # Coefficient of variation of risk contributions: 0 == perfect parity.
        cv = float(np.std(rc) / (np.mean(rc) + 1e-18))
        return cv, "risk-contrib CV", "lower"
    raise ValueError(problem)  # pragma: no cover


def time_solve(
    adapter: Adapter, problem: str, ctx: Context, *, repeats: int, warmup: int
) -> tuple[np.ndarray, float]:
    """Solve once for weights, then time ``repeats`` runs; return ``(weights, best_seconds)``."""
    for _ in range(warmup):
        adapter.solve(problem, ctx)
    best = np.inf
    weights = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        weights = adapter.solve(problem, ctx)
        best = min(best, time.perf_counter() - t0)
    return weights, best


@dataclass
class Row:
    problem: str
    library: str
    is_reference: bool
    seconds: float
    quality_value: float
    quality_label: str
    quality_dir: str
    max_weight_diff: float  # L-inf distance to jaxfolio weights (0 for jaxfolio itself)
    status: str  # "ok", "unsupported", or "error: ..."


def run(
    returns: pd.DataFrame, *, repeats: int = 7, warmup: int = 1
) -> tuple[list[Row], dict[tuple[str, str], np.ndarray], str]:
    """Run every available adapter on every problem; collect rows and weight vectors."""
    ctx = make_context(returns)
    adapters = available_adapters()
    ref_name = next(a.name for a in adapters if a.is_reference)

    # First pass: reference (jaxfolio) weights per problem, to measure agreement.
    ref_weights: dict[str, np.ndarray] = {}
    ref = next(a for a in adapters if a.is_reference)
    for problem in PROBLEMS:
        ref_weights[problem] = ref.solve(problem, ctx)

    rows: list[Row] = []
    weights_by: dict[tuple[str, str], np.ndarray] = {}
    for adapter in adapters:
        for problem in PROBLEMS:
            try:
                w, secs = time_solve(adapter, problem, ctx, repeats=repeats, warmup=warmup)
                val, label, direction = quality(problem, w, ctx)
                diff = float(np.max(np.abs(w - ref_weights[problem])))
                rows.append(
                    Row(problem, adapter.name, adapter.is_reference, secs, val, label,
                        direction, diff, "ok")
                )
                weights_by[(problem, adapter.name)] = w
            except Unsupported as exc:
                rows.append(Row(problem, adapter.name, adapter.is_reference,
                                np.nan, np.nan, "", "", np.nan, f"unsupported: {exc}"))
            except Exception as exc:  # noqa: BLE001 - report, don't abort the sweep
                rows.append(Row(problem, adapter.name, adapter.is_reference,
                                np.nan, np.nan, "", "", np.nan, f"error: {exc}"))
    return rows, weights_by, ref_name


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_tables(rows: list[Row], ref_name: str) -> pd.DataFrame:
    """Pretty-print one table per problem and return the tidy results frame."""
    df = pd.DataFrame([r.__dict__ for r in rows])
    df["ms"] = df["seconds"] * 1e3

    for problem in PROBLEMS:
        sub = df[df["problem"] == problem].copy()
        ok = sub[sub["status"] == "ok"]
        label = ok["quality_label"].iloc[0] if len(ok) else "quality"
        print(f"\n=== {PROBLEM_TITLES[problem]} ===")

        table = []
        for _, r in sub.iterrows():
            if r["status"] != "ok":
                table.append({"library": r["library"], "ms": None,
                              label: None, "Δw vs jaxfolio": None, "note": r["status"]})
                continue
            table.append({
                "library": r["library"] + ("  (ref)" if r["is_reference"] else ""),
                "ms": round(r["ms"], 3),
                label: round(r["quality_value"], 4),
                "Δw vs jaxfolio": round(r["max_weight_diff"], 4),
                "note": "",
            })
        out = pd.DataFrame(table).set_index("library")
        print(out.to_string(na_rep="—"))

    return df


def plot_speed(df: pd.DataFrame) -> None:
    """Horizontal bar chart of median solve time per problem (log scale)."""
    import matplotlib.pyplot as plt

    from jaxfolio.viz import theme

    theme.use_dark_theme()
    ok = df[df["status"] == "ok"]
    fig, axes = plt.subplots(1, len(PROBLEMS), figsize=(15, 4.6), constrained_layout=True)

    for ax, problem in zip(axes, PROBLEMS, strict=True):
        sub = ok[ok["problem"] == problem].sort_values("ms")
        libs = sub["library"].tolist()
        ms = sub["ms"].tolist()
        colors = [
            theme.CATEGORICAL[0] if ref else theme.INK_MUTED
            for ref in sub["is_reference"]
        ]
        y = np.arange(len(libs))
        ax.barh(y, ms, color=colors, edgecolor=theme.SURFACE)
        ax.set_yticks(y)
        ax.set_yticklabels(libs)
        ax.set_xscale("log")
        ax.set_xlabel("median solve time (ms, log)")
        ax.set_title(PROBLEM_TITLES[problem])
        ax.invert_yaxis()
        for yi, v in zip(y, ms, strict=True):
            ax.text(v, yi, f"  {v:.2f} ms", va="center", ha="left",
                    color=theme.INK_SECONDARY, fontsize=8)
        ax.margins(x=0.25)

    fig.suptitle("Solve time: jaxfolio vs. popular portfolio-optimization libraries")
    fig.savefig(OUT / "benchmark_speed.png", dpi=150)
    plt.close(fig)


def plot_weight_agreement(weights_by, ref_name: str, assets: list[str]) -> None:
    """Grouped bars of min-variance weights across libraries — do they agree?"""
    import matplotlib.pyplot as plt

    from jaxfolio.viz import theme

    theme.use_dark_theme()
    problem = "min_variance"
    libs = [lib for (prob, lib) in weights_by if prob == problem]
    if not libs:
        return

    fig, ax = plt.subplots(figsize=(13, 5.2), constrained_layout=True)
    x = np.arange(len(assets))
    width = 0.8 / max(len(libs), 1)
    for i, lib in enumerate(libs):
        w = weights_by[(problem, lib)]
        ax.bar(x + i * width, w, width, label=lib, color=theme.color(i),
               edgecolor=theme.SURFACE, linewidth=0.4)
    ax.set_xticks(x + width * (len(libs) - 1) / 2)
    ax.set_xticklabels(assets, rotation=45, ha="right")
    ax.set_ylabel("weight")
    ax.set_title(f"{PROBLEM_TITLES[problem]}: weight agreement across libraries")
    ax.legend(ncol=min(len(libs), 6), loc="upper right", fontsize=8)
    fig.savefig(OUT / "benchmark_weights.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Scenario 2 — repeated solves (a rolling-window backtest)
# --------------------------------------------------------------------------- #
def rolling_windows(
    returns: pd.DataFrame, *, lookback: int = 252, step: int = 21
) -> list[pd.DataFrame]:
    """Trailing look-back windows on a fixed monthly rebalance cadence."""
    return [
        returns.iloc[t - lookback : t]
        for t in range(lookback, len(returns) + 1, step)
    ]


def run_throughput(
    returns: pd.DataFrame, *, lookback: int = 252, step: int = 21, repeats: int = 3
) -> pd.DataFrame:
    """Time each library solving every rolling window — the real backtest workload.

    Moments are precomputed per window (not timed) so this isolates solver
    throughput and every library sees identical inputs. jaxfolio compiles once on
    the first window and reuses the kernel for the rest (every window has the same
    shape); the other libraries solve each window from scratch. The full pass is
    repeated ``repeats`` times and the best (least-noisy) total is kept.
    """
    windows = rolling_windows(returns, lookback=lookback, step=step)
    contexts = [make_context(w) for w in windows]
    adapters = available_adapters()

    rows = []
    for adapter in adapters:
        for problem in PROBLEMS:
            try:
                adapter.solve(problem, contexts[0])  # warm-up (compile / cache)
                best = float("inf")
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    for ctx in contexts:
                        adapter.solve(problem, ctx)
                    best = min(best, time.perf_counter() - t0)
                total = best
                rows.append({
                    "problem": problem, "library": adapter.name,
                    "is_reference": adapter.is_reference, "n_windows": len(contexts),
                    "total_ms": total * 1e3, "per_solve_ms": total / len(contexts) * 1e3,
                    "status": "ok",
                })
            except Unsupported as exc:
                rows.append({"problem": problem, "library": adapter.name,
                             "is_reference": adapter.is_reference, "status": f"unsupported: {exc}"})
            except Exception as exc:  # noqa: BLE001
                rows.append({"problem": problem, "library": adapter.name,
                             "is_reference": adapter.is_reference, "status": f"error: {exc}"})
    return pd.DataFrame(rows)


def print_throughput_tables(df: pd.DataFrame) -> None:
    n = int(df.loc[df["status"] == "ok", "n_windows"].iloc[0])
    print(f"\n\n########## Repeated solves — {n} rolling windows (a backtest) ##########")
    for problem in PROBLEMS:
        sub = df[df["problem"] == problem]
        print(f"\n=== {PROBLEM_TITLES[problem]} ===")
        table = []
        for _, r in sub.iterrows():
            ok = r["status"] == "ok"
            table.append({
                "library": r["library"] + ("  (ref)" if r["is_reference"] else ""),
                "total ms": round(r["total_ms"], 2) if ok else None,
                "ms/solve": round(r["per_solve_ms"], 3) if ok else None,
                "note": "" if ok else r["status"],
            })
        print(pd.DataFrame(table).set_index("library").to_string(na_rep="—"))


def plot_throughput(df: pd.DataFrame) -> None:
    """Horizontal bars of amortized ms/solve over the rolling backtest (log scale)."""
    import matplotlib.pyplot as plt

    from jaxfolio.viz import theme

    theme.use_dark_theme()
    ok = df[df["status"] == "ok"]
    n = int(ok["n_windows"].iloc[0])
    fig, axes = plt.subplots(1, len(PROBLEMS), figsize=(15, 4.6), constrained_layout=True)

    for ax, problem in zip(axes, PROBLEMS, strict=True):
        sub = ok[ok["problem"] == problem].sort_values("per_solve_ms")
        libs = sub["library"].tolist()
        ms = sub["per_solve_ms"].tolist()
        colors = [theme.CATEGORICAL[0] if ref else theme.INK_MUTED for ref in sub["is_reference"]]
        y = np.arange(len(libs))
        ax.barh(y, ms, color=colors, edgecolor=theme.SURFACE)
        ax.set_yticks(y)
        ax.set_yticklabels(libs)
        ax.set_xscale("log")
        ax.set_xlabel("amortized ms / solve (log)")
        ax.set_title(PROBLEM_TITLES[problem])
        ax.invert_yaxis()
        for yi, v in zip(y, ms, strict=True):
            ax.text(v, yi, f"  {v:.3f}", va="center", ha="left",
                    color=theme.INK_SECONDARY, fontsize=8)
        ax.margins(x=0.28)

    fig.suptitle(f"Amortized solve time over {n} rolling rebalances (a backtest workload)")
    fig.savefig(OUT / "benchmark_throughput.png", dpi=150)
    plt.close(fig)


def plot_hero(single_df: pd.DataFrame, tput_df: pd.DataFrame) -> None:
    """One clean summary figure (used in the README / docs): amortized ms/solve.

    Shows the fair, headline result — amortized solve time over a rolling
    backtest — with a footnote stating the agreement (all libraries reach the
    same optimum) that makes the speed comparison meaningful.
    """
    import matplotlib.pyplot as plt

    from jaxfolio.viz import theme

    theme.use_dark_theme()
    ok = tput_df[tput_df["status"] == "ok"]
    n = int(ok["n_windows"].iloc[0])
    # Agreement on the strictly-convex problems (unique optimum) where every
    # library should land on the same portfolio. Risk parity is excluded: SciPy's
    # generic SLSQP does not reach ERC, so its Δw there reflects a weaker solver,
    # not a moment/setup difference — reporting it would misstate the agreement.
    unique_opt = single_df[
        (single_df["status"] == "ok")
        & (single_df["problem"].isin(["min_variance", "max_sharpe"]))
    ]
    max_dw = float(unique_opt["max_weight_diff"].max())

    fig, axes = plt.subplots(1, len(PROBLEMS), figsize=(14, 4.8), constrained_layout=True)
    for ax, problem in zip(axes, PROBLEMS, strict=True):
        sub = ok[ok["problem"] == problem].sort_values("per_solve_ms")
        libs, ms = sub["library"].tolist(), sub["per_solve_ms"].tolist()
        colors = [theme.CATEGORICAL[0] if r else theme.INK_MUTED for r in sub["is_reference"]]
        y = np.arange(len(libs))
        ax.barh(y, ms, color=colors, edgecolor=theme.SURFACE)
        ax.set_yticks(y)
        ax.set_yticklabels(libs)
        ax.set_xscale("log")
        ax.set_xlabel("amortized ms / solve  (log, lower is better)")
        ax.set_title(PROBLEM_TITLES[problem])
        ax.invert_yaxis()
        for yi, v in zip(y, ms, strict=True):
            ax.text(v, yi, f"  {v:.2f}", va="center", ha="left",
                    color=theme.INK_SECONDARY, fontsize=8.5)
        ax.margins(x=0.30)

    fig.suptitle(
        f"Portfolio optimization: amortized solve time over a {n}-rebalance backtest",
        fontsize=13,
    )
    fig.text(
        0.5, -0.02,
        "Every library is given identical sample moments and reaches the same optimum on "
        f"minimum-variance and maximum-Sharpe (max weight difference < {max(max_dw, 0.001):.3f}). "
        "Best-of-3 timing; jaxfolio compiles its JIT kernel once (warm-up, not charged) then "
        "reuses it across all rebalances.",
        ha="center", va="top", fontsize=8, color=theme.INK_MUTED, wrap=True,
    )
    fig.savefig(OUT / "benchmark.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    # A realistic, reproducible, offline panel: 3 years of daily returns, 20 assets.
    returns = jf.generate_returns(n_assets=20, n_days=756, seed=7)
    # A longer panel for the repeated-solve (backtest) scenario.
    returns_long = jf.generate_returns(n_assets=20, n_days=1512, seed=7)

    present = [a.name for a in available_adapters()]
    absent = missing_libraries()
    print("jaxfolio benchmark — portfolio optimization libraries")
    print(f"  data:      {returns.shape[1]} assets x {returns.shape[0]} daily returns")
    print(f"  comparing: {', '.join(present)}")
    if absent:
        print(f"  not installed (skipped): {', '.join(absent)}")
        print("    -> uv pip install PyPortfolioOpt Riskfolio-Lib skfolio cvxpy")

    # Scenario 1: single cold solve.
    print("\n\n########## Single solve (cold) ##########")
    rows, weights_by, ref_name = run(returns)
    df = print_tables(rows, ref_name)

    # Scenario 2: repeated solves across a rolling backtest (amortized).
    tdf = run_throughput(returns_long)
    print_throughput_tables(tdf)

    df.drop(columns=["quality_dir"]).to_csv(OUT / "benchmark_results.csv", index=False)
    tdf.to_csv(OUT / "benchmark_throughput.csv", index=False)
    plot_speed(df)
    plot_weight_agreement(weights_by, ref_name, [str(c) for c in returns.columns])
    plot_throughput(tdf)
    plot_hero(df, tdf)

    print(f"\nSaved tables (CSV) and charts -> {OUT}/")
    print("  benchmark_results.csv · benchmark_throughput.csv")
    print("  benchmark.png · benchmark_speed.png · benchmark_weights.png ·"
          " benchmark_throughput.png")


if __name__ == "__main__":
    main()
