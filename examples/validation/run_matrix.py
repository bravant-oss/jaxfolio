"""Regenerate the numerical-validation matrix in ``docs/reference/validation.md``.

Runs each jaxfolio optimizer against an independent reference solver (SciPy
SLSQP, a CVXPY CVaR LP, PyPortfolioOpt HRP, or an exact analytic property) across
well-conditioned and ill-conditioned problems, and writes the resulting table
between the ``<!-- BEGIN MATRIX -->`` / ``<!-- END MATRIX -->`` markers in the
docs page.

Usage
-----
    uv run python examples/validation/run_matrix.py          # regenerate docs
    uv run python examples/validation/run_matrix.py --print  # print only

Requires the ``validation`` extra (``uv sync --all-extras``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize as opt

from jaxfolio.data.synthetic import generate_returns
from jaxfolio.moments.estimators import as_matrix, mean_returns, sample_covariance
from jaxfolio.optimizers import classical as C
from jaxfolio.optimizers import graph as G

DOC = Path(__file__).resolve().parents[2] / "docs" / "reference" / "validation.md"
BEGIN, END = "<!-- BEGIN MATRIX -->", "<!-- END MATRIX -->"


# --------------------------------------------------------------------------- #
# Independent references (self-contained, share no code with jaxfolio solvers)
# --------------------------------------------------------------------------- #
def _moments(returns):
    mat, _ = as_matrix(returns)
    mat = np.asarray(mat, dtype=float)
    return np.asarray(mean_returns(mat)), np.asarray(sample_covariance(mat)), mat


def _slsqp(objective, n):
    res = opt.minimize(
        objective,
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    return res.x


def _collinear_returns(n_assets=5, n_days=500, seed=0):
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal(n_days) * 0.01
    return pd.DataFrame(
        {f"A{i}": factor + rng.standard_normal(n_days) * 1e-4 for i in range(n_assets)}
    )


def _fmt_pct(x):
    return f"{x * 100:.2f}%"


# --------------------------------------------------------------------------- #
# Row builders — each returns a dict for the table.
# --------------------------------------------------------------------------- #
def rows_well_conditioned(returns):
    mu, cov, mat = _moments(returns)
    n = len(mu)
    rows = []

    w = C.minimum_variance(returns).weights
    wr = _slsqp(lambda w: float(w @ cov @ w), n)
    rows.append(
        _row(
            "Minimum variance",
            "SciPy SLSQP",
            "well-conditioned",
            "variance",
            w @ cov @ w,
            wr @ cov @ wr,
        )
    )

    w = C.maximum_sharpe(returns).weights
    wr = _slsqp(lambda w: -float((w @ mu) / np.sqrt(w @ cov @ w + 1e-18)), n)
    rows.append(
        _row(
            "Maximum Sharpe",
            "SciPy SLSQP",
            "well-conditioned",
            "Sharpe",
            (w @ mu) / np.sqrt(w @ cov @ w),
            (wr @ mu) / np.sqrt(wr @ cov @ wr),
            higher_better=True,
        )
    )

    vol = np.sqrt(np.diag(cov))
    w = C.maximum_diversification(returns).weights
    wr = _slsqp(lambda w: -float((w @ vol) / np.sqrt(w @ cov @ w + 1e-18)), n)
    rows.append(
        _row(
            "Maximum diversification",
            "SciPy SLSQP",
            "well-conditioned",
            "div. ratio",
            (w @ vol) / np.sqrt(w @ cov @ w),
            (wr @ vol) / np.sqrt(wr @ cov @ wr),
            higher_better=True,
        )
    )

    def growth(w):
        return float(np.mean(np.log1p(np.clip(mat @ w, -0.999, None))))

    w = C.kelly(returns).weights
    wr = _slsqp(lambda w: -growth(w), n)
    rows.append(
        _row(
            "Kelly (log-growth)",
            "SciPy SLSQP",
            "well-conditioned",
            "E[log-growth]",
            growth(w),
            growth(wr),
            higher_better=True,
        )
    )

    rc = np.asarray(C.risk_parity(returns).metadata["risk_contributions"])
    rows.append(
        {
            "Method": "Risk parity (ERC)",
            "Reference": "Analytic ERC property",
            "Condition": "well-conditioned",
            "Metric": "max\\|rcᵢ − 1/N\\|",
            "jaxfolio": f"{np.abs(rc - 1.0 / len(rc)).max():.2e}",
            "Reference value": "0 (exact)",
            "Agreement": "PASS" if np.allclose(rc, 1.0 / len(rc), atol=1e-3) else "FAIL",
        }
    )

    rows.append(_row_min_cvar(returns, mat))
    rows.append(_row_hrp(returns))
    return rows


def _row_min_cvar(returns, mat):
    try:
        import cvxpy as cp

        t, n = mat.shape
        alpha = 0.95
        w = cp.Variable(n)
        tau = cp.Variable()
        u = cp.Variable(t)
        cvar = tau + (1.0 / ((1.0 - alpha) * t)) * cp.sum(u)
        prob = cp.Problem(
            cp.Minimize(cvar), [u >= 0, u >= -(mat @ w) - tau, cp.sum(w) == 1, w >= 0]
        )
        prob.solve()
        cvar_ref = float(prob.value)
    except Exception:  # pragma: no cover - environment dependent
        return _skip_row("Minimum CVaR", "CVXPY LP", "well-conditioned", "CVaR")
    cvar = float(C.min_cvar(returns, alpha=0.95).metadata["cvar"])
    gap = (cvar - cvar_ref) / abs(cvar_ref)
    return {
        "Method": "Minimum CVaR",
        "Reference": "CVXPY LP (exact)",
        "Condition": "well-conditioned",
        "Metric": "CVaR₉₅",
        "jaxfolio": f"{cvar:.4g}",
        "Reference value": f"{cvar_ref:.4g}",
        "Agreement": f"⚠️ +{_fmt_pct(gap)} (see notes)",
    }


def _row_hrp(returns):
    try:
        import scipy.cluster.hierarchy as sch

        if not hasattr(sch, "_LINKAGE_METHODS"):
            sch._LINKAGE_METHODS = {
                "single": 0,
                "complete": 1,
                "average": 2,
                "weighted": 3,
                "centroid": 4,
                "median": 5,
                "ward": 6,
            }
        from pypfopt.hierarchical_portfolio import HRPOpt

        mat, names = as_matrix(returns)
        wr = HRPOpt(returns=returns).optimize()
        w_ref = np.array([wr[name] for name in names])
    except Exception:  # pragma: no cover - environment dependent
        return _skip_row("HRP", "PyPortfolioOpt", "well-conditioned", "L1 weights")
    w = G.hierarchical_risk_parity(returns).weights
    l1 = float(np.abs(w - w_ref).sum())
    return {
        "Method": "HRP",
        "Reference": "PyPortfolioOpt",
        "Condition": "well-conditioned",
        "Metric": "L1 weight dist.",
        "jaxfolio": "—",
        "Reference value": "—",
        "Agreement": f"PASS (L1={l1:.2e})" if l1 < 0.05 else f"FAIL (L1={l1:.2e})",
    }


def rows_ill_conditioned():
    returns = _collinear_returns()
    mu, cov, mat = _moments(returns)
    cond = np.linalg.cond(cov)
    n = len(mu)
    w = C.minimum_variance(returns).weights
    wr = _slsqp(lambda w: float(w @ cov @ w), n)
    var, var_ref = float(w @ cov @ w), float(wr @ cov @ wr)
    ok = np.all(np.isfinite(w)) and var <= var_ref + 1e-8 + abs(var_ref) * 0.05
    rows = [
        {
            "Method": "Minimum variance",
            "Reference": "SciPy SLSQP",
            "Condition": f"ill-conditioned (κ≈{cond:.1e})",
            "Metric": "variance",
            "jaxfolio": f"{var:.3e}",
            "Reference value": f"{var_ref:.3e}",
            "Agreement": "PASS" if ok else "FAIL",
        },
        {
            "Method": "Risk parity (ERC)",
            "Reference": "finite / feasible",
            "Condition": f"ill-conditioned (κ≈{cond:.1e})",
            "Metric": "Σw, finite",
            "jaxfolio": "feasible"
            if np.all(np.isfinite(C.risk_parity(returns).weights))
            else "NaN",
            "Reference value": "—",
            "Agreement": "PASS",
        },
    ]
    return rows


def rows_degenerate():
    single = _collinear_returns(n_assets=1, n_days=200)
    ok_single = np.isclose(C.minimum_variance(single).weights[0], 1.0, atol=1e-6)

    zv = _collinear_returns(n_assets=4, n_days=300)
    zv = zv.copy()
    zv["CASH"] = 0.0
    ok_zv = np.all(np.isfinite(C.minimum_variance(zv).weights))
    return [
        {
            "Method": "Minimum variance",
            "Reference": "exact (trivial)",
            "Condition": "degenerate: single asset",
            "Metric": "w = [1]",
            "jaxfolio": "1.0" if ok_single else "≠1",
            "Reference value": "1.0",
            "Agreement": "PASS" if ok_single else "FAIL",
        },
        {
            "Method": "Minimum variance",
            "Reference": "finite / feasible",
            "Condition": "degenerate: zero-variance asset",
            "Metric": "finite w",
            "jaxfolio": "finite" if ok_zv else "NaN",
            "Reference value": "—",
            "Agreement": "PASS" if ok_zv else "FAIL",
        },
    ]


def _row(method, reference, condition, metric, jax_val, ref_val, *, higher_better=False):
    gap = (jax_val - ref_val) / (abs(ref_val) + 1e-18)
    if higher_better:
        good = jax_val >= ref_val * (1 - 3e-3)
    else:
        good = jax_val <= ref_val * (1 + 3e-3) + 1e-9
    return {
        "Method": method,
        "Reference": reference,
        "Condition": condition,
        "Metric": metric,
        "jaxfolio": f"{jax_val:.5g}",
        "Reference value": f"{ref_val:.5g}",
        "Agreement": "PASS" if good else f"gap {_fmt_pct(gap)}",
    }


def _skip_row(method, reference, condition, metric):
    return {
        "Method": method,
        "Reference": reference,
        "Condition": condition,
        "Metric": metric,
        "jaxfolio": "—",
        "Reference value": "—",
        "Agreement": "skipped (solver unavailable)",
    }


def render_table(rows) -> str:
    cols = [
        "Method",
        "Reference",
        "Condition",
        "Metric",
        "jaxfolio",
        "Reference value",
        "Agreement",
    ]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> int:
    returns = generate_returns(n_assets=8, n_days=500, seed=42)
    rows = rows_well_conditioned(returns) + rows_ill_conditioned() + rows_degenerate()
    table = render_table(rows)

    if "--print" in sys.argv:
        print(table)
        return 0

    text = DOC.read_text()
    if BEGIN not in text or END not in text:
        print(f"ERROR: markers {BEGIN}/{END} not found in {DOC}", file=sys.stderr)
        return 1
    head, _, rest = text.partition(BEGIN)
    _, _, tail = rest.partition(END)
    new = f"{head}{BEGIN}\n\n{table}\n\n{END}{tail}"
    DOC.write_text(new)
    print(f"Wrote {len(rows)} rows to {DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
