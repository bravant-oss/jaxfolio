"""Adapters wrapping each portfolio-optimization library behind one interface.

Every adapter exposes ``solve(problem, ctx) -> np.ndarray`` returning long-only,
fully-invested weights aligned to ``ctx.assets`` for one of three canonical
problems — ``"min_variance"``, ``"max_sharpe"``, ``"risk_parity"``. A library
that does not natively support a problem raises :class:`Unsupported`; a library
that is not installed is simply absent from :func:`available_adapters`.

To keep the comparison apples-to-apples, every adapter is fed the *same* moments
(``ctx.mu`` / ``ctx.cov`` — per-period sample estimates matching jaxfolio's
``sample_covariance``) wherever the library's API allows injecting them. skfolio
estimates its own moments internally (its default empirical prior matches these);
that difference is part of an honest comparison and shows up as small weight
disagreements, not as an unfair advantage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PROBLEMS = ("min_variance", "max_sharpe", "risk_parity")


class Unsupported(Exception):
    """Raised by an adapter when it has no native solver for a problem."""


@dataclass
class Context:
    """Everything an adapter needs to solve, precomputed once per dataset."""

    returns: pd.DataFrame  # (T x N) simple returns
    assets: list[str]  # column order every adapter must align to
    mu: np.ndarray  # (N,) per-period sample mean
    cov: np.ndarray  # (N, N) per-period sample covariance (ddof=1)
    risk_free: float = 0.0  # per-period risk-free rate

    @property
    def n(self) -> int:
        return len(self.assets)

    def mu_series(self) -> pd.Series:
        return pd.Series(self.mu, index=self.assets)

    def cov_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.cov, index=self.assets, columns=self.assets)


def make_context(returns: pd.DataFrame, risk_free: float = 0.0) -> Context:
    """Build a :class:`Context` with moments that match jaxfolio's estimators."""
    assets = [str(c) for c in returns.columns]
    mat = returns.to_numpy(dtype=float)
    mu = mat.mean(axis=0)
    cov = np.cov(mat, rowvar=False, ddof=1)
    return Context(returns=returns, assets=assets, mu=mu, cov=cov, risk_free=risk_free)


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #
class Adapter:
    """One optimization library behind a uniform ``solve`` interface."""

    name: str = "adapter"
    is_reference: bool = False  # jaxfolio is the reference every other lib is diffed against

    @classmethod
    def is_available(cls) -> bool:  # pragma: no cover - trivial
        raise NotImplementedError

    def solve(self, problem: str, ctx: Context) -> np.ndarray:
        raise NotImplementedError


def _normalize(w: np.ndarray) -> np.ndarray:
    """Clip tiny negatives from solver noise and renormalize to sum 1."""
    w = np.asarray(w, dtype=float).reshape(-1)
    w = np.where(np.abs(w) < 1e-9, 0.0, w)
    total = w.sum()
    return w / total if total != 0 else w


# --------------------------------------------------------------------------- #
# jaxfolio — the reference implementation
# --------------------------------------------------------------------------- #
class JaxfolioAdapter(Adapter):
    name = "jaxfolio"
    is_reference = True

    @classmethod
    def is_available(cls) -> bool:
        try:
            import jaxfolio  # noqa: F401

            return True
        except ImportError:
            return False

    def solve(self, problem: str, ctx: Context) -> np.ndarray:
        import jaxfolio as jf

        if problem == "min_variance":
            res = jf.minimum_variance(ctx.returns)
        elif problem == "max_sharpe":
            res = jf.maximum_sharpe(ctx.returns)
        elif problem == "risk_parity":
            res = jf.risk_parity(ctx.returns)
        else:  # pragma: no cover - guarded by PROBLEMS
            raise Unsupported(problem)
        return _normalize(res.weights)


# --------------------------------------------------------------------------- #
# SciPy — the DIY SLSQP baseline (always available; jaxfolio depends on scipy)
# --------------------------------------------------------------------------- #
class ScipyAdapter(Adapter):
    name = "SciPy (SLSQP)"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import scipy.optimize  # noqa: F401

            return True
        except ImportError:
            return False

    def solve(self, problem: str, ctx: Context) -> np.ndarray:
        from scipy.optimize import minimize

        n, mu, cov, rf = ctx.n, ctx.mu, ctx.cov, ctx.risk_free
        w0 = np.full(n, 1.0 / n)
        bounds = [(0.0, 1.0)] * n
        budget = {"type": "eq", "fun": lambda w: w.sum() - 1.0}

        if problem == "min_variance":
            obj = lambda w: float(w @ cov @ w)  # noqa: E731
        elif problem == "max_sharpe":
            obj = lambda w: -float((w @ mu - rf) / np.sqrt(w @ cov @ w + 1e-18))  # noqa: E731
        elif problem == "risk_parity":
            # Minimize dispersion of (normalized) risk contributions around 1/N.
            def obj(w):
                rc = w * (cov @ w)
                rc = rc / (rc.sum() + 1e-18)
                return float(((rc - 1.0 / n) ** 2).sum())
        else:  # pragma: no cover
            raise Unsupported(problem)

        res = minimize(
            obj,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=[budget],
            options={"maxiter": 500, "ftol": 1e-12},
        )
        return _normalize(res.x)


# --------------------------------------------------------------------------- #
# PyPortfolioOpt
# --------------------------------------------------------------------------- #
class PyPortfolioOptAdapter(Adapter):
    name = "PyPortfolioOpt"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import pypfopt  # noqa: F401

            return True
        except ImportError:
            return False

    def solve(self, problem: str, ctx: Context) -> np.ndarray:
        from pypfopt import EfficientFrontier

        if problem == "risk_parity":
            # PyPortfolioOpt ships HRP, not equal-risk-contribution; skip for a
            # like-for-like ERC comparison rather than compare different methods.
            raise Unsupported("PyPortfolioOpt has no native ERC risk-parity solver")

        ef = EfficientFrontier(ctx.mu_series(), ctx.cov_frame(), weight_bounds=(0.0, 1.0))
        if problem == "min_variance":
            ef.min_volatility()
        elif problem == "max_sharpe":
            ef.max_sharpe(risk_free_rate=ctx.risk_free)
        else:  # pragma: no cover
            raise Unsupported(problem)
        # ef.weights is a positional numpy array aligned to ef.tickers.
        w = pd.Series(ef.weights, index=list(ef.tickers)).reindex(ctx.assets).to_numpy(dtype=float)
        return _normalize(w)


# --------------------------------------------------------------------------- #
# Riskfolio-Lib
# --------------------------------------------------------------------------- #
class RiskfolioAdapter(Adapter):
    name = "Riskfolio-Lib"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import riskfolio  # noqa: F401

            return True
        except ImportError:
            return False

    def _portfolio(self, ctx: Context):
        import riskfolio as rp

        port = rp.Portfolio(returns=ctx.returns)
        port.assets_stats(method_mu="hist", method_cov="hist")
        # Override with the shared moments so the comparison is apples-to-apples.
        port.mu = pd.DataFrame(ctx.mu.reshape(1, -1), columns=ctx.assets)
        port.cov = ctx.cov_frame()
        return port

    def solve(self, problem: str, ctx: Context) -> np.ndarray:
        port = self._portfolio(ctx)
        if problem == "min_variance":
            w = port.optimization(model="Classic", rm="MV", obj="MinRisk", hist=True)
        elif problem == "max_sharpe":
            port.rf = ctx.risk_free
            w = port.optimization(
                model="Classic", rm="MV", obj="Sharpe", rf=ctx.risk_free, hist=True
            )
        elif problem == "risk_parity":
            w = port.rp_optimization(model="Classic", rm="MV", hist=True)
        else:  # pragma: no cover
            raise Unsupported(problem)
        if w is None:
            raise Unsupported(f"Riskfolio-Lib returned no solution for {problem}")
        aligned = w["weights"].reindex(ctx.assets).to_numpy(dtype=float)
        return _normalize(aligned)


# --------------------------------------------------------------------------- #
# skfolio
# --------------------------------------------------------------------------- #
class SkfolioAdapter(Adapter):
    name = "skfolio"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import skfolio  # noqa: F401

            return True
        except ImportError:
            return False

    def solve(self, problem: str, ctx: Context) -> np.ndarray:
        from skfolio import RiskMeasure
        from skfolio.optimization import MeanRisk, ObjectiveFunction, RiskBudgeting

        if problem == "min_variance":
            model = MeanRisk(
                objective_function=ObjectiveFunction.MINIMIZE_RISK,
                risk_measure=RiskMeasure.VARIANCE,
            )
        elif problem == "max_sharpe":
            model = MeanRisk(
                objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
                risk_measure=RiskMeasure.VARIANCE,
                risk_free_rate=ctx.risk_free,
            )
        elif problem == "risk_parity":
            model = RiskBudgeting(risk_measure=RiskMeasure.VARIANCE)
        else:  # pragma: no cover
            raise Unsupported(problem)

        model.fit(ctx.returns)
        # skfolio keeps the fitted feature (asset) order; realign defensively.
        cols = list(getattr(model, "feature_names_in_", ctx.assets))
        w = pd.Series(np.asarray(model.weights_, dtype=float), index=cols)
        return _normalize(w.reindex(ctx.assets).to_numpy(dtype=float))


# --------------------------------------------------------------------------- #
# CVXPY — the general-purpose convex-modeling reference
# --------------------------------------------------------------------------- #
class CvxpyAdapter(Adapter):
    name = "CVXPY"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import cvxpy  # noqa: F401

            return True
        except ImportError:
            return False

    def solve(self, problem: str, ctx: Context) -> np.ndarray:
        import cvxpy as cp

        n, mu = ctx.n, ctx.mu
        cov = 0.5 * (ctx.cov + ctx.cov.T) + 1e-12 * np.eye(n)  # symmetrize + ridge for PSD

        if problem == "min_variance":
            w = cp.Variable(n)
            prob = cp.Problem(cp.Minimize(cp.quad_form(w, cov)), [cp.sum(w) == 1, w >= 0])
            prob.solve()
            return _normalize(w.value)

        if problem == "max_sharpe":
            # Schur/Cornuejols reformulation: min y'Σy s.t. (μ-rf)'y = 1, y >= 0,
            # then w = y / sum(y) is the long-only tangency portfolio.
            y = cp.Variable(n)
            excess = mu - ctx.risk_free
            prob = cp.Problem(cp.Minimize(cp.quad_form(y, cov)), [excess @ y == 1, y >= 0])
            prob.solve()
            if y.value is None:
                raise Unsupported(
                    "CVXPY tangency reformulation infeasible (no positive-excess asset)"
                )
            return _normalize(y.value / y.value.sum())

        if problem == "risk_parity":
            # Convex log-barrier program (Spinu 2013): min 0.5 x'Σx - (1/N) Σ log x.
            x = cp.Variable(n, pos=True)
            obj = 0.5 * cp.quad_form(x, cov) - (1.0 / n) * cp.sum(cp.log(x))
            prob = cp.Problem(cp.Minimize(obj))
            prob.solve()
            return _normalize(x.value / x.value.sum())

        raise Unsupported(problem)  # pragma: no cover


# Ordered so the reference (jaxfolio) is first and the always-present scipy
# baseline is second; optional libraries follow.
ALL_ADAPTERS: list[type[Adapter]] = [
    JaxfolioAdapter,
    ScipyAdapter,
    PyPortfolioOptAdapter,
    RiskfolioAdapter,
    SkfolioAdapter,
    CvxpyAdapter,
]


def available_adapters() -> list[Adapter]:
    """Instantiate every adapter whose backing library is importable."""
    return [cls() for cls in ALL_ADAPTERS if cls.is_available()]


def missing_libraries() -> list[str]:
    """Names of optional libraries that are not installed (for a friendly note)."""
    return [cls.name for cls in ALL_ADAPTERS if not cls.is_available()]
