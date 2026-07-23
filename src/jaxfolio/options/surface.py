"""Implied-volatility surfaces.

A :class:`VolSurface` turns a discrete grid of implied vols (strikes x expiries)
into a callable ``iv(strike, ttm)`` by interpolating **in total variance**
(``w = iv^2 * ttm``) across time and across strike — the representation in which
a well-behaved surface is smooth and arbitrage checks are natural. Build one from
market option prices (:meth:`VolSurface.from_chain`, which inverts prices with the
existing Newton :func:`~jaxfolio.options.pricing.implied_volatility`) or fit a
parametric **raw-SVI** slice per expiry (:meth:`VolSurface.fit_svi`).

The surface reuses the package pricer/Greeks: :meth:`VolSurface.price` and
:meth:`VolSurface.greeks` read the vol off the surface and call
:func:`~jaxfolio.options.pricing.black_scholes_price` /
:func:`~jaxfolio.options.greeks.all_greeks`, so nothing is re-derived.

Arbitrage diagnostics (:meth:`VolSurface.arbitrage_report`) flag **butterfly**
violations (call price must be convex in strike) and **calendar** violations
(total variance must not decrease with maturity).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.optimize as opt

from jaxfolio.options.greeks import all_greeks
from jaxfolio.options.pricing import black_scholes_price, implied_volatility


@dataclass
class SVIParams:
    """Raw-SVI parameters for a single expiry slice (Gatheral, 2004).

    Total variance as a function of log-moneyness ``k = log(strike / spot)``::

        w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

    Attributes
    ----------
    a:
        Vertical level of the variance smile (>= 0 region after fit).
    b:
        Slope / wing steepness (``b >= 0``).
    rho:
        Skew, ``-1 < rho < 1`` (negative = downward equity skew).
    m:
        Horizontal shift of the smile minimum.
    sigma:
        Curvature at the minimum (``sigma > 0``).
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_variance(self, k: np.ndarray) -> np.ndarray:
        """Total implied variance ``w(k)`` at log-moneyness ``k``."""
        k = np.asarray(k, dtype=float)
        return self.a + self.b * (
            self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.sigma**2)
        )


def calibrate_svi(k: np.ndarray, w: np.ndarray) -> SVIParams:
    """Least-squares-fit raw-SVI parameters to observed total variances.

    Parameters
    ----------
    k:
        Log-moneyness of each observation.
    w:
        Observed total implied variance (``iv^2 * ttm``) at each ``k``.

    Returns
    -------
    SVIParams
        The fitted slice. Bounds keep the fit in the no-vertical-arbitrage region
        (``b >= 0``, ``|rho| < 1``, ``sigma > 0``, ``a >= 0``).
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    w_mean = float(np.mean(w))

    def resid(theta):
        a, b, rho, m, sigma = theta
        model = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))
        return model - w

    x0 = [max(w_mean * 0.5, 1e-6), 0.1, -0.3, 0.0, 0.1]
    lower = [0.0, 0.0, -0.999, k.min() - 1.0, 1e-4]
    upper = [max(w.max(), 1e-3) * 2 + 1e-6, 10.0, 0.999, k.max() + 1.0, 5.0]
    res = opt.least_squares(resid, x0, bounds=(lower, upper), max_nfev=2000)
    a, b, rho, m, sigma = res.x
    return SVIParams(a=float(a), b=float(b), rho=float(rho), m=float(m), sigma=float(sigma))


@dataclass
class VolSurface:
    """An interpolated implied-volatility surface.

    Attributes
    ----------
    strikes:
        Strictly increasing strike grid, shape ``(K,)``.
    ttms:
        Strictly increasing time-to-maturity grid (years), shape ``(M,)``.
    iv_grid:
        Implied vols, shape ``(M, K)`` — row ``i`` is the smile at ``ttms[i]``.
    spot:
        Reference spot used for log-moneyness.
    rate, div:
        Continuous risk-free rate and dividend yield used when pricing off the
        surface.
    svi:
        Optional per-expiry SVI slices (populated by :meth:`fit_svi`); when
        present, ``iv`` is evaluated from SVI across strike instead of grid
        interpolation.
    """

    strikes: np.ndarray
    ttms: np.ndarray
    iv_grid: np.ndarray
    spot: float
    rate: float = 0.0
    div: float = 0.0
    svi: list[SVIParams] | None = None

    def __post_init__(self) -> None:
        self.strikes = np.asarray(self.strikes, dtype=float)
        self.ttms = np.asarray(self.ttms, dtype=float)
        self.iv_grid = np.asarray(self.iv_grid, dtype=float)
        if self.iv_grid.shape != (len(self.ttms), len(self.strikes)):
            raise ValueError(
                f"iv_grid shape {self.iv_grid.shape} != (len(ttms), len(strikes)) "
                f"({len(self.ttms)}, {len(self.strikes)})"
            )
        if np.any(np.diff(self.strikes) <= 0) or np.any(np.diff(self.ttms) <= 0):
            raise ValueError("strikes and ttms must be strictly increasing")

    # -- construction ------------------------------------------------------- #
    @classmethod
    def from_iv_grid(cls, strikes, ttms, iv_grid, spot, *, rate=0.0, div=0.0) -> VolSurface:
        """Build directly from a known implied-vol grid."""
        return cls(strikes, ttms, iv_grid, spot, rate=rate, div=div)

    @classmethod
    def from_chain(
        cls,
        spot: float,
        strikes,
        ttms,
        prices: np.ndarray,
        *,
        rate: float = 0.0,
        div: float = 0.0,
        is_call: bool = True,
    ) -> VolSurface:
        """Invert a grid of market option prices into an implied-vol surface.

        ``prices`` has shape ``(len(ttms), len(strikes))``. Each entry is inverted
        with the Newton solver; non-invertible quotes (below intrinsic) become
        ``NaN`` and are then filled by nearest-valid interpolation along the smile.
        """
        strikes = np.asarray(strikes, dtype=float)
        ttms = np.asarray(ttms, dtype=float)
        prices = np.asarray(prices, dtype=float)
        iv_grid = np.empty((len(ttms), len(strikes)))
        for i, t in enumerate(ttms):
            for j, k in enumerate(strikes):
                iv_grid[i, j] = float(
                    implied_volatility(prices[i, j], spot, k, t, rate, div, is_call)
                )
            iv_grid[i] = _fill_nans(strikes, iv_grid[i])
        return cls(strikes, ttms, iv_grid, spot, rate=rate, div=div)

    def fit_svi(self) -> VolSurface:
        """Return a copy whose smiles are replaced by fitted raw-SVI slices.

        Each expiry's ``(log-moneyness, total-variance)`` points are fit with
        :func:`calibrate_svi`; the grid is re-evaluated from the fits so ``iv``
        becomes smooth and extrapolates sensibly beyond the quoted strikes.
        """
        k = np.log(self.strikes / self.spot)
        sections: list[SVIParams] = []
        new_grid = np.empty_like(self.iv_grid)
        for i, t in enumerate(self.ttms):
            w = self.iv_grid[i] ** 2 * t
            params = calibrate_svi(k, w)
            sections.append(params)
            new_grid[i] = np.sqrt(np.clip(params.total_variance(k), 1e-12, None) / t)
        return VolSurface(
            self.strikes, self.ttms, new_grid, self.spot, rate=self.rate, div=self.div, svi=sections
        )

    # -- evaluation --------------------------------------------------------- #
    def _slice_iv(self, i: int, strike: float) -> float:
        """Implied vol at ``strike`` on expiry-slice ``i``."""
        if self.svi is not None:
            k = np.log(strike / self.spot)
            w = float(self.svi[i].total_variance(k))
            return float(np.sqrt(max(w, 1e-12) / self.ttms[i]))
        return float(np.interp(strike, self.strikes, self.iv_grid[i]))

    def iv(self, strike: float, ttm: float) -> float:
        """Interpolated implied vol at ``(strike, ttm)``.

        Interpolates in **total variance** across maturity (flat-forward beyond
        the quoted range) and along the smile at each bracketing expiry.
        """
        ttms = self.ttms
        if ttm <= ttms[0]:
            return self._slice_iv(0, strike)
        if ttm >= ttms[-1]:
            return self._slice_iv(len(ttms) - 1, strike)
        hi = int(np.searchsorted(ttms, ttm))
        lo = hi - 1
        t_lo, t_hi = ttms[lo], ttms[hi]
        w_lo = self._slice_iv(lo, strike) ** 2 * t_lo
        w_hi = self._slice_iv(hi, strike) ** 2 * t_hi
        frac = (ttm - t_lo) / (t_hi - t_lo)
        w = w_lo + frac * (w_hi - w_lo)
        return float(np.sqrt(max(w, 1e-12) / ttm))

    def price(self, strike: float, ttm: float, *, is_call: bool = True) -> float:
        """Black-Scholes price of an option, using the surface's implied vol."""
        vol = self.iv(strike, ttm)
        return float(black_scholes_price(self.spot, strike, ttm, vol, self.rate, self.div, is_call))

    def greeks(self, strike: float, ttm: float, *, is_call: bool = True) -> dict[str, float]:
        """Full Greeks of an option, using the surface's implied vol."""
        vol = self.iv(strike, ttm)
        return all_greeks(self.spot, strike, ttm, vol, self.rate, self.div, is_call)

    # -- arbitrage diagnostics --------------------------------------------- #
    def arbitrage_report(self, tol: float = 1e-6) -> dict[str, object]:
        """Check static no-arbitrage conditions on the surface.

        Returns a dict with ``butterfly_ok`` (call price convex in strike at each
        expiry), ``calendar_ok`` (total variance non-decreasing in maturity at
        each strike), an overall ``arbitrage_free`` flag, and the lists of
        offending ``(ttm, strike)`` / ``(strike, ttm_pair)`` locations.
        """
        butterfly_violations = []
        for i, t in enumerate(self.ttms):
            calls = np.array(
                [
                    black_scholes_price(
                        self.spot, k, t, self._slice_iv(i, k), self.rate, self.div, True
                    )
                    for k in self.strikes
                ]
            )
            # Convexity: second difference of call price in strike must be >= 0.
            second_diff = calls[:-2] - 2 * calls[1:-1] + calls[2:]
            for j in np.where(second_diff < -tol)[0]:
                butterfly_violations.append((float(t), float(self.strikes[j + 1])))

        calendar_violations = []
        for k in self.strikes:
            w = np.array([self._slice_iv(i, k) ** 2 * self.ttms[i] for i in range(len(self.ttms))])
            dec = np.where(np.diff(w) < -tol)[0]
            for i in dec:
                calendar_violations.append(
                    (float(k), (float(self.ttms[i]), float(self.ttms[i + 1])))
                )

        return {
            "butterfly_ok": not butterfly_violations,
            "calendar_ok": not calendar_violations,
            "arbitrage_free": not butterfly_violations and not calendar_violations,
            "butterfly_violations": butterfly_violations,
            "calendar_violations": calendar_violations,
        }

    def is_arbitrage_free(self, tol: float = 1e-6) -> bool:
        """Convenience boolean: ``arbitrage_report(...)['arbitrage_free']``."""
        return bool(self.arbitrage_report(tol)["arbitrage_free"])


def _fill_nans(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fill NaNs in ``y`` by interpolating against ``x`` (nearest at the edges)."""
    y = np.asarray(y, dtype=float)
    good = np.isfinite(y)
    if good.all():
        return y
    if not good.any():
        raise ValueError("cannot fill a fully-NaN implied-vol slice (all quotes below intrinsic?)")
    return np.interp(x, x[good], y[good])
