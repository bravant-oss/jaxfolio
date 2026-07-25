"""Graph- and hierarchy-based portfolio optimizers.

* :func:`hierarchical_risk_parity` — López de Prado's HRP: cluster assets by a
  correlation-distance dendrogram, quasi-diagonalize, then recursively bisect and
  allocate by inverse-variance.
* :func:`hierarchical_equal_risk` — HERC: like HRP but splits capital across the
  dendrogram's natural clusters by equal risk contribution.
* :func:`mst_centrality` — build the minimum spanning tree of the correlation
  network and weight assets by inverse degree/centrality (peripheral, less
  systemically coupled assets get more capital).

SciPy provides linkage and the MST; all allocation math stays in numpy.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import squareform

from jaxfolio.moments.estimators import (
    as_matrix,
    correlation_from_covariance,
    sample_covariance,
)
from jaxfolio.results import finalize_result
from jaxfolio.types import PortfolioResult


def _corr_distance(corr: np.ndarray) -> np.ndarray:
    """Correlation distance ``sqrt(0.5 * (1 - rho))`` used by HRP."""
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    return dist


def _quasi_diagonal_order(link: np.ndarray) -> list[int]:
    """Return the leaf order that quasi-diagonalizes the correlation matrix."""
    link = link.astype(int)
    n = link[-1, 3]  # total number of original leaves

    def recurse(node: int) -> list[int]:
        if node < n:
            return [node]
        left, right = link[node - n, 0], link[node - n, 1]
        return recurse(left) + recurse(right)

    root = len(link) + n - 1
    return recurse(root)


def _inverse_variance_weights(cov: np.ndarray) -> np.ndarray:
    """Inverse-variance weights for a cluster's covariance sub-block."""
    ivp = 1.0 / np.diag(cov)
    return ivp / ivp.sum()


def _cluster_variance(cov: np.ndarray, idx: list[int]) -> float:
    """Variance of the inverse-variance sub-portfolio over ``idx``."""
    sub = cov[np.ix_(idx, idx)]
    w = _inverse_variance_weights(sub)
    return float(w @ sub @ w)


def _recursive_bisection(cov: np.ndarray, order: list[int]) -> np.ndarray:
    """HRP recursive bisection: split capital by inverse cluster variance."""
    n = cov.shape[0]
    weights = np.ones(n)
    clusters = [order]
    while clusters:
        clusters = [
            half
            for cl in clusters
            for half in (cl[: len(cl) // 2], cl[len(cl) // 2 :])
            if len(cl) > 1
        ]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            var_left = _cluster_variance(cov, left)
            var_right = _cluster_variance(cov, right)
            alpha = 1.0 - var_left / (var_left + var_right)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
    return weights


def _finalize(w, names, method, mat, metadata=None) -> PortfolioResult:
    """Attach annualized diagnostics to a graph-method weight vector."""
    return finalize_result(w, names, method, returns=mat, metadata=metadata)


def _trivial_single_asset(names, mat_np, method):
    """Return the degenerate 1-asset allocation (linkage needs >= 2 assets)."""
    return _finalize(np.ones(1), names, method, mat_np, {"note": "single asset"})


def hierarchical_risk_parity(returns, *, linkage_method: str = "single") -> PortfolioResult:
    """Hierarchical Risk Parity (López de Prado, 2016).

    A three-stage allocator that avoids inverting the (often ill-conditioned)
    covariance matrix. First, assets are clustered by the correlation distance
    ``sqrt(0.5 * (1 - rho))`` into a dendrogram (**tree clustering**). Second, the
    covariance matrix is **quasi-diagonalized** by reordering rows/columns to the
    dendrogram's leaf order, placing similar assets adjacent. Third, capital is
    assigned by **recursive bisection**: each cluster is split in two and capital
    is allocated between the halves in inverse proportion to their inverse-
    variance sub-portfolio variance. The result is a long-only, fully-invested
    portfolio that is more robust out-of-sample than a direct minimum-variance
    solve. Validated against PyPortfolioOpt's HRP (see the
    [validation matrix](validation.md)).

    Parameters
    ----------
    returns:
        Asset return panel (``PortfolioResult``-compatible input, e.g. a
        DataFrame of periodic returns).
    linkage_method:
        SciPy hierarchical-clustering linkage method used to build the
        dendrogram. ``"single"`` (the default) matches López de Prado's original
        formulation; ``"average"``, ``"complete"``, or ``"ward"`` are also valid.

    Returns
    -------
    PortfolioResult
        Weights plus metadata: the ``linkage`` matrix, the quasi-diagonal leaf
        ``order``, and the ``ordered_assets`` names. A single-asset panel returns
        the trivial ``[1.0]`` allocation.
    """
    mat, names = as_matrix(returns)
    mat_np = np.asarray(mat)
    if len(names) == 1:
        return _trivial_single_asset(names, mat_np, "Hierarchical Risk Parity")
    cov = np.asarray(sample_covariance(mat))
    corr = np.asarray(correlation_from_covariance(cov))

    dist = _corr_distance(corr)
    link = linkage(squareform(dist, checks=False), method=linkage_method)
    order = _quasi_diagonal_order(link)
    w = _recursive_bisection(cov, order)
    w = w / w.sum()

    ordered_names = [names[i] for i in order]
    return _finalize(
        w,
        names,
        "Hierarchical Risk Parity",
        mat_np,
        {"linkage": link.tolist(), "order": order, "ordered_assets": ordered_names},
    )


def hierarchical_equal_risk(
    returns,
    *,
    n_clusters: int = 4,
    linkage_method: str = "ward",
) -> PortfolioResult:
    """Hierarchical Equal Risk Contribution (HERC).

    Cut the dendrogram into ``n_clusters`` groups, allocate capital across
    clusters by inverse cluster-variance, and within each cluster by inverse
    variance. A robust, less order-sensitive cousin of HRP.
    """
    from scipy.cluster.hierarchy import fcluster

    mat, names = as_matrix(returns)
    mat_np = np.asarray(mat)
    if len(names) == 1:
        return _trivial_single_asset(names, mat_np, "Hierarchical Equal Risk (HERC)")
    cov = np.asarray(sample_covariance(mat))
    corr = np.asarray(correlation_from_covariance(cov))
    n = len(names)
    n_clusters = min(n_clusters, n)

    dist = _corr_distance(corr)
    link = linkage(squareform(dist, checks=False), method=linkage_method)
    labels = fcluster(link, t=n_clusters, criterion="maxclust")

    weights = np.zeros(n)
    cluster_idx = {c: np.where(labels == c)[0].tolist() for c in np.unique(labels)}

    # Across-cluster: inverse cluster variance.
    cvars = {c: _cluster_variance(cov, idx) for c, idx in cluster_idx.items()}
    inv = {c: 1.0 / v for c, v in cvars.items()}
    total = sum(inv.values())
    for c, idx in cluster_idx.items():
        cluster_cap = inv[c] / total
        sub = cov[np.ix_(idx, idx)]
        wr = _inverse_variance_weights(sub)
        weights[idx] = cluster_cap * wr

    weights = weights / weights.sum()
    return _finalize(
        weights,
        names,
        "Hierarchical Equal Risk (HERC)",
        mat_np,
        {"labels": labels.tolist(), "n_clusters": int(n_clusters)},
    )


def mst_centrality(returns, *, alpha: float = 1.0) -> PortfolioResult:
    """Minimum-spanning-tree centrality allocation.

    Build the MST of the correlation-distance network; assets with lower degree
    centrality (more peripheral, less coupled to the market core) receive more
    weight: ``w_i ∝ 1 / (degree_i)^alpha``.
    """
    mat, names = as_matrix(returns)
    mat_np = np.asarray(mat)
    corr = np.asarray(correlation_from_covariance(sample_covariance(mat)))

    dist = _corr_distance(corr)
    mst = minimum_spanning_tree(dist).toarray()
    # Symmetrize the (upper-triangular) MST adjacency and compute degrees.
    adj = (mst + mst.T) > 0
    degree = adj.sum(axis=1).astype(float)
    degree = np.clip(degree, 1.0, None)

    inv = 1.0 / degree**alpha
    w = inv / inv.sum()

    # Eigenvector centrality on the MST (for diagnostics / plotting).
    try:
        vals, vecs = np.linalg.eigh(adj.astype(float))
        centrality = np.abs(vecs[:, -1])
    except np.linalg.LinAlgError:  # pragma: no cover
        centrality = degree

    return _finalize(
        w,
        names,
        "MST Centrality",
        mat_np,
        {
            "degree": degree.tolist(),
            "eigenvector_centrality": centrality.tolist(),
            "mst_adjacency": adj.astype(int).tolist(),
        },
    )
