"""Learning-based portfolio optimizers.

Two representative methods:

* :func:`deep_sharpe` — an end-to-end differentiable allocation policy: a small
  MLP maps a window of recent returns to portfolio weights (via softmax), and is
  trained by gradient ascent to maximize the realized Sharpe ratio of the
  resulting portfolio. Pure JAX + optax, no flax dependency.
* :func:`online_gradient` — the exponentiated-gradient online portfolio (a
  Cover-style universal portfolio update) with sub-linear regret guarantees.

Both return a :class:`PortfolioResult` holding the final allocation; ``deep_sharpe``
additionally stores the trained parameters so it can be rolled forward.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxfolio.moments.estimators import as_matrix, mean_returns, sample_covariance
from jaxfolio.optimizers.base import portfolio_return, portfolio_volatility
from jaxfolio.types import PortfolioResult

Array = jnp.ndarray
_PPY = 252


def _init_mlp(key, sizes: list[int]) -> list[tuple[Array, Array]]:
    """Glorot-initialize a simple MLP as a list of ``(W, b)`` layers."""
    params = []
    keys = jax.random.split(key, len(sizes) - 1)
    for k, (din, dout) in zip(keys, zip(sizes[:-1], sizes[1:], strict=True), strict=True):
        scale = jnp.sqrt(2.0 / (din + dout))
        w = jax.random.normal(k, (din, dout)) * scale
        b = jnp.zeros(dout)
        params.append((w, b))
    return params


def _mlp_forward(params, x: Array) -> Array:
    """Forward pass with tanh hidden activations and a softmax output → weights."""
    h = x
    for w, b in params[:-1]:
        h = jnp.tanh(h @ w + b)
    w, b = params[-1]
    logits = h @ w + b
    return jax.nn.softmax(logits)


def deep_sharpe(
    returns,
    *,
    lookback: int = 60,
    hidden: tuple[int, ...] = (64, 32),
    epochs: int = 300,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> PortfolioResult:
    """Train a differentiable MLP allocation policy to maximize in-sample Sharpe.

    The policy consumes the flattened trailing ``lookback`` window of returns and
    emits long-only weights. Training maximizes the annualized Sharpe of the
    strategy's realized returns across all windows. The reported allocation is
    the policy applied to the most recent window.

    Parameters
    ----------
    lookback:
        Length of the trailing return window fed to the policy.
    hidden:
        Hidden layer widths of the MLP.
    epochs:
        Number of full-batch gradient ascent steps.
    """
    mat, names = as_matrix(returns)
    t, n = mat.shape
    if t <= lookback + 1:
        raise ValueError("Not enough observations for the requested lookback")

    # Build (window -> next-period return) supervised windows.
    windows = jnp.stack([mat[i : i + lookback].reshape(-1) for i in range(t - lookback)])
    next_rets = mat[lookback:]  # (num_windows, n)

    key = jax.random.PRNGKey(seed)
    sizes = [lookback * n, *hidden, n]
    params = _init_mlp(key, sizes)

    def strategy_returns(params) -> Array:
        weights = jax.vmap(lambda x: _mlp_forward(params, x))(windows)
        return jnp.sum(weights * next_rets, axis=1)

    def neg_sharpe(params) -> Array:
        r = strategy_returns(params)
        mean = jnp.mean(r)
        std = jnp.std(r) + 1e-8
        return -(mean / std) * jnp.sqrt(_PPY)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    loss_fn = jax.jit(jax.value_and_grad(neg_sharpe))

    @jax.jit
    def step(params, opt_state):
        loss, grads = loss_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    history = []
    for _ in range(epochs):
        params, opt_state, loss = step(params, opt_state)
        history.append(float(loss))

    # Allocation = policy on the most recent window.
    last_window = mat[-lookback:].reshape(-1)
    w = _mlp_forward(params, last_window)

    mu = mean_returns(mat)
    cov = sample_covariance(mat)
    w_np = np.asarray(w, dtype=float)
    ann_mu = float(portfolio_return(w, mu) * _PPY)
    ann_vol = float(portfolio_volatility(w, cov) * np.sqrt(_PPY))
    return PortfolioResult(
        weights=w_np,
        assets=names,
        method="Deep Sharpe (MLP policy)",
        expected_return=ann_mu,
        volatility=ann_vol,
        sharpe=ann_mu / ann_vol if ann_vol > 0 else None,
        metadata={
            "final_train_sharpe": -history[-1] if history else None,
            "epochs": epochs,
            "lookback": lookback,
            "params": params,  # retained so the policy can be rolled forward
        },
    )


def online_gradient(
    returns,
    *,
    eta: float = 0.05,
) -> PortfolioResult:
    """Exponentiated-gradient (EG) universal online portfolio (Helmbold et al.).

    Sequentially updates weights multiplicatively by the realized asset returns:
    ``w_{t+1,i} ∝ w_{t,i} * exp(eta * r_{t,i} / (w_t . r_t))``. This achieves
    sub-linear regret versus the best constant-rebalanced portfolio in hindsight.
    The reported allocation is the final weight vector; the wealth path is stored
    in the metadata.
    """
    mat, names = as_matrix(returns)
    t, n = mat.shape

    def update(w, r):
        gross = 1.0 + r
        port = jnp.dot(w, gross)
        grad = gross / port
        w_new = w * jnp.exp(eta * grad)
        w_new = w_new / jnp.sum(w_new)
        return w_new, port

    w0 = jnp.full(n, 1.0 / n)
    w_final, wealth_steps = jax.lax.scan(update, w0, mat)
    wealth = jnp.cumprod(wealth_steps)

    mu = mean_returns(mat)
    cov = sample_covariance(mat)
    ann_mu = float(portfolio_return(w_final, mu) * _PPY)
    ann_vol = float(portfolio_volatility(w_final, cov) * np.sqrt(_PPY))
    return PortfolioResult(
        weights=np.asarray(w_final, dtype=float),
        assets=names,
        method="Online EG Portfolio",
        expected_return=ann_mu,
        volatility=ann_vol,
        sharpe=ann_mu / ann_vol if ann_vol > 0 else None,
        metadata={
            "eta": eta,
            "final_wealth": float(wealth[-1]),
            "wealth_path": np.asarray(wealth, dtype=float).tolist(),
        },
    )
