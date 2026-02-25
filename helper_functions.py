"""
Helper functions for Gaussian-kernel smoothing of IDR/EasyUQ discrete predictive distributions.

Includes:
- one-fit bandwidth tuning via the logarithmic score, LogS(F,y) = -log f(y),
- Gaussian-kernel smoothed PIT.

The bandwidth tuning follows the one-fit approach described in Walz et al. (2024).

Reference
---------
Walz, E.-M., Henzi, A., Ziegel, J., and Gneiting, T. (2024).
Easy Uncertainty Quantification (EasyUQ): Generating Predictive Distributions from Single-Valued Model Output.
SIAM Review, 66(1), 91–122.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

def _logsumexp(a: np.ndarray) -> float:
    """
    Stable log(sum(exp(a))).
    """
    a = np.asarray(a, dtype=float)
    m = np.max(a)
    if not np.isfinite(m):
        return -np.inf
    return float(m + np.log(np.sum(np.exp(a - m))))


def _log_kernel_mixture_density(y: float, support: np.ndarray, weights: np.ndarray, h: float) -> float:
    """
    Log density of a Gaussian-kernel mixture computed in a numerically stable way.
    """
    support = np.asarray(support, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mask = weights > 0
    if not np.any(mask):
        return -np.inf

    z = (y - support[mask]) / h
    log_components = (
        np.log(weights[mask])
        - 0.5 * z * z
        - np.log(np.sqrt(2.0 * np.pi) * h)
    )
    return _logsumexp(log_components)


def tune_gaussian_bandwidth_onefit(preds, y, bounds=None):
    """
    One-fit bandwidth tuning for Gaussian-kernel smoothing as described in .

    Minimizes the mean log score of the Gaussian-kernel smoothed predictive distribution induced by IDR.

    Parameters
    ----------
    preds : idrpredict-like
        Must provide `preds.predictions[i].points` and `.ecdf`.
    y : array_like
        Observations, same length as `preds.predictions`.
    bounds : tuple(float, float) or None
        Bounds for the bandwidth search. If None, uses (tiny, max(y)).

    Returns
    -------
    h_opt : float
        Optimal bandwidth.
    obj_min : float
        Minimum mean log score.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n == 0:
        raise ValueError("y must be non-empty")

    tiny = np.finfo(float).tiny
    if bounds is None:
        hi = float(np.max(y))
        if not np.isfinite(hi) or hi <= 0.0:
            hi = 1.0
        bounds = (tiny, hi)

    def objective(h: float) -> float:
        total = 0.0

        for pred, yi in zip(preds.predictions, y):
            pts = np.asarray(pred.points, dtype=float)
            ecdf = np.asarray(pred.ecdf, dtype=float)
            w = np.diff(np.concatenate(([0.0], ecdf)))  # weights from CDF

            # Leave-one-out style adjustment: if yi is in the support, drop its mass and renormalize.
            w_adj = w
            hit = np.flatnonzero(pts == yi)
            if hit.size > 0:
                w_adj = w.copy()
                w_adj[hit] = 0.0
                s = float(np.sum(w_adj))
                if s > 0.0:
                    w_adj /= s
                else:
                    w_adj = w # degenerate case (e.g., point mass at yi): fall back to original weights.

            logf = _log_kernel_mixture_density(float(yi), pts, w_adj, float(h))
            total += -logf

        return total / n

    res = minimize_scalar(objective, method="bounded", bounds=bounds)
    return float(res.x), float(res.fun)


def pit_gaussian_kernel(preds, y, h):
    """
    Gaussian-kernel smoothed PIT.

    Parameters
    ----------
    preds : idrpredict
        Object with `preds.predictions[i].points` and `.ecdf`.
    y : array_like
        Observations, same length as `preds.predictions`.
    h : float
        Bandwidth.

    Returns
    -------
    ndarray
        Smoothed PIT values.
    """
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y, dtype=float)

    for i, (pred, yi) in enumerate(zip(preds.predictions, y)):
        pts = np.asarray(pred.points, dtype=float)
        ecdf = np.asarray(pred.ecdf, dtype=float)
        w = np.diff(np.concatenate(([0.0], ecdf)))
        out[i] = float(np.sum(w * norm.cdf(yi, loc=pts, scale=h)))

    return out