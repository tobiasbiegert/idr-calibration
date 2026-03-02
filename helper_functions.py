"""
Helper functions for Gaussian-kernel smoothing and linear interpolation of IDR/EasyUQ discrete predictive distributions.

Includes:
- one-fit bandwidth tuning for Gaussian-kernel smoothing via the logarithmic score, LogS(F,y) = -log f(y),
- Gaussian-kernel smoothed PIT,
- PIT values based on a linearly interpolated predictive CDF,
- CRPS values based on a linearly interpolated predictive CDF.

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
    One-fit bandwidth tuning for Gaussian-kernel smoothing.

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

    if y.ndim > 1:
        raise ValueError("y must be a 1-D array")
    if y.size != len(preds.predictions):
        raise ValueError("y must have same length as predictions")
    if y.size == 0:
        raise ValueError("y must be non-empty")
        
    n = y.size

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
                    w_adj = w  # degenerate case (e.g., point mass at yi): fall back to original weights.

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
    preds : idrpredict-like
        Must provide `preds.predictions[i].points` and `.ecdf`.
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
    if y.ndim > 1:
        raise ValueError("y must be a 1-D array")
    if y.size != len(preds.predictions):
        raise ValueError("y must have same length as predictions")
    
    out = np.empty_like(y, dtype=float)

    for i, (pred, yi) in enumerate(zip(preds.predictions, y)):
        pts = np.asarray(pred.points, dtype=float)
        ecdf = np.asarray(pred.ecdf, dtype=float)
        w = np.diff(np.concatenate(([0.0], ecdf)))
        out[i] = float(np.sum(w * norm.cdf(yi, loc=pts, scale=h)))

    return out

def interp_adapt_linear(x, p, threshold):
    """
    Linear interpolation of a predictive CDF at a single threshold.

    Parameters
    ----------
    x : array_like
        One-dimensional, increasing support points.
    p : array_like
        CDF values at `x`.
    threshold : float
        Point at which the interpolant is evaluated.

    Returns
    -------
    float
        Interpolated CDF value at `threshold`.
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)
    threshold = float(threshold)

    if threshold < np.min(x):
        return 0.0
    if threshold > np.max(x):
        return float(p[-1])
    if x.size == 1:
        return float(p[0])

    return float(np.interp(threshold, x, p))

def pit_linear(preds, y):
    """
    PIT values based on a linearly interpolated predictive CDF.

    Parameters
    ----------
    preds : idrpredict-like
        Must provide `preds.predictions[i].points` and `.ecdf`.
    y : array_like
        Observations, same length as `preds.predictions`.

    Returns
    -------
    ndarray
        PIT values obtained from the linearly interpolated predictive CDF.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim > 1:
        raise ValueError("y must be a 1-D array")
    if y.size != len(preds.predictions):
        raise ValueError("y must have same length as predictions")

    def pit0(pred, yi):
        return interp_adapt_linear(
            x=pred.points,
            p=pred.ecdf,
            threshold=yi,
        )

    return np.array(list(map(pit0, preds.predictions, y)), dtype=float)

def _integral_affine_square(a, b, lo, hi):
    """
    Integral of (a + b z)^2 over [lo, hi].
    """
    return (b * b / 3.0) * (hi**3 - lo**3) + (a * b) * (hi**2 - lo**2) + (a * a) * (hi - lo)


def _integrate_piecewise_linear_cdf(x, p, lo, hi, which="F2"):
    """
    Integrate F(z)^2 or (1 - F(z))^2 over [lo, hi], where F is piecewise linear.
    """
    if hi <= lo:
        return 0.0

    total = 0.0

    for j in range(len(x) - 1):
        seg_lo = max(lo, x[j])
        seg_hi = min(hi, x[j + 1])
        if seg_hi <= seg_lo:
            continue

        b = (p[j + 1] - p[j]) / (x[j + 1] - x[j])
        a = p[j] - b * x[j]

        if which == "F2":
            total += _integral_affine_square(a, b, seg_lo, seg_hi)
        elif which == "1-F2":
            total += _integral_affine_square(1.0 - a, -b, seg_lo, seg_hi)
        else:
            raise ValueError("which must be 'F2' or '1-F2'")

    return total


def crps_linear(preds, y):
    """
    CRPS for a predictive CDF obtained by linear interpolation of (points, ecdf).

    Parameters
    ----------
    preds : idrpredict-like
        Must provide `preds.predictions[i].points` and `.ecdf`.
    y : array_like
        Observations, same length as `preds.predictions`.

    Returns
    -------
    ndarray
        CRPS values for the linearly interpolated predictive CDF.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim > 1:
        raise ValueError("y must be a 1-D array")
        
    predictions = preds.predictions
    if y.size != len(predictions):
        raise ValueError("y must have same length as predictions")

    out = np.empty(len(predictions), dtype=float)

    for i, (pred, yi) in enumerate(zip(predictions, y)):
        x = np.asarray(pred.points, dtype=float)
        p = np.asarray(pred.ecdf, dtype=float)

        if yi <= x[0]:
            out[i] = (x[0] - yi) + _integrate_piecewise_linear_cdf(x, p, x[0], x[-1], which="1-F2")
        elif yi >= x[-1]:
            out[i] = _integrate_piecewise_linear_cdf(x, p, x[0], x[-1], which="F2") + (yi - x[-1])
        else:
            out[i] = (
                _integrate_piecewise_linear_cdf(x, p, x[0], yi, which="F2")
                + _integrate_piecewise_linear_cdf(x, p, yi, x[-1], which="1-F2")
            )

    return out