import numpy as np
from typing import Iterable


__all__ = [
    "roc_auc",
    "pr_auc",
]


def roc_auc(curve: Iterable[tuple[float, float]]) -> float:
    """Calculate the ROC AUC (Area Under Curve) from a curve of points (false positive rate, true positive rate)."""
    # add points (0, 0) and (1, 1) to the curve for completeness
    x = np.array([0.0, *[p[0] for p in curve], 1.0], dtype=np.float32)
    y = np.array([0.0, *[p[1] for p in curve], 1.0], dtype=np.float32)
    # sort the points by x value
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    auc = np.trapezoid(y, x)  # integrate using trapezoidal rule
    return float(auc)


def pr_auc(curve: Iterable[tuple[float, float]]) -> float:
    """Calculate the PR AUC (Area Under Curve) from a curve of points (recall, precision)."""
    # add points (0, 1) and (1, 0) to the curve for completeness
    x = np.array([0.0, *[p[0] for p in curve], 1.0], dtype=np.float32)
    y = np.array([1.0, *[p[1] for p in curve], 0.0], dtype=np.float32)
    # sort the points by x value
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    auc = np.trapezoid(y, x)  # integrate using trapezoidal rule
    return float(auc)
