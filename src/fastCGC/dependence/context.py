from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FitContext:
    data: np.ndarray
    shifted: np.ndarray
    standardized: np.ndarray
    valid_rows: np.ndarray
    allowed_pairs: np.ndarray
    n_variables: int
    n_pasts: int
    n_samples: int
    row_variable: np.ndarray
    row_lag: np.ndarray
    row_mean: np.ndarray
    row_scale: np.ndarray



def build_fit_context(data: np.ndarray, n_pasts: int) -> FitContext:
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError("data must have shape (n_variables, n_samples).")
    if array.shape[1] <= n_pasts:
        raise ValueError("data must contain more samples than n_pasts.")
    if not np.all(np.isfinite(array)):
        raise ValueError("data must contain only finite values.")

    n_variables = array.shape[0]
    if n_pasts == 0:
        shifted = array.copy()
    else:
        shifted = array[:, n_pasts:]
        for index in range(n_pasts):
            start = n_pasts - 1 - index
            stop = -index - 1
            shifted = np.r_[shifted, array[:, start:stop]]

    means = shifted.mean(axis=1, keepdims=True)
    centered = shifted - means
    scales = np.sqrt(np.mean(centered**2, axis=1))
    tolerance = np.finfo(float).eps * np.maximum(1.0, np.abs(means[:, 0]))
    valid = scales > tolerance
    standardized = np.zeros_like(centered)
    standardized[valid] = centered[valid] / scales[valid, None]

    rows = shifted.shape[0]
    allowed = valid[:, None] & valid[:n_variables][None, :]
    diagonal_size = min(rows, n_variables)
    diagonal = np.arange(diagonal_size)
    allowed[diagonal, diagonal] = False

    row_indices = np.arange(rows, dtype=int)
    return FitContext(
        data=array,
        shifted=shifted,
        standardized=standardized,
        valid_rows=valid,
        allowed_pairs=allowed,
        n_variables=n_variables,
        n_pasts=n_pasts,
        n_samples=shifted.shape[1],
        row_variable=row_indices % n_variables,
        row_lag=row_indices // n_variables,
        row_mean=means[:, 0],
        row_scale=scales,
    )
