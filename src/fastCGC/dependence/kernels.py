"""Kernel utilities for nonlinear dependence backends."""

from __future__ import annotations

from collections import OrderedDict
from typing import Generic, Hashable, TypeVar

import numpy as np

Array = np.ndarray
_T = TypeVar("_T")


def _as_2d_float(array: Array) -> Array:
    values = np.asarray(array, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError("expected a one- or two-dimensional array")
    return values


def pairwise_squared_distances(x: Array, y: Array | None = None) -> Array:
    """Return squared Euclidean distances between sample rows."""

    left = _as_2d_float(x)
    right = left if y is None else _as_2d_float(y)
    if left.shape[1] != right.shape[1]:
        raise ValueError("x and y must share the same feature dimension")
    left_norm = np.sum(left * left, axis=1, keepdims=True)
    right_norm = np.sum(right * right, axis=1, keepdims=True).T
    squared = left_norm + right_norm - 2.0 * left @ right.T
    return np.maximum(squared, 0.0)


def center_gram(gram: Array) -> Array:
    """Double-center a Gram matrix."""

    values = np.asarray(gram, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("gram must be a square matrix")
    row_mean = values.mean(axis=1, keepdims=True)
    col_mean = values.mean(axis=0, keepdims=True)
    grand_mean = float(values.mean())
    return values - row_mean - col_mean + grand_mean


def median_bandwidth(
    samples: Array,
    *,
    max_pairs: int = 4_096,
    random_state: int | None = 0,
) -> float:
    """Estimate an RBF bandwidth with a deterministic median heuristic."""

    data = _as_2d_float(samples)
    n_samples = data.shape[0]
    if n_samples < 2:
        return 1.0

    rng = np.random.default_rng(random_state)
    total_pairs = n_samples * (n_samples - 1) // 2
    if total_pairs <= max_pairs:
        squared = pairwise_squared_distances(data)
        upper = squared[np.triu_indices(n_samples, k=1)]
    else:
        left = rng.integers(0, n_samples, size=max_pairs)
        right = rng.integers(0, n_samples - 1, size=max_pairs)
        right = np.where(right >= left, right + 1, right)
        delta = data[left] - data[right]
        upper = np.sum(delta * delta, axis=1)
    upper = upper[upper > 0.0]
    if upper.size == 0:
        return 1.0
    bandwidth = float(np.sqrt(np.median(upper)))
    return bandwidth if np.isfinite(bandwidth) and bandwidth > 0.0 else 1.0


def estimate_gram_bytes(
    n_rows: int,
    n_cols: int | None = None,
    *,
    dtype: np.dtype | type[np.floating] = np.float64,
    n_buffers: int = 1,
) -> int:
    """Estimate the in-memory size of one or more Gram matrices."""

    cols = n_rows if n_cols is None else n_cols
    return int(n_rows) * int(cols) * np.dtype(dtype).itemsize * int(n_buffers)


def rbf_kernel(
    x: Array,
    y: Array | None = None,
    *,
    bandwidth: float | None = None,
    block_size: int | None = None,
) -> Array:
    """Return a dense or blocked Gaussian RBF kernel matrix."""

    left = _as_2d_float(x)
    right = left if y is None else _as_2d_float(y)
    if left.shape[1] != right.shape[1]:
        raise ValueError("x and y must share the same feature dimension")
    sigma = float(
        bandwidth
        if bandwidth is not None
        else median_bandwidth(
            left if y is None else np.vstack([left, right])
        )
    )
    if sigma <= 0.0 or not np.isfinite(sigma):
        raise ValueError("bandwidth must be finite and positive")

    gamma = 0.5 / (sigma * sigma)
    if block_size is None or block_size >= left.shape[0]:
        return np.exp(-gamma * pairwise_squared_distances(left, right))

    output = np.empty((left.shape[0], right.shape[0]), dtype=float)
    for start in range(0, left.shape[0], block_size):
        stop = min(start + block_size, left.shape[0])
        output[start:stop] = np.exp(
            -gamma * pairwise_squared_distances(left[start:stop], right)
        )
    return output


class ByteBudgetCache(Generic[_T]):
    """A tiny LRU cache that never exceeds its configured byte budget."""

    def __init__(self, max_bytes: int) -> None:
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        self.max_bytes = int(max_bytes)
        self._cache: OrderedDict[Hashable, tuple[_T, int]] = OrderedDict()
        self.current_bytes = 0
        self.hits = 0
        self.misses = 0

    def _value_nbytes(self, value: _T) -> int:
        nbytes = getattr(value, "nbytes", None)
        if nbytes is None:
            raise TypeError("cached values must expose an nbytes attribute")
        return int(nbytes)

    def get(self, key: Hashable) -> _T | None:
        entry = self._cache.get(key)
        if entry is None:
            self.misses += 1
            return None
        self._cache.move_to_end(key)
        self.hits += 1
        return entry[0]

    def set(self, key: Hashable, value: _T) -> _T:
        nbytes = self._value_nbytes(value)
        if key in self._cache:
            _, previous_nbytes = self._cache.pop(key)
            self.current_bytes -= previous_nbytes
        if self.max_bytes == 0 or nbytes > self.max_bytes:
            self._cache.clear()
            self.current_bytes = 0
            return value
        while self._cache and self.current_bytes + nbytes > self.max_bytes:
            _, (_, removed_nbytes) = self._cache.popitem(last=False)
            self.current_bytes -= removed_nbytes
        self._cache[key] = (value, nbytes)
        self.current_bytes += nbytes
        return value

    def stats(self) -> dict[str, int]:
        return {
            "entries": len(self._cache),
            "current_bytes": self.current_bytes,
            "hits": self.hits,
            "misses": self.misses,
        }
