"""HSIC helpers and randomized marginal dependence tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable

import numpy as np

from .kernels import ByteBudgetCache, _as_2d_float, center_gram, median_bandwidth, rbf_kernel
from .random_features import (
    complete_shift_feature_statistics,
    feature_cross_covariance_norm,
    random_fourier_features,
)
from .resampling import circular_shift_indices, permutation_pvalue

Array = np.ndarray


@dataclass(frozen=True)
class PairwiseTestResult:
    """Pairwise dependence test output."""

    statistic: Array
    effect: Array
    pvalue: Array
    evaluated: Array
    metadata: dict[str, Any] = field(default_factory=dict)


def _is_constant(samples: Array) -> bool:
    values = np.asarray(samples, dtype=float)
    return values.shape[0] < 2 or np.allclose(values, values[0])


def biased_hsic(
    x: Array,
    y: Array,
    *,
    bandwidth_x: float | None = None,
    bandwidth_y: float | None = None,
    block_size: int | None = None,
) -> float:
    """Return the biased HSIC statistic."""

    left = _as_2d_float(x)
    right = _as_2d_float(y)
    if left.shape[0] != right.shape[0]:
        raise ValueError("x and y must share the same sample axis")
    if left.shape[0] < 2:
        return 0.0
    gram_x = center_gram(rbf_kernel(left, bandwidth=bandwidth_x, block_size=block_size))
    gram_y = center_gram(rbf_kernel(right, bandwidth=bandwidth_y, block_size=block_size))
    return float(np.sum(gram_x * gram_y) / (left.shape[0] - 1) ** 2)


def _complete_shift_gram_statistics(gram_x: Array, gram_y: Array) -> Array:
    """Evaluate the complete same-axis circular-shift orbit in O(n² log n)."""

    left = np.asarray(gram_x, dtype=float)
    right = np.asarray(gram_y, dtype=float)
    if left.ndim != 2 or left.shape[0] != left.shape[1] or right.shape != left.shape:
        raise ValueError("gram_x and gram_y must be square matrices with matching shapes")
    n_samples = left.shape[0]
    if n_samples < 2:
        return np.zeros(n_samples, dtype=float)

    indices = np.arange(n_samples)
    statistics = np.zeros(n_samples, dtype=float)
    for diagonal in range(n_samples):
        left_values = left[indices, (indices + diagonal) % n_samples]
        right_values = right[indices, (indices + diagonal) % n_samples]
        statistics += np.fft.irfft(
            np.fft.rfft(left_values) * np.conj(np.fft.rfft(right_values)),
            n=n_samples,
        )
    statistics = np.concatenate([statistics[:1], statistics[:0:-1]])
    return statistics / (n_samples - 1) ** 2


def hsic_test(
    x: Array,
    y: Array,
    *,
    bandwidth_x: float | None = None,
    bandwidth_y: float | None = None,
    block_size: int | None = None,
    n_shifts: int | None = None,
    random_state: int | None = 0,
    memory_budget_mb: int = 512,
    max_samples: int | None = 512,
) -> PairwiseTestResult:
    """Evaluate exact-kernel HSIC with a circular-shift randomization test."""

    left = _as_2d_float(x)
    right = _as_2d_float(y)
    if left.shape[0] != right.shape[0]:
        raise ValueError("x and y must share the same sample axis")
    n_samples = left.shape[0]
    evaluated = np.array([not (_is_constant(left) or _is_constant(right))], dtype=bool)
    if not evaluated[0]:
        return PairwiseTestResult(
            statistic=np.array([0.0]),
            effect=np.array([0.0]),
            pvalue=np.array([1.0]),
            evaluated=evaluated,
            metadata={"backend": "hsic", "null_method": "complete_shift"},
        )
    if memory_budget_mb < 1:
        raise ValueError("memory_budget_mb must be positive")
    if max_samples is not None and n_samples > max_samples:
        raise ValueError(
            "Exact HSIC sample guard exceeded; use the rff_hsic backend."
        )
    required_bytes = 4 * n_samples * n_samples * np.dtype(float).itemsize
    if required_bytes > memory_budget_mb * 1024**2:
        raise ValueError(
            "Exact HSIC memory budget exceeded before allocation; "
            f"required {required_bytes / 1024**2:.1f} MiB for {n_samples} samples."
        )

    x_bandwidth = bandwidth_x or median_bandwidth(left, random_state=random_state)
    y_bandwidth = bandwidth_y or median_bandwidth(right, random_state=random_state)
    gram_x = center_gram(
        rbf_kernel(left, bandwidth=x_bandwidth, block_size=block_size)
    )
    gram_y = center_gram(
        rbf_kernel(right, bandwidth=y_bandwidth, block_size=block_size)
    )
    orbit = _complete_shift_gram_statistics(gram_x, gram_y)
    observed = float(max(orbit[0], 0.0))
    shifts = circular_shift_indices(
        n_samples,
        n_shifts=n_shifts,
        random_state=random_state,
    )
    null = orbit[shifts]
    return PairwiseTestResult(
        statistic=np.array([observed]),
        effect=np.array([np.sqrt(observed)]),
        pvalue=np.array([permutation_pvalue(observed, null)]),
        evaluated=evaluated,
        metadata={
            "backend": "hsic",
            "bandwidth_x": x_bandwidth,
            "bandwidth_y": y_bandwidth,
            "null_method": "complete_shift" if n_shifts is None else "sampled_shift",
            "n_shifts": int(shifts.size),
        },
    )


def rff_hsic_statistic(
    x: Array,
    y: Array,
    *,
    n_features: int = 64,
    bandwidth_x: float | None = None,
    bandwidth_y: float | None = None,
    random_state: int | None = 0,
    key: object | None = None,
) -> float:
    """Return the randomized-feature HSIC statistic."""

    left = _as_2d_float(x)
    right = _as_2d_float(y)
    x_bandwidth = bandwidth_x or median_bandwidth(left, random_state=random_state)
    y_bandwidth = bandwidth_y or median_bandwidth(right, random_state=random_state)
    x_features = random_fourier_features(
        left,
        n_features=n_features,
        bandwidth=x_bandwidth,
        random_state=random_state,
        key=(key, "x"),
    )
    y_features = random_fourier_features(
        right,
        n_features=n_features,
        bandwidth=y_bandwidth,
        random_state=random_state,
        key=(key, "y"),
    )
    return feature_cross_covariance_norm(x_features, y_features)


def rff_hsic_test(
    x: Array,
    y: Array,
    *,
    n_features: int = 64,
    bandwidth_x: float | None = None,
    bandwidth_y: float | None = None,
    random_state: int | None = 0,
    key: object | None = None,
    feature_pair_batch_size: int | None = None,
) -> PairwiseTestResult:
    """Evaluate complete-shift RFF-HSIC for one pair."""

    left = _as_2d_float(x)
    right = _as_2d_float(y)
    if left.shape[0] != right.shape[0]:
        raise ValueError("x and y must share the same sample axis")
    evaluated = np.array([not (_is_constant(left) or _is_constant(right))], dtype=bool)
    if not evaluated[0]:
        return PairwiseTestResult(
            statistic=np.array([0.0]),
            effect=np.array([0.0]),
            pvalue=np.array([1.0]),
            evaluated=evaluated,
            metadata={"backend": "rff_hsic", "null_method": "complete_shift"},
        )

    x_bandwidth = bandwidth_x or median_bandwidth(left, random_state=random_state)
    y_bandwidth = bandwidth_y or median_bandwidth(right, random_state=random_state)
    x_features = random_fourier_features(
        left,
        n_features=n_features,
        bandwidth=x_bandwidth,
        random_state=random_state,
        key=(key, "x"),
    )
    y_features = random_fourier_features(
        right,
        n_features=n_features,
        bandwidth=y_bandwidth,
        random_state=random_state,
        key=(key, "y"),
    )
    null_statistics = complete_shift_feature_statistics(
        x_features,
        y_features,
        feature_pair_batch_size=feature_pair_batch_size,
    )
    observed = float(null_statistics[0])
    exceedances = np.count_nonzero(null_statistics[1:] >= observed - 1e-12)
    pvalue = float((exceedances + 1.0) / null_statistics.size)
    return PairwiseTestResult(
        statistic=np.array([observed]),
        effect=np.array([np.sqrt(max(observed, 0.0))]),
        pvalue=np.array([pvalue]),
        evaluated=evaluated,
        metadata={
            "backend": "rff_hsic",
            "bandwidth_x": x_bandwidth,
            "bandwidth_y": y_bandwidth,
            "n_features": n_features,
            "null_method": "complete_shift",
        },
    )


@dataclass
class RffHsicBackend:
    """Reusable cached RFF-HSIC evaluator for lagged rows."""

    n_features: int = 64
    bandwidth: float | str = "median"
    random_state: int | None = 0
    cache_max_bytes: int = 128 * 1024 * 1024
    feature_pair_batch_size: int | None = None

    def __post_init__(self) -> None:
        self._cache = ByteBudgetCache[Array](self.cache_max_bytes)

    def _resolve_bandwidth(self, samples: Array, side_key: Hashable) -> float:
        if self.bandwidth == "median":
            return median_bandwidth(samples, random_state=self.random_state)
        bandwidth = float(self.bandwidth)
        if bandwidth <= 0.0:
            raise ValueError(f"{side_key}: bandwidth must be positive")
        return bandwidth

    def _features(self, samples: Array, cache_key: Hashable) -> tuple[Array, float]:
        data = _as_2d_float(samples)
        bandwidth = self._resolve_bandwidth(data, cache_key)
        key = ("rff_hsic", cache_key, self.n_features, bandwidth)
        cached = self._cache.get(key)
        if cached is not None:
            return cached, bandwidth
        features = random_fourier_features(
            data,
            n_features=self.n_features,
            bandwidth=bandwidth,
            random_state=self.random_state,
            key=cache_key,
        )
        return self._cache.set(key, features), bandwidth

    def evaluate_pairs(
        self,
        x_rows: Array,
        y_rows: Array,
        *,
        row_keys: list[tuple[Hashable, Hashable]] | None = None,
    ) -> PairwiseTestResult:
        """Evaluate one or more paired univariate rows."""

        left = np.asarray(x_rows, dtype=float)
        right = np.asarray(y_rows, dtype=float)
        if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
            raise ValueError("x_rows and y_rows must share a two-dimensional shape")
        keys = row_keys or [(f"x-{idx}", f"y-{idx}") for idx in range(left.shape[0])]
        if len(keys) != left.shape[0]:
            raise ValueError("row_keys must match the number of row pairs")

        statistics = np.zeros(left.shape[0], dtype=float)
        effects = np.zeros(left.shape[0], dtype=float)
        pvalues = np.ones(left.shape[0], dtype=float)
        evaluated = np.zeros(left.shape[0], dtype=bool)
        bandwidths_x: list[float] = []
        bandwidths_y: list[float] = []
        for index, ((x_key, y_key), x_row, y_row) in enumerate(zip(keys, left, right)):
            if _is_constant(x_row) or _is_constant(y_row):
                continue
            x_features, bandwidth_x = self._features(x_row[:, None], x_key)
            y_features, bandwidth_y = self._features(y_row[:, None], y_key)
            null_statistics = complete_shift_feature_statistics(
                x_features,
                y_features,
                feature_pair_batch_size=self.feature_pair_batch_size,
            )
            observed = float(null_statistics[0])
            exceedances = np.count_nonzero(
                null_statistics[1:] >= observed - 1e-12
            )
            bandwidths_x.append(bandwidth_x)
            bandwidths_y.append(bandwidth_y)
            statistics[index] = observed
            effects[index] = np.sqrt(max(observed, 0.0))
            pvalues[index] = (exceedances + 1.0) / null_statistics.size
            evaluated[index] = True
        metadata = self._cache.stats() | {
            "backend": "rff_hsic",
            "n_features": self.n_features,
            "bandwidth_x": bandwidths_x,
            "bandwidth_y": bandwidths_y,
        }
        return PairwiseTestResult(
            statistic=statistics,
            effect=effects,
            pvalue=pvalues,
            evaluated=evaluated,
            metadata=metadata,
        )
