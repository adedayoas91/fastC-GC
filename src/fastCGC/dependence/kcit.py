"""Reference kernel conditional-independence helpers with exact-kernel guards."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .resampling import (
    PermutationTestResult,
    apply_permutation_indices,
    local_permutation_indices,
    permutation_pvalue,
)

__all__ = [
    "KcitResult",
    "estimate_exact_kernel_bytes",
    "guard_exact_kcit_feasibility",
    "kcit_statistic",
    "kcit_test",
]


@dataclass(frozen=True)
class KcitResult(PermutationTestResult):
    """KCIT-style result with kernel metadata."""

    bandwidth_x: float
    bandwidth_y: float
    bandwidth_z: float


def _as_2d(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=float)
    if values.ndim == 1:
        return values[:, None]
    if values.ndim != 2:
        raise ValueError("Expected a one- or two-dimensional array.")
    return values


def estimate_exact_kernel_bytes(
    n_samples: int,
    *,
    n_matrices: int = 6,
    dtype: np.dtype = np.dtype(np.float64),
) -> int:
    """Estimate dense exact-kernel memory in bytes."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive.")
    if n_matrices < 1:
        raise ValueError("n_matrices must be positive.")
    return int(n_samples * n_samples * np.dtype(dtype).itemsize * n_matrices)


def guard_exact_kcit_feasibility(
    n_samples: int,
    *,
    memory_budget_mb: int = 512,
    max_samples: int | None = 512,
) -> None:
    """Reject exact KCIT configurations before large allocations occur."""

    if memory_budget_mb < 1:
        raise ValueError("memory_budget_mb must be positive.")
    if max_samples is not None and n_samples > max_samples:
        raise ValueError(
            "Exact KCIT sample guard exceeded; use a lower-rank or approximate backend."
        )

    required_mb = estimate_exact_kernel_bytes(n_samples) / (1024.0**2)
    if required_mb > memory_budget_mb:
        raise ValueError(
            "Exact KCIT memory budget exceeded before allocation; "
            f"required {required_mb:.1f} MiB for {n_samples} samples."
        )


def _median_bandwidth(values: np.ndarray) -> float:
    x = _as_2d(values)
    if x.shape[0] < 2:
        return 1.0
    step = max(1, x.shape[0] // 128)
    sample = x[::step]
    diffs = sample[:, None, :] - sample[None, :, :]
    sq_dist = np.sum(diffs * diffs, axis=2)
    upper = sq_dist[np.triu_indices_from(sq_dist, k=1)]
    positive = upper[upper > 0]
    if positive.size == 0:
        return 1.0
    return float(np.sqrt(0.5 * np.median(positive)))


def _rbf_kernel(values: np.ndarray, bandwidth: float) -> np.ndarray:
    x = _as_2d(values)
    diffs = x[:, None, :] - x[None, :, :]
    sq_dist = np.sum(diffs * diffs, axis=2)
    scale = max(float(bandwidth), np.finfo(float).eps)
    return np.exp(-sq_dist / (2.0 * scale * scale))


def _center_gram(gram: np.ndarray) -> np.ndarray:
    row_mean = gram.mean(axis=1, keepdims=True)
    col_mean = gram.mean(axis=0, keepdims=True)
    overall = gram.mean()
    return gram - row_mean - col_mean + overall


def _residualize_kernel(gram: np.ndarray, conditioner: np.ndarray, ridge: float) -> np.ndarray:
    identity = np.eye(gram.shape[0], dtype=float)
    solve = np.linalg.solve(conditioner + ridge * identity, gram)
    residual = gram - conditioner @ solve
    return _center_gram((residual + residual.T) * 0.5)


def kcit_statistic(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    bandwidth_x: float | None = None,
    bandwidth_y: float | None = None,
    bandwidth_z: float | None = None,
    ridge: float = 1e-6,
    memory_budget_mb: int = 512,
    max_samples: int | None = 512,
) -> float:
    """Return a guarded exact-kernel conditional-dependence statistic."""

    x_2d = _as_2d(x)
    y_2d = _as_2d(y)
    z_2d = _as_2d(z)
    if not (x_2d.shape[0] == y_2d.shape[0] == z_2d.shape[0]):
        raise ValueError("x, y, and z must share the same sample count.")
    if ridge <= 0.0:
        raise ValueError("ridge must be positive.")

    n_samples = x_2d.shape[0]
    guard_exact_kcit_feasibility(
        n_samples,
        memory_budget_mb=memory_budget_mb,
        max_samples=max_samples,
    )

    sigma_x = _median_bandwidth(x_2d) if bandwidth_x is None else float(bandwidth_x)
    sigma_y = _median_bandwidth(y_2d) if bandwidth_y is None else float(bandwidth_y)
    sigma_z = _median_bandwidth(z_2d) if bandwidth_z is None else float(bandwidth_z)

    kx = _center_gram(_rbf_kernel(x_2d, sigma_x))
    ky = _center_gram(_rbf_kernel(y_2d, sigma_y))
    kz = _center_gram(_rbf_kernel(z_2d, sigma_z))
    rx = _residualize_kernel(kx, kz, ridge)
    ry = _residualize_kernel(ky, kz, ridge)
    statistic = np.trace(rx @ ry) / float(max((n_samples - 1) ** 2, 1))
    return float(max(0.0, statistic))


def kcit_test(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    bandwidth_x: float | None = None,
    bandwidth_y: float | None = None,
    bandwidth_z: float | None = None,
    ridge: float = 1e-6,
    memory_budget_mb: int = 512,
    max_samples: int | None = 512,
    local_permutation_neighbors: int = 5,
    n_permutations: int = 64,
    random_state: int | np.random.Generator | None = 0,
) -> KcitResult:
    """Calibrate the reference KCIT statistic with a local permutation null."""

    x_2d = _as_2d(x)
    y_array = np.asarray(y, dtype=float)
    z_2d = _as_2d(z)
    sigma_x = _median_bandwidth(x_2d) if bandwidth_x is None else float(bandwidth_x)
    sigma_y = _median_bandwidth(y_array) if bandwidth_y is None else float(bandwidth_y)
    sigma_z = _median_bandwidth(z_2d) if bandwidth_z is None else float(bandwidth_z)

    statistic = kcit_statistic(
        x_2d,
        y_array,
        z_2d,
        bandwidth_x=sigma_x,
        bandwidth_y=sigma_y,
        bandwidth_z=sigma_z,
        ridge=ridge,
        memory_budget_mb=memory_budget_mb,
        max_samples=max_samples,
    )
    permutations = local_permutation_indices(
        z_2d,
        n_permutations=n_permutations,
        n_neighbors=local_permutation_neighbors,
        random_state=random_state,
    )
    null = np.empty(n_permutations, dtype=float)
    for draw, indices in enumerate(np.atleast_2d(permutations)):
        null[draw] = kcit_statistic(
            x_2d,
            apply_permutation_indices(y_array, indices),
            z_2d,
            bandwidth_x=sigma_x,
            bandwidth_y=sigma_y,
            bandwidth_z=sigma_z,
            ridge=ridge,
            memory_budget_mb=memory_budget_mb,
            max_samples=max_samples,
        )

    return KcitResult(
        statistic=statistic,
        pvalue=permutation_pvalue(statistic, null),
        null_distribution=null,
        bandwidth_x=sigma_x,
        bandwidth_y=sigma_y,
        bandwidth_z=sigma_z,
    )
