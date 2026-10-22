"""Nearest-neighbor mutual-information helpers."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

from .resampling import (
    PermutationTestResult,
    apply_permutation_indices,
    local_permutation_indices,
    permutation_pvalue,
)

__all__ = [
    "estimate_conditional_mutual_information",
    "estimate_mutual_information",
    "mutual_information_test",
    "conditional_mutual_information_test",
]


def _as_2d(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=float)
    if values.ndim == 1:
        return values[:, None]
    if values.ndim != 2:
        raise ValueError("Expected a one- or two-dimensional array.")
    return values


def _validate_knn_inputs(
    x: np.ndarray, y: np.ndarray, *, n_neighbors: int
) -> tuple[np.ndarray, np.ndarray]:
    x_2d = _as_2d(x)
    y_2d = _as_2d(y)
    if x_2d.shape[0] != y_2d.shape[0]:
        raise ValueError("x and y must share the same sample count.")
    if x_2d.shape[0] <= n_neighbors:
        raise ValueError("n_neighbors must be smaller than the sample count.")
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be positive.")
    return x_2d, y_2d


def _count_neighbors(
    tree: cKDTree, points: np.ndarray, radii: np.ndarray
) -> np.ndarray:
    shrunk = np.maximum(np.nextafter(radii, 0.0), 0.0)
    return np.fromiter(
        (
            len(tree.query_ball_point(point, radius, p=np.inf)) - 1
            for point, radius in zip(points, shrunk, strict=False)
        ),
        dtype=int,
        count=points.shape[0],
    )


def estimate_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_neighbors: int = 5,
) -> float:
    """Estimate mutual information via the KSG nearest-neighbor estimator."""

    x_2d, y_2d = _validate_knn_inputs(x, y, n_neighbors=n_neighbors)
    n_samples = x_2d.shape[0]

    joint = np.concatenate((x_2d, y_2d), axis=1)
    joint_tree = cKDTree(joint)
    distances, _ = joint_tree.query(joint, k=n_neighbors + 1, p=np.inf)
    radii = np.asarray(distances[:, -1], dtype=float)

    x_tree = cKDTree(x_2d)
    y_tree = cKDTree(y_2d)
    nx = _count_neighbors(x_tree, x_2d, radii)
    ny = _count_neighbors(y_tree, y_2d, radii)

    estimate = (
        digamma(n_neighbors)
        + digamma(n_samples)
        - np.mean(digamma(nx + 1) + digamma(ny + 1))
    )
    return float(max(0.0, estimate))


def estimate_conditional_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    n_neighbors: int = 5,
) -> float:
    """Estimate conditional mutual information using a kNN max-norm estimator."""

    x_2d, y_2d = _validate_knn_inputs(x, y, n_neighbors=n_neighbors)
    z_2d = _as_2d(z)
    if z_2d.shape[0] != x_2d.shape[0]:
        raise ValueError("z must share the same sample count as x and y.")

    xyz = np.concatenate((x_2d, y_2d, z_2d), axis=1)
    xz = np.concatenate((x_2d, z_2d), axis=1)
    yz = np.concatenate((y_2d, z_2d), axis=1)

    xyz_tree = cKDTree(xyz)
    distances, _ = xyz_tree.query(xyz, k=n_neighbors + 1, p=np.inf)
    radii = np.asarray(distances[:, -1], dtype=float)

    xz_tree = cKDTree(xz)
    yz_tree = cKDTree(yz)
    z_tree = cKDTree(z_2d)
    nxz = _count_neighbors(xz_tree, xz, radii)
    nyz = _count_neighbors(yz_tree, yz, radii)
    nz = _count_neighbors(z_tree, z_2d, radii)

    estimate = np.mean(
        digamma(n_neighbors) + digamma(nz + 1) - digamma(nxz + 1) - digamma(nyz + 1)
    )
    return float(max(0.0, estimate))


def mutual_information_test(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_neighbors: int = 5,
    n_permutations: int = 64,
    random_state: int | np.random.Generator | None = 0,
) -> PermutationTestResult:
    """Estimate MI and calibrate it with an unconditional permutation null."""

    statistic = estimate_mutual_information(x, y, n_neighbors=n_neighbors)
    rng = np.random.default_rng(random_state)
    y_values = np.asarray(y)
    null = np.empty(n_permutations, dtype=float)
    for draw in range(n_permutations):
        null[draw] = estimate_mutual_information(
            x,
            rng.permutation(y_values),
            n_neighbors=n_neighbors,
        )
    return PermutationTestResult(
        statistic=statistic,
        pvalue=permutation_pvalue(statistic, null),
        null_distribution=null,
    )


def conditional_mutual_information_test(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    n_neighbors: int = 5,
    local_permutation_neighbors: int = 5,
    n_permutations: int = 64,
    random_state: int | np.random.Generator | None = 0,
) -> PermutationTestResult:
    """Estimate CMI and calibrate it with a deterministic local permutation null."""

    statistic = estimate_conditional_mutual_information(
        x,
        y,
        z,
        n_neighbors=n_neighbors,
    )
    permutations = local_permutation_indices(
        z,
        n_permutations=n_permutations,
        n_neighbors=local_permutation_neighbors,
        random_state=random_state,
    )
    y_array = np.asarray(y)
    null = np.empty(n_permutations, dtype=float)
    for draw, indices in enumerate(np.atleast_2d(permutations)):
        null[draw] = estimate_conditional_mutual_information(
            x,
            apply_permutation_indices(y_array, indices),
            z,
            n_neighbors=n_neighbors,
        )
    return PermutationTestResult(
        statistic=statistic,
        pvalue=permutation_pvalue(statistic, null),
        null_distribution=null,
    )
