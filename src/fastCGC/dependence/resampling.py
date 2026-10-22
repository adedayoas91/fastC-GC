"""Deterministic resampling helpers for nonlinear dependence backends."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

__all__ = [
    "PermutationTestResult",
    "apply_permutation_indices",
    "circular_shift_indices",
    "local_permutation_indices",
    "permutation_pvalue",
]


@dataclass(frozen=True)
class PermutationTestResult:
    """Container for sampled-null test outputs."""

    statistic: float
    pvalue: float
    null_distribution: np.ndarray


def _as_2d(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=float)
    if values.ndim == 1:
        return values[:, None]
    if values.ndim != 2:
        raise ValueError("Expected a one- or two-dimensional array.")
    return values


def circular_shift_indices(
    n_samples: int,
    *,
    n_shifts: int | None = None,
    include_zero: bool = False,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Return deterministic circular-shift offsets.

    When ``n_shifts`` is omitted, the complete orbit is returned.
    """

    if n_samples < 2:
        raise ValueError("n_samples must be at least two.")

    candidates = np.arange(0 if include_zero else 1, n_samples, dtype=int)
    if n_shifts is None or n_shifts >= candidates.size:
        return candidates

    rng = np.random.default_rng(random_state)
    chosen = np.sort(rng.choice(candidates, size=n_shifts, replace=False))
    return chosen.astype(int, copy=False)


def local_permutation_indices(
    conditioning: np.ndarray,
    *,
    n_permutations: int = 1,
    n_neighbors: int = 5,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Return local-permutation indices conditioned on neighborhood structure.

    Each output row contains a permutation-like index map for one sampled null.
    Indices are preferentially chosen from each sample's nearest conditioning
    neighbors, while avoiding repeated assignments whenever feasible.
    """

    if n_permutations < 1:
        raise ValueError("n_permutations must be positive.")
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be positive.")

    z = _as_2d(conditioning)
    n_samples = z.shape[0]
    if n_samples < 2:
        raise ValueError("conditioning must contain at least two samples.")

    max_neighbors = min(n_neighbors, n_samples - 1)
    tree = cKDTree(z)
    _, neighbors = tree.query(z, k=max_neighbors + 1, p=np.inf)
    if neighbors.ndim == 1:
        neighbors = neighbors[:, None]
    candidate_pool = np.asarray(neighbors[:, 1:], dtype=int)
    rng = np.random.default_rng(random_state)

    permutations = np.empty((n_permutations, n_samples), dtype=int)
    for draw in range(n_permutations):
        assigned = np.full(n_samples, -1, dtype=int)
        used = np.zeros(n_samples, dtype=bool)
        for sample in rng.permutation(n_samples):
            candidates = np.array(candidate_pool[sample], copy=True)
            rng.shuffle(candidates)
            pick = -1
            for candidate in candidates:
                if not used[candidate]:
                    pick = int(candidate)
                    break
            if pick < 0:
                pick = int(candidates[0])
            assigned[sample] = pick
            used[pick] = True
        permutations[draw] = assigned

    if n_permutations == 1:
        return permutations[0]
    return permutations


def apply_permutation_indices(
    values: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    """Apply sampled indices to a one- or two-dimensional value array."""

    array = np.asarray(values)
    if array.ndim == 1:
        return array[np.asarray(indices, dtype=int)]
    if array.ndim == 2:
        return array[np.asarray(indices, dtype=int), :]
    raise ValueError("values must be one- or two-dimensional.")


def permutation_pvalue(
    observed: float,
    null_distribution: np.ndarray,
) -> float:
    """Return the upper-tail randomization p-value with a plus-one correction."""

    null = np.asarray(null_distribution, dtype=float)
    if null.ndim != 1 or null.size == 0:
        raise ValueError("null_distribution must be a non-empty vector.")
    return float(
        (np.count_nonzero(null >= observed - 1e-12) + 1.0)
        / (null.size + 1.0)
    )
