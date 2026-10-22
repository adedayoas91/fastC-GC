"""Tests for deterministic nonlinear-backend resampling helpers."""

from __future__ import annotations

import numpy as np

from fastCGC.dependence.resampling import (
    circular_shift_indices,
    local_permutation_indices,
    permutation_pvalue,
)


def test_circular_shift_indices_are_deterministic_and_skip_zero() -> None:
    first = circular_shift_indices(12, n_shifts=4, random_state=7)
    second = circular_shift_indices(12, n_shifts=4, random_state=7)

    np.testing.assert_array_equal(first, second)
    assert 0 not in first
    assert len(np.unique(first)) == first.size


def test_local_permutation_indices_are_reproducible_and_local() -> None:
    z = np.linspace(0.0, 1.0, 12)[:, None]
    first = local_permutation_indices(
        z,
        n_permutations=3,
        n_neighbors=2,
        random_state=11,
    )
    second = local_permutation_indices(
        z,
        n_permutations=3,
        n_neighbors=2,
        random_state=11,
    )

    np.testing.assert_array_equal(first, second)
    for draw in first:
        for source, target in enumerate(draw):
            assert abs(int(source) - int(target)) <= 2


def test_permutation_pvalue_uses_plus_one_correction() -> None:
    pvalue = permutation_pvalue(0.9, np.array([0.1, 0.2, 0.95, 0.4]))
    assert pvalue == 2.0 / 5.0
