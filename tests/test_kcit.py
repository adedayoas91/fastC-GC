"""Tests for the guarded reference KCIT helpers."""

from __future__ import annotations

import numpy as np
import pytest

from fastCGC.dependence.kcit import (
    estimate_exact_kernel_bytes,
    guard_exact_kcit_feasibility,
    kcit_test,
)


def test_exact_kernel_memory_estimate_scales_quadratically() -> None:
    small = estimate_exact_kernel_bytes(32)
    large = estimate_exact_kernel_bytes(64)
    assert large == small * 4


def test_exact_kcit_guard_rejects_excessive_budget() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        guard_exact_kcit_feasibility(128, memory_budget_mb=0)

    with pytest.raises(ValueError, match="sample guard exceeded"):
        guard_exact_kcit_feasibility(33, memory_budget_mb=64, max_samples=32)

    with pytest.raises(ValueError, match="memory budget exceeded"):
        guard_exact_kcit_feasibility(256, memory_budget_mb=1, max_samples=None)


def test_kcit_distinguishes_conditional_null_from_dependence() -> None:
    rng = np.random.default_rng(21)
    z = rng.normal(size=90)
    x_null = z + 0.15 * rng.normal(size=90)
    y_null = z - 0.15 * rng.normal(size=90)
    x_dep = z + 0.15 * rng.normal(size=90)
    y_dep = np.sin(x_dep) + 0.15 * z + 0.05 * rng.normal(size=90)

    null_result = kcit_test(
        x_null,
        y_null,
        z,
        n_permutations=24,
        local_permutation_neighbors=6,
        random_state=2,
    )
    dependent_result = kcit_test(
        x_dep,
        y_dep,
        z,
        n_permutations=24,
        local_permutation_neighbors=6,
        random_state=2,
    )

    assert dependent_result.statistic > null_result.statistic
    assert dependent_result.pvalue < null_result.pvalue


def test_kcit_is_deterministic_for_fixed_seed() -> None:
    rng = np.random.default_rng(7)
    z = rng.normal(size=72)
    x = z + 0.1 * rng.normal(size=72)
    y = x**2 + 0.05 * rng.normal(size=72)

    first = kcit_test(
        x,
        y,
        z,
        n_permutations=16,
        local_permutation_neighbors=5,
        random_state=4,
    )
    second = kcit_test(
        x,
        y,
        z,
        n_permutations=16,
        local_permutation_neighbors=5,
        random_state=4,
    )

    assert first.statistic == second.statistic
    assert first.pvalue == second.pvalue
    np.testing.assert_allclose(first.null_distribution, second.null_distribution)
