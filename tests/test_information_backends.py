"""Tests for nearest-neighbor information backends."""

from __future__ import annotations

import numpy as np

from fastCGC.dependence.information import (
    conditional_mutual_information_test,
    estimate_conditional_mutual_information,
    estimate_mutual_information,
    mutual_information_test,
)


def test_mutual_information_increases_on_dependent_data() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=180)
    y_independent = rng.normal(size=180)
    y_dependent = np.sin(x) + 0.05 * rng.normal(size=180)

    independent = estimate_mutual_information(x, y_independent, n_neighbors=5)
    dependent = estimate_mutual_information(x, y_dependent, n_neighbors=5)

    assert dependent > independent + 0.2


def test_conditional_mutual_information_shrinks_after_conditioning() -> None:
    rng = np.random.default_rng(9)
    z = rng.normal(size=220)
    x = z + 0.1 * rng.normal(size=220)
    y = z - 0.1 * rng.normal(size=220)

    marginal = estimate_mutual_information(x, y, n_neighbors=5)
    conditional = estimate_conditional_mutual_information(x, y, z, n_neighbors=5)

    assert marginal > 0.4
    assert conditional < 0.1


def test_information_tests_are_deterministic_for_fixed_seed() -> None:
    rng = np.random.default_rng(13)
    z = rng.normal(size=120)
    x = z + 0.2 * rng.normal(size=120)
    y = x * z + 0.1 * rng.normal(size=120)

    first = mutual_information_test(
        x,
        y,
        n_neighbors=4,
        n_permutations=16,
        random_state=3,
    )
    second = mutual_information_test(
        x,
        y,
        n_neighbors=4,
        n_permutations=16,
        random_state=3,
    )
    assert first.statistic == second.statistic
    assert first.pvalue == second.pvalue
    np.testing.assert_allclose(first.null_distribution, second.null_distribution)

    first_cmi = conditional_mutual_information_test(
        x,
        y,
        z,
        n_neighbors=4,
        local_permutation_neighbors=6,
        n_permutations=16,
        random_state=5,
    )
    second_cmi = conditional_mutual_information_test(
        x,
        y,
        z,
        n_neighbors=4,
        local_permutation_neighbors=6,
        n_permutations=16,
        random_state=5,
    )
    assert first_cmi.statistic == second_cmi.statistic
    assert first_cmi.pvalue == second_cmi.pvalue
    np.testing.assert_allclose(
        first_cmi.null_distribution,
        second_cmi.null_distribution,
    )
