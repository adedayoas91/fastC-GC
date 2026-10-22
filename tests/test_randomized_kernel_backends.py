"""Tests for RFF-HSIC, RCIT, and RCoT helpers."""

from __future__ import annotations

import numpy as np

from fastCGC.dependence.hsic import RffHsicBackend, biased_hsic, rff_hsic_test
from fastCGC.dependence.rcit import (
    RandomizedConditionalBackend,
    rcit_test,
    rcot_test,
    residualize_matrix,
)


def _nonlinear_series(seed: int = 0, n_samples: int = 128) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n_samples, 1))
    x_null = z + 0.2 * rng.normal(size=(n_samples, 1))
    y_null = np.sin(z) + 0.2 * rng.normal(size=(n_samples, 1))
    x_alt = rng.normal(size=(n_samples, 1))
    y_alt = 0.7 * x_alt**2 + 0.2 * z + 0.1 * rng.normal(size=(n_samples, 1))
    y_independent = rng.normal(size=(n_samples, 1))
    return x_null, y_null, z, x_alt, y_alt, y_independent


def test_biased_hsic_is_larger_for_nonlinear_dependence_than_independence() -> None:
    _, _, _, x_alt, y_alt, y_independent = _nonlinear_series(11)
    dependent = biased_hsic(x_alt, y_alt, block_size=32)
    independent = biased_hsic(x_alt, y_independent, block_size=32)
    assert dependent > independent


def test_rff_hsic_complete_shift_test_detects_dependence() -> None:
    _, _, _, x_alt, y_alt, y_independent = _nonlinear_series(13)
    dependent = rff_hsic_test(x_alt, y_alt, n_features=48, random_state=5)
    independent = rff_hsic_test(
        x_alt, y_independent, n_features=48, random_state=5
    )
    assert dependent.statistic[0] > independent.statistic[0]
    assert dependent.pvalue[0] < independent.pvalue[0]


def test_rff_hsic_backend_reuses_cached_row_features() -> None:
    rng = np.random.default_rng(17)
    rows = rng.normal(size=(2, 64))
    backend = RffHsicBackend(n_features=24, random_state=3, cache_max_bytes=4096)
    first = backend.evaluate_pairs(rows, rows, row_keys=[("a", "b"), ("a", "b")])
    second = backend.evaluate_pairs(rows, rows, row_keys=[("a", "b"), ("a", "b")])
    assert second.metadata["hits"] >= first.metadata["hits"]
    assert second.metadata["entries"] <= 2


def test_residualize_matrix_matches_columnwise_solution() -> None:
    rng = np.random.default_rng(19)
    targets = rng.normal(size=(81, 6))
    conditioning = rng.normal(size=(81, 4))
    residualized = residualize_matrix(targets, conditioning, ridge=1e-5)

    expected_columns = []
    centered_targets = targets - targets.mean(axis=0, keepdims=True)
    centered_conditioning = conditioning - conditioning.mean(axis=0, keepdims=True)
    gram = centered_conditioning.T @ centered_conditioning / targets.shape[0]
    gram.flat[:: gram.shape[0] + 1] += 1e-5
    for column in range(targets.shape[1]):
        rhs = centered_conditioning.T @ centered_targets[:, [column]] / targets.shape[0]
        coefficients = np.linalg.solve(gram, rhs)
        expected = centered_targets[:, [column]] - centered_conditioning @ coefficients
        expected_columns.append(expected[:, 0])
    expected_matrix = np.column_stack(expected_columns)
    expected_matrix -= expected_matrix.mean(axis=0, keepdims=True)
    np.testing.assert_allclose(residualized, expected_matrix, atol=1e-10, rtol=1e-10)


def test_rcot_conditions_away_shared_nonlinear_driver() -> None:
    x_null, y_null, z, _, _, _ = _nonlinear_series(23)
    conditioned = rcot_test(x_null, y_null, z, random_state=7)
    misconditioned = rcot_test(
        x_null,
        y_null,
        np.zeros_like(z),
        random_state=7,
    )
    assert conditioned.statistic[0] < misconditioned.statistic[0]
    assert 0.0 <= conditioned.pvalue[0] <= 1.0


def test_rcit_and_rcot_detect_conditional_dependence() -> None:
    _, _, z, x_alt, y_alt, _ = _nonlinear_series(29)
    rcot = rcot_test(x_alt, y_alt, z, random_state=11)
    rcit = rcit_test(x_alt, y_alt, z, random_state=11)
    null = rcot_test(x_alt, np.random.default_rng(31).normal(size=y_alt.shape), z, random_state=11)
    assert rcot.statistic[0] > null.statistic[0]
    assert rcit.statistic[0] > 0.0
    assert rcit.metadata["backend"] == "rcit"
    assert rcot.metadata["backend"] == "rcot"


def test_randomized_conditional_backend_reports_cache_stats() -> None:
    _, _, z, x_alt, y_alt, _ = _nonlinear_series(37)
    backend = RandomizedConditionalBackend(mode="rcot", random_state=13)
    first = backend.evaluate(x_alt, y_alt, z, key="candidate")
    second = backend.evaluate(x_alt, y_alt, z, key="candidate")
    assert second.metadata["cache"]["hits"] >= first.metadata["cache"]["hits"]
    assert 0.0 <= second.pvalue[0] <= 1.0
