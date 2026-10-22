"""Unit tests for nonlinear kernel primitives."""

from __future__ import annotations

import numpy as np

from fastCGC.dependence.kernels import (
    ByteBudgetCache,
    center_gram,
    median_bandwidth,
    pairwise_squared_distances,
    rbf_kernel,
)


def test_pairwise_squared_distances_are_symmetric_and_non_negative() -> None:
    data = np.array([[0.0, 1.0], [2.0, -1.0], [3.0, 0.5]])
    distances = pairwise_squared_distances(data)
    np.testing.assert_allclose(distances, distances.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(distances), 0.0, atol=1e-12)
    assert np.all(distances >= 0.0)


def test_blocked_rbf_kernel_matches_dense() -> None:
    rng = np.random.default_rng(7)
    data = rng.normal(size=(23, 3))
    bandwidth = median_bandwidth(data, random_state=11)
    dense = rbf_kernel(data, bandwidth=bandwidth)
    blocked = rbf_kernel(data, bandwidth=bandwidth, block_size=5)
    np.testing.assert_allclose(blocked, dense, rtol=1e-12, atol=1e-12)


def test_center_gram_has_zero_row_and_column_sums() -> None:
    rng = np.random.default_rng(13)
    gram = rbf_kernel(rng.normal(size=(19, 2)), bandwidth=0.8)
    centered = center_gram(gram)
    np.testing.assert_allclose(centered.sum(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(centered.sum(axis=1), 0.0, atol=1e-12)


def test_median_bandwidth_is_deterministic_and_positive() -> None:
    rng = np.random.default_rng(17)
    samples = rng.normal(size=(64, 4))
    first = median_bandwidth(samples, max_pairs=32, random_state=5)
    second = median_bandwidth(samples, max_pairs=32, random_state=5)
    assert first > 0.0
    assert first == second


def test_byte_budget_cache_evicts_oldest_entries() -> None:
    cache = ByteBudgetCache[np.ndarray](max_bytes=3 * 8)
    first = np.arange(2.0)
    second = np.arange(3.0)
    cache.set("first", first)
    assert cache.get("first") is not None
    cache.set("second", second)
    assert cache.get("second") is not None
    assert cache.get("first") is None
    assert cache.current_bytes <= cache.max_bytes
