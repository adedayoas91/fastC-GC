"""Unit tests for random-feature primitives."""

from __future__ import annotations

import numpy as np

from fastCGC.dependence.random_features import (
    complete_shift_feature_statistics,
    random_fourier_features,
    stable_seed,
)


def _brute_force_shift_statistics(
    x_features: np.ndarray,
    y_features: np.ndarray,
) -> np.ndarray:
    n_samples = x_features.shape[0]
    statistics = np.zeros(n_samples, dtype=float)
    for shift in range(n_samples):
        rolled = np.roll(y_features, -shift, axis=0)
        covariance = x_features.T @ rolled / n_samples
        statistics[shift] = np.sum(covariance * covariance)
    return statistics


def test_stable_seed_changes_with_semantic_key() -> None:
    first = stable_seed(3, "row", 1)
    second = stable_seed(3, "row", 2)
    assert first != second
    assert first == stable_seed(3, "row", 1)


def test_random_fourier_features_are_deterministic_and_centered() -> None:
    rng = np.random.default_rng(19)
    samples = rng.normal(size=(41, 2))
    first = random_fourier_features(
        samples,
        n_features=16,
        bandwidth=0.7,
        random_state=9,
        key="shared",
    )
    second = random_fourier_features(
        samples,
        n_features=16,
        bandwidth=0.7,
        random_state=9,
        key="shared",
    )
    np.testing.assert_allclose(first, second, atol=1e-12)
    np.testing.assert_allclose(first.mean(axis=0), 0.0, atol=1e-12)


def test_complete_shift_statistics_match_brute_force_orbit() -> None:
    rng = np.random.default_rng(23)
    x_features = rng.normal(size=(17, 5))
    y_features = rng.normal(size=(17, 4))
    expected = _brute_force_shift_statistics(
        x_features - x_features.mean(axis=0, keepdims=True),
        y_features - y_features.mean(axis=0, keepdims=True),
    )
    observed = complete_shift_feature_statistics(
        x_features,
        y_features,
        feature_pair_batch_size=2,
    )
    np.testing.assert_allclose(observed, expected, atol=1e-10, rtol=1e-10)


def test_complete_shift_statistics_are_batch_invariant() -> None:
    rng = np.random.default_rng(29)
    x_features = rng.normal(size=(21, 7))
    y_features = rng.normal(size=(21, 6))
    first = complete_shift_feature_statistics(
        x_features,
        y_features,
        feature_pair_batch_size=2,
    )
    second = complete_shift_feature_statistics(
        x_features,
        y_features,
        feature_pair_batch_size=5,
    )
    np.testing.assert_allclose(first, second, atol=1e-12, rtol=1e-12)
