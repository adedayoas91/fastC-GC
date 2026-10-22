"""End-to-end tests for selectable nonlinear FastGcStar backends."""

from __future__ import annotations

import numpy as np
import pytest

from fastCGC import FastGcStar
from fastCGC.dependence.hsic import (
    _complete_shift_gram_statistics,
    hsic_test,
)
from fastCGC.dependence.kernels import center_gram, rbf_kernel


BACKEND_OPTIONS = {
    "hsic": {"n_shifts": 3, "max_samples": 64},
    "kcit": {
        "n_permutations": 2,
        "local_permutation_neighbors": 2,
        "max_samples": 64,
    },
    "rff_hsic": {"n_features": 8},
    "rcit": {
        "n_features_x": 4,
        "n_features_y": 4,
        "n_features_z": 6,
        "n_features_joint": 6,
    },
    "rcot": {
        "n_features_x": 4,
        "n_features_y": 4,
        "n_features_z": 6,
    },
    "mi_knn": {"n_neighbors": 2, "n_permutations": 2},
    "cmi_knn": {
        "n_neighbors": 2,
        "n_permutations": 2,
        "local_permutation_neighbors": 2,
    },
}


def _series(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = 24
    driver = rng.normal(size=samples)
    return np.vstack(
        (
            driver + 0.1 * rng.normal(size=samples),
            np.sin(driver) + 0.1 * rng.normal(size=samples),
            rng.normal(size=samples),
        )
    )


@pytest.mark.parametrize("method", ["cgc", "fcgc"])
@pytest.mark.parametrize(
    ("profile", "marginal", "conditional"),
    [
        ("linear", "pearson", "ridge_residual"),
        ("kernel", "hsic", "kcit"),
        ("randomized_kernel_rcit", "rff_hsic", "rcit"),
        ("randomized_kernel", "rff_hsic", "rcot"),
        ("information", "mi_knn", "cmi_knn"),
    ],
)
def test_every_profile_runs_with_both_conditioning_policies(
    method: str,
    profile: str,
    marginal: str,
    conditional: str,
) -> None:
    estimator = FastGcStar(
        n_pasts=1,
        method=method,
        dependence_profile=profile,
        backend_options=BACKEND_OPTIONS,
        random_state=7,
    ).fit(_series(3), alpha=1.0)

    assert estimator.marginal_pvalue_.shape == (6, 3)
    assert estimator.conditional_pvalue_.shape == (6, 3)
    assert estimator.backend_metadata_["marginal_backend"] == marginal
    assert estimator.backend_metadata_["conditional_backend"] == conditional
    assert np.all(np.isfinite(estimator.marginal_pvalue_))
    assert np.all(np.isfinite(estimator.conditional_pvalue_))


def test_marginal_and_conditional_backends_can_be_overridden_independently() -> None:
    estimator = FastGcStar(
        n_pasts=1,
        method="cgc",
        dependence_profile="linear",
        marginal_test="rff_hsic",
        conditional_test="cmi_knn",
        backend_options=BACKEND_OPTIONS,
        evaluation_policy="full",
        random_state=11,
    ).fit(_series(5))

    assert estimator.backend_metadata_["marginal_backend"] == "rff_hsic"
    assert estimator.backend_metadata_["conditional_backend"] == "cmi_knn"


def test_exact_hsic_fft_orbit_matches_brute_force_shifts() -> None:
    rng = np.random.default_rng(13)
    gram_x = center_gram(rbf_kernel(rng.normal(size=(11, 2)), bandwidth=0.8))
    gram_y = center_gram(rbf_kernel(rng.normal(size=(11, 1)), bandwidth=1.2))
    expected = np.array(
        [
            np.sum(
                gram_x
                * np.roll(
                    np.roll(gram_y, -shift, axis=0),
                    -shift,
                    axis=1,
                )
            )
            / 100.0
            for shift in range(11)
        ]
    )
    observed = _complete_shift_gram_statistics(gram_x, gram_y)
    np.testing.assert_allclose(observed, expected, atol=1e-12, rtol=1e-12)


def test_exact_hsic_guard_recommends_random_features() -> None:
    rng = np.random.default_rng(17)
    with pytest.raises(ValueError, match="rff_hsic"):
        hsic_test(
            rng.normal(size=(20, 1)),
            rng.normal(size=(20, 1)),
            max_samples=10,
        )
