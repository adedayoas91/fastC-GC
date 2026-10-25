"""Tests for normalized causal-method adapters."""

from __future__ import annotations

import numpy as np
import pytest

from fastCGC import GcStar
from fastCGC.experiments.causal_benchmark import simulate_sparse_var
from fastCGC.experiments.causal_methods import (
    METHOD_SPECS,
    MethodResult,
    fit_method,
)


def test_reference_residualization_includes_an_intercept() -> None:
    time = np.linspace(-1.0, 1.0, 40)
    conditioning = np.vstack([2.0 + time, -3.0 + time**2])
    x = 7.0 + 0.5 * conditioning[0] - conditioning[1] + np.sin(time)
    y = -4.0 - conditioning[0] + 0.25 * conditioning[1] + np.cos(time)

    x_residual, y_residual = GcStar._regression_residual_pair(
        x, y, conditioning
    )

    assert abs(float(x_residual.mean())) < 1e-12
    assert abs(float(y_residual.mean())) < 1e-12
    design = np.column_stack([conditioning.T, np.ones(time.size)])
    np.testing.assert_allclose(design.T @ x_residual, 0.0, atol=1e-11)
    np.testing.assert_allclose(design.T @ y_residual, 0.0, atol=1e-11)


def test_reference_permutation_results_are_reproducible_with_a_seed() -> None:
    data = simulate_sparse_var(
        n_variables=3,
        n_samples=50,
        order=1,
        seed=31,
    ).series
    first = GcStar(
        n_perm=12,
        n_pasts=1,
        n_lags=1,
        method="cgc",
        random_state=9,
    ).fit(data)
    second = GcStar(
        n_perm=12,
        n_pasts=1,
        n_lags=1,
        method="cgc",
        random_state=9,
    ).fit(data)

    np.testing.assert_array_equal(first.pVal_corr_, second.pVal_corr_)
    np.testing.assert_array_equal(first.pVal_inv_corr_, second.pVal_inv_corr_)


def test_study_adapters_use_lag_source_target_orientation() -> None:
    dataset = simulate_sparse_var(
        n_variables=4,
        n_samples=120,
        order=2,
        seed=41,
    )
    for method in ("fast-cgc", "fast-cgc-star"):
        result = fit_method(
            method,
            dataset.series,
            order=2,
            alpha=0.05,
            beta=0.05,
            seed=41,
            n_permutations=20,
        )
        assert result.lag_adjacency.shape == dataset.lag_truth.shape
        assert result.lag_scores.shape == dataset.lag_truth.shape
        assert result.lag_marginal_pvalues is not None
        assert result.lag_conditional_pvalues is not None
        assert result.lag_marginal_pvalues.shape == dataset.lag_truth.shape
        assert result.lag_conditional_pvalues.shape == dataset.lag_truth.shape
        np.testing.assert_allclose(
            result.lag_pvalues,
            np.maximum(
                result.lag_marginal_pvalues,
                result.lag_conditional_pvalues,
            ),
        )
        expected = (
            (result.lag_marginal_pvalues <= 0.05)
            & (result.lag_conditional_pvalues <= 0.05)
        )
        np.testing.assert_array_equal(result.lag_adjacency, expected)
        assert np.all(np.isfinite(result.lag_scores))


def test_study_adapter_records_nondefault_ridge_and_full_policy() -> None:
    dataset = simulate_sparse_var(
        n_variables=3,
        n_samples=60,
        order=1,
        seed=43,
    )

    result = fit_method(
        "fast-cgc-star",
        dataset.series,
        order=1,
        alpha=0.05,
        beta=0.05,
        ridge=1e-4,
        evaluation_policy="full",
        seed=43,
    )

    assert result.metadata["ridge"] == pytest.approx(1e-4)
    assert result.metadata["evaluation_policy"] == "full"


def test_method_manifest_tracks_upstream_execution_boundaries() -> None:
    by_key = {spec.key: spec for spec in METHOD_SPECS}

    assert by_key["fast-cgc"].display_name == "fastC-GC"
    assert by_key["fast-cgc-star"].display_name == "fastC-GC*"

    for method in (
        "var-gc",
        "lasso-var",
        "cuts-plus",
        "var-lingam",
    ):
        assert by_key[method].track.startswith("external-repository")
        assert not by_key[method].runnable

    for method in ("cgc", "cgc-star"):
        assert by_key[method].track == "installed-vcs-aggregate"
        assert by_key[method].dependency.startswith("gc-extension @ git+")
        assert not by_key[method].runnable

    for method in ("pcmci-plus", "bagged-pcmci-plus"):
        assert by_key[method].track == "installed-package"
        assert by_key[method].dependency == "tigramite==5.2.10.1"
        assert not by_key[method].runnable

    assert by_key["mssd"].track == "pending-upstream"
    assert not by_key["mssd"].runnable
    assert by_key["dynotears"].track == "historical-only"
    assert not by_key["dynotears"].runnable
    assert by_key["lpcmci"].track == "pag"
    assert not by_key["lpcmci"].runnable


def test_external_comparison_method_has_no_local_fitter() -> None:
    dataset = simulate_sparse_var(
        n_variables=3,
        n_samples=50,
        order=1,
        seed=19,
    )

    with pytest.raises(NotImplementedError, match="comparison notebook"):
        fit_method(
            "pcmci-plus",
            dataset.series,
            order=1,
            alpha=0.05,
            beta=0.05,
            seed=19,
            n_permutations=20,
        )


def test_method_result_rejects_orientation_shape_mismatches() -> None:
    adjacency = np.zeros((1, 3, 3), dtype=bool)
    try:
        MethodResult(
            method="broken",
            lag_adjacency=adjacency,
            lag_scores=np.zeros((3, 3)),
        )
    except ValueError as exc:
        assert "lag_scores" in str(exc)
    else:  # pragma: no cover - protects the validation contract
        raise AssertionError("shape mismatch should fail")


@pytest.mark.parametrize(
    "field",
    ("lag_marginal_pvalues", "lag_conditional_pvalues"),
)
def test_method_result_rejects_component_pvalue_shape_mismatches(
    field: str,
) -> None:
    adjacency = np.zeros((1, 3, 3), dtype=bool)
    arguments = {
        "method": "broken",
        "lag_adjacency": adjacency,
        "lag_scores": np.zeros_like(adjacency, dtype=float),
        field: np.zeros((3, 3)),
    }

    with pytest.raises(ValueError, match=field):
        MethodResult(**arguments)
