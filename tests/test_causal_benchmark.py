"""Tests for the common causal-comparison data and scoring contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from fastCGC.experiments.causal_benchmark import (
    aggregate_lags,
    companion_spectral_radius,
    comparison_metrics,
    graph_metrics,
    simulate_sparse_var,
)


def _load_cli_module():
    script = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "benchmarks"
        / "benchmark_causal_methods.py"
    )
    spec = importlib.util.spec_from_file_location("causal_benchmark_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_metadata_converts_numpy_values_to_json_types() -> None:
    metadata = {
        "rank": np.int64(4),
        "scores": np.array([0.25, 0.75]),
        "nested": (np.bool_(True),),
    }

    converted = _load_cli_module()._json_ready(metadata)

    assert converted == {
        "rank": 4,
        "scores": [0.25, 0.75],
        "nested": [True],
    }
    json.dumps(converted)


def test_comparison_runner_records_software_without_hardware_ranking() -> None:
    module = _load_cli_module()

    provenance = module._software_provenance()

    assert provenance["python"].startswith("3.14.")
    assert provenance["python_implementation"] == "CPython"
    assert "packages" in provenance
    assert "machine" not in provenance
    assert "cpu_brand" not in provenance
    assert "thread_limits" not in provenance


def test_exported_dataset_bundle_preserves_arrays_and_conventions(tmp_path) -> None:
    dataset = simulate_sparse_var(
        n_variables=3,
        n_samples=50,
        order=2,
        seed=13,
    )
    module = _load_cli_module()

    path = module._save_dataset_bundle(tmp_path, "example", dataset)

    with np.load(path, allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored["series"], dataset.series)
        np.testing.assert_array_equal(
            stored["coefficients"], dataset.coefficients
        )
        np.testing.assert_array_equal(stored["lag_truth"], dataset.lag_truth)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["data_digest"] == dataset.digest
    assert metadata["series_layout"] == "variable, time"
    assert metadata["truth_layout"] == "lag, source, target"


def test_sparse_var_is_reproducible_stable_and_lag_resolved() -> None:
    first = simulate_sparse_var(
        n_variables=5,
        n_samples=120,
        order=2,
        seed=17,
        noise="gaussian",
    )
    second = simulate_sparse_var(
        n_variables=5,
        n_samples=120,
        order=2,
        seed=17,
        noise="gaussian",
    )

    np.testing.assert_array_equal(first.series, second.series)
    np.testing.assert_array_equal(first.coefficients, second.coefficients)
    np.testing.assert_array_equal(first.lag_truth, second.lag_truth)
    np.testing.assert_array_equal(
        first.lag_truth,
        np.transpose(first.coefficients != 0.0, (0, 2, 1)),
    )
    assert first.series.shape == (5, 120)
    assert first.lag_truth.shape == (2, 5, 5)
    assert first.digest == second.digest
    assert first.spectral_radius <= 0.8 + 1e-12
    assert companion_spectral_radius(first.coefficients) <= 0.8 + 1e-12


def test_sparse_var_noise_families_share_graph_for_a_seed() -> None:
    gaussian = simulate_sparse_var(
        n_variables=4,
        n_samples=80,
        order=2,
        seed=23,
        noise="gaussian",
    )
    laplace = simulate_sparse_var(
        n_variables=4,
        n_samples=80,
        order=2,
        seed=23,
        noise="laplace",
    )

    np.testing.assert_array_equal(gaussian.coefficients, laplace.coefficients)
    np.testing.assert_array_equal(gaussian.lag_truth, laplace.lag_truth)
    assert not np.array_equal(gaussian.series, laplace.series)


def test_graph_metrics_include_ranking_and_structural_measures() -> None:
    truth = np.zeros((2, 3, 3), dtype=bool)
    truth[0, 0, 1] = True
    truth[1, 2, 1] = True
    selected = np.zeros_like(truth)
    selected[0, 0, 1] = True
    selected[0, 1, 2] = True
    scores = np.linspace(0.0, 1.0, truth.size).reshape(truth.shape)
    scores[truth] = np.array([1.2, 1.1])

    metrics = graph_metrics(truth, selected, scores=scores, scope="cross")

    assert metrics["true_positive"] == 1.0
    assert metrics["false_positive"] == 1.0
    assert metrics["false_negative"] == 1.0
    assert metrics["shd"] == 2.0
    assert 0.0 <= metrics["mcc"] <= 1.0
    assert 0.0 <= metrics["auroc"] <= 1.0
    assert 0.0 <= metrics["auprc"] <= 1.0


def test_binary_decision_is_not_reused_as_an_auroc_curve() -> None:
    truth = np.array([[False, True], [False, False]])
    selected = np.array([[False, True], [True, False]])
    metrics = graph_metrics(
        truth,
        selected,
        scores=selected.astype(float),
        scope="cross",
    )

    assert np.isnan(metrics["auroc"])
    assert np.isnan(metrics["auprc"])


def test_comparison_metrics_separate_exact_aggregate_and_self_edges() -> None:
    truth = np.zeros((2, 3, 3), dtype=bool)
    truth[0, 0, 1] = True
    truth[1, 1, 1] = True
    selected = np.zeros_like(truth)
    selected[1, 0, 1] = True
    selected[1, 1, 1] = True
    scores = np.arange(truth.size, dtype=float).reshape(truth.shape)

    metrics = comparison_metrics(truth, selected, scores)

    assert metrics["exact_f1"] == 0.0
    assert metrics["aggregate_f1"] == 1.0
    assert metrics["self_f1"] == 1.0
    np.testing.assert_array_equal(
        aggregate_lags(truth, binary=True),
        np.array(
            [
                [False, True, False],
                [False, True, False],
                [False, False, False],
            ]
        ),
    )
