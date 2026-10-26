"""Tests for statistical-validation data and scoring contracts."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from fastCGC.experiments.statistical_validation import (
    null_calibration_metrics,
    recovery_metrics,
    simulate_null_series,
    threshold_adjacency,
    truth_for_fitted_order,
)


def _load_validation_cli():
    script = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "validation"
        / "run_statistical_validation.py"
    )
    spec = importlib.util.spec_from_file_location("statistical_validation_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "regime",
    ("iid-null", "stationary-ar-null", "drifting-null"),
)
def test_null_generators_are_deterministic_and_finite(regime: str) -> None:
    first = simulate_null_series(
        regime=regime,
        n_variables=4,
        n_samples=80,
        seed=17,
    )
    second = simulate_null_series(
        regime=regime,
        n_variables=4,
        n_samples=80,
        seed=17,
    )

    np.testing.assert_array_equal(first.series, second.series)
    assert first.digest == second.digest
    assert first.series.shape == (4, 80)
    assert np.all(np.isfinite(first.series))


def test_threshold_adjacency_uses_both_component_pvalues() -> None:
    marginal = np.array([[[0.001, 0.02], [0.001, 0.001]]])
    conditional = np.array([[[0.006, 0.001], [0.004, 0.009]]])

    selected = threshold_adjacency(
        marginal,
        conditional,
        alpha=0.01,
        beta=0.005,
    )

    expected = np.array([[[False, False], [True, False]]])
    np.testing.assert_array_equal(selected, expected)


def test_truth_alignment_truncates_or_pads_lags() -> None:
    truth = np.zeros((2, 3, 3), dtype=bool)
    truth[0, 0, 1] = True
    truth[1, 1, 2] = True

    truncated, omitted = truth_for_fitted_order(truth, 1)
    padded, padded_omitted = truth_for_fitted_order(truth, 4)

    np.testing.assert_array_equal(truncated, truth[:1])
    assert omitted == 1
    np.testing.assert_array_equal(padded[:2], truth)
    assert not np.any(padded[2:])
    assert padded_omitted == 0


def test_null_metrics_exclude_self_edges() -> None:
    marginal = np.ones((1, 2, 2))
    conditional = np.ones((1, 2, 2))
    marginal[0, 0, 0] = 0.001
    conditional[0, 0, 0] = 0.001

    metrics = null_calibration_metrics(
        marginal,
        conditional,
        alpha=0.01,
        beta=0.005,
    )

    assert metrics["candidate_count"] == 2.0
    assert metrics["joint_false_positive_rate"] == 0.0
    assert metrics["graph_familywise_error"] == 0.0


def test_recovery_metrics_require_aligned_truth() -> None:
    marginal = np.full((1, 2, 2), 0.5)
    conditional = np.full((1, 2, 2), 0.5)

    with pytest.raises(ValueError, match="identical shapes"):
        recovery_metrics(
            np.zeros((2, 2, 2), dtype=bool),
            marginal,
            conditional,
            alpha=0.01,
            beta=0.005,
        )


def test_validation_cli_resumes_at_completed_fit_boundary(tmp_path) -> None:
    module = _load_validation_cli()
    args = module._parser().parse_args(
        [
            "--arms",
            "calibration",
            "--methods",
            "fast-cgc-star",
            "--n-variables",
            "3",
            "--calibration-samples",
            "30",
            "--true-order",
            "1",
            "--betas",
            "0.05",
            "--replicates",
            "1",
            "--max-new-fits",
            "1",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert module.run(args) == 0
    first = sorted((tmp_path / "checkpoints").glob("*.json"))
    assert len(first) == 1

    assert module.run(args) == 0
    second = sorted((tmp_path / "checkpoints").glob("*.json"))
    assert len(second) == 2
