from __future__ import annotations

import json

import numpy as np
import pytest

from fastCGC.experiments.benchmark_data import (
    generate_trial_data,
    load_or_generate_trial_data,
    load_trial_data,
    save_trial_data,
)


def _small_trial(scenario: str = "singleLag-Markovian"):
    return generate_trial_data(
        scenario,
        3,
        seed=91,
        n_vars_range=(10, 10),
        length_range=(24, 24),
    )


def test_generation_is_deterministic_and_preserves_global_rng() -> None:
    np.random.seed(1234)
    expected_next = np.random.random(5)

    np.random.seed(1234)
    first = _small_trial()
    actual_next = np.random.random(5)
    second = _small_trial()

    np.testing.assert_array_equal(actual_next, expected_next)
    np.testing.assert_array_equal(first.X, second.X)
    np.testing.assert_array_equal(first.A, second.A)
    assert first.digest == second.digest
    assert first.effective_seed == 94


@pytest.mark.parametrize(
    ("scenario", "n_lags"),
    [
        ("singleLag-Markovian", 1),
        ("varLags-NonMarkovian", 2),
    ],
)
def test_lag_truth_uses_source_target_orientation(
    scenario: str,
    n_lags: int,
) -> None:
    trial = _small_trial(scenario)

    assert trial.lag_coefficients.shape == (n_lags, 10, 10)
    assert trial.lag_truth.shape == (n_lags, 10, 10)
    np.testing.assert_array_equal(
        trial.lag_truth,
        np.transpose(trial.lag_coefficients != 0, (0, 2, 1)),
    )


def test_shared_bundle_roundtrip_and_configuration_guard(tmp_path) -> None:
    generated = load_or_generate_trial_data(
        tmp_path,
        "singleLag-Markovian",
        0,
        seed=42,
        n_vars_range=(10, 10),
        length_range=(20, 20),
    )
    reused = load_or_generate_trial_data(
        tmp_path,
        "singleLag-Markovian",
        0,
        seed=42,
        n_vars_range=(10, 10),
        length_range=(20, 20),
    )

    assert reused.digest == generated.digest
    np.testing.assert_array_equal(reused.X, generated.X)
    with pytest.raises(ValueError, match="configuration does not match"):
        load_or_generate_trial_data(
            tmp_path,
            "singleLag-Markovian",
            0,
            seed=43,
            n_vars_range=(10, 10),
            length_range=(20, 20),
        )


def test_corrupted_bundle_metadata_is_rejected(tmp_path) -> None:
    trial = _small_trial()
    path = save_trial_data(trial, tmp_path)
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["digest"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="digest mismatch"):
        load_trial_data(path)
