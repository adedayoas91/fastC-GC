"""Deterministic, shareable datasets for benchmark notebooks.

Each trial is generated once, stored with its positive-lag truth, and identified
by a SHA-256 digest. Method notebooks load the stored bundle rather than
maintaining separate simulation cells. This makes the input data identical
across upstream repositories and keeps estimator randomness separate from data
generation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from ..simulation import adj_mtx, continuous_noise_fun

DEFAULT_SEED: int = 42
DEFAULT_N_VARS_RANGE: tuple[int, int] = (18, 25)
DEFAULT_LENGTH_RANGE: tuple[int, int] = (3000, 6000)
DATASET_FORMAT_VERSION = 1

VALID_SCENARIOS: tuple[str, ...] = (
    "singleLag-Markovian",
    "singleLag-NonMarkovian",
    "varLags-Markovian",
    "varLags-NonMarkovian",
)


def _array_digest(hasher: Any, name: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    hasher.update(name.encode("utf-8"))
    hasher.update(str(array.dtype).encode("ascii"))
    hasher.update(json.dumps(array.shape).encode("ascii"))
    hasher.update(array.tobytes())


@dataclass(frozen=True, slots=True)
class TrialData:
    """One deterministic trial and its aggregate and lag-resolved truth."""

    X: np.ndarray
    A: np.ndarray
    lag_coefficients: np.ndarray
    lag_truth: np.ndarray
    n_vars: int
    length: int
    scenario: str
    trial_id: int
    seed: int
    n_vars_range: tuple[int, int]
    length_range: tuple[int, int]

    @property
    def effective_seed(self) -> int:
        """Seed used by this trial's data generator."""

        return self.seed + self.trial_id

    @property
    def digest(self) -> str:
        """Return a content and configuration digest for the trial."""

        hasher = hashlib.sha256()
        metadata = {
            "format_version": DATASET_FORMAT_VERSION,
            "scenario": self.scenario,
            "trial_id": self.trial_id,
            "seed": self.seed,
            "effective_seed": self.effective_seed,
            "n_vars_range": list(self.n_vars_range),
            "length_range": list(self.length_range),
        }
        hasher.update(json.dumps(metadata, sort_keys=True).encode("utf-8"))
        _array_digest(hasher, "series", self.X)
        _array_digest(hasher, "aggregate_coefficients", self.A)
        _array_digest(hasher, "lag_coefficients", self.lag_coefficients)
        _array_digest(hasher, "lag_truth", self.lag_truth)
        return hasher.hexdigest()


def _validate_range(name: str, values: tuple[int, int]) -> tuple[int, int]:
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two integers.")
    low, high = (int(values[0]), int(values[1]))
    if low < 1 or high < low:
        raise ValueError(f"{name} must satisfy 1 <= low <= high.")
    return low, high


def _draw_inclusive(low: int, high: int) -> int:
    return int(np.random.randint(low, high + 1))


def generate_trial_data(
    scenario: str,
    trial_id: int,
    *,
    seed: int = DEFAULT_SEED,
    n_vars_range: tuple[int, int] = DEFAULT_N_VARS_RANGE,
    length_range: tuple[int, int] = DEFAULT_LENGTH_RANGE,
) -> TrialData:
    """Generate one trial without changing the caller's NumPy RNG state."""

    if scenario not in VALID_SCENARIOS:
        raise ValueError(
            f"Unknown scenario {scenario!r}; expected one of {VALID_SCENARIOS}."
        )
    if trial_id < 0:
        raise ValueError("trial_id must be non-negative.")
    n_vars_range = _validate_range("n_vars_range", n_vars_range)
    length_range = _validate_range("length_range", length_range)

    random_state = np.random.get_state()
    np.random.seed(int(seed) + int(trial_id))
    try:
        n_vars = _draw_inclusive(*n_vars_range)
        length = _draw_inclusive(*length_range)
        non_markovian = "NonMarkovian" in scenario
        variable_lags = scenario.startswith("varLags")

        disturbance = (
            continuous_noise_fun(num=n_vars, length=length)
            if non_markovian
            else None
        )
        coefficients = np.asarray(adj_mtx(n_vars))

        if variable_lags:
            lag_mask = np.random.random(coefficients.shape) > 0.55
            np.fill_diagonal(lag_mask, True)
            lag_one = coefficients * lag_mask
            lag_two = coefficients * ~lag_mask
            lag_coefficients = np.stack((lag_one, lag_two))
            aggregate_coefficients = np.logical_or(lag_one, lag_two)
        else:
            lag_coefficients = coefficients[None, :, :]
            aggregate_coefficients = coefficients

        series_time_first = np.zeros((length, n_vars), dtype=float)
        series_time_first[0] = np.random.randn(n_vars)
        for time_index in range(length - 1):
            if variable_lags:
                step = (
                    lag_coefficients[0] @ series_time_first[time_index]
                    + lag_coefficients[1] @ series_time_first[time_index - 1]
                )
            else:
                step = coefficients @ series_time_first[time_index]
            step = step + np.random.normal(0, 0.25, n_vars)
            if disturbance is not None:
                step = step + disturbance[:, time_index]
            series_time_first[time_index + 1] = step

        series = series_time_first.T
        lag_truth = np.transpose(lag_coefficients != 0, (0, 2, 1))
    finally:
        np.random.set_state(random_state)

    return TrialData(
        X=series,
        A=aggregate_coefficients,
        lag_coefficients=lag_coefficients,
        lag_truth=lag_truth,
        n_vars=n_vars,
        length=length,
        scenario=scenario,
        trial_id=trial_id,
        seed=int(seed),
        n_vars_range=n_vars_range,
        length_range=length_range,
    )


def trial_bundle_path(root: Path, scenario: str, trial_id: int) -> Path:
    """Return the canonical path for one shared trial bundle."""

    if scenario not in VALID_SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}.")
    if trial_id < 0:
        raise ValueError("trial_id must be non-negative.")
    return Path(root) / scenario / f"trial-{trial_id:05d}.npz"


def _metadata(trial: TrialData) -> dict[str, Any]:
    return {
        "format_version": DATASET_FORMAT_VERSION,
        "generator": "fastCGC.experiments.benchmark_data.generate_trial_data",
        "scenario": trial.scenario,
        "trial_id": trial.trial_id,
        "master_seed": trial.seed,
        "effective_seed": trial.effective_seed,
        "n_vars": trial.n_vars,
        "length": trial.length,
        "n_vars_range": list(trial.n_vars_range),
        "length_range": list(trial.length_range),
        "digest": trial.digest,
        "series_layout": "variable, time",
        "aggregate_coefficient_layout": "target, source",
        "lag_coefficient_layout": "lag, target, source",
        "truth_layout": "lag, source, target",
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def save_trial_data(trial: TrialData, root: Path) -> Path:
    """Atomically persist a trial bundle and its provenance metadata."""

    path = trial_bundle_path(root, trial.scenario, trial.trial_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        np.savez_compressed(
            handle,
            series=np.asarray(trial.X, dtype=float),
            aggregate_coefficients=np.asarray(trial.A),
            lag_coefficients=np.asarray(trial.lag_coefficients),
            lag_truth=np.asarray(trial.lag_truth, dtype=bool),
        )
        temporary = Path(handle.name)
    os.replace(temporary, path)
    _atomic_json(path.with_suffix(".json"), _metadata(trial))
    return path


def load_trial_data(path: Path) -> TrialData:
    """Load a trial bundle and reject incomplete or corrupted provenance."""

    path = Path(path)
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Incomplete trial bundle at {path}.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != DATASET_FORMAT_VERSION:
        raise ValueError(f"Unsupported dataset format in {metadata_path}.")

    with np.load(path, allow_pickle=False) as stored:
        trial = TrialData(
            X=np.asarray(stored["series"], dtype=float),
            A=np.asarray(stored["aggregate_coefficients"]),
            lag_coefficients=np.asarray(stored["lag_coefficients"]),
            lag_truth=np.asarray(stored["lag_truth"], dtype=bool),
            n_vars=int(metadata["n_vars"]),
            length=int(metadata["length"]),
            scenario=str(metadata["scenario"]),
            trial_id=int(metadata["trial_id"]),
            seed=int(metadata["master_seed"]),
            n_vars_range=tuple(int(value) for value in metadata["n_vars_range"]),
            length_range=tuple(int(value) for value in metadata["length_range"]),
        )

    if trial.X.shape != (trial.n_vars, trial.length):
        raise ValueError(f"Series shape does not match metadata in {path}.")
    expected_truth_shape = np.transpose(
        trial.lag_coefficients, (0, 2, 1)
    ).shape
    if trial.lag_truth.shape != expected_truth_shape:
        raise ValueError(f"Lag truth shape does not match coefficients in {path}.")
    if trial.digest != metadata.get("digest"):
        raise ValueError(f"Dataset digest mismatch for {path}.")
    return trial


def load_or_generate_trial_data(
    root: Path,
    scenario: str,
    trial_id: int,
    *,
    seed: int = DEFAULT_SEED,
    n_vars_range: tuple[int, int] = DEFAULT_N_VARS_RANGE,
    length_range: tuple[int, int] = DEFAULT_LENGTH_RANGE,
) -> TrialData:
    """Reuse a matching bundle or generate it once when it is absent."""

    expected_n_vars = _validate_range("n_vars_range", n_vars_range)
    expected_length = _validate_range("length_range", length_range)
    path = trial_bundle_path(root, scenario, trial_id)
    if path.exists() or path.with_suffix(".json").exists():
        trial = load_trial_data(path)
        requested = (
            scenario,
            trial_id,
            int(seed),
            expected_n_vars,
            expected_length,
        )
        recorded = (
            trial.scenario,
            trial.trial_id,
            trial.seed,
            trial.n_vars_range,
            trial.length_range,
        )
        if recorded != requested:
            raise ValueError(
                f"Existing dataset configuration does not match request: {path}."
            )
        return trial

    trial = generate_trial_data(
        scenario,
        trial_id,
        seed=seed,
        n_vars_range=expected_n_vars,
        length_range=expected_length,
    )
    save_trial_data(trial, root)
    return trial


__all__ = [
    "DATASET_FORMAT_VERSION",
    "DEFAULT_LENGTH_RANGE",
    "DEFAULT_N_VARS_RANGE",
    "DEFAULT_SEED",
    "VALID_SCENARIOS",
    "TrialData",
    "generate_trial_data",
    "load_or_generate_trial_data",
    "load_trial_data",
    "save_trial_data",
    "trial_bundle_path",
]
