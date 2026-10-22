"""Small deterministic timing harness for nonlinear dependence helpers."""

from __future__ import annotations

import json
import platform
from pathlib import Path
import time

import numpy as np
import scipy

from fastCGC import FastGcStar
from fastCGC.dependence.hsic import hsic_test, rff_hsic_test
from fastCGC.dependence.information import (
    estimate_conditional_mutual_information,
    estimate_mutual_information,
)
from fastCGC.dependence.kcit import kcit_statistic
from fastCGC.dependence.rcit import rcit_test, rcot_test

OUTPUT_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "outputs"
    / "backend-benchmarks"
    / "benchmark_dependence_backends.json"
)


def _time_call(fn, *args, repeat: int = 3, **kwargs) -> dict[str, float]:
    timings = []
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        timings.append(time.perf_counter() - start)
    return {
        "result": float(result),
        "median_seconds": float(np.median(timings)),
        "max_seconds": float(np.max(timings)),
    }


def _time_profile(data: np.ndarray, profile: str) -> float:
    options = {
        "rff_hsic": {"n_features": 32},
        "rcot": {
            "n_features_x": 16,
            "n_features_y": 16,
            "n_features_z": 32,
        },
    }
    estimator = FastGcStar(
        n_pasts=1,
        dependence_profile=profile,
        evaluation_policy="full",
        backend_options=options,
        random_state=42,
    ).fit(data)
    return float(np.count_nonzero(estimator.conditional_evaluated_))


def main() -> None:
    rng = np.random.default_rng(42)
    z = rng.normal(size=128)
    x = z + 0.1 * rng.normal(size=128)
    y = np.sin(x) + 0.1 * z + 0.05 * rng.normal(size=128)
    profile_data = np.vstack((x, y, z, rng.normal(size=128)))

    payload = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "mi_knn": _time_call(estimate_mutual_information, x, y, n_neighbors=5),
        "cmi_knn": _time_call(
            estimate_conditional_mutual_information,
            x,
            y,
            z,
            n_neighbors=5,
        ),
        "kcit": _time_call(
            kcit_statistic,
            x,
            y,
            z,
            max_samples=256,
            memory_budget_mb=64,
        ),
        "hsic_exact": _time_call(
            lambda: hsic_test(
                x,
                y,
                n_shifts=32,
                max_samples=256,
                memory_budget_mb=64,
            ).statistic[0]
        ),
        "rff_hsic": _time_call(
            lambda: rff_hsic_test(
                x,
                y,
                n_features=64,
                random_state=42,
            ).statistic[0]
        ),
        "rcit": _time_call(
            lambda: rcit_test(
                x,
                y,
                z,
                n_features_x=16,
                n_features_y=16,
                n_features_z=32,
                n_features_joint=32,
                random_state=42,
            ).statistic[0]
        ),
        "rcot": _time_call(
            lambda: rcot_test(
                x,
                y,
                z,
                n_features_x=16,
                n_features_y=16,
                n_features_z=32,
                random_state=42,
            ).statistic[0]
        ),
        "fastgcstar_linear": _time_call(
            _time_profile,
            profile_data,
            "linear",
        ),
        "fastgcstar_randomized_kernel": _time_call(
            _time_profile,
            profile_data,
            "randomized_kernel",
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
