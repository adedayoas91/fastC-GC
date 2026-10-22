"""Quick nonlinear-backend calibration and power checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from fastCGC.dependence.information import (
    conditional_mutual_information_test,
    mutual_information_test,
)
from fastCGC.dependence.kcit import kcit_test

OUTPUT_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "outputs"
    / "backend-benchmarks"
    / "validate_nonlinear_backends.json"
)


@dataclass(frozen=True)
class ValidationSummary:
    seeds: int
    mi_null_mean: float
    mi_power: float
    cmi_null_mean: float
    cmi_power: float
    kcit_null_mean: float
    kcit_power: float


def _run_validation() -> ValidationSummary:
    mi_null_pvalues: list[float] = []
    mi_alt_pvalues: list[float] = []
    cmi_null_pvalues: list[float] = []
    cmi_alt_pvalues: list[float] = []
    kcit_null_pvalues: list[float] = []
    kcit_alt_pvalues: list[float] = []

    for seed in range(6):
        rng = np.random.default_rng(seed)
        z = rng.normal(size=96)
        x = z + 0.1 * rng.normal(size=96)
        y_null = z + 0.1 * rng.normal(size=96)
        y_alt = np.sin(x) + 0.15 * z + 0.05 * rng.normal(size=96)
        w_null = rng.normal(size=96)
        w_alt = np.sin(x) + 0.1 * rng.normal(size=96)

        mi_null_pvalues.append(
            mutual_information_test(
                x,
                w_null,
                n_neighbors=4,
                n_permutations=24,
                random_state=seed,
            ).pvalue
        )
        mi_alt_pvalues.append(
            mutual_information_test(
                x,
                w_alt,
                n_neighbors=4,
                n_permutations=24,
                random_state=seed,
            ).pvalue
        )
        cmi_null_pvalues.append(
            conditional_mutual_information_test(
                x,
                y_null,
                z,
                n_neighbors=4,
                local_permutation_neighbors=6,
                n_permutations=24,
                random_state=seed,
            ).pvalue
        )
        cmi_alt_pvalues.append(
            conditional_mutual_information_test(
                x,
                y_alt,
                z,
                n_neighbors=4,
                local_permutation_neighbors=6,
                n_permutations=24,
                random_state=seed,
            ).pvalue
        )
        kcit_null_pvalues.append(
            kcit_test(
                x,
                y_null,
                z,
                n_permutations=24,
                local_permutation_neighbors=6,
                random_state=seed,
            ).pvalue
        )
        kcit_alt_pvalues.append(
            kcit_test(
                x,
                y_alt,
                z,
                n_permutations=24,
                local_permutation_neighbors=6,
                random_state=seed,
            ).pvalue
        )

    alpha = 0.05
    return ValidationSummary(
        seeds=6,
        mi_null_mean=float(np.mean(mi_null_pvalues)),
        mi_power=float(np.mean(np.asarray(mi_alt_pvalues) <= alpha)),
        cmi_null_mean=float(np.mean(cmi_null_pvalues)),
        cmi_power=float(np.mean(np.asarray(cmi_alt_pvalues) <= alpha)),
        kcit_null_mean=float(np.mean(kcit_null_pvalues)),
        kcit_power=float(np.mean(np.asarray(kcit_alt_pvalues) <= alpha)),
    )


def main() -> None:
    summary = _run_validation()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(summary)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
