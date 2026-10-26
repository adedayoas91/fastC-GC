#!/usr/bin/env python3
"""Run resumable calibration, ridge, and lag-sensitivity experiments.

Each method/configuration/seed fit is an atomic checkpoint. Rerunning the same
command reuses verified successful fits and continues at the next unfinished
job. Component p-values are saved so all declared beta thresholds are scored
from one full-evaluation fit.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "outputs" / "statistical-validation"

try:
    import fastCGC  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name != "fastCGC":
        raise
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastCGC.experiments.causal_benchmark import simulate_sparse_var  # noqa: E402
from fastCGC.experiments.causal_methods import fit_method  # noqa: E402
from fastCGC.experiments.notebook_checkpoint import (  # noqa: E402
    configuration_hash,
    file_sha256,
    read_checkpoints,
    reusable_success,
    write_checkpoint,
    write_npz_checkpoint,
)
from fastCGC.experiments.statistical_validation import (  # noqa: E402
    null_calibration_metrics,
    recovery_metrics,
    simulate_null_series,
    truth_for_fitted_order,
)


SCHEMA_VERSION = 1
METHODS = ("fast-cgc", "fast-cgc-star")
NULL_REGIMES = ("iid-null", "stationary-ar-null", "drifting-null")


@dataclass(frozen=True, slots=True)
class ValidationJob:
    arm: str
    method: str
    regime: str
    n_variables: int
    n_samples: int
    true_order: int
    fitted_order: int
    ridge: float
    seed: int

    @property
    def configuration(self) -> dict[str, Any]:
        return {"validation_schema": SCHEMA_VERSION, **asdict(self)}

    @property
    def identifier(self) -> str:
        digest = configuration_hash(self.configuration)[:16]
        return f"{self.arm}-{self.method}-{self.regime}-{digest}"


def _parse_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return parsed


def _parse_floats(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one number.")
    return parsed


def _parse_strings(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one value.")
    return parsed


def _require_python_314() -> None:
    if sys.version_info[:2] != (3, 14) or platform.python_implementation() != "CPython":
        raise RuntimeError(
            "The validation environment requires CPython 3.14; found "
            f"{platform.python_implementation()} {platform.python_version()}."
        )


def _jobs(args: argparse.Namespace) -> list[ValidationJob]:
    seeds = range(args.seed_start, args.seed_start + args.replicates)
    jobs: list[ValidationJob] = []
    if "calibration" in args.arms:
        for method in args.methods:
            for regime in NULL_REGIMES:
                for n_samples in args.calibration_samples:
                    for seed in seeds:
                        jobs.append(
                            ValidationJob(
                                arm="calibration",
                                method=method,
                                regime=regime,
                                n_variables=args.n_variables,
                                n_samples=n_samples,
                                true_order=args.true_order,
                                fitted_order=args.true_order,
                                ridge=args.default_ridge,
                                seed=seed,
                            )
                        )
    if "lambda" in args.arms:
        for method in args.methods:
            for regime in ("stationary-ar-null", "sparse-var"):
                for ridge in args.ridges:
                    for seed in seeds:
                        jobs.append(
                            ValidationJob(
                                arm="lambda",
                                method=method,
                                regime=regime,
                                n_variables=args.n_variables,
                                n_samples=args.sensitivity_samples,
                                true_order=args.true_order,
                                fitted_order=args.true_order,
                                ridge=ridge,
                                seed=seed,
                            )
                        )
    if "lag" in args.arms:
        for method in args.methods:
            for fitted_order in args.fitted_orders:
                for seed in seeds:
                    jobs.append(
                        ValidationJob(
                            arm="lag",
                            method=method,
                            regime="sparse-var",
                            n_variables=args.n_variables,
                            n_samples=args.sensitivity_samples,
                            true_order=args.true_order,
                            fitted_order=fitted_order,
                            ridge=args.default_ridge,
                            seed=seed,
                        )
                    )
    return jobs


def _data_for_job(
    job: ValidationJob,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray | None, str]:
    if job.regime == "sparse-var":
        generated = simulate_sparse_var(
            n_variables=job.n_variables,
            n_samples=job.n_samples,
            order=job.true_order,
            seed=job.seed,
            edge_probability=args.edge_probability,
            noise="gaussian",
            max_radius=args.max_radius,
        )
        return generated.series, generated.lag_truth, generated.digest
    generated = simulate_null_series(
        regime=job.regime,
        n_variables=job.n_variables,
        n_samples=job.n_samples,
        seed=job.seed,
        ar_coefficient=args.ar_coefficient,
        drift_strength=args.drift_strength,
    )
    return generated.series, None, generated.digest


def _score_job(
    job: ValidationJob,
    args: argparse.Namespace,
    marginal_pvalues: np.ndarray,
    conditional_pvalues: np.ndarray,
    lag_truth: np.ndarray | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    effective_length = job.n_samples - job.fitted_order
    for beta in args.betas:
        common = {
            **asdict(job),
            "alpha": args.alpha,
            "beta": beta,
            "effective_length": effective_length,
            "shift_pvalue_floor": 1.0 / effective_length,
            "beta_attainable": beta >= 1.0 / effective_length,
        }
        if lag_truth is None:
            metrics = null_calibration_metrics(
                marginal_pvalues,
                conditional_pvalues,
                alpha=args.alpha,
                beta=beta,
            )
            rows.append({**common, "omitted_true_edges": 0, **metrics})
        else:
            fitted_truth, omitted = truth_for_fitted_order(
                lag_truth,
                job.fitted_order,
            )
            metrics = recovery_metrics(
                fitted_truth,
                marginal_pvalues,
                conditional_pvalues,
                alpha=args.alpha,
                beta=beta,
            )
            rows.append({**common, "omitted_true_edges": omitted, **metrics})
    return rows


def _run_job(
    job: ValidationJob,
    args: argparse.Namespace,
    native_path: Path,
) -> tuple[list[dict[str, Any]], str, str]:
    series, lag_truth, data_digest = _data_for_job(job, args)
    result = fit_method(
        job.method,
        series,
        order=job.fitted_order,
        alpha=args.alpha,
        beta=max(args.betas),
        ridge=job.ridge,
        evaluation_policy="full",
        seed=job.seed,
    )
    if (
        result.lag_marginal_pvalues is None
        or result.lag_conditional_pvalues is None
    ):
        raise RuntimeError("The fitted method did not expose component p-values.")
    arrays: dict[str, np.ndarray] = {
        "lag_adjacency": np.asarray(result.lag_adjacency, dtype=bool),
        "lag_scores": np.asarray(result.lag_scores, dtype=float),
        "lag_pvalues": np.asarray(result.lag_pvalues, dtype=float),
        "lag_marginal_pvalues": np.asarray(
            result.lag_marginal_pvalues,
            dtype=float,
        ),
        "lag_conditional_pvalues": np.asarray(
            result.lag_conditional_pvalues,
            dtype=float,
        ),
        "alpha": np.asarray(args.alpha),
        "fit_beta": np.asarray(max(args.betas)),
        "ridge": np.asarray(job.ridge),
    }
    if lag_truth is not None:
        arrays["lag_truth"] = np.asarray(lag_truth, dtype=bool)
    native_digest = write_npz_checkpoint(native_path, arrays)
    metrics = _score_job(
        job,
        args,
        arrays["lag_marginal_pvalues"],
        arrays["lag_conditional_pvalues"],
        lag_truth,
    )
    return metrics, data_digest, native_digest


def _atomic_write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _collect_metrics(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for checkpoint in read_checkpoints(output_dir / "checkpoints"):
        if checkpoint.get("status") != "ok":
            continue
        rows.extend(checkpoint.get("metrics", []))
    return sorted(
        rows,
        key=lambda row: (
            str(row["arm"]),
            str(row["method"]),
            str(row["regime"]),
            int(row["n_samples"]),
            int(row["fitted_order"]),
            float(row["ridge"]),
            int(row["seed"]),
            float(row["beta"]),
        ),
    )


def _study_configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "validation_schema": SCHEMA_VERSION,
        "methods": list(args.methods),
        "arms": list(args.arms),
        "n_variables": args.n_variables,
        "calibration_samples": list(args.calibration_samples),
        "sensitivity_samples": args.sensitivity_samples,
        "true_order": args.true_order,
        "fitted_orders": list(args.fitted_orders),
        "ridges": list(args.ridges),
        "default_ridge": args.default_ridge,
        "alpha": args.alpha,
        "betas": list(args.betas),
        "replicates": args.replicates,
        "seed_start": args.seed_start,
        "edge_probability": args.edge_probability,
        "max_radius": args.max_radius,
        "ar_coefficient": args.ar_coefficient,
        "drift_strength": args.drift_strength,
        "evaluation_policy": "full",
    }


def _write_or_validate_manifest(
    output_dir: Path,
    configuration: dict[str, Any],
) -> None:
    path = output_dir / "validation-protocol.json"
    payload = {
        "configuration_hash": configuration_hash(configuration),
        "configuration": configuration,
        "python": platform.python_version(),
    }
    if path.exists():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        if recorded != payload:
            raise ValueError(
                f"Existing validation protocol does not match this run: {path}."
            )
        return
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> int:
    _require_python_314()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_or_validate_manifest(output_dir, _study_configuration(args))
    jobs = _jobs(args)
    completed_new = 0
    for index, job in enumerate(jobs, start=1):
        checkpoint_path = output_dir / "checkpoints" / f"{job.identifier}.json"
        native_path = output_dir / "native" / f"{job.identifier}.npz"
        expected = {
            "job_id": job.identifier,
            "config_hash": configuration_hash(job.configuration),
        }
        if args.resume:
            reusable = reusable_success(checkpoint_path, expected=expected)
            if reusable is not None:
                recorded_native = Path(str(reusable["native_output"]))
                if not recorded_native.is_absolute():
                    recorded_native = output_dir / recorded_native
                if not recorded_native.is_file():
                    raise FileNotFoundError(recorded_native)
                if file_sha256(recorded_native) != reusable["native_output_sha256"]:
                    raise ValueError(f"Native output digest mismatch: {recorded_native}")
                print(f"[{index}/{len(jobs)}] reuse {job.identifier}")
                continue
        if args.max_new_fits is not None and completed_new >= args.max_new_fits:
            print("Reached --max-new-fits; rerun the same command to resume.")
            break
        print(f"[{index}/{len(jobs)}] fit   {job.identifier}")
        try:
            metrics, data_digest, native_digest = _run_job(
                job,
                args,
                native_path,
            )
        except KeyboardInterrupt:
            print("Interrupted; completed job checkpoints are safe.")
            break
        except Exception as error:  # preserve failure provenance for later audit
            write_checkpoint(
                checkpoint_path,
                {
                    **expected,
                    **job.configuration,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            if args.fail_fast:
                raise
            print(f"failed: {type(error).__name__}: {error}")
            continue
        write_checkpoint(
            checkpoint_path,
            {
                **expected,
                **job.configuration,
                "status": "ok",
                "data_digest": data_digest,
                "native_output": str(native_path.relative_to(output_dir)),
                "native_output_sha256": native_digest,
                "metrics": metrics,
            },
        )
        completed_new += 1

    rows = _collect_metrics(output_dir)
    _atomic_write_csv(output_dir / "validation-metrics.csv", rows)
    print(f"Collected {len(rows)} threshold rows in {output_dir}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arms",
        type=_parse_strings,
        default=("calibration", "lambda", "lag"),
    )
    parser.add_argument("--methods", type=_parse_strings, default=METHODS)
    parser.add_argument("--n-variables", type=int, default=5)
    parser.add_argument(
        "--calibration-samples",
        type=_parse_ints,
        default=(250, 500, 1000),
    )
    parser.add_argument("--sensitivity-samples", type=int, default=500)
    parser.add_argument("--true-order", type=int, default=2)
    parser.add_argument("--fitted-orders", type=_parse_ints, default=(1, 2, 4))
    parser.add_argument(
        "--ridges",
        type=_parse_floats,
        default=(1e-8, 1e-6, 1e-4, 1e-2),
    )
    parser.add_argument("--default-ridge", type=float, default=1e-6)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument(
        "--betas",
        type=_parse_floats,
        default=(0.001, 0.005, 0.01),
    )
    parser.add_argument("--replicates", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=30_000)
    parser.add_argument("--edge-probability", type=float, default=0.12)
    parser.add_argument("--max-radius", type=float, default=0.8)
    parser.add_argument("--ar-coefficient", type=float, default=0.8)
    parser.add_argument("--drift-strength", type=float, default=1.5)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-new-fits", type=int)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--fail-fast", action="store_true")
    parser.set_defaults(resume=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    unknown_arms = set(args.arms) - {"calibration", "lambda", "lag"}
    if unknown_arms:
        raise ValueError(f"Unknown validation arms: {sorted(unknown_arms)}")
    unknown_methods = set(args.methods) - set(METHODS)
    if unknown_methods:
        raise ValueError(f"Unsupported methods: {sorted(unknown_methods)}")
    if args.replicates < 1:
        raise ValueError("--replicates must be positive.")
    if args.max_new_fits is not None and args.max_new_fits < 1:
        raise ValueError("--max-new-fits must be positive.")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
