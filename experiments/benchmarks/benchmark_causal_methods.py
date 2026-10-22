"""Export paired datasets and compare the in-repository study methods.

The master process writes raw method-level metrics and edge-level predictions.
Each local method/dataset pair is fitted in a fresh process so failures remain
isolated. External comparison methods must be run unchanged from pinned author
repositories against the exported datasets. Statistical summaries and
external-output conversion belong in separate steps.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata as importlib_metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "outputs" / "causal-comparison"
EXPECTED_PYTHON = (3, 14)


def _package_version(distribution: str) -> str:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return "not-installed"


def _software_provenance() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "packages": {
            name: _package_version(name)
            for name in (
                "fastcgc",
                "matplotlib",
                "numpy",
                "pandas",
                "scikit-learn",
                "scipy",
                "tqdm",
            )
        },
    }


def _require_python_314() -> None:
    if sys.version_info[:2] != EXPECTED_PYTHON:
        raise RuntimeError(
            "The comparison requires CPython 3.14; "
            f"found {platform.python_version()}."
        )
    if platform.python_implementation() != "CPython":
        raise RuntimeError("The comparison requires CPython.")


try:
    import fastCGC  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name != "fastCGC":
        raise
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import fastCGC  # noqa: F401

from fastCGC.experiments.causal_benchmark import (  # noqa: E402
    CausalBenchmarkData,
    comparison_metrics,
    simulate_sparse_var,
)
from fastCGC.experiments.causal_methods import (  # noqa: E402
    METHOD_SPECS,
    MethodResult,
    fit_method,
    method_spec,
)


DEFAULT_METHODS = (
    "fast-cgc",
    "fast-cgc-star",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _json_ready(value: Any) -> Any:
    """Convert NumPy metadata values to JSON-native scalars and containers."""

    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _save_dataset_bundle(
    output_dir: Path,
    dataset_id: str,
    dataset: CausalBenchmarkData,
) -> Path:
    """Write one method-neutral dataset and its graph convention metadata."""

    dataset_dir = output_dir / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    data_path = dataset_dir / f"{dataset_id}.npz"
    np.savez_compressed(
        data_path,
        series=np.asarray(dataset.series, dtype=float),
        coefficients=np.asarray(dataset.coefficients, dtype=float),
        lag_truth=np.asarray(dataset.lag_truth, dtype=bool),
    )
    metadata = {
        "dataset_id": dataset_id,
        "data_digest": dataset.digest,
        "seed": dataset.seed,
        "noise": dataset.noise,
        "spectral_radius": dataset.spectral_radius,
        "series_layout": "variable, time",
        "coefficient_layout": "lag, target, source",
        "truth_layout": "lag, source, target",
        "external_method_policy": (
            "Run unchanged code from a pinned author repository; retain the "
            "native raw output and command before conversion."
        ),
    }
    data_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return data_path


def _save_worker_result(
    path: Path,
    result: MethodResult,
    *,
    digest: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        lag_adjacency=np.asarray(result.lag_adjacency, dtype=bool),
        lag_scores=np.asarray(result.lag_scores, dtype=float),
        lag_pvalues=(
            np.asarray(result.lag_pvalues, dtype=float)
            if result.lag_pvalues is not None
            else np.empty((0,), dtype=float)
        ),
    )
    metadata_path = path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "method": result.method,
                "data_digest": digest,
                "metadata": _json_ready(result.metadata),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _worker(args: argparse.Namespace) -> int:
    _require_python_314()
    dataset = simulate_sparse_var(
        n_variables=args.n_variables,
        n_samples=args.n_samples,
        order=args.order,
        seed=args.seed,
        edge_probability=args.edge_probability,
        noise=args.noise,
        max_radius=args.max_radius,
    )
    result = fit_method(
        args.method,
        dataset.series,
        order=args.order,
        alpha=args.alpha,
        beta=args.beta,
        seed=args.seed,
        n_permutations=args.n_permutations,
    )
    _save_worker_result(
        Path(args.worker_output),
        result,
        digest=dataset.digest,
    )
    return 0


def _worker_command(
    args: argparse.Namespace,
    *,
    method: str,
    n_variables: int,
    n_samples: int,
    order: int,
    seed: int,
    noise: str,
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--method",
        method,
        "--n-variables",
        str(n_variables),
        "--n-samples",
        str(n_samples),
        "--order",
        str(order),
        "--seed",
        str(seed),
        "--noise",
        noise,
        "--edge-probability",
        str(args.edge_probability),
        "--max-radius",
        str(args.max_radius),
        "--alpha",
        str(args.alpha),
        "--beta",
        str(args.beta),
        "--n-permutations",
        str(args.n_permutations),
        "--worker-output",
        str(output),
    ]


def _edge_rows(
    *,
    dataset_id: str,
    method: str,
    truth: np.ndarray,
    result: MethodResult,
) -> Iterable[dict[str, Any]]:
    n_lags, n_variables, _ = truth.shape
    pvalues = result.lag_pvalues
    for lag in range(n_lags):
        for source in range(n_variables):
            for target in range(n_variables):
                yield {
                    "dataset_id": dataset_id,
                    "method": method,
                    "lag": lag + 1,
                    "source": source,
                    "target": target,
                    "is_self": source == target,
                    "truth": int(truth[lag, source, target]),
                    "selected": int(result.lag_adjacency[lag, source, target]),
                    "score": float(result.lag_scores[lag, source, target]),
                    "pvalue": (
                        ""
                        if pvalues is None
                        else float(pvalues[lag, source, target])
                    ),
                }


def _load_worker_result(path: Path) -> tuple[MethodResult, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as stored:
        pvalues = stored["lag_pvalues"]
        result = MethodResult(
            method=json.loads(path.with_suffix(".json").read_text())["method"],
            lag_adjacency=stored["lag_adjacency"],
            lag_scores=stored["lag_scores"],
            lag_pvalues=None if pvalues.size == 0 else pvalues,
        )
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return result, metadata


def _dataset_cells(args: argparse.Namespace) -> Iterable[tuple[int, int, int, int, str]]:
    for n_variables in args.variables:
        for n_samples in args.samples:
            for order in args.orders:
                for seed in range(args.seed_start, args.seed_start + args.replicates):
                    for noise in args.noises:
                        yield n_variables, n_samples, order, seed, noise


def _run(args: argparse.Namespace) -> int:
    _require_python_314()
    unknown = [method for method in args.methods if method not in DEFAULT_METHODS]
    if unknown:
        raise ValueError(
            f"Methods without a local runner: {', '.join(sorted(unknown))}. "
            "Run comparison methods from their pinned author repositories."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="causal-comparison-") as temporary:
        temporary_dir = Path(temporary)
        for cell_index, (n_variables, n_samples, order, seed, noise) in enumerate(
            _dataset_cells(args)
        ):
            dataset = simulate_sparse_var(
                n_variables=n_variables,
                n_samples=n_samples,
                order=order,
                seed=seed,
                edge_probability=args.edge_probability,
                noise=noise,
                max_radius=args.max_radius,
            )
            dataset_id = (
                f"var{n_variables}-t{n_samples}-p{order}-{noise}-seed{seed}"
            )
            _save_dataset_bundle(output_dir, dataset_id, dataset)
            for method_index, method in enumerate(args.methods):
                worker_output = temporary_dir / f"{cell_index}-{method_index}.npz"
                command = _worker_command(
                    args,
                    method=method,
                    n_variables=n_variables,
                    n_samples=n_samples,
                    order=order,
                    seed=seed,
                    noise=noise,
                    output=worker_output,
                )
                completed = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout_seconds,
                    check=False,
                )
                common = {
                    "dataset_id": dataset_id,
                    "method": method,
                    "method_name": method_spec(method).display_name,
                    "family": method_spec(method).family,
                    "track": method_spec(method).track,
                    "n_variables": n_variables,
                    "n_samples": n_samples,
                    "order": order,
                    "noise": noise,
                    "seed": seed,
                    "spectral_radius": dataset.spectral_radius,
                    "data_digest": dataset.digest,
                }
                if completed.returncode != 0:
                    metric_rows.append(
                        {
                            **common,
                            "status": "failed",
                            "failure_aware_exact_f1": 0.0,
                            "error": completed.stderr.strip()[-4000:],
                        }
                    )
                    continue

                result, worker_metadata = _load_worker_result(worker_output)
                if worker_metadata["data_digest"] != dataset.digest:
                    raise RuntimeError(
                        f"Paired input mismatch for {dataset_id} and {method}."
                    )
                metrics = comparison_metrics(
                    dataset.lag_truth,
                    result.lag_adjacency,
                    result.lag_scores,
                )
                metric_rows.append(
                    {
                        **common,
                        "status": "ok",
                        "failure_aware_exact_f1": metrics["exact_f1"],
                        "error": "",
                        **metrics,
                    }
                )
                edge_rows.extend(
                    _edge_rows(
                        dataset_id=dataset_id,
                        method=method,
                        truth=dataset.lag_truth,
                        result=result,
                    )
                )

    _write_csv(output_dir / "method_metrics.csv", metric_rows)
    _write_csv(output_dir / "edge_predictions.csv", edge_rows)
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "software_provenance": _software_provenance(),
        "methods": list(args.methods),
        "variables": list(args.variables),
        "samples": list(args.samples),
        "orders": list(args.orders),
        "noises": list(args.noises),
        "replicates": args.replicates,
        "seed_start": args.seed_start,
        "edge_probability": args.edge_probability,
        "max_radius": args.max_radius,
        "alpha": args.alpha,
        "beta": args.beta,
        "n_permutations": args.n_permutations,
        "timeout_seconds": args.timeout_seconds,
        "orientation": "lag, source, target",
        "primary_estimand": "positive-lag cross-variable exact-edge graph",
        "ranking_metrics": {
            "auprc": "primary because graphs are sparse",
            "auroc": "secondary; only continuous scores are accepted",
        },
        "external_method_policy": (
            "Clone and pin each authors' repository, run its native code on "
            "the exported datasets, and preserve raw outputs before conversion."
        ),
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Wrote {len(metric_rows)} method rows to {output_dir}")
    return 0


def _list_methods() -> int:
    for spec in METHOD_SPECS:
        if spec.runnable:
            state = "local runner"
        elif spec.track.startswith("external-repository"):
            state = "external repo"
        else:
            state = "separate track"
        print(
            f"{spec.display_name:34} key={spec.key:24} "
            f"{state:14} {spec.family:36} "
            f"[{spec.dependency}] {spec.note}"
        )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-methods", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--method", default="fast-cgc")
    parser.add_argument("--methods", type=_parse_strings, default=DEFAULT_METHODS)
    parser.add_argument("--variables", type=_parse_ints, default=(5, 10, 20))
    parser.add_argument("--samples", type=_parse_ints, default=(250, 500, 1000))
    parser.add_argument("--orders", type=_parse_ints, default=(1, 2, 5))
    parser.add_argument("--noises", type=_parse_strings, default=("gaussian", "laplace"))
    parser.add_argument("--replicates", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--n-variables", type=int, default=5, help=argparse.SUPPRESS)
    parser.add_argument("--n-samples", type=int, default=500, help=argparse.SUPPRESS)
    parser.add_argument("--order", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=10_000, help=argparse.SUPPRESS)
    parser.add_argument("--noise", choices=("gaussian", "laplace"), default="gaussian", help=argparse.SUPPRESS)
    parser.add_argument("--edge-probability", type=float, default=0.12)
    parser.add_argument("--max-radius", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--worker-output", default="", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.list_methods:
        return _list_methods()
    if args.worker:
        if not args.worker_output:
            raise ValueError("--worker-output is required in worker mode.")
        return _worker(args)
    if args.replicates < 1:
        raise ValueError("replicates must be positive.")
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
