#!/usr/bin/env python3
"""Run a narrow, resumable wall-clock comparison of the four RCCP methods.

This script deliberately excludes unrelated CSL baselines. It compares the two
fast algorithms only with the c-GC/c-GC* implementations whose conditioning
sets they adopt. Each grid cell runs in a fresh process, performs an untimed
warm-up, and records repeated fit-plus-graph-construction timings.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "outputs" / "rccp-runtime"
METHODS = ("cgc", "fast-cgc", "cgc-star", "fast-cgc-star")

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
)
from fastCGC.experiments.upstream import (  # noqa: E402
    load_installed_conventional_cgc,
)


def _parse_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return parsed


def _parse_strings(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one value.")
    return parsed


def _require_python_314() -> None:
    if sys.version_info[:2] != (3, 14) or platform.python_implementation() != "CPython":
        raise RuntimeError(
            "The runtime comparison requires CPython 3.14; found "
            f"{platform.python_implementation()} {platform.python_version()}."
        )


def _fit_once(
    method: str,
    series: np.ndarray,
    *,
    order: int,
    alpha: float,
    beta: float,
    seed: int,
    n_permutations: int,
    conventional_class: type[Any] | None,
) -> None:
    if method.startswith("fast-"):
        fit_method(
            method,
            series,
            order=order,
            alpha=alpha,
            beta=beta,
            ridge=1e-6,
            evaluation_policy="full",
            seed=seed,
        )
        return

    if conventional_class is None:
        raise RuntimeError("Conventional c-GC was not prepared before timing.")
    estimator = conventional_class(
        n_perm=n_permutations,
        n_pasts=order,
        n_lags=order,
        temporal=True,
        method="fcgc" if method == "cgc-star" else "cgc",
    )
    np.random.seed(seed)
    estimator.fit(series, verbose=0)
    estimator.get_connectivity_matrix(
        simulation=True,
        alpha=alpha,
        beta=beta,
    )


def _worker(args: argparse.Namespace) -> int:
    _require_python_314()
    dataset = simulate_sparse_var(
        n_variables=args.n_variables,
        n_samples=args.n_samples,
        order=args.order,
        seed=args.seed,
        edge_probability=args.edge_probability,
        noise="gaussian",
        max_radius=args.max_radius,
    )
    conventional_class: type[Any] | None = None
    if args.method.startswith("fast-"):
        source_revision = "local-source:" + ":".join(
            file_sha256(path)
            for path in (
                PROJECT_ROOT / "src" / "fastCGC" / "core" / "fast_gc_star.py",
                PROJECT_ROOT / "src" / "fastCGC" / "experiments" / "causal_methods.py",
            )
        )
    else:
        conventional_class, revision = load_installed_conventional_cgc()
        source_revision = revision.commit
    for _ in range(args.warmups):
        _fit_once(
            args.method,
            dataset.series,
            order=args.order,
            alpha=args.alpha,
            beta=args.beta,
            seed=args.seed,
            n_permutations=args.n_permutations,
            conventional_class=conventional_class,
        )
    timings = []
    for repeat in range(args.repeats):
        started = time.perf_counter()
        _fit_once(
            args.method,
            dataset.series,
            order=args.order,
            alpha=args.alpha,
            beta=args.beta,
            seed=args.seed + repeat,
            n_permutations=args.n_permutations,
            conventional_class=conventional_class,
        )
        timings.append(time.perf_counter() - started)
    Path(args.worker_output).write_text(
        json.dumps(
            {
                "data_digest": dataset.digest,
                "source_revision": source_revision,
                "timings_seconds": timings,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def _jobs(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    index = 0
    for n_variables in args.variables:
        for n_samples in args.samples:
            for order in args.orders:
                seed = args.seed_start + index
                index += 1
                for method in args.methods:
                    yield {
                        "method": method,
                        "n_variables": n_variables,
                        "n_samples": n_samples,
                        "order": order,
                        "seed": seed,
                    }


def _worker_command(
    args: argparse.Namespace,
    job: dict[str, Any],
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--method",
        str(job["method"]),
        "--n-variables",
        str(job["n_variables"]),
        "--n-samples",
        str(job["n_samples"]),
        "--order",
        str(job["order"]),
        "--seed",
        str(job["seed"]),
        "--alpha",
        str(args.alpha),
        "--beta",
        str(args.beta),
        "--n-permutations",
        str(args.n_permutations),
        "--edge-probability",
        str(args.edge_probability),
        "--max-radius",
        str(args.max_radius),
        "--warmups",
        str(args.warmups),
        "--repeats",
        str(args.repeats),
        "--worker-output",
        str(output),
    ]


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
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
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _collect_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for checkpoint in read_checkpoints(output_dir / "checkpoints"):
        if checkpoint.get("status") != "ok":
            continue
        for repeat, elapsed in enumerate(checkpoint["timings_seconds"]):
            rows.append(
                {
                    "method": checkpoint["method"],
                    "n_variables": checkpoint["n_variables"],
                    "n_samples": checkpoint["n_samples"],
                    "order": checkpoint["order"],
                    "seed": checkpoint["seed"],
                    "repeat": repeat,
                    "wall_seconds": elapsed,
                    "data_digest": checkpoint["data_digest"],
                    "source_revision": checkpoint["source_revision"],
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            row["n_variables"],
            row["n_samples"],
            row["order"],
            row["method"],
            row["repeat"],
        ),
    )


def _run(args: argparse.Namespace) -> int:
    _require_python_314()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configuration = {
        "methods": list(args.methods),
        "variables": list(args.variables),
        "samples": list(args.samples),
        "orders": list(args.orders),
        "seed_start": args.seed_start,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "alpha": args.alpha,
        "beta": args.beta,
        "n_permutations": args.n_permutations,
        "edge_probability": args.edge_probability,
        "max_radius": args.max_radius,
        "thread_limits": 1,
    }
    manifest_path = output_dir / "runtime-protocol.json"
    manifest = {
        "configuration_hash": configuration_hash(configuration),
        "configuration": configuration,
        "scope": "four RCCP methods only; fit plus graph construction",
    }
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError(f"Existing runtime protocol does not match: {manifest_path}")
    else:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    jobs = list(_jobs(args))
    new_cells = 0
    with tempfile.TemporaryDirectory(prefix="rccp-runtime-") as temporary:
        temporary_dir = Path(temporary)
        for index, job in enumerate(jobs, start=1):
            job_config = {**configuration, **job}
            job_id = configuration_hash(job_config)[:16]
            checkpoint_path = output_dir / "checkpoints" / f"{job_id}.json"
            expected = {"job_id": job_id, **job}
            if args.resume and reusable_success(checkpoint_path, expected=expected):
                print(f"[{index}/{len(jobs)}] reuse {job['method']} {job_id}")
                continue
            if args.max_new_cells is not None and new_cells >= args.max_new_cells:
                print("Reached --max-new-cells; rerun the same command to resume.")
                break
            worker_output = temporary_dir / f"{job_id}.json"
            environment = os.environ.copy()
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                environment[name] = "1"
            completed = subprocess.run(
                _worker_command(args, job, worker_output),
                cwd=PROJECT_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=args.timeout_seconds,
                check=False,
            )
            if completed.returncode != 0:
                write_checkpoint(
                    checkpoint_path,
                    {
                        **expected,
                        "status": "failed",
                        "error": completed.stderr.strip()[-4000:],
                    },
                )
                if args.fail_fast:
                    raise RuntimeError(completed.stderr)
                continue
            result = json.loads(worker_output.read_text(encoding="utf-8"))
            write_checkpoint(
                checkpoint_path,
                {**expected, "status": "ok", **result},
            )
            new_cells += 1
            print(f"[{index}/{len(jobs)}] fit   {job['method']} {job_id}")

    rows = _collect_rows(output_dir)
    _atomic_write_csv(output_dir / "runtime-measurements.csv", rows)
    checkpoints = read_checkpoints(output_dir / "checkpoints")
    failed = sum(checkpoint.get("status") != "ok" for checkpoint in checkpoints)
    print(
        f"Collected {len(rows)} timing rows with {failed} failed cells in "
        f"{output_dir}"
    )
    return int(failed > 0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--methods", type=_parse_strings, default=METHODS)
    parser.add_argument("--method", default="fast-cgc", help=argparse.SUPPRESS)
    parser.add_argument("--variables", type=_parse_ints, default=(5, 20))
    parser.add_argument("--samples", type=_parse_ints, default=(250, 1000))
    parser.add_argument("--orders", type=_parse_ints, default=(1, 5))
    parser.add_argument("--seed-start", type=int, default=40_000)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.005)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--edge-probability", type=float, default=0.12)
    parser.add_argument("--max-radius", type=float, default=0.8)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-new-cells", type=int)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--n-variables", type=int, default=5, help=argparse.SUPPRESS)
    parser.add_argument("--n-samples", type=int, default=250, help=argparse.SUPPRESS)
    parser.add_argument("--order", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=40_000, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", default="", help=argparse.SUPPRESS)
    parser.set_defaults(resume=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.worker:
        if not args.worker_output:
            raise ValueError("--worker-output is required in worker mode.")
        return _worker(args)
    unknown = set(args.methods) - set(METHODS)
    if unknown:
        raise ValueError(f"Unsupported runtime methods: {sorted(unknown)}")
    if args.warmups < 0 or args.repeats < 2:
        raise ValueError("Use at least two repeats and non-negative warm-ups.")
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
