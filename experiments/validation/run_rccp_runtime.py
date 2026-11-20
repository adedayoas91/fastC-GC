#!/usr/bin/env python3
"""Run resumable wall-clock comparisons of RCCP methods or dependence variants.

The default run compares fastC-GC and fastC-GC* with their conventional
counterparts. With --dependence-variants, it compares test configurations
within each fast method. Each grid cell runs in a fresh process with untimed
warm-ups and records repeated fit-plus-graph-construction timings.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
from statistics import mean, stdev
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "outputs" / "rccp-runtime"
DEPENDENCE_OUTPUT = DEFAULT_OUTPUT.with_name("rccp-dependence-runtime")
METHODS = ("cgc", "fast-cgc", "cgc-star", "fast-cgc-star")
FAST_METHODS = ("fast-cgc", "fast-cgc-star")

try:
    import fastCGC  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name != "fastCGC":
        raise
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastCGC.experiments.causal_benchmark import simulate_sparse_var  # noqa: E402
from fastCGC.experiments.causal_methods import fit_method  # noqa: E402
from fastCGC.experiments.dependence_comparison import (  # noqa: E402
    DEPENDENCE_TESTS,
    fit_dependence_variant,
)
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
    dependence_variant: str | None = None,
) -> None:
    if dependence_variant is not None:
        fit_dependence_variant(
            method,
            series,
            order=order,
            alpha=alpha,
            beta=beta,
            seed=seed,
            variant=dependence_variant,
        )
        return
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


def _fast_source_revision(*, dependence: bool) -> str:
    source = PROJECT_ROOT / "src" / "fastCGC"
    paths = [
        source / "core" / "fast_gc_star.py",
        source / "experiments" / "causal_methods.py",
    ]
    if dependence:
        paths.extend(
            [
                source / "core" / "gc_star.py",
                source / "experiments" / "causal_benchmark.py",
                source / "experiments" / "dependence_comparison.py",
                *sorted((source / "dependence").glob("*.py")),
            ]
        )
    return "local-source:" + ":".join(file_sha256(path) for path in paths)


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
        source_revision = _fast_source_revision(dependence=args.dependence_variant is not None)
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
            dependence_variant=args.dependence_variant,
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
            dependence_variant=args.dependence_variant,
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
                    job = {
                        "method": method,
                        "n_variables": n_variables,
                        "n_samples": n_samples,
                        "order": order,
                        "seed": seed,
                    }
                    if args.dependence_variants is None:
                        yield job
                    else:
                        for variant in args.dependence_variants:
                            yield {**job, "dependence_variant": variant}


def _worker_command(
    args: argparse.Namespace,
    job: dict[str, Any],
    output: Path,
) -> list[str]:
    command = [
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
    if "dependence_variant" in job:
        command.extend(("--dependence-variant", job["dependence_variant"]))
    return command


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
        dependence = {}
        if "dependence_variant" in checkpoint:
            variant = checkpoint["dependence_variant"]
            marginal, conditional = DEPENDENCE_TESTS[variant]
            dependence = {
                "dependence_variant": variant,
                "marginal_test": marginal,
                "conditional_test": conditional,
            }
        for repeat, elapsed in enumerate(checkpoint["timings_seconds"]):
            rows.append(
                {
                    "method": checkpoint["method"],
                    **dependence,
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
            row.get("dependence_variant", ""),
            row["repeat"],
        ),
    )


def _summarize_dependence_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cell_fields = ("n_variables", "n_samples", "order", "seed", "method", "data_digest")
    timings: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = tuple(row[name] for name in cell_fields) + (row["dependence_variant"],)
        timings[key].append(row["wall_seconds"])

    summary = []
    for key, values in timings.items():
        variant = key[-1]
        marginal, conditional = DEPENDENCE_TESTS[variant]
        linear = timings.get((*key[:-1], "linear"))
        elapsed = mean(values)
        summary.append(
            {
                **dict(zip(cell_fields, key[:-1], strict=True)),
                "dependence_variant": variant,
                "marginal_test": marginal,
                "conditional_test": conditional,
                "repeats": len(values),
                "mean_wall_seconds": elapsed,
                "sd_wall_seconds": stdev(values),
                "wall_time_ratio_vs_linear": elapsed / mean(linear) if linear else None,
            }
        )
    return summary


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
    scope = "four RCCP methods only; fit plus graph construction"
    if args.dependence_variants is not None:
        configuration.update(
            {
                "dependence_variants": list(args.dependence_variants),
                "evaluation_policy": "full",
                "source_revision": _fast_source_revision(dependence=True),
                "environment": {
                    "platform": platform.platform(),
                    "processor": platform.processor(),
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "scipy": version("scipy"),
                },
            }
        )
        scope = "fastC-GC dependence variants; fit plus graph construction"
    manifest_path = output_dir / "runtime-protocol.json"
    manifest = {
        "configuration_hash": configuration_hash(configuration),
        "configuration": configuration,
        "scope": scope,
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
            label = job["method"]
            if "dependence_variant" in job:
                label += f"/{job['dependence_variant']}"
            job_config = {**configuration, **job}
            job_id = configuration_hash(job_config)[:16]
            checkpoint_path = output_dir / "checkpoints" / f"{job_id}.json"
            expected = {"job_id": job_id, **job}
            if args.resume and reusable_success(checkpoint_path, expected=expected):
                print(f"[{index}/{len(jobs)}] reuse {label} {job_id}")
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
            try:
                completed = subprocess.run(
                    _worker_command(args, job, worker_output),
                    cwd=PROJECT_ROOT,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout_seconds,
                    check=False,
                )
                error = completed.stderr.strip()[-4000:] if completed.returncode else None
            except subprocess.TimeoutExpired:
                error = f"Worker exceeded the {args.timeout_seconds:g} s timeout."
            if error is not None:
                write_checkpoint(
                    checkpoint_path,
                    {
                        **expected,
                        "status": "failed",
                        "error": error,
                    },
                )
                if args.fail_fast:
                    raise RuntimeError(error)
                print(f"[{index}/{len(jobs)}] failed {label}: {error}")
                continue
            result = json.loads(worker_output.read_text(encoding="utf-8"))
            write_checkpoint(
                checkpoint_path,
                {**expected, "status": "ok", **result},
            )
            new_cells += 1
            print(f"[{index}/{len(jobs)}] fit   {label} {job_id}")

    rows = _collect_rows(output_dir)
    _atomic_write_csv(output_dir / "runtime-measurements.csv", rows)
    if args.dependence_variants is not None:
        _atomic_write_csv(output_dir / "runtime-summary.csv", _summarize_dependence_rows(rows))
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
    parser.add_argument("--methods", type=_parse_strings)
    parser.add_argument(
        "--dependence-variants",
        type=_parse_strings,
        help=f"Comma-separated variants for the fast methods: {', '.join(DEPENDENCE_TESTS)}.",
    )
    parser.add_argument("--dependence-variant", choices=DEPENDENCE_TESTS, help=argparse.SUPPRESS)
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
    parser.add_argument("--output-dir")
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
    benchmark_dependence = args.dependence_variants is not None
    supported_methods = FAST_METHODS if benchmark_dependence else METHODS
    args.methods = args.methods or supported_methods
    args.output_dir = args.output_dir or (
        DEPENDENCE_OUTPUT if benchmark_dependence else DEFAULT_OUTPUT
    )
    unknown = set(args.methods) - set(supported_methods)
    if unknown:
        raise ValueError(f"Unsupported runtime methods: {sorted(unknown)}")
    if args.warmups < 0 or args.repeats < 2:
        raise ValueError("Use at least two repeats and non-negative warm-ups.")
    if benchmark_dependence:
        variants = args.dependence_variants
        unknown = set(variants) - DEPENDENCE_TESTS.keys()
        if unknown:
            raise ValueError(f"Unknown dependence variants: {sorted(unknown)}")
        if len(variants) != len(set(variants)):
            raise ValueError("Each dependence variant may appear only once.")
        if Path(args.output_dir).resolve() == DEFAULT_OUTPUT.resolve():
            raise ValueError("Use a separate --output-dir for dependence-variant timings.")
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
