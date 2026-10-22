"""Synchronize conventional c-GC notebooks with the upstream-run contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks" / "simulations" / "linear"
SCENARIOS = (
    "singleLag-Markovian",
    "singleLag-NonMarkovian",
    "varLags-Markovian",
    "varLags-NonMarkovian",
)


def _source(text: str) -> list[str]:
    return text.strip().splitlines(keepends=True)


def _markdown(text: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _source(text),
    }


def _code(text: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


def _configuration(method: str, scenario: str) -> dict[str, Any]:
    star = method == "fcgc"
    variable_lags = scenario.startswith("varLags")
    method_slug = "published-cgc-star" if star else "published-cgc"
    return {
        "method": method,
        "method_slug": method_slug,
        "display_name": "Published c-GC*" if star else "Published c-GC",
        "scenario": scenario,
        "n_lags": 2 if variable_lags else 1,
        "n_pasts": "list(range(2, 8))" if variable_lags else "list(range(1, 8))",
        "result_stem": "published_cgcstar" if star else "published_cgc",
    }


def _cells(configuration: dict[str, Any]) -> list[dict[str, Any]]:
    display_name = configuration["display_name"]
    method = configuration["method"]
    method_slug = configuration["method_slug"]
    scenario = configuration["scenario"]
    n_lags = configuration["n_lags"]
    n_pasts = configuration["n_pasts"]
    result_stem = configuration["result_stem"]

    introduction = f"""
# {display_name}: {scenario}

This notebook executes the authors' published `GcStar` implementation from
`https://github.com/adedayoas91/causalized-GC` at the recorded commit. It
does not copy or reimplement the estimator.

Every comparison notebook reads the same hash-verified dataset bundles under
`data/shared/causal-comparison-notebooks/`. A completed
`(scenario, trial, n_past)` fit is checkpointed immediately. Rerunning the
notebook skips only successful checkpoints whose dataset hash, settings, and
upstream commit still match. The upstream implementation has no mid-fit
checkpoint API, so an interrupted fit restarts from its beginning.

The data seed is deterministic and the stored input is verified by hash. The
notebook also records a seed before fitting, but the upstream estimator uses
randomness inside concurrent Numba code and has no random-state argument.
Estimator repeatability must therefore be measured rather than assumed.

This notebook requires CPython 3.14, records package versions for
reproducibility, and reports graph-recovery metrics only.
"""

    imports = """
from __future__ import annotations

import importlib.metadata as importlib_metadata
import json
import os
import platform
import sys
from pathlib import Path

EXPECTED_PYTHON = (3, 14)

if sys.version_info[:2] != EXPECTED_PYTHON:
    raise RuntimeError(
        "This notebook requires CPython 3.14; "
        f"found {platform.python_version()}."
    )
if platform.python_implementation() != "CPython":
    raise RuntimeError("This notebook requires CPython.")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = next(
    (
        path
        for path in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
        if (path / "src" / "fastCGC" / "__init__.py").exists()
    ),
    None,
)
if PROJECT_ROOT is None:
    raise FileNotFoundError("Could not locate the fastC-GC project root.")
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from fastCGC.experiments.benchmark_data import load_or_generate_trial_data
from fastCGC.experiments.notebook_checkpoint import (
    checkpoint_path,
    configuration_hash,
    derive_algorithm_seed,
    metric_rows,
    read_checkpoints,
    reusable_success,
    write_checkpoint,
    write_npz_checkpoint,
)
from fastCGC.experiments.upstream import load_published_cgc


def package_version(distribution: str) -> str:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return "not-installed"


PACKAGE_VERSIONS = {
    name: package_version(name)
    for name in (
        "cdt",
        "fastcgc",
        "llvmlite",
        "matplotlib",
        "numba",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "statsmodels",
        "torch",
        "tqdm",
    )
}
SOFTWARE_FINGERPRINT = {
    "python": platform.python_version(),
    "python_implementation": platform.python_implementation(),
    "packages": PACKAGE_VERSIONS,
}

UPSTREAM_ROOT = Path(
    os.environ.get(
        "CAUSALIZED_GC_ROOT",
        PROJECT_ROOT / "external" / "causalized-GC",
    )
).expanduser()
GcStar, UPSTREAM_REVISION = load_published_cgc(UPSTREAM_ROOT)

print(f"Project root: {PROJECT_ROOT}")
print(f"Upstream root: {UPSTREAM_REVISION.root}")
print(f"Upstream commit: {UPSTREAM_REVISION.commit}")
print(f"Software: {SOFTWARE_FINGERPRINT}")
"""

    settings = f"""
METHOD = "{method}"
METHOD_SLUG = "{method_slug}"
SCENARIO = "{scenario}"
MASTER_SEED = 42
N_VARS_RANGE = (18, 25)
LENGTH_RANGE = (3000, 6000)
N_PASTS = {n_pasts}
N_LAGS = {n_lags}
NUM_TRIALS = 10
N_PERM = 1000
ALPHA = 0.01
BETA = 0.001

# Pause/resume controls. Set a positive limit for a bounded session; rerun the
# cell to continue. Successful, matching checkpoints are reused when RESUME is
# true. Failed fits remain recorded but are retried.
RESUME = True
MAX_NEW_FITS: int | None = None

SHARED_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "shared"
    / "causal-comparison-notebooks"
    / "linear-large"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "causal-comparison-notebooks"
    / METHOD_SLUG
    / SCENARIO
)
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
NATIVE_OUTPUT_DIR = OUTPUT_DIR / "native"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRIC_NAMES = [
    "accuracy",
    "precision",
    "recall",
    "fpr",
    "balanced_accuracy",
    "f1",
]
METRIC_LABELS = {{
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "fpr": "False Positive Rate",
    "balanced_accuracy": "Balanced Accuracy",
    "f1": "F1 Score",
}}

RUN_CONFIGURATION = {{
    "method": METHOD,
    "scenario": SCENARIO,
    "master_seed": MASTER_SEED,
    "n_vars_range": N_VARS_RANGE,
    "length_range": LENGTH_RANGE,
    "n_pasts": N_PASTS,
    "n_lags": N_LAGS,
    "n_perm": N_PERM,
    "alpha": ALPHA,
    "beta": BETA,
    "upstream_url": UPSTREAM_REVISION.remote_url,
    "upstream_commit": UPSTREAM_REVISION.commit,
    "output_estimand": "aggregate positive-lag graph",
    "software_fingerprint": SOFTWARE_FINGERPRINT,
}}
CONFIG_HASH = configuration_hash(RUN_CONFIGURATION)

print(f"Method: {{METHOD_SLUG}}")
print(f"Scenario: {{SCENARIO}}")
print(f"Data seed: {{MASTER_SEED}} + trial_id")
print(f"Configuration hash: {{CONFIG_HASH}}")
print(f"Output directory: {{OUTPUT_DIR}}")
"""

    execution = f"""
new_fits = 0
pause_requested = False

for trial_id in tqdm(range(NUM_TRIALS), desc=f"{{METHOD_SLUG}} trials"):
    trial = load_or_generate_trial_data(
        SHARED_DATA_DIR,
        SCENARIO,
        trial_id,
        seed=MASTER_SEED,
        n_vars_range=N_VARS_RANGE,
        length_range=LENGTH_RANGE,
    )
    tqdm.write(
        f"[trial {{trial_id + 1}}/{{NUM_TRIALS}}] "
        f"{{trial.n_vars}} variables, {{trial.length}} timepoints, "
        f"digest={{trial.digest[:12]}}"
    )

    for n_past in N_PASTS:
        result_path = checkpoint_path(
            CHECKPOINT_DIR,
            trial_id=trial_id,
            n_past=n_past,
        )
        expected = {{
            "method": METHOD_SLUG,
            "scenario": SCENARIO,
            "trial_id": trial_id,
            "n_past": n_past,
            "dataset_digest": trial.digest,
            "config_hash": CONFIG_HASH,
            "upstream_commit": UPSTREAM_REVISION.commit,
            "software_fingerprint": SOFTWARE_FINGERPRINT,
        }}
        if RESUME and reusable_success(result_path, expected=expected) is not None:
            tqdm.write(f"    n_past={{n_past}}: checkpoint reused")
            continue
        if MAX_NEW_FITS is not None and new_fits >= MAX_NEW_FITS:
            pause_requested = True
            break

        algorithm_seed = derive_algorithm_seed(
            "published-cgc",
            MASTER_SEED,
            METHOD,
            SCENARIO,
            trial_id,
            n_past,
        )
        np.random.seed(algorithm_seed)
        try:
            estimator = GcStar(
                n_perm=N_PERM,
                n_pasts=n_past,
                n_lags=N_LAGS,
                temporal=True,
                method=METHOD,
            )
            estimator.fit(trial.X, verbose=0)
            estimator.get_connectivity_matrix(
                simulation=True,
                alpha=ALPHA,
                beta=BETA,
            )
            estimator.compute_confusion_matrix(trial.A, simulation=True)
            metric_values = np.asarray(estimator.all_metrics(), dtype=float)
            metrics = dict(zip(METRIC_NAMES, metric_values.tolist(), strict=True))
            native_path = (
                NATIVE_OUTPUT_DIR
                / f"trial-{{trial_id:05d}}-past-{{n_past:02d}}.npz"
            )
            native_digest = write_npz_checkpoint(
                native_path,
                {{
                    "conn_mat": estimator.conn_mat,
                    "corr": estimator.corr_,
                    "pvalue_corr": estimator.pVal_corr_,
                    "conditional_corr": estimator.inv_corr_,
                    "pvalue_conditional": estimator.pVal_inv_corr_,
                }},
            )
            write_checkpoint(
                result_path,
                {{
                    **expected,
                    "status": "ok",
                    "data_seed": trial.effective_seed,
                    "algorithm_seed": algorithm_seed,
                    "metrics": metrics,
                    "native_output": str(native_path.relative_to(PROJECT_ROOT)),
                    "native_output_sha256": native_digest,
                    "upstream_url": UPSTREAM_REVISION.remote_url,
                    "error": "",
                }},
            )
            new_fits += 1
        except KeyboardInterrupt:
            pause_requested = True
            tqdm.write(
                "Interrupted during a fit. Completed checkpoints are safe; "
                "rerun this cell to continue."
            )
            break
        except Exception as error:
            write_checkpoint(
                result_path,
                {{
                    **expected,
                    "status": "failed",
                    "data_seed": trial.effective_seed,
                    "algorithm_seed": algorithm_seed,
                    "metrics": {{}},
                    "native_output": "",
                    "native_output_sha256": "",
                    "upstream_url": UPSTREAM_REVISION.remote_url,
                    "error": f"{{type(error).__name__}}: {{error}}",
                }},
            )
            tqdm.write(f"    n_past={{n_past}} failed: {{error}}")

    if pause_requested:
        break

records = read_checkpoints(CHECKPOINT_DIR)
rows = metric_rows(
    records,
    method=METHOD_SLUG,
    scenario=SCENARIO,
    config_hash=CONFIG_HASH,
    upstream_commit=UPSTREAM_REVISION.commit,
)
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_DIR / "{result_stem}_results.csv", index=False)
ok_df = df[df["status"] == "ok"].copy() if not df.empty else df.copy()

aggregated = {{}}
for n_past in N_PASTS:
    subset = ok_df[ok_df["n_past"] == n_past] if not ok_df.empty else ok_df
    aggregated[str(n_past)] = {{
        metric: {{
            "mean": (
                float(np.round(subset[metric].mean(), 4))
                if metric in subset and not subset.empty
                else float("nan")
            ),
            "std": (
                float(np.round(subset[metric].std(ddof=0), 4))
                if metric in subset and not subset.empty
                else float("nan")
            ),
        }}
        for metric in METRIC_NAMES
    }}

with (OUTPUT_DIR / "{result_stem}_aggregated.json").open(
    "w", encoding="utf-8"
) as handle:
    json.dump(aggregated, handle, indent=2)

expected_fits = NUM_TRIALS * len(N_PASTS)
completed_fits = int((df["status"] == "ok").sum()) if not df.empty else 0
failed_fits = int((df["status"] == "failed").sum()) if not df.empty else 0
summary = {{
    **RUN_CONFIGURATION,
    "config_hash": CONFIG_HASH,
    "expected_fits": expected_fits,
    "completed_fits": completed_fits,
    "failed_fits": failed_fits,
    "complete": completed_fits == expected_fits,
    "pause_requested": pause_requested,
    "software_fingerprint": SOFTWARE_FINGERPRINT,
}}
with (OUTPUT_DIR / "{result_stem}_summary.json").open(
    "w", encoding="utf-8"
) as handle:
    json.dump(summary, handle, indent=2)

print(
    f"Completed {{completed_fits}}/{{expected_fits}} fits; "
    f"failed records: {{failed_fits}}."
)
if pause_requested:
    print("Run this cell again to resume from the next unfinished fit.")
"""

    plotting = f"""
if ok_df.empty:
    print("No successful fits are available to plot yet.")
else:
    data_by_metric = {{
        metric: {{
            "mean": [
                aggregated[str(n_past)][metric]["mean"]
                for n_past in N_PASTS
            ],
            "std": [
                aggregated[str(n_past)][metric]["std"]
                for n_past in N_PASTS
            ],
        }}
        for metric in METRIC_NAMES
    }}

    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "{display_name} performance versus $n_{{pasts}}$",
        fontsize=16,
        fontweight="bold",
    )
    axes = ax.flatten()
    for index, metric in enumerate(METRIC_NAMES):
        axis = axes[index]
        means = data_by_metric[metric]["mean"]
        standard_deviations = data_by_metric[metric]["std"]
        axis.plot(N_PASTS, means, marker=".")
        axis.errorbar(N_PASTS, means, yerr=standard_deviations, fmt="o")
        axis.set_xticks(N_PASTS)
        axis.set_xlabel("$n_{{pasts}}$")
        axis.set_ylabel("Proportion")
        axis.set_title(METRIC_LABELS[metric])
        axis.grid(True, alpha=0.3)
        axis.set_ylim([0, 1.05])

    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.savefig(
        OUTPUT_DIR / "{result_stem}_metrics.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
"""

    return [
        _markdown(introduction),
        _code(imports),
        _code(settings),
        _code(execution),
        _code(plotting),
    ]


def synchronize() -> list[Path]:
    """Rewrite the eight conventional linear notebooks deterministically."""

    updated: list[Path] = []
    for method_directory, prefix, method in (
        ("c-GC", "c-GC", "cgc"),
        ("c-GC-star", "c-GC-star", "fcgc"),
    ):
        for scenario in SCENARIOS:
            path = NOTEBOOK_ROOT / method_directory / f"{prefix}-{scenario}.ipynb"
            notebook = json.loads(path.read_text(encoding="utf-8"))
            notebook["cells"] = _cells(_configuration(method, scenario))
            notebook.setdefault("metadata", {})["kernelspec"] = {
                "display_name": "fastcgc (Python 3.14, Apple Silicon)",
                "language": "python",
                "name": "fastcgc-py314-arm64",
            }
            language_info = notebook["metadata"].setdefault(
                "language_info", {}
            )
            language_info["name"] = "python"
            language_info["version"] = "3.14"
            path.write_text(
                json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            updated.append(path)
    return updated


def main() -> int:
    for path in synchronize():
        print(path.relative_to(PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
