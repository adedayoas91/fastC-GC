"""Add shared datasets and resumable fits to nonlinear study notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks" / "simulations" / "nonlinear"


def _source(text: str) -> list[str]:
    return text.strip().splitlines(keepends=True)


def _markdown(text: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}


def _code(text: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


def _cells(method: str) -> list[dict[str, Any]]:
    star = method == "fcgc"
    label = "fastC-GC*" if star else "fastC-GC"
    slug = "nonlinear-fastcgc-star" if star else "nonlinear-fastcgc"
    directory = "fastC-GC-star" if star else "fastC-GC"
    filename = (
        "fastC-GC-star-singleLag-Markovian.ipynb"
        if star
        else "fastC-GC-singleLag-Markovian.ipynb"
    )
    result_stem = "nonlinear_fastcgcstar" if star else "nonlinear_fastcgc"

    introduction = f"""
# {label}: nonlinear dependence profiles

This notebook evaluates the local `FastGcStar` study method with nonlinear
dependence profiles. It is not the conventional published c-GC/c-GC*
implementation; the conventional linear notebooks load that implementation
from its pinned authors' repository.

The nonlinear fastC-GC and fastC-GC* notebooks share immutable dataset bundles in the
`nonlinear-small` suite. Each completed `(profile, trial)` fit has an atomic
checkpoint and native-output archive. Set `MAX_NEW_FITS` to limit one session,
or interrupt and rerun the execution cell to continue.

This notebook requires CPython 3.14, records package versions for
reproducibility, and reports graph-recovery metrics only.
"""

    imports = """
from __future__ import annotations

import importlib.metadata as importlib_metadata
import json
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

from fastCGC import FastGcStar
from fastCGC.dependence.config import supported_profiles
from fastCGC.experiments.benchmark_data import load_or_generate_trial_data
from fastCGC.experiments.notebook_checkpoint import (
    checkpoint_path,
    configuration_hash,
    derive_algorithm_seed,
    file_sha256,
    metric_rows,
    read_checkpoints,
    reusable_success,
    write_checkpoint,
    write_npz_checkpoint,
)


def package_version(distribution: str) -> str:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return "not-installed"


PACKAGE_VERSIONS = {
    name: package_version(name)
    for name in (
        "fastcgc",
        "matplotlib",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "tqdm",
    )
}
SOFTWARE_FINGERPRINT = {
    "python": platform.python_version(),
    "python_implementation": platform.python_implementation(),
    "packages": PACKAGE_VERSIONS,
}

LOCAL_SOURCE_DIGEST = file_sha256(
    PROJECT_ROOT / "src" / "fastCGC" / "core" / "fast_gc_star.py"
)
LOCAL_REVISION = f"local-source:{LOCAL_SOURCE_DIGEST}"

print(f"Project root: {PROJECT_ROOT}")
print(f"FastGcStar source digest: {LOCAL_SOURCE_DIGEST}")
print(f"Supported profiles: {supported_profiles()}")
print(f"Software: {SOFTWARE_FINGERPRINT}")
"""

    settings = f"""
METHOD = "{method}"
METHOD_SLUG = "{slug}"
SCENARIO = "singleLag-Markovian"
MASTER_SEED = 42
N_VARS_RANGE = (12, 14)
LENGTH_RANGE = (200, 300)
N_PAST = 1
N_LAGS = 1
NUM_TRIALS = 2
N_PERM = 1000
ALPHA = 0.05
BETA = 0.05
EVALUATION_POLICY = "auto"

PROFILES = [
    "linear",
    "randomized_kernel",
    "randomized_kernel_rcit",
    # "information",
    # "kernel",
]
BACKEND_OPTIONS = {{
    "rff_hsic": {{"n_features": 32}},
    "rcot": {{"n_features_x": 16, "n_features_y": 16, "n_features_z": 32}},
    "rcit": {{
        "n_features_x": 16,
        "n_features_y": 16,
        "n_features_z": 32,
        "n_features_joint": 32,
    }},
    "mi_knn": {{"n_neighbors": 4, "n_permutations": 24}},
    "cmi_knn": {{
        "n_neighbors": 4,
        "n_permutations": 24,
        "local_permutation_neighbors": 6,
    }},
}}

RESUME = True
MAX_NEW_FITS: int | None = None

SHARED_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "shared"
    / "causal-comparison-notebooks"
    / "nonlinear-small"
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
    "n_past": N_PAST,
    "n_lags": N_LAGS,
    "n_perm": N_PERM,
    "alpha": ALPHA,
    "beta": BETA,
    "profiles": PROFILES,
    "backend_options": BACKEND_OPTIONS,
    "evaluation_policy": EVALUATION_POLICY,
    "local_source_digest": LOCAL_SOURCE_DIGEST,
    "software_fingerprint": SOFTWARE_FINGERPRINT,
}}
CONFIG_HASH = configuration_hash(RUN_CONFIGURATION)

print(f"Method: {{METHOD_SLUG}}")
print(f"Scenario: {{SCENARIO}}")
print(f"Configuration hash: {{CONFIG_HASH}}")
print(f"Output directory: {{OUTPUT_DIR}}")
"""

    execution = f"""
new_fits = 0
pause_requested = False

for profile in tqdm(PROFILES, desc="profiles"):
    for trial_id in range(NUM_TRIALS):
        trial = load_or_generate_trial_data(
            SHARED_DATA_DIR,
            SCENARIO,
            trial_id,
            seed=MASTER_SEED,
            n_vars_range=N_VARS_RANGE,
            length_range=LENGTH_RANGE,
        )
        result_path = checkpoint_path(
            CHECKPOINT_DIR / profile,
            trial_id=trial_id,
            n_past=N_PAST,
        )
        expected = {{
            "method": METHOD_SLUG,
            "scenario": SCENARIO,
            "trial_id": trial_id,
            "n_past": N_PAST,
            "profile": profile,
            "dataset_digest": trial.digest,
            "config_hash": CONFIG_HASH,
            "upstream_commit": LOCAL_REVISION,
            "software_fingerprint": SOFTWARE_FINGERPRINT,
        }}
        if RESUME and reusable_success(result_path, expected=expected) is not None:
            tqdm.write(f"[{{profile}}] trial {{trial_id}}: checkpoint reused")
            continue
        if MAX_NEW_FITS is not None and new_fits >= MAX_NEW_FITS:
            pause_requested = True
            break

        algorithm_seed = derive_algorithm_seed(
            "nonlinear-fastcgc",
            MASTER_SEED,
            METHOD,
            profile,
            trial_id,
        )
        try:
            estimator = FastGcStar(
                n_perm=N_PERM,
                n_pasts=N_PAST,
                n_lags=N_LAGS,
                temporal=True,
                method=METHOD,
                dependence_profile=profile,
                backend_options=BACKEND_OPTIONS,
                evaluation_policy=EVALUATION_POLICY,
                random_state=algorithm_seed,
            )
            estimator.fit(
                trial.X,
                verbose=0,
                alpha=ALPHA,
                beta=BETA,
            )
            estimator.get_connectivity_matrix(
                alpha=ALPHA,
                beta=BETA,
                simulation=True,
            )
            estimator.compute_confusion_matrix(trial.A, simulation=True)
            metric_values = np.asarray(estimator.compute_metrics(), dtype=float)
            metrics = dict(zip(METRIC_NAMES, metric_values.tolist(), strict=True))
            native_path = (
                NATIVE_OUTPUT_DIR
                / profile
                / f"trial-{{trial_id:05d}}.npz"
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
                    "error": f"{{type(error).__name__}}: {{error}}",
                }},
            )
            tqdm.write(f"[{{profile}}] trial {{trial_id}} failed: {{error}}")

    if pause_requested:
        break

records = read_checkpoints(CHECKPOINT_DIR)
rows = metric_rows(
    records,
    method=METHOD_SLUG,
    scenario=SCENARIO,
    config_hash=CONFIG_HASH,
    upstream_commit=LOCAL_REVISION,
)
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_DIR / "{result_stem}_results.csv", index=False)
ok_df = df[df["status"] == "ok"].copy() if not df.empty else df.copy()

aggregated = {{}}
for profile in PROFILES:
    subset = ok_df[ok_df["profile"] == profile] if not ok_df.empty else ok_df
    aggregated[profile] = {{
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

expected_fits = NUM_TRIALS * len(PROFILES)
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

    profile_plot = f"""
if ok_df.empty:
    print("No successful fits are available to plot yet.")
else:
    data_by_metric = {{
        metric: {{
            "mean": [aggregated[profile][metric]["mean"] for profile in PROFILES],
            "std": [aggregated[profile][metric]["std"] for profile in PROFILES],
        }}
        for metric in METRIC_NAMES
    }}
    x = np.arange(len(PROFILES))
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "{label} performance by dependence profile",
        fontsize=16,
        fontweight="bold",
    )
    axes = ax.flatten()
    for index, metric in enumerate(METRIC_NAMES):
        axis = axes[index]
        means = data_by_metric[metric]["mean"]
        standard_deviations = data_by_metric[metric]["std"]
        axis.bar(x, means, yerr=standard_deviations, capsize=4, alpha=0.85)
        axis.set_xticks(x)
        axis.set_xticklabels(PROFILES, rotation=30, ha="right", fontsize=9)
        axis.set_ylabel("Proportion")
        axis.set_title(METRIC_LABELS[metric])
        axis.grid(True, axis="y", alpha=0.3)
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
        _code(profile_plot),
    ]


def synchronize() -> list[Path]:
    updated: list[Path] = []
    for method, directory, filename in (
        ("cgc", "fastC-GC", "fastC-GC-singleLag-Markovian.ipynb"),
        ("fcgc", "fastC-GC-star", "fastC-GC-star-singleLag-Markovian.ipynb"),
    ):
        path = NOTEBOOK_ROOT / directory / filename
        notebook = json.loads(path.read_text(encoding="utf-8"))
        notebook["cells"] = _cells(method)
        notebook.setdefault("metadata", {})["kernelspec"] = {
            "display_name": "fastcgc (Python 3.14, Apple Silicon)",
            "language": "python",
            "name": "fastcgc-py314-arm64",
        }
        language_info = notebook["metadata"].setdefault("language_info", {})
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
