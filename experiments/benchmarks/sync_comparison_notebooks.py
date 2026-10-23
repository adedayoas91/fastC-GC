"""Generate the dataset-first causal-comparison notebook matrix.

The notebooks are deliberately thin launch surfaces. Local study methods call
the installed ``fastCGC`` package, while comparison methods call unchanged
entry points from verified author checkouts. No external estimator is copied
into this repository.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks"

TIGRAMITE_URL = "https://github.com/jakobrunge/tigramite.git"
TIGRAMITE_COMMIT = "ff3ff13e1481073b8c5833a6fde1c304627a208e"
UNN_URL = "https://github.com/jarrycyx/UNN.git"
UNN_COMMIT = "2212724cce8a7752db564089a7537fecedb0e36d"


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    key: str
    label: str
    truth: str


@dataclass(frozen=True, slots=True)
class MethodSpec:
    key: str
    filename: str
    label: str
    source: str


DATASETS = (
    DatasetSpec(
        "sparseVAR",
        "stable sparse VAR",
        "exact positive-lag graph",
    ),
    DatasetSpec(
        "lorenz96",
        "nonlinear Lorenz-96",
        "aggregate structural-dependency graph",
    ),
    DatasetSpec(
        "dream3",
        "DREAM3 gene-network",
        "aggregate directed gene-regulatory graph",
    ),
)

METHODS = (
    MethodSpec("fast-cgc", "fastC-GC.ipynb", "fastC-GC", "local package"),
    MethodSpec(
        "fast-cgc-star",
        "fastC-GC-star.ipynb",
        "fastC-GC*",
        "local package",
    ),
    MethodSpec(
        "cgc",
        "c-GC.ipynb",
        "conventional c-GC",
        "adedayoas91/causalized-GC",
    ),
    MethodSpec(
        "cgc-star",
        "c-GC-star.ipynb",
        "conventional c-GC*",
        "adedayoas91/causalized-GC",
    ),
    MethodSpec(
        "pcmci-plus",
        "PCMCI-plus.ipynb",
        "PCMCI+",
        "jakobrunge/tigramite",
    ),
    MethodSpec(
        "bagged-pcmci-plus",
        "Bagged-PCMCI-plus.ipynb",
        "Bagged PCMCI+",
        "jakobrunge/tigramite",
    ),
    MethodSpec(
        "cuts-plus",
        "CUTS-plus.ipynb",
        "CUTS+",
        "jarrycyx/UNN",
    ),
)


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


def _notebook(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "fastcgc (Python 3.14, Apple Silicon)",
                "language": "python",
                "name": "fastcgc-py314-arm64",
            },
            "language_info": {"name": "python", "version": "3.14"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


COMMON_IMPORTS = r'''
from __future__ import annotations

import hashlib
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

import numpy as np

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

from fastCGC.experiments.notebook_checkpoint import (
    configuration_hash,
    derive_algorithm_seed,
    file_sha256,
    reusable_success,
    write_checkpoint,
    write_npz_checkpoint,
)
'''


def _settings(dataset: DatasetSpec, method: MethodSpec) -> str:
    return f'''
DATASET_KEY = "{dataset.key}"
METHOD_KEY = "{method.key}"
MASTER_SEED = 42
ALPHA = 0.01
BETA = 0.001
N_PERMUTATIONS = 1000
BOOTSTRAP_SAMPLES = 200

# Pause/resume boundary: one complete method/dataset fit.
RESUME = True
MAX_NEW_FITS: int | None = None

DATASET_DIR = (
    PROJECT_ROOT / "data" / "shared" / "causal-comparison"
    / DATASET_KEY / "datasets"
)
OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "outputs" / "causal-comparison"
    / DATASET_KEY / METHOD_KEY
)
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
NATIVE_OUTPUT_DIR = OUTPUT_DIR / "native"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_CONFIGURATION = {{
    "dataset": DATASET_KEY,
    "method": METHOD_KEY,
    "master_seed": MASTER_SEED,
    "alpha": ALPHA,
    "beta": BETA,
    "n_permutations": N_PERMUTATIONS,
    "bootstrap_samples": BOOTSTRAP_SAMPLES,
    "truth_scope": "{dataset.truth}",
}}
CONFIG_HASH = configuration_hash(RUN_CONFIGURATION)
print(f"Dataset directory: {{DATASET_DIR}}")
print(f"Output directory: {{OUTPUT_DIR}}")
'''


LOCAL_RUNNER = r'''
from fastCGC.experiments.causal_methods import fit_method

SOURCE_REVISION = "local-source:" + file_sha256(
    PROJECT_ROOT / "src" / "fastCGC" / "core" / "fast_gc_star.py"
)


def run_native(series, fit_order, algorithm_seed, run_dir):
    result = fit_method(
        METHOD_KEY,
        series,
        order=fit_order,
        alpha=ALPHA,
        beta=BETA,
        seed=algorithm_seed,
        n_permutations=N_PERMUTATIONS,
    )
    arrays = {
        "lag_adjacency": result.lag_adjacency,
        "lag_scores": result.lag_scores,
    }
    if result.lag_pvalues is not None:
        arrays["lag_pvalues"] = result.lag_pvalues
    return arrays
'''


CONVENTIONAL_RUNNER = r'''
from fastCGC.experiments.upstream import load_conventional_cgc

UPSTREAM_ROOT = Path(
    os.environ.get(
        "CAUSALIZED_GC_ROOT",
        PROJECT_ROOT / "external" / "causalized-GC",
    )
).expanduser()
GcStar, UPSTREAM_REVISION = load_conventional_cgc(UPSTREAM_ROOT)
SOURCE_REVISION = UPSTREAM_REVISION.commit


def run_native(series, fit_order, algorithm_seed, run_dir):
    np.random.seed(algorithm_seed)
    upstream_method = "fcgc" if METHOD_KEY.endswith("-star") else "cgc"
    estimator = GcStar(
        n_perm=N_PERMUTATIONS,
        n_pasts=fit_order,
        n_lags=fit_order,
        temporal=True,
        method=upstream_method,
    )
    estimator.fit(series, verbose=0)
    estimator.get_connectivity_matrix(
        simulation=True,
        alpha=ALPHA,
        beta=BETA,
    )
    return {
        "conn_mat": estimator.conn_mat,
        "corr": estimator.corr_,
        "pvalue_corr": estimator.pVal_corr_,
        "conditional_corr": estimator.inv_corr_,
        "pvalue_conditional": estimator.pVal_inv_corr_,
    }
'''


PCMCI_RUNNER = f'''
from fastCGC.experiments.upstream import verify_upstream_revision

UPSTREAM_ROOT = Path(
    os.environ.get("TIGRAMITE_ROOT", PROJECT_ROOT / "external" / "tigramite")
).expanduser()
UPSTREAM_REVISION = verify_upstream_revision(
    UPSTREAM_ROOT,
    expected_url="{TIGRAMITE_URL}",
    expected_commit="{TIGRAMITE_COMMIT}",
)
if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

SOURCE_REVISION = UPSTREAM_REVISION.commit


def run_native(series, fit_order, algorithm_seed, run_dir):
    dataframe = pp.DataFrame(series.T)
    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=ParCorr(significance="analytic"),
        verbosity=0,
    )
    results = pcmci.run_pcmciplus(
        tau_min=1,
        tau_max=fit_order,
        pc_alpha=ALPHA,
    )
    return {{
        "graph": results["graph"],
        "val_matrix": results["val_matrix"],
        "p_matrix": results["p_matrix"],
    }}
'''


BAGGED_PCMCI_RUNNER = f'''
from fastCGC.experiments.upstream import verify_upstream_revision

UPSTREAM_ROOT = Path(
    os.environ.get("TIGRAMITE_ROOT", PROJECT_ROOT / "external" / "tigramite")
).expanduser()
UPSTREAM_REVISION = verify_upstream_revision(
    UPSTREAM_ROOT,
    expected_url="{TIGRAMITE_URL}",
    expected_commit="{TIGRAMITE_COMMIT}",
)
if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

SOURCE_REVISION = UPSTREAM_REVISION.commit


def run_native(series, fit_order, algorithm_seed, run_dir):
    dataframe = pp.DataFrame(series.T)
    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=ParCorr(significance="analytic"),
        verbosity=0,
    )
    results = pcmci.run_bootstrap_of(
        method="run_pcmciplus",
        method_args={{
            "tau_min": 1,
            "tau_max": fit_order,
            "pc_alpha": ALPHA,
        }},
        boot_samples=BOOTSTRAP_SAMPLES,
        boot_blocklength=1,
        seed=algorithm_seed,
    )
    summary = results["summary_results"]
    return {{
        "most_frequent_links": summary["most_frequent_links"],
        "link_frequency": summary["link_frequency"],
        "val_matrix_mean": summary["val_matrix_mean"],
        "val_matrix_interval": summary["val_matrix_interval"],
    }}
'''


CUTS_RUNNER = f'''
from copy import deepcopy

from fastCGC.experiments.upstream import verify_upstream_revision

UNN_ROOT = Path(
    os.environ.get("UNN_ROOT", PROJECT_ROOT / "external" / "UNN")
).expanduser()
UPSTREAM_REVISION = verify_upstream_revision(
    UNN_ROOT,
    expected_url="{UNN_URL}",
    expected_commit="{UNN_COMMIT}",
)
CUTS_ROOT = UNN_ROOT / "CUTS_Plus"
if str(CUTS_ROOT) not in sys.path:
    sys.path.insert(0, str(CUTS_ROOT))

import torch
from omegaconf import OmegaConf
import cuts_plus
from utils.logger import MyLogger
from utils.misc import reproduc

SOURCE_REVISION = UPSTREAM_REVISION.commit


def run_native(series, fit_order, algorithm_seed, run_dir):
    options = OmegaConf.load(CUTS_ROOT / "opt" / "lorenz_example.yaml")
    native_options = deepcopy(options.sota.cuts_plus)
    native_options.n_nodes = int(series.shape[0])
    reproduc(seed=algorithm_seed, benchmark=False, deterministic=True)

    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    log = MyLogger(
        log_dir=str(run_dir / "upstream-log"),
        stdout=False,
        stderr=False,
        tensorboard=False,
    )
    data = np.asarray(series.T, dtype=float)
    mask = np.ones_like(data)
    try:
        graph = cuts_plus.main(
            data,
            mask,
            None,
            native_options,
            log,
            device=device,
        )
    finally:
        log.close()
    return {{"graph": np.asarray(graph)}}
'''


EXECUTION = r'''
bundle_paths = sorted(DATASET_DIR.glob("*.npz"))
if not bundle_paths:
    raise FileNotFoundError(
        f"No frozen {DATASET_KEY} bundles found in {DATASET_DIR}. "
        "Run 00-prepare-data.ipynb first."
    )

new_fits = 0
pause_requested = False
for bundle_path in bundle_paths:
    metadata_path = bundle_path.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata for {bundle_path}.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(bundle_path, allow_pickle=False) as stored:
        series = np.asarray(stored["series"], dtype=float)
        if "lag_truth" in stored.files:
            default_order = int(stored["lag_truth"].shape[0])
        else:
            default_order = 1
    digest = hashlib.sha256(
        np.ascontiguousarray(series, dtype=np.float64).tobytes()
    ).hexdigest()
    recorded_digest = metadata.get("data_digest", metadata.get("digest"))
    if digest != recorded_digest:
        raise ValueError(f"Dataset digest mismatch for {bundle_path}.")
    fit_order = int(metadata.get("fit_order", default_order))

    checkpoint = CHECKPOINT_DIR / f"{bundle_path.stem}.json"
    expected = {
        "method": METHOD_KEY,
        "dataset": DATASET_KEY,
        "dataset_id": bundle_path.stem,
        "dataset_digest": digest,
        "fit_order": fit_order,
        "config_hash": CONFIG_HASH,
        "upstream_commit": SOURCE_REVISION,
    }
    if RESUME and reusable_success(checkpoint, expected=expected) is not None:
        print(f"{bundle_path.stem}: checkpoint reused")
        continue
    if MAX_NEW_FITS is not None and new_fits >= MAX_NEW_FITS:
        pause_requested = True
        break

    algorithm_seed = derive_algorithm_seed(
        "causal-comparison",
        MASTER_SEED,
        DATASET_KEY,
        METHOD_KEY,
        bundle_path.stem,
    )
    run_dir = OUTPUT_DIR / "runs" / bundle_path.stem
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        arrays = run_native(series, fit_order, algorithm_seed, run_dir)
        native_path = NATIVE_OUTPUT_DIR / f"{bundle_path.stem}.npz"
        native_digest = write_npz_checkpoint(native_path, arrays)
        write_checkpoint(
            checkpoint,
            {
                **expected,
                "status": "ok",
                "algorithm_seed": algorithm_seed,
                "native_output": str(native_path.relative_to(PROJECT_ROOT)),
                "native_output_sha256": native_digest,
                "error": "",
            },
        )
        new_fits += 1
        print(f"{bundle_path.stem}: complete")
    except KeyboardInterrupt:
        pause_requested = True
        print("Interrupted. Completed dataset checkpoints are safe.")
        break
    except Exception as error:
        write_checkpoint(
            checkpoint,
            {
                **expected,
                "status": "failed",
                "algorithm_seed": algorithm_seed,
                "native_output": "",
                "native_output_sha256": "",
                "error": f"{type(error).__name__}: {error}",
            },
        )
        print(f"{bundle_path.stem}: failed: {error}")

print(f"Completed {new_fits} new fits in this session.")
if pause_requested:
    print("Rerun this cell to resume at the next unfinished dataset.")
'''


def _runner(method: MethodSpec) -> str:
    if method.key.startswith("fast-"):
        return LOCAL_RUNNER
    if method.key in {"cgc", "cgc-star"}:
        return CONVENTIONAL_RUNNER
    if method.key == "pcmci-plus":
        return PCMCI_RUNNER
    if method.key == "bagged-pcmci-plus":
        return BAGGED_PCMCI_RUNNER
    if method.key == "cuts-plus":
        return CUTS_RUNNER
    raise KeyError(method.key)


def _method_notebook(dataset: DatasetSpec, method: MethodSpec) -> dict[str, Any]:
    introduction = f'''
# {method.label} on {dataset.label}

This notebook is the `{dataset.key}` copy of the `{method.key}` experiment.
It reads the same frozen, hash-verified bundles as every other method in this
directory and writes one atomic checkpoint per completed dataset fit.

Source: **{method.source}**. External methods call a pinned authors' checkout;
their estimator code is neither copied nor reimplemented here. Native arrays
are saved before any common graph-orientation conversion or scoring. This
study records causal-structure efficacy only and does not benchmark runtime.
'''
    return _notebook(
        [
            _markdown(introduction),
            _code(COMMON_IMPORTS),
            _code(_settings(dataset, method)),
            _code(_runner(method)),
            _code(EXECUTION),
        ]
    )


PREPARE_IMPORTS = r'''
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

EXPECTED_PYTHON = (3, 14)
if sys.version_info[:2] != EXPECTED_PYTHON:
    raise RuntimeError(
        "This notebook requires CPython 3.14; "
        f"found {platform.python_version()}."
    )

import numpy as np

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

from fastCGC.experiments.upstream import verify_upstream_revision
'''


SPARSE_PREPARE = r'''
DATASET_KEY = "sparseVAR"
DATASET_ROOT = (
    PROJECT_ROOT / "data" / "shared" / "causal-comparison" / DATASET_KEY
)

# Confirmatory grid. Every method notebook reads the resulting files.
VARIABLES = (5, 10, 20)
SAMPLES = (250, 500, 1000)
ORDERS = (1, 2, 5)
NOISES = ("gaussian", "laplace")
REPLICATES = 50
SEED_START = 10_000

command = [
    sys.executable,
    str(PROJECT_ROOT / "experiments" / "benchmarks" / "benchmark_causal_methods.py"),
    "--datasets-only",
    "--variables", ",".join(map(str, VARIABLES)),
    "--samples", ",".join(map(str, SAMPLES)),
    "--orders", ",".join(map(str, ORDERS)),
    "--noises", ",".join(NOISES),
    "--replicates", str(REPLICATES),
    "--seed-start", str(SEED_START),
    "--output-dir", str(DATASET_ROOT),
]
subprocess.run(command, cwd=PROJECT_ROOT, check=True)
print(f"Prepared {len(list((DATASET_ROOT / 'datasets').glob('*.npz')))} bundles.")
'''


SHARED_EXTERNAL_PREPARE_HELPERS = r'''
def series_digest(series):
    return hashlib.sha256(
        np.ascontiguousarray(series, dtype=np.float64).tobytes()
    ).hexdigest()


def save_bundle(path, *, series, aggregate_truth, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = series_digest(series)
    metadata = {**metadata, "data_digest": digest}
    metadata_path = path.with_suffix(".json")
    if path.exists() or metadata_path.exists():
        if not path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Incomplete dataset bundle at {path}.")
        recorded = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(path, allow_pickle=False) as stored:
            saved_series = np.asarray(stored["series"], dtype=float)
        if series_digest(saved_series) != digest or recorded != metadata:
            raise ValueError(f"Existing bundle does not match request: {path}.")
        return False

    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        np.savez_compressed(
            handle,
            series=np.asarray(series, dtype=float),
            aggregate_truth=np.asarray(aggregate_truth, dtype=bool),
        )
        temporary_data = Path(handle.name)
    os.replace(temporary_data, path)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{metadata_path.name}.",
        delete=False,
    ) as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary_metadata = Path(handle.name)
    os.replace(temporary_metadata, metadata_path)
    return True
'''


LORENZ_PREPARE = f'''
DATASET_KEY = "lorenz96"
DATASET_DIR = (
    PROJECT_ROOT / "data" / "shared" / "causal-comparison"
    / DATASET_KEY / "datasets"
)
UNN_ROOT = Path(
    os.environ.get("UNN_ROOT", PROJECT_ROOT / "external" / "UNN")
).expanduser()
UPSTREAM_REVISION = verify_upstream_revision(
    UNN_ROOT,
    expected_url="{UNN_URL}",
    expected_commit="{UNN_COMMIT}",
)
CUTS_ROOT = UNN_ROOT / "CUTS_Plus"
if str(CUTS_ROOT) not in sys.path:
    sys.path.insert(0, str(CUTS_ROOT))
from data.simu_data import simulate_lorenz_96

VARIABLES = (10, 20)
SAMPLES = (250, 500, 1000)
REPLICATES = 50
SEED_START = 20_000
FIT_ORDER = 5
MAX_NEW_DATASETS: int | None = None

created = 0
for n_variables in VARIABLES:
    for n_samples in SAMPLES:
        for seed in range(SEED_START, SEED_START + REPLICATES):
            if MAX_NEW_DATASETS is not None and created >= MAX_NEW_DATASETS:
                break
            data, target_source_truth = simulate_lorenz_96(
                p=n_variables,
                T=n_samples,
                F=10.0,
                delta_t=0.1,
                sd=0.1,
                burn_in=1000,
                seed=seed,
            )
            dataset_id = f"lorenz96-var{{n_variables}}-t{{n_samples}}-seed{{seed}}"
            was_created = save_bundle(
                DATASET_DIR / f"{{dataset_id}}.npz",
                series=np.asarray(data, dtype=float).T,
                aggregate_truth=np.asarray(target_source_truth, dtype=bool).T,
                metadata={{
                    "dataset_id": dataset_id,
                    "dataset": DATASET_KEY,
                    "generator": "UNN/CUTS_Plus/data/simu_data.py:simulate_lorenz_96",
                    "upstream_commit": UPSTREAM_REVISION.commit,
                    "seed": seed,
                    "n_variables": n_variables,
                    "n_samples": n_samples,
                    "fit_order": FIT_ORDER,
                    "series_layout": "variable, time",
                    "truth_layout": "source, target",
                    "truth_scope": "aggregate structural dependency",
                }},
            )
            created += int(was_created)
        if MAX_NEW_DATASETS is not None and created >= MAX_NEW_DATASETS:
            break
    if MAX_NEW_DATASETS is not None and created >= MAX_NEW_DATASETS:
        break
print(f"Created {{created}} new Lorenz-96 bundles. Rerun to resume.")
'''


DREAM_PREPARE = f'''
DATASET_KEY = "dream3"
DATASET_DIR = (
    PROJECT_ROOT / "data" / "shared" / "causal-comparison"
    / DATASET_KEY / "datasets"
)
UNN_ROOT = Path(
    os.environ.get("UNN_ROOT", PROJECT_ROOT / "external" / "UNN")
).expanduser()
UPSTREAM_REVISION = verify_upstream_revision(
    UNN_ROOT,
    expected_url="{UNN_URL}",
    expected_commit="{UNN_COMMIT}",
)
CUTS_ROOT = UNN_ROOT / "CUTS_Plus"
if str(CUTS_ROOT) not in sys.path:
    sys.path.insert(0, str(CUTS_ROOT))
from data.simu_data import load_dream_data

# The UNN source references these files but does not distribute them. Place the
# DREAM3 files under UNN_ROOT/causal_discov_sota/SRU_for_GCI/data/dream3/
# before executing this cell.
DREAM_DATA_ROOT = (
    UNN_ROOT / "causal_discov_sota" / "SRU_for_GCI" / "data" / "dream3"
)
if not DREAM_DATA_ROOT.is_dir():
    raise FileNotFoundError(
        "DREAM3 source files are absent. Populate the path documented above; "
        "the benchmark will not fabricate or download an unverified copy."
    )

DATASET_IDS = tuple(range(5))
FIT_ORDER = 1
MAX_NEW_DATASETS: int | None = None

created = 0
original_cwd = Path.cwd()
try:
    os.chdir(UNN_ROOT)
    for dataset_id_number in DATASET_IDS:
        if MAX_NEW_DATASETS is not None and created >= MAX_NEW_DATASETS:
            break
        data, target_source_truth = load_dream_data(dataset_id_number)
        dataset_id = f"dream3-size100-{{dataset_id_number}}"
        was_created = save_bundle(
            DATASET_DIR / f"{{dataset_id}}.npz",
            series=np.asarray(data, dtype=float).T,
            aggregate_truth=np.asarray(target_source_truth, dtype=bool).T,
            metadata={{
                "dataset_id": dataset_id,
                "dataset": DATASET_KEY,
                "loader": "UNN/CUTS_Plus/data/simu_data.py:load_dream_data",
                "upstream_commit": UPSTREAM_REVISION.commit,
                "dataset_index": dataset_id_number,
                "fit_order": FIT_ORDER,
                "series_layout": "variable, time",
                "truth_layout": "source, target",
                "truth_scope": "aggregate directed gene-regulatory graph",
            }},
        )
        created += int(was_created)
finally:
    os.chdir(original_cwd)
print(f"Created {{created}} new DREAM3 bundles. Rerun to resume.")
'''


def _prepare_notebook(dataset: DatasetSpec) -> dict[str, Any]:
    introduction = f'''
# Prepare shared {dataset.label} data

This notebook creates or verifies the frozen `{dataset.key}` bundles used by
all seven method notebooks in this directory. Data generation and acquisition
are kept separate from model fitting so every method reads byte-identical
observations and the same declared ground truth.
'''
    if dataset.key == "sparseVAR":
        cells = [
            _markdown(introduction),
            _code(PREPARE_IMPORTS),
            _code(SPARSE_PREPARE),
        ]
    elif dataset.key == "lorenz96":
        cells = [
            _markdown(introduction),
            _code(PREPARE_IMPORTS),
            _code(SHARED_EXTERNAL_PREPARE_HELPERS),
            _code(LORENZ_PREPARE),
        ]
    elif dataset.key == "dream3":
        cells = [
            _markdown(
                introduction
                + "\nThe CUTS+ repository references DREAM3 files that it does not "
                "distribute. This notebook therefore validates a user-supplied, "
                "licensed copy instead of downloading an unverified mirror."
            ),
            _code(PREPARE_IMPORTS),
            _code(SHARED_EXTERNAL_PREPARE_HELPERS),
            _code(DREAM_PREPARE),
        ]
    else:
        raise KeyError(dataset.key)
    return _notebook(cells)


SCORING_IMPORTS = r'''
from __future__ import annotations

import platform
import sys
from pathlib import Path

EXPECTED_PYTHON = (3, 14)
if sys.version_info[:2] != EXPECTED_PYTHON:
    raise RuntimeError(
        "This notebook requires CPython 3.14; "
        f"found {platform.python_version()}."
    )

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

from fastCGC.experiments.comparison_scoring import score_dataset_outputs
'''


def _scoring_notebook(dataset: DatasetSpec) -> dict[str, Any]:
    method_keys = ",\n    ".join(f'"{method.key}"' for method in METHODS)
    settings = f'''
DATASET_KEY = "{dataset.key}"
METHODS = (
    {method_keys},
)

# Collection is safe to rerun. It reads immutable bundles and completed native
# outputs, records absent/failed fits explicitly, and atomically replaces only
# the two derived score tables.
artifacts = score_dataset_outputs(
    PROJECT_ROOT,
    DATASET_KEY,
    methods=METHODS,
)
run_metrics = artifacts.run_metrics
summary = artifacts.summary

print(f"Run-level rows: {{len(run_metrics)}}")
print(run_metrics["status"].value_counts(dropna=False).to_string())
print(f"Run-level metrics: {{artifacts.run_metrics_path}}")
print(f"Mean (SD) summary: {{artifacts.summary_path}}")
'''
    preview = r'''
mean_std_columns = [
    "auprc_mean_std",
    "auroc_mean_std",
    "f1_mean_std",
    "mcc_mean_std",
    "normalized_shd_mean_std",
    "failure_aware_f1_mean_std",
]
identifier_columns = [
    "dataset",
    "method",
    "estimand",
    "n_variables",
    "n_samples",
    "fit_order",
    "noise",
    "expected",
    "missing",
    "attempted",
    "completed",
    "failed",
    "success_rate",
]
summary[identifier_columns + mean_std_columns]
'''
    introduction = f'''
# Collect and score {dataset.label} results

Run this notebook after any or all method notebooks in this directory. It
validates dataset and native-output hashes, converts each package's native
graph orientation in one shared tested module, and writes run-level and
grouped summary CSV files. Missing and failed fits remain visible.

The summary reports the mean and sample standard deviation across independent
dataset instances in each experimental cell. Exact-lag and aggregate-any-lag
recovery are separate estimands; aggregate-only methods are never assigned a
fabricated exact lag. AUROC and AUPRC are reported only when a method supplies
usable continuous scores and both edge classes are present.
'''
    return _notebook(
        [
            _markdown(introduction),
            _code(SCORING_IMPORTS),
            _code(settings),
            _code(preview),
        ]
    )


def synchronize() -> list[Path]:
    """Write all dataset-first notebooks deterministically."""

    updated: list[Path] = []
    for dataset in DATASETS:
        dataset_root = NOTEBOOK_ROOT / dataset.key
        dataset_root.mkdir(parents=True, exist_ok=True)
        notebooks = {
            "00-prepare-data.ipynb": _prepare_notebook(dataset),
            **{
                method.filename: _method_notebook(dataset, method)
                for method in METHODS
            },
            "90-collect-and-score.ipynb": _scoring_notebook(dataset),
        }
        for filename, notebook in notebooks.items():
            path = dataset_root / filename
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
