"""Generate the dataset-first causal-comparison notebook matrix.

The notebooks are deliberately thin launch surfaces. Local study methods call
the installed ``fastCGC`` package. Comparison methods use pinned installed
distributions when upstream is packageable and verified source checkouts only
when no usable package exists. No external estimator is copied into this
repository.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks"

TIGRAMITE_VERSION = "5.2.10.1"
CAUSALTS_COMMIT = "0d1fae061b2445a51d0c8b75fc565d897dd54fb7"
CAUSALTS_VERSION = "0.27.0"
UNN_URL = "https://github.com/jarrycyx/UNN.git"
UNN_COMMIT = "2212724cce8a7752db564089a7537fecedb0e36d"
DREAM3_DATA_URL = (
    "https://raw.githubusercontent.com/sakhanna/SRU_for_GCI"
)
DREAM3_DATA_COMMIT = "7fe4b24880a947056d7753cf51f69d5e4f0ee5fb"
DREAM3_DATA_FILES = {
    "Dream3TensorData/Size100Ecoli1.pt": (
        "c312c1a4d2570bfb0fbeae075868de9547f50efd70c2903697009586c5b70c97"
    ),
    "Dream3TensorData/Size100Ecoli2.pt": (
        "77d7cf39a0e78396f73922f3345b1650d99cd1445173c4bb87a2c86ee41e04a5"
    ),
    "Dream3TensorData/Size100Yeast1.pt": (
        "94844418b42c262ca9f378e673f124ec982cd0b635260c200f56f886091b9d3b"
    ),
    "Dream3TensorData/Size100Yeast2.pt": (
        "29e81a6e98b6b190590eed911d7e13c8e68ada53eba668dca6cb5bec9973c9ec"
    ),
    "Dream3TensorData/Size100Yeast3.pt": (
        "ba509214bb86fa1d4a8160f95b0d384182aab38afed7545d47aaee09bc73d36b"
    ),
    "TrueGeneNetworks/InSilicoSize100-Ecoli1.tsv": (
        "a3ff157a8398f52a13dd26af8407817ec81c0ac8d4582ccb38534e384f68cd73"
    ),
    "TrueGeneNetworks/InSilicoSize100-Ecoli2.tsv": (
        "f5448add4389180a105e2e75f730278cf2d946b6948823091569f386fe97fde1"
    ),
    "TrueGeneNetworks/InSilicoSize100-Yeast1.tsv": (
        "00cebe38ad7a596c605dcc8db021f1b23e3ac309bc4f42775ef2490515212712"
    ),
    "TrueGeneNetworks/InSilicoSize100-Yeast2.tsv": (
        "7320839722d86d161522f9343d80513a8ee94dda38ffad823b4e668ae467ccae"
    ),
    "TrueGeneNetworks/InSilicoSize100-Yeast3.tsv": (
        "296e70fd0ee2aacb20b1079a8a070b0f633a421384b32f009011b75b0b75746d"
    ),
}
BAGGED_PCMCI_DATA_URL = (
    "https://github.com/EyringMLClimateGroup/"
    "debeire24clear_Bagged_TimeSeries_Causality.git"
)
BAGGED_PCMCI_DATA_COMMIT = "8e16fb89b8f3b4db47ff57f2bd09dcd5c9f7e4c9"
CAUSAL_CHAMBER_PACKAGE_URL = (
    "https://github.com/juangamella/causal-chamber-package.git"
)
CAUSAL_CHAMBER_VERSION = "0.2.8"
CAUSAL_CHAMBER_DATA_URL = (
    "https://causalchamber.s3.eu-central-1.amazonaws.com/"
    "downloadables/wt_walks_v1.zip"
)
CAUSAL_CHAMBER_DATA_SHA256 = (
    "1ef92d39fff7645a9e6e8cd41311e708362606491590a1534c6a5135d207736e"
)
CAUSAL_RIVERS_URL = "https://github.com/CausalRivers/causalrivers.git"
CAUSAL_RIVERS_COMMIT = "a30154fbb9a2e03bae133c643d7e978df63b3766"
CAUSAL_RIVERS_DATA_URL = (
    "https://github.com/CausalRivers/causalrivers/"
    "releases/download/First_release/product.zip"
)
CAUSAL_RIVERS_DATA_SHA256 = (
    "4f779fa9222653af092046eacba39a955cc65638bfb70fb7e31a32d9d8373b55"
)


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    key: str
    label: str
    truth: str
    notebook_subdir: tuple[str, ...]
    evaluation: str


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
        ("simulations", "sparseVAR"),
        "recovery",
    ),
    DatasetSpec(
        "lorenz96",
        "Lorenz-96 process",
        "aggregate structural-dependency graph",
        ("simulations", "lorenz96"),
        "recovery",
    ),
    DatasetSpec(
        "autocorrelatedSCM",
        "autocorrelated additive structural causal process",
        "exact positive-lag graph",
        ("simulations", "autocorrelatedSCM"),
        "recovery",
    ),
    DatasetSpec(
        "dream3",
        "DREAM3 in-silico gene-network",
        "aggregate directed gene-regulatory graph",
        ("simulations", "dream3"),
        "recovery",
    ),
    DatasetSpec(
        "causalChamber",
        "Causal Chambers wind-tunnel measurements",
        "physical reference network",
        ("empirical", "causalChamber"),
        "qualitative",
    ),
    DatasetSpec(
        "causalRivers",
        "CausalRivers discharge measurements",
        "river-network reference",
        ("empirical", "causalRivers"),
        "qualitative",
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
        "gc-extension from adedayoas91/causalized-GC (installed Git distribution)",
    ),
    MethodSpec(
        "cgc-star",
        "c-GC-star.ipynb",
        "conventional c-GC*",
        "gc-extension from adedayoas91/causalized-GC (installed Git distribution)",
    ),
    MethodSpec(
        "pcmci-plus",
        "PCMCI-plus.ipynb",
        "PCMCI+",
        f"tigramite=={TIGRAMITE_VERSION} (installed distribution)",
    ),
    MethodSpec(
        "bagged-pcmci-plus",
        "Bagged-PCMCI-plus.ipynb",
        "Bagged PCMCI+",
        f"tigramite=={TIGRAMITE_VERSION} (installed distribution)",
    ),
    MethodSpec(
        "cedar",
        "CEDAR.ipynb",
        "CEDAR",
        (
            f"causalts=={CAUSALTS_VERSION} from bloomberg/causal-ts "
            f"commit {CAUSALTS_COMMIT} (installed Git distribution)"
        ),
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
    cedar_configuration = ""
    if method.key == "cedar":
        cedar_configuration = '''    "alpha_cond1": ALPHA,
    "alpha_cond2": ALPHA,
    "lag_method": "partial_dcor",
    "lag_pvalue_method": "t_test",
    "lag_alpha": 0.05,
    "lag_n_permutations": 200,
    "normalize": "minmax",
    "prune": True,
    "include_lag0": False,
    "multi_lag": True,
    "multi_lag_keep": "first",
    "include_autoreg": True,
    "assume_ar1": True,
'''
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
{cedar_configuration}    **({{"native_output_schema": 2}} if METHOD_KEY.startswith("fast-") else {{}}),
}}
CONFIG_HASH = configuration_hash(RUN_CONFIGURATION)
print(f"Dataset directory: {{DATASET_DIR}}")
print(f"Output directory: {{OUTPUT_DIR}}")
'''


LOCAL_RUNNER = r'''
from fastCGC.experiments.causal_methods import fit_method

SOURCE_REVISION = "local-source:" + ":".join(
    file_sha256(path)
    for path in (
        PROJECT_ROOT / "src" / "fastCGC" / "core" / "fast_gc_star.py",
        PROJECT_ROOT / "src" / "fastCGC" / "experiments" / "causal_methods.py",
    )
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
    if result.lag_marginal_pvalues is not None:
        arrays["lag_marginal_pvalues"] = result.lag_marginal_pvalues
    if result.lag_conditional_pvalues is not None:
        arrays["lag_conditional_pvalues"] = result.lag_conditional_pvalues
    arrays["alpha"] = np.asarray(ALPHA)
    arrays["beta"] = np.asarray(BETA)
    arrays["ridge"] = np.asarray(result.metadata["ridge"])
    return arrays
'''


CONVENTIONAL_RUNNER = r'''
from fastCGC.experiments.upstream import load_installed_conventional_cgc

GcStar, UPSTREAM_REVISION = load_installed_conventional_cgc()
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
        "alpha": np.asarray(ALPHA),
        "beta": np.asarray(BETA),
    }
'''


PCMCI_RUNNER = f'''
from fastCGC.experiments.upstream import verify_installed_distribution

SOURCE_REVISION = verify_installed_distribution(
    "tigramite",
    expected_version="{TIGRAMITE_VERSION}",
)

from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI


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
from fastCGC.experiments.upstream import verify_installed_distribution

SOURCE_REVISION = verify_installed_distribution(
    "tigramite",
    expected_version="{TIGRAMITE_VERSION}",
)

from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI


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


CEDAR_RUNNER = f'''
import pandas as pd

from fastCGC.experiments.upstream import (
    CAUSALTS_VERSION,
    load_installed_cedar,
)

run_cedar, ParCorrGPU, UPSTREAM_REVISION = load_installed_cedar()
SOURCE_REVISION = UPSTREAM_REVISION.commit


def run_native(series, fit_order, algorithm_seed, run_dir):
    frame = pd.DataFrame(
        np.asarray(series.T, dtype=float),
        columns=[f"X{{index}}" for index in range(series.shape[0])],
    )
    ci_test = ParCorrGPU(
        frame.to_numpy(copy=False),
        device="cpu",
        seed=algorithm_seed,
    )
    result = run_cedar(
        df=frame,
        ci_test=ci_test,
        max_lag=fit_order,
        alpha_cond1=ALPHA,
        alpha_cond2=ALPHA,
        lag_method="partial_dcor",
        lag_pvalue_method="t_test",
        lag_alpha=0.05,
        n_permutations=200,
        normalize="minmax",
        filter_children=False,
        prune=True,
        sweep_all_lags=False,
        max_prune_path=4,
        include_lag0=False,
        multi_lag=True,
        multi_lag_keep="first",
        n_jobs=1,
        include_C=False,
        include_autoreg=True,
        assume_ar1=True,
        verbose=False,
    )
    return {{
        "cg_tig": np.asarray(result.cg_tig),
        "val_matrix": np.asarray(result.val_matrix),
        "pvalue_matrix": np.asarray(result.pvalue_matrix),
        "lag_importance": np.asarray(result.lag_importance),
        "lag_pvalues": np.asarray(result.lag_pvalues),
        "selected_lags": np.asarray(result.selected_lags),
        "var_names": np.asarray(result.var_names),
        "max_lag": np.asarray(result.max_lag),
        "alpha_cond1": np.asarray(result.alpha_cond1),
        "alpha_cond2": np.asarray(result.alpha_cond2),
        "n_tests": np.asarray(result.n_tests),
        "pruned": np.asarray(result.pruned),
        "causalts_version": np.asarray(CAUSALTS_VERSION),
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
    configured_groups = int(native_options.n_groups)
    native_options.n_groups = min(configured_groups, native_options.n_nodes)
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
    if method.key == "cedar":
        return CEDAR_RUNNER
    if method.key == "cuts-plus":
        return CUTS_RUNNER
    raise KeyError(method.key)


def _method_notebook(dataset: DatasetSpec, method: MethodSpec) -> dict[str, Any]:
    downstream = (
        "common graph-orientation conversion and recovery scoring"
        if dataset.evaluation == "recovery"
        else "common graph-orientation conversion and visualization"
    )
    evaluation_statement = (
        "This study records causal-structure recovery and does not benchmark runtime."
        if dataset.evaluation == "recovery"
        else (
            "This empirical case study compares the inferred directed graphs "
            "qualitatively; recovery metrics are not required."
        )
    )
    introduction = f'''
# {method.label} on {dataset.label}

This notebook is the `{dataset.key}` copy of the `{method.key}` experiment.
It reads the same frozen, hash-verified bundles as every other method in this
directory and writes one atomic checkpoint per completed dataset fit.

Source: **{method.source}**. Comparison methods use a pinned installed
distribution when upstream is packageable; repository-only methods call a
verified checkout. Their estimator code is neither copied nor reimplemented
here. Native arrays are saved before {downstream}. {evaluation_statement}
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
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import types
import urllib.request
import zipfile
import pickle
from pathlib import Path

EXPECTED_PYTHON = (3, 14)
if sys.version_info[:2] != EXPECTED_PYTHON:
    raise RuntimeError(
        "This notebook requires CPython 3.14; "
        f"found {platform.python_version()}."
    )

import numpy as np
import pandas as pd
import scipy.special

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


def save_bundle(
    path,
    *,
    series,
    metadata,
    aggregate_truth=None,
    lag_truth=None,
    reference_graph=None,
    extra_arrays=None,
):
    graph_arrays = (aggregate_truth, lag_truth, reference_graph)
    if sum(value is not None for value in graph_arrays) > 1:
        raise ValueError("Provide at most one truth or reference graph.")
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"series": np.asarray(series, dtype=float)}
    if lag_truth is not None:
        arrays["lag_truth"] = np.asarray(lag_truth, dtype=bool)
    elif aggregate_truth is not None:
        arrays["aggregate_truth"] = np.asarray(
            aggregate_truth,
            dtype=bool,
        )
    elif reference_graph is not None:
        arrays["reference_graph"] = np.asarray(
            reference_graph,
            dtype=bool,
        )
    for name, value in (extra_arrays or {}).items():
        if name in arrays:
            raise ValueError(f"Duplicate bundle array name: {name}.")
        arrays[name] = np.asarray(value)
    digest = series_digest(arrays["series"])
    metadata = {**metadata, "data_digest": digest}
    metadata_path = path.with_suffix(".json")
    if path.exists() or metadata_path.exists():
        if not path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Incomplete dataset bundle at {path}.")
        recorded = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(path, allow_pickle=False) as stored:
            saved = {name: np.asarray(stored[name]) for name in stored.files}
        arrays_match = set(saved) == set(arrays) and all(
            np.array_equal(saved[name], value)
            for name, value in arrays.items()
        )
        if not arrays_match or recorded != metadata:
            raise ValueError(f"Existing bundle does not match request: {path}.")
        return False

    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        np.savez_compressed(handle, **arrays)
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


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(url, path, expected_sha256):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        observed = file_digest(path)
        if observed != expected_sha256:
            raise ValueError(
                f"File digest mismatch for {path}: {observed}."
            )
        return False

    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        with urllib.request.urlopen(url, timeout=120) as response:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
    try:
        observed = file_digest(temporary)
        if observed != expected_sha256:
            raise ValueError(
                f"Downloaded file digest mismatch: {observed}."
            )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
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
                    "regime": "forcing10-dt0.1-sd0.1",
                    "noise": "gaussian-sd0.1",
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


AUTOCORRELATED_PREPARE = f'''
DATASET_KEY = "autocorrelatedSCM"
DATASET_DIR = (
    PROJECT_ROOT / "data" / "shared" / "causal-comparison"
    / DATASET_KEY / "datasets"
)
GENERATOR_ROOT = Path(
    os.environ.get(
        "BAGGED_PCMCI_DATA_ROOT",
        PROJECT_ROOT
        / "external"
        / "debeire24clear_Bagged_TimeSeries_Causality",
    )
).expanduser()
UPSTREAM_REVISION = verify_upstream_revision(
    GENERATOR_ROOT,
    expected_url="{BAGGED_PCMCI_DATA_URL}",
    expected_commit="{BAGGED_PCMCI_DATA_COMMIT}",
)
generator_path = GENERATOR_ROOT / "generate_data_mod.py"
spec = importlib.util.spec_from_file_location(
    "debeire24_generate_data_mod",
    generator_path,
)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load the author generator at {{generator_path}}.")
author_generator = importlib.util.module_from_spec(spec)
temporary_networkx_module = False
if importlib.util.find_spec("networkx") is None:
    generator_source = generator_path.read_text(encoding="utf-8")
    if "nx." in generator_source:
        raise ModuleNotFoundError(
            "The pinned generator now uses networkx; install the author "
            "repository's dependencies before running this notebook."
        )
    # The pinned file imports networkx but never references its `nx` alias.
    # Supplying an empty module avoids adding an unused project dependency
    # while leaving every executed generator definition unchanged.
    sys.modules["networkx"] = types.ModuleType("networkx")
    temporary_networkx_module = True
try:
    spec.loader.exec_module(author_generator)
finally:
    if temporary_networkx_module:
        sys.modules.pop("networkx", None)


def identity_link(x):
    return x


def quadratic_taper_link(x):
    return x + 5.0 * x**2 * np.exp(-(x**2) / 20.0)


def make_noises(random_state, choices, n_variables):
    functions = []
    assigned = []
    scales = []
    weibull_shape = 2.0
    weibull_mean = scipy.special.gamma(1.0 / weibull_shape + 1.0)
    weibull_variance = (
        scipy.special.gamma(2.0 / weibull_shape + 1.0)
        - weibull_mean**2
    )
    for _ in range(n_variables):
        noise_type = str(random_state.choice(choices))
        scale = float(random_state.uniform(0.5, 2.0))
        if noise_type == "gaussian":
            def draw(size, scale=scale):
                return scale * random_state.randn(size)
        elif noise_type == "weibull":
            def draw(size, scale=scale):
                values = random_state.weibull(weibull_shape, size=size)
                return scale * (values - weibull_mean) / np.sqrt(
                    weibull_variance
                )
        else:
            raise ValueError(f"Unknown noise type: {{noise_type}}.")
        functions.append(draw)
        assigned.append(noise_type)
        scales.append(scale)
    return functions, assigned, scales


def lag_arrays(links, *, n_variables, fit_order):
    truth = np.zeros((fit_order, n_variables, n_variables), dtype=bool)
    coefficients = np.zeros((fit_order, n_variables, n_variables), dtype=float)
    function_codes = np.full(
        (fit_order, n_variables, n_variables),
        -1,
        dtype=np.int8,
    )
    for target, parents in links.items():
        for (source, signed_lag), coefficient, function in parents:
            if signed_lag >= 0:
                raise ValueError("The matched comparison arm must contain only lags.")
            lag_index = abs(int(signed_lag)) - 1
            truth[lag_index, source, target] = True
            coefficients[lag_index, source, target] = float(coefficient)
            function_codes[lag_index, source, target] = (
                0 if function.__name__ in {{"lin", "identity_link"}} else 1
            )
    return truth, coefficients, function_codes


# One-factor-at-a-time design around the paper-matched reference cell. Link
# functions and innovation families are described directly rather than being
# split into separate linear/nonlinear experiment trees.
REGIMES = (
    {{"regime": "reference", "n_variables": 10, "n_samples": 500,
      "max_auto": 0.60, "functions": ("identity",),
      "noise_types": ("gaussian",)}},
    {{"regime": "variables-5", "n_variables": 5, "n_samples": 500,
      "max_auto": 0.60, "functions": ("identity",),
      "noise_types": ("gaussian",)}},
    {{"regime": "variables-20", "n_variables": 20, "n_samples": 500,
      "max_auto": 0.60, "functions": ("identity",),
      "noise_types": ("gaussian",)}},
    {{"regime": "samples-250", "n_variables": 10, "n_samples": 250,
      "max_auto": 0.60, "functions": ("identity",),
      "noise_types": ("gaussian",)}},
    {{"regime": "samples-1000", "n_variables": 10, "n_samples": 1000,
      "max_auto": 0.60, "functions": ("identity",),
      "noise_types": ("gaussian",)}},
    {{"regime": "autocorrelation-0.20", "n_variables": 10,
      "n_samples": 500, "max_auto": 0.20,
      "functions": ("identity",), "noise_types": ("gaussian",)}},
    {{"regime": "autocorrelation-0.95", "n_variables": 10,
      "n_samples": 500, "max_auto": 0.95,
      "functions": ("identity",), "noise_types": ("gaussian",)}},
    {{"regime": "mixed-link-functions", "n_variables": 10,
      "n_samples": 500, "max_auto": 0.60,
      "functions": ("identity", "quadratic-taper"),
      "noise_types": ("gaussian",)}},
    {{"regime": "mixed-innovations", "n_variables": 10,
      "n_samples": 500, "max_auto": 0.60,
      "functions": ("identity",),
      "noise_types": ("gaussian", "weibull")}},
    {{"regime": "cross-link-null", "n_variables": 10,
      "n_samples": 500, "max_auto": 0.60,
      "functions": ("identity",), "noise_types": ("gaussian",),
      "cross_link_factor": 0.0}},
)
FIT_ORDER = 5
REPLICATES = 50
SEED_START = 30_000
MAX_STATIONARITY_ATTEMPTS = 1000
MAX_NEW_DATASETS: int | None = None

created = 0
stop_requested = False
for regime_index, regime in enumerate(REGIMES):
    n_variables = int(regime["n_variables"])
    n_samples = int(regime["n_samples"])
    n_cross_links = math.floor(
        float(regime.get("cross_link_factor", 1.5)) * n_variables
    )
    functions = [
        identity_link if name == "identity" else quadratic_taper_link
        for name in regime["functions"]
    ]
    positive_couplings = np.arange(0.1, 0.51, 0.1).tolist()
    couplings = positive_couplings + [-value for value in positive_couplings]
    auto_coefficients = np.arange(
        max(0.0, float(regime["max_auto"]) - 0.3),
        float(regime["max_auto"]) + 0.01,
        0.05,
    ).tolist()

    for replicate in range(REPLICATES):
        if MAX_NEW_DATASETS is not None and created >= MAX_NEW_DATASETS:
            stop_requested = True
            break
        requested_seed = SEED_START + regime_index * REPLICATES + replicate
        model_seed = requested_seed
        nonstationary = True
        for retry in range(MAX_STATIONARITY_ATTEMPTS):
            random_state = np.random.RandomState(model_seed)
            links = author_generator.generate_random_contemp_model(
                N=n_variables,
                L=n_cross_links,
                coupling_coeffs=couplings,
                coupling_funcs=functions,
                auto_coeffs=auto_coefficients,
                tau_max=FIT_ORDER,
                contemp_fraction=0.0,
                random_state=random_state,
            )
            noises, assigned_noises, noise_scales = make_noises(
                random_state,
                regime["noise_types"],
                n_variables,
            )
            generated, nonstationary = (
                author_generator.generate_nonlinear_contemp_timeseries(
                    links=links,
                    T=n_samples + 10_000,
                    noises=noises,
                    random_state=random_state,
                )
            )
            if not nonstationary:
                series_time_variable = np.asarray(generated[:n_samples], dtype=float)
                break
            model_seed += 10_000
        if nonstationary:
            raise RuntimeError(
                f"No stationary process found for {{regime['regime']}} "
                f"replicate {{replicate}} after {{MAX_STATIONARITY_ATTEMPTS}} attempts."
            )

        lag_truth, lag_coefficients, lag_function_codes = lag_arrays(
            links,
            n_variables=n_variables,
            fit_order=FIT_ORDER,
        )
        dataset_id = (
            f"acscm-{{regime['regime']}}-seed{{requested_seed}}"
        )
        was_created = save_bundle(
            DATASET_DIR / f"{{dataset_id}}.npz",
            series=series_time_variable.T,
            lag_truth=lag_truth,
            extra_arrays={{
                "lag_coefficients": lag_coefficients,
                "lag_function_codes": lag_function_codes,
            }},
            metadata={{
                "dataset_id": dataset_id,
                "dataset": DATASET_KEY,
                "generator": "generate_data_mod.py",
                "generator_source": "{BAGGED_PCMCI_DATA_URL}",
                "upstream_commit": UPSTREAM_REVISION.commit,
                "regime": regime["regime"],
                "seed": requested_seed,
                "accepted_model_seed": model_seed,
                "stationarity_retries": retry,
                "n_variables": n_variables,
                "n_samples": n_samples,
                "fit_order": FIT_ORDER,
                "cross_link_count": n_cross_links,
                "cross_link_factor": regime.get("cross_link_factor", 1.5),
                "maximum_autoregressive_coefficient": regime["max_auto"],
                "available_coupling_coefficients": couplings,
                "available_link_functions": list(regime["functions"]),
                "available_noise_types": list(regime["noise_types"]),
                "noise": "+".join(regime["noise_types"]),
                "assigned_noise_types": assigned_noises,
                "assigned_noise_scales": noise_scales,
                "source_paper_contemporaneous_fraction": 0.30,
                "used_contemporaneous_fraction": 0.0,
                "adaptation": (
                    "Contemporaneous links are disabled so every method is "
                    "evaluated on the common positive-lag estimand."
                ),
                "series_layout": "variable, time",
                "truth_layout": "lag, source, target",
                "coefficient_layout": "lag, source, target",
                "function_code_map": {{"0": "identity", "1": "quadratic-taper"}},
                "truth_scope": "exact positive-lag graph",
            }},
        )
        created += int(was_created)
    if stop_requested:
        break
print(
    f"Created {{created}} new autocorrelated-SCM bundles. "
    "Rerun to verify existing bundles and resume."
)
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

# CUTS+ delegates this loader to the SRU_for_GCI layout, but the pinned UNN
# checkout does not contain those files. Fetch the exact preprocessed tensors
# and reference networks used by that loader from the original authors' pinned
# repository revision, verifying every file before use.
DREAM_DATA_ROOT = (
    UNN_ROOT / "causal_discov_sota" / "SRU_for_GCI" / "data" / "dream3"
)
DREAM3_FILES = {DREAM3_DATA_FILES!r}
DREAM3_RAW_ROOT = (
    "{DREAM3_DATA_URL}/{DREAM3_DATA_COMMIT}/data/dream3"
)
for relative_path, expected_sha256 in DREAM3_FILES.items():
    download_verified(
        f"{{DREAM3_RAW_ROOT}}/{{relative_path}}",
        DREAM_DATA_ROOT / relative_path,
        expected_sha256,
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
                "data_source": (
                    "https://github.com/sakhanna/SRU_for_GCI/tree/"
                    "{DREAM3_DATA_COMMIT}/data/dream3"
                ),
                "data_source_commit": "{DREAM3_DATA_COMMIT}",
                "dataset_index": dataset_id_number,
                "fit_order": FIT_ORDER,
                "regime": "dream3-size100",
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


CAUSAL_CHAMBER_PREPARE = f'''
from fastCGC.experiments.upstream import verify_installed_distribution

DATASET_KEY = "causalChamber"
DATASET_DIR = (
    PROJECT_ROOT / "data" / "shared" / "causal-comparison"
    / DATASET_KEY / "datasets"
)
PACKAGE_REVISION = verify_installed_distribution(
    "causalchamber",
    expected_version="{CAUSAL_CHAMBER_VERSION}",
)
from causalchamber.ground_truth import graph
ARCHIVE_PATH = Path(
    os.environ.get(
        "CAUSAL_CHAMBER_ARCHIVE",
        PROJECT_ROOT / "external" / "datasets" / "wt_walks_v1.zip",
    )
).expanduser()
download_verified(
    "{CAUSAL_CHAMBER_DATA_URL}",
    ARCHIVE_PATH,
    "{CAUSAL_CHAMBER_DATA_SHA256}",
)

VARIABLES = (
    "hatch", "load_in", "load_out", "pot_1", "pot_2",
    "current_in", "current_out", "pressure_downwind",
    "pressure_upwind", "rpm_in", "rpm_out", "mic",
    "pressure_intake", "pressure_ambient", "signal_1", "signal_2",
)
EXPERIMENTS = ("actuators_random_walk_1",)
FIT_ORDER = 10
MAX_NEW_DATASETS: int | None = None

reference_frame = graph("wt", "standard")
reference_graph = reference_frame.loc[list(VARIABLES), list(VARIABLES)].to_numpy(
    dtype=bool
)

created = 0
with zipfile.ZipFile(ARCHIVE_PATH) as archive:
    for experiment in EXPERIMENTS:
        if MAX_NEW_DATASETS is not None and created >= MAX_NEW_DATASETS:
            break
        member = f"wt_walks_v1/{{experiment}}.csv"
        if member not in archive.namelist():
            raise FileNotFoundError(f"{{member}} is absent from {{ARCHIVE_PATH}}.")
        with archive.open(member) as handle:
            frame = pd.read_csv(handle, usecols=list(VARIABLES))
        minima = frame.min(axis=0)
        spans = frame.max(axis=0) - minima
        if (spans <= 0).any():
            constant = spans.index[spans <= 0].tolist()
            raise ValueError(f"Constant wind-tunnel variables: {{constant}}.")
        normalized = (frame - minima) / spans
        series = normalized.to_numpy(dtype=float).T
        dataset_id = f"causal-chamber-wt-{{experiment.replace('_', '-')}}"
        was_created = save_bundle(
            DATASET_DIR / f"{{dataset_id}}.npz",
            series=series,
            reference_graph=reference_graph,
            metadata={{
                "dataset_id": dataset_id,
                "dataset": DATASET_KEY,
                "data_source": "{CAUSAL_CHAMBER_DATA_URL}",
                "data_archive_sha256": "{CAUSAL_CHAMBER_DATA_SHA256}",
                "data_license": "CC BY 4.0",
                "ground_truth_package_source": "{CAUSAL_CHAMBER_PACKAGE_URL}",
                "upstream_version": PACKAGE_REVISION,
                "experiment": experiment,
                "fit_order": FIT_ORDER,
                "regime": "wind-tunnel-actuator-random-walk",
                "variable_names": list(VARIABLES),
                "n_variables": int(series.shape[0]),
                "n_samples": int(series.shape[1]),
                "preprocessing": "per-variable min-max normalization",
                "series_layout": "variable, time",
                "reference_layout": "source, target",
                "reference_scope": (
                    "physical causal model supplied by the Causal Chambers "
                    "authors; used as a contextual graph, not as a mandatory "
                    "lag-resolved scoring target"
                ),
                "evaluation": "qualitative inferred-graph comparison",
            }},
        )
        created += int(was_created)
print(
    f"Created {{created}} new Causal Chambers bundles. "
    "Rerun to verify existing bundles and resume."
)
'''


CAUSAL_RIVERS_PREPARE = f'''
DATASET_KEY = "causalRivers"
DATASET_DIR = (
    PROJECT_ROOT / "data" / "shared" / "causal-comparison"
    / DATASET_KEY / "datasets"
)
UPSTREAM_ROOT = Path(
    os.environ.get(
        "CAUSAL_RIVERS_ROOT",
        PROJECT_ROOT / "external" / "causalrivers",
    )
).expanduser()
UPSTREAM_REVISION = verify_upstream_revision(
    UPSTREAM_ROOT,
    expected_url="{CAUSAL_RIVERS_URL}",
    expected_commit="{CAUSAL_RIVERS_COMMIT}",
)
if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))
try:
    import networkx as nx
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "CausalRivers preparation requires networkx from the official "
        "CausalRivers environment. Install that environment before running "
        "this notebook; networkx is not a fastC-GC estimator dependency."
    ) from error
from tools.graph_sampling_tools import get_all_subgraphs

ARCHIVE_PATH = Path(
    os.environ.get(
        "CAUSAL_RIVERS_ARCHIVE",
        PROJECT_ROOT / "external" / "datasets" / "causalrivers-product.zip",
    )
).expanduser()
download_verified(
    "{CAUSAL_RIVERS_DATA_URL}",
    ARCHIVE_PATH,
    "{CAUSAL_RIVERS_DATA_SHA256}",
)

REGIONS = {{
    "east-germany": (
        "product/rivers_east_germany.p",
        "product/rivers_ts_east_germany.csv",
    ),
    "bavaria": (
        "product/rivers_bavaria.p",
        "product/rivers_ts_bavaria.csv",
    ),
}}
NODES_PER_SUBGRAPH = 5
SUBGRAPHS_PER_REGION = 5
FIT_ORDER = 5
RESOLUTION = "6h"
MAX_NEW_DATASETS: int | None = None


def deterministic_subgraphs(graph):
    candidates = sorted(
        tuple(sorted(int(node) for node in candidate))
        for candidate in get_all_subgraphs(graph, n_vars=NODES_PER_SUBGRAPH)
    )
    if len(candidates) < SUBGRAPHS_PER_REGION:
        raise ValueError(
            f"Only {{len(candidates)}} candidate subgraphs were found."
        )
    indices = np.linspace(
        0,
        len(candidates) - 1,
        SUBGRAPHS_PER_REGION,
        dtype=int,
    )
    return candidates, [(int(index), candidates[int(index)]) for index in indices]


created = 0
stop_requested = False
with zipfile.ZipFile(ARCHIVE_PATH) as archive:
    for region, (graph_member, series_member) in REGIONS.items():
        # The archive hash is checked before loading this official NetworkX pickle.
        graph = pickle.loads(archive.read(graph_member))
        candidates, selections = deterministic_subgraphs(graph)
        required_nodes = sorted({{
            node for _, nodes in selections for node in nodes
        }})
        with archive.open(series_member) as handle:
            raw = pd.read_csv(
                handle,
                index_col="datetime",
                usecols=["datetime", *map(str, required_nodes)],
            )
        raw["dt"] = pd.to_datetime(raw.index).round(RESOLUTION).values
        processed = raw.groupby("dt").mean()
        minima = processed.min(axis=0)
        spans = processed.max(axis=0) - minima
        if spans.isna().any() or (spans <= 0).any():
            invalid = spans.index[spans.isna() | (spans <= 0)].tolist()
            raise ValueError(f"Unusable river stations in {{region}}: {{invalid}}.")
        processed = (processed - minima) / spans
        processed = processed.interpolate()

        for selection_rank, (candidate_index, nodes) in enumerate(selections):
            if MAX_NEW_DATASETS is not None and created >= MAX_NEW_DATASETS:
                stop_requested = True
                break
            names = [str(node) for node in nodes]
            sample = processed[names]
            complete_rows = np.flatnonzero(~sample.isna().any(axis=1).to_numpy())
            if not len(complete_rows):
                raise ValueError(
                    f"No complete observations for {{region}} nodes {{nodes}}."
                )
            sample = sample.iloc[complete_rows[0] : complete_rows[-1] + 1]
            if sample.isna().any().any():
                raise ValueError(
                    f"Interior missing values remain for {{region}} nodes {{nodes}}."
                )

            node_index = {{node: index for index, node in enumerate(nodes)}}
            reference_graph = np.zeros(
                (NODES_PER_SUBGRAPH, NODES_PER_SUBGRAPH),
                dtype=bool,
            )
            for source, target in graph.subgraph(nodes).edges:
                reference_graph[node_index[source], node_index[target]] = True
            positions = np.asarray(
                [graph.nodes[node].get("p", (np.nan, np.nan)) for node in nodes],
                dtype=float,
            )
            dataset_id = (
                f"causal-rivers-{{region}}-n{{NODES_PER_SUBGRAPH}}-"
                f"sample{{selection_rank + 1:02d}}"
            )
            was_created = save_bundle(
                DATASET_DIR / f"{{dataset_id}}.npz",
                series=sample.to_numpy(dtype=float).T,
                reference_graph=reference_graph,
                extra_arrays={{"node_positions": positions}},
                metadata={{
                    "dataset_id": dataset_id,
                    "dataset": DATASET_KEY,
                    "data_source": "{CAUSAL_RIVERS_DATA_URL}",
                    "data_archive_sha256": "{CAUSAL_RIVERS_DATA_SHA256}",
                    "data_license": "Data licence Germany (dl-de)",
                    "sampling_source": "{CAUSAL_RIVERS_URL}",
                    "upstream_commit": UPSTREAM_REVISION.commit,
                    "region": region,
                    "regime": region,
                    "fit_order": FIT_ORDER,
                    "resolution": RESOLUTION,
                    "variable_names": names,
                    "node_ids": list(nodes),
                    "candidate_index": candidate_index,
                    "candidate_count": len(candidates),
                    "selection_rank": selection_rank,
                    "sampling_strategy": (
                        "deterministic evenly spaced selection from the sorted "
                        "output of the authors' get_all_subgraphs function"
                    ),
                    "preprocessing": (
                        "6-hour rounded means, per-station min-max normalization, "
                        "linear interpolation, and removal of incomplete end rows"
                    ),
                    "n_variables": NODES_PER_SUBGRAPH,
                    "n_samples": int(sample.shape[0]),
                    "series_layout": "variable, time",
                    "reference_layout": "source, target",
                    "reference_scope": (
                        "constructed river-connectivity network supplied with "
                        "CausalRivers; used as a contextual graph rather than a "
                        "mandatory exhaustive causal scoring target"
                    ),
                    "evaluation": "qualitative inferred-graph comparison",
                }},
            )
            created += int(was_created)
        if stop_requested:
            break
print(
    f"Created {{created}} new CausalRivers bundles. "
    "Rerun to verify existing bundles and resume."
)
'''


def _prepare_notebook(dataset: DatasetSpec) -> dict[str, Any]:
    shared_graph = (
        "the same declared ground truth"
        if dataset.evaluation == "recovery"
        else "the same optional contextual reference network"
    )
    introduction = f'''
# Prepare shared {dataset.label} data

This notebook creates or verifies the frozen `{dataset.key}` bundles used by
all eight method notebooks in this directory. Data generation and acquisition
are kept separate from model fitting so every method reads byte-identical
observations and {shared_graph}.
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
    elif dataset.key == "autocorrelatedSCM":
        cells = [
            _markdown(
                introduction
                + r'''

The process follows the simulation family used by Debeire et al. (2024):

\[
X_t^j = a_j X_{t-1}^j
  + \sum_i c_i f_i(X_{t-\tau_i}^i) + \eta_t^j.
\]

The identity link is \(f(x)=x\). The second link option is
\(f(x)=x+5x^2\exp(-x^2/20)\). Cross-link coefficients are sampled from
\(\{\pm0.1,\ldots,\pm0.5\}\), lags range from 1 to 5, innovation scales
range from 0.5 to 2, and each accepted process passes the authors'
stationarity check. The study varies variable count, sample count,
autocorrelation, link-function set, and innovation family one factor at a
time around a reference cell, with 50 seeds per cell. A cross-link-free cell
measures false positives when the cross-variable graph is empty.

The paper's broad simulation includes 30% contemporaneous cross-links. This
matched comparison arm sets that fraction to zero because all eight methods
can be scored on a shared positive-lag graph, whereas they do not expose a
common contemporaneous estimand. This is a declared adaptation, not a claim of
literal reproduction. The notebook calls the official generator from its
pinned authors' checkout and records every accepted seed and rejected-model
count.
'''
            ),
            _code(PREPARE_IMPORTS),
            _code(SHARED_EXTERNAL_PREPARE_HELPERS),
            _code(AUTOCORRELATED_PREPARE),
        ]
    elif dataset.key == "dream3":
        cells = [
            _markdown(
                introduction
                + "\nDREAM3 Size 100 is an in-silico GeneNetWeaver benchmark, "
                "not a real-world measurement dataset. The CUTS+ repository "
                "references DREAM3 files that it does not "
                "distribute. This notebook therefore validates a user-supplied, "
                "licensed copy instead of downloading an unverified mirror."
            ),
            _code(PREPARE_IMPORTS),
            _code(SHARED_EXTERNAL_PREPARE_HELPERS),
            _code(DREAM_PREPARE),
        ]
    elif dataset.key == "causalChamber":
        cells = [
            _markdown(
                introduction
                + "\nThis case study uses measured wind-tunnel time series from "
                "the Causal Chambers project. The supplied physical causal "
                "model is retained as a separately labelled reference panel; "
                "the empirical comparison does not require recovery metrics."
            ),
            _code(PREPARE_IMPORTS),
            _code(SHARED_EXTERNAL_PREPARE_HELPERS),
            _code(CAUSAL_CHAMBER_PREPARE),
        ]
    elif dataset.key == "causalRivers":
        cells = [
            _markdown(
                introduction
                + "\nThis case study uses measured river-discharge time series. "
                "Five connected five-station subgraphs are selected "
                "deterministically from each of the eastern-Germany and Bavaria "
                "networks using the authors' sampling routine. The constructed "
                "river network is visual context, not a required exhaustive "
                "causal scoring target."
            ),
            _code(PREPARE_IMPORTS),
            _code(SHARED_EXTERNAL_PREPARE_HELPERS),
            _code(CAUSAL_RIVERS_PREPARE),
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

# Leave both values as None to reproduce the archived decisions. After the
# independent calibration arm is complete, set both values (for example,
# DECISION_ALPHA = 0.01 and DECISION_BETA = 0.005) to write a separate
# rethresholded score directory. New fast-method archives must contain both
# component p-value arrays.
DECISION_ALPHA: float | None = None
DECISION_BETA: float | None = None

# Collection is safe to rerun. It reads immutable bundles and completed native
# outputs, records absent/failed fits explicitly, and atomically replaces only
# the two derived score tables.
artifacts = score_dataset_outputs(
    PROJECT_ROOT,
    DATASET_KEY,
    methods=METHODS,
    decision_alpha=DECISION_ALPHA,
    decision_beta=DECISION_BETA,
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
    "regime",
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


EMPIRICAL_VISUALIZATION_IMPORTS = r'''
from __future__ import annotations

import json
import math
import platform
import sys
from pathlib import Path

EXPECTED_PYTHON = (3, 14)
if sys.version_info[:2] != EXPECTED_PYTHON:
    raise RuntimeError(
        "This notebook requires CPython 3.14; "
        f"found {platform.python_version()}."
    )

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
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

from fastCGC.experiments.comparison_scoring import collect_dataset_graphs
'''


def _visualization_notebook(dataset: DatasetSpec) -> dict[str, Any]:
    method_keys = ",\n    ".join(f'"{method.key}"' for method in METHODS)
    method_labels = ",\n    ".join(
        f'"{method.key}": "{method.label}"' for method in METHODS
    )
    settings = f'''
DATASET_KEY = "{dataset.key}"
METHODS = (
    {method_keys},
)
METHOD_LABELS = {{
    {method_labels},
}}

# Collection is safe to rerun. No recovery metric is computed here.
artifacts = collect_dataset_graphs(
    PROJECT_ROOT,
    DATASET_KEY,
    methods=METHODS,
)
edge_table = artifacts.edge_table
status_table = artifacts.status_table
reference_edges = artifacts.reference_edges

print(status_table["status"].value_counts(dropna=False).to_string())
print(f"Inferred edge table: {{artifacts.edge_table_path}}")
print(f"Run-status table: {{artifacts.status_table_path}}")
print(f"Reference edge table: {{artifacts.reference_edges_path}}")
'''
    plotting = r'''
# Choose one measured system for the figure. Set an explicit identifier to
# reproduce a different bundle; the edge tables retain every completed bundle.
PLOT_DATASET_ID: str | None = None

dataset_ids = sorted(status_table["dataset_id"].unique())
if not dataset_ids:
    raise ValueError("No empirical dataset bundles were collected.")
plot_dataset_id = PLOT_DATASET_ID or dataset_ids[0]
if plot_dataset_id not in dataset_ids:
    raise ValueError(f"Unknown PLOT_DATASET_ID: {plot_dataset_id}.")

bundle_path = (
    PROJECT_ROOT
    / "data"
    / "shared"
    / "causal-comparison"
    / DATASET_KEY
    / "datasets"
    / f"{plot_dataset_id}.npz"
)
metadata = json.loads(bundle_path.with_suffix(".json").read_text(encoding="utf-8"))
names = [str(name) for name in metadata["variable_names"]]
with np.load(bundle_path, allow_pickle=False) as stored:
    stored_positions = (
        np.asarray(stored["node_positions"], dtype=float)
        if "node_positions" in stored.files
        else None
    )


def graph_positions(node_names, positions=None):
    if positions is not None and positions.shape == (len(node_names), 2):
        if np.isfinite(positions).all() and np.ptp(positions, axis=0).max() > 0:
            centered = positions - positions.mean(axis=0, keepdims=True)
            scale = np.max(np.linalg.norm(centered, axis=1))
            if scale > 0:
                return centered / scale
    angles = np.linspace(0.0, 2.0 * np.pi, len(node_names), endpoint=False)
    return np.column_stack((np.cos(angles), np.sin(angles)))


def plot_directed_graph(ax, node_names, edges, title, positions=None):
    coordinates = graph_positions(node_names, positions)
    index = {name: position for name, position in zip(node_names, coordinates)}
    directed_pairs = set(zip(edges.get("source", []), edges.get("target", [])))
    for edge in edges.itertuples(index=False):
        start = index[str(edge.source)]
        end = index[str(edge.target)]
        reverse_exists = (str(edge.target), str(edge.source)) in directed_pairs
        score = getattr(edge, "score", np.nan)
        strength = float(score) if np.isfinite(score) else 0.65
        strength = float(np.clip(abs(strength), 0.15, 1.0))
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.7 + 1.3 * strength,
            color="#2f5597",
            alpha=0.35 + 0.6 * strength,
            connectionstyle=f"arc3,rad={0.13 if reverse_exists else 0.0}",
            shrinkA=14,
            shrinkB=14,
            zorder=1,
        )
        ax.add_patch(arrow)
    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=120,
        facecolor="#f4b183",
        edgecolor="#7f6000",
        linewidth=0.8,
        zorder=2,
    )
    for name, (x_value, y_value) in zip(node_names, coordinates):
        horizontal_alignment = (
            "left" if x_value > 0.08 else "right" if x_value < -0.08 else "center"
        )
        vertical_alignment = "bottom" if y_value >= 0 else "top"
        ax.text(
            x_value * 1.08,
            y_value * 1.08,
            name,
            ha=horizontal_alignment,
            va=vertical_alignment,
            fontsize=6,
            clip_on=False,
        )
    if edges.empty:
        ax.text(
            0.5,
            0.03,
            "No selected cross-variable edges",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=7,
            color="#666666",
        )
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.axis("off")


panels = []
reference_for_plot = reference_edges[
    reference_edges["dataset_id"] == plot_dataset_id
]
if not reference_for_plot.empty:
    panels.append(("Reference network (context only)", reference_for_plot))
for method in METHODS:
    method_edges = edge_table[
        (edge_table["dataset_id"] == plot_dataset_id)
        & (edge_table["method"] == method)
    ]
    panels.append((METHOD_LABELS[method], method_edges))

n_columns = 3
n_rows = math.ceil(len(panels) / n_columns)
figure, axes = plt.subplots(
    n_rows,
    n_columns,
    figsize=(5.4 * n_columns, 5.0 * n_rows),
    squeeze=False,
)
for axis, (title, edges) in zip(axes.flat, panels):
    plot_directed_graph(axis, names, edges, title, stored_positions)
for axis in axes.flat[len(panels) :]:
    axis.axis("off")
figure.suptitle(f"{DATASET_KEY}: {plot_dataset_id}", fontsize=12)
figure.tight_layout()

figure_dir = (
    PROJECT_ROOT
    / "data"
    / "outputs"
    / "causal-comparison"
    / DATASET_KEY
    / "graphs"
)
figure_dir.mkdir(parents=True, exist_ok=True)
pdf_path = figure_dir / f"{plot_dataset_id}-inferred-graphs.pdf"
png_path = figure_dir / f"{plot_dataset_id}-inferred-graphs.png"
figure.savefig(pdf_path, bbox_inches="tight")
figure.savefig(png_path, dpi=300, bbox_inches="tight")
plt.show()
print(pdf_path)
print(png_path)
'''
    introduction = f'''
# Collect and visualize {dataset.label} results

Run this notebook after any or all method notebooks in this directory. It
validates dataset and native-output hashes, converts every completed output to
the shared source-to-target orientation, and exports the inferred cross-variable
edges. Missing and failed fits remain visible in a separate status table.

This is an empirical case study and does not require graph-recovery scores.
The main output is a set of directly comparable directed
graph panels. If the dataset supplies a physical or constructed reference
network, that network appears in a separately labelled contextual panel and is
not assumed to be an exhaustive lag-resolved causal truth.
'''
    return _notebook(
        [
            _markdown(introduction),
            _code(EMPIRICAL_VISUALIZATION_IMPORTS),
            _code(settings),
            _code(plotting),
        ]
    )


def synchronize() -> list[Path]:
    """Write all dataset-first notebooks deterministically."""

    updated: list[Path] = []
    for dataset in DATASETS:
        dataset_root = NOTEBOOK_ROOT.joinpath(*dataset.notebook_subdir)
        dataset_root.mkdir(parents=True, exist_ok=True)
        terminal_notebook = (
            {"90-collect-and-score.ipynb": _scoring_notebook(dataset)}
            if dataset.evaluation == "recovery"
            else {
                "90-collect-and-visualize.ipynb": _visualization_notebook(dataset)
            }
        )
        notebooks = {
            "00-prepare-data.ipynb": _prepare_notebook(dataset),
            **{
                method.filename: _method_notebook(dataset, method)
                for method in METHODS
            },
            **terminal_notebook,
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
