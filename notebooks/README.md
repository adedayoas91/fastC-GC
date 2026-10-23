# Notebooks

This directory contains the runnable experiment notebooks. Reusable estimator,
dataset, metric, and checkpoint code remains in `src/fastCGC/`; command-line
experiment drivers and notebook generators remain in `experiments/`.

## Dataset-first comparison layout

```text
sparseVAR/
lorenz96/
dream3/
```

Each dataset directory contains the same notebook matrix:

```text
00-prepare-data.ipynb
fastC-GC.ipynb
fastC-GC-star.ipynb
c-GC.ipynb
c-GC-star.ipynb
PCMCI-plus.ipynb
Bagged-PCMCI-plus.ipynb
CUTS-plus.ipynb
90-collect-and-score.ipynb
```

Run `00-prepare-data.ipynb` first. It creates or validates the immutable dataset
bundles used by every method notebook in that directory. The method notebooks
save native outputs before common graph conversion and scoring. Conventional
c-GC/c-GC*, PCMCI+, Bagged PCMCI+, and CUTS+ are loaded from pinned authors'
checkouts under the ignored `external/` directory; their estimators are not
copied into this repository.

Run `90-collect-and-score.ipynb` after the method notebooks. It validates the
saved dataset and native-output hashes, records missing or failed fits, and
writes `run_metrics.csv` plus `summary_mean_std.csv` under
`data/outputs/causal-comparison/<dataset>/scores/`. Summary rows report the
mean and sample standard deviation across dataset instances in the same
experimental cell. Exact-lag and aggregate-any-lag results remain separate;
methods with aggregate-only native outputs are not assigned synthetic lags.

The DREAM3 source files are not distributed by the CUTS+ repository. Its data
preparation notebook therefore validates a user-supplied, licensed copy rather
than downloading data from an unverified mirror.

The comparison notebooks expect these verified checkouts by default:

- `external/causalized-GC` at `575ac1dc67a0282a5ca88bc11a2804df052f674d`;
- `external/tigramite` at `ff3ff13e1481073b8c5833a6fde1c304627a208e`;
- `external/UNN` at `2212724cce8a7752db564089a7537fecedb0e36d`.

Override their locations with `CAUSALIZED_GC_ROOT`, `TIGRAMITE_ROOT`, and
`UNN_ROOT`, respectively. The checkouts themselves remain ignored by Git.

The earlier scenario notebooks remain under `simulations/`. They cover the
original single-lag/variable-lag and dependence-profile studies and are not the
cross-method comparison matrix.

All notebooks use deterministic, hash-verified dataset bundles. Successful
fits are checkpointed, so a stopped run can continue without repeating
completed work. Set `MAX_NEW_FITS` to a positive integer for a bounded session.

Regenerate the committed notebooks after changing their shared templates:

```bash
uv run --extra dev python experiments/benchmarks/sync_comparison_notebooks.py
uv run --extra dev python experiments/benchmarks/sync_conventional_cgc_notebooks.py
uv run --extra dev python experiments/benchmarks/sync_nonlinear_notebooks.py
```

Notebook outputs are intentionally cleared before commit. Experimental data,
checkpoints, native estimator outputs, and figures are written under `data/`
rather than beside the notebooks.
