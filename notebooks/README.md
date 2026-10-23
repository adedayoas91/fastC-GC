# Notebooks

This directory contains the runnable experiment notebooks. Reusable estimator,
dataset, metric, and checkpoint code remains in `src/fastCGC/`; command-line
experiment drivers and notebook generators remain in `experiments/`.

## Dataset-first comparison layout

```text
simulations/
├── sparseVAR/
├── lorenz96/
├── autocorrelatedSCM/
└── dream3/
empirical/
├── causalChamber/
└── causalRivers/
```

Every directory contains the same preparation notebook and seven method
notebooks:

```text
00-prepare-data.ipynb
fastC-GC.ipynb
fastC-GC-star.ipynb
c-GC.ipynb
c-GC-star.ipynb
PCMCI-plus.ipynb
Bagged-PCMCI-plus.ipynb
CUTS-plus.ipynb
```

Simulation and in-silico directories end with
`90-collect-and-score.ipynb`. Empirical directories end with
`90-collect-and-visualize.ipynb`.

Run `00-prepare-data.ipynb` first. It creates or validates the immutable dataset
bundles used by every method notebook in that directory. The method notebooks
save native outputs before common graph conversion. Conventional
c-GC/c-GC*, PCMCI+, Bagged PCMCI+, and CUTS+ are loaded from pinned authors'
checkouts under the ignored `external/` directory; their estimators are not
copied into this repository.

For simulations and DREAM3, run `90-collect-and-score.ipynb` after the method
notebooks. It validates the
saved dataset and native-output hashes, records missing or failed fits, and
writes `run_metrics.csv` plus `summary_mean_std.csv` under
`data/outputs/causal-comparison/<dataset>/scores/`. Summary rows report the
mean and sample standard deviation across dataset instances in the same
experimental cell. Exact-lag and aggregate-any-lag results remain separate;
methods with aggregate-only native outputs are not assigned synthetic lags.

For Causal Chambers and CausalRivers, run
`90-collect-and-visualize.ipynb`. It exports every inferred cross-variable edge,
keeps missing and failed fits visible, and draws the methods' directed graphs
with a common node layout. These empirical case studies do not require
graph-recovery metrics. A supplied physical or constructed network is shown in
a separately labelled reference panel rather than assumed to be an exhaustive
lag-resolved causal truth.

The DREAM3 Size 100 networks are in-silico GeneNetWeaver benchmarks, not
real-world measurements. Their source files are not distributed by the CUTS+
repository. The data
preparation notebook therefore validates a user-supplied, licensed copy rather
than downloading data from an unverified mirror.

The comparison notebooks expect these verified checkouts by default:

- `external/causalized-GC` at `575ac1dc67a0282a5ca88bc11a2804df052f674d`;
- `external/tigramite` at `ff3ff13e1481073b8c5833a6fde1c304627a208e`;
- `external/UNN` at `2212724cce8a7752db564089a7537fecedb0e36d`;
- `external/debeire24clear_Bagged_TimeSeries_Causality` at
  `8e16fb89b8f3b4db47ff57f2bd09dcd5c9f7e4c9` for the autocorrelated-SCM
  generator used by Debeire et al. (2024);
- `external/causal-chamber-package` at
  `9fb5d82e391bb91a89a64f2e1c9b6ae8e7aed6d6` for the wind-tunnel reference
  network;
- `external/causalrivers` at
  `a30154fbb9a2e03bae133c643d7e978df63b3766` for the authors' connected-subgraph
  sampler.

Override their locations with `CAUSALIZED_GC_ROOT`, `TIGRAMITE_ROOT`, and
`UNN_ROOT`. Set `BAGGED_PCMCI_DATA_ROOT`,
`CAUSAL_CHAMBER_PACKAGE_ROOT`, or `CAUSAL_RIVERS_ROOT` for the other checkouts.
The preparation notebooks download the official measurement archives into the
ignored `external/datasets/` directory and verify their recorded SHA-256
digests. Existing downloads can be selected with `CAUSAL_CHAMBER_ARCHIVE` and
`CAUSAL_RIVERS_ARCHIVE`.

The CausalRivers preparation notebook imports `networkx` and the subgraph
sampler from the pinned CausalRivers checkout. Run that notebook in an
environment containing the official repository's data dependencies; `networkx`
is not added as a dependency of the fastC-GC estimator package.

The simulation tree is organized by data-generating process, not by dependence
profile. The preparation notebook describes the process and records its
parameters. Identity and curved link functions, when present, are analysis
factors within that process rather than separate directory classes.

The autocorrelated-SCM notebooks use the authors' generator from
[Debeire et al. (2024)](https://proceedings.mlr.press/v236/debeire24a.html).
The common comparison arm sets the generator's contemporaneous-link fraction
to zero so all seven methods are scored on the same positive-lag graph. This
adaptation is stated in the notebook and stored in each dataset's metadata. A
cross-link-free cell is included to measure false positives directly.

All notebooks use deterministic, hash-verified dataset bundles. Successful
fits are checkpointed, so a stopped run can continue without repeating
completed work. Set `MAX_NEW_FITS` to a positive integer for a bounded session.

Regenerate the committed notebooks after changing their shared templates:

```bash
uv run --extra dev python experiments/benchmarks/sync_comparison_notebooks.py
```

Notebook outputs are intentionally cleared before commit. Experimental data,
checkpoints, native estimator outputs, and figures are written under `data/`
rather than beside the notebooks.
