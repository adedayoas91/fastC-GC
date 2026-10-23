# fastCGC

`FastGcStar` implements fastC-GC and fastC-GC*. The `fastC-` prefix identifies
this vectorized implementation; it does not identify a dependence model or
change the conventional c-GC/c-GC* decision rule. The estimator supports linear,
kernel, randomized-kernel, and information-theoretic dependence tests while
keeping the c-GC/c-GC* conditioning rules unchanged.
The `fastC-` names are reserved for this portable NumPy/SciPy implementation.
Separate NVIDIA/cuML implementations retain the conventional c-GC and c-GC*
names and are outside this repository.

The project targets CPython 3.14. External comparison methods use isolated
environments supplied by their own repositories; those checkouts and
environments are not committed here.

## Package layout

The installable package uses the standard `src` layout:

```text
src/fastCGC/
├── core/          # GcStar and FastGcStar estimators
├── dependence/    # linear, kernel, and information-theoretic backends
├── experiments/   # reusable dataset, scoring, and checkpoint support
├── simulation.py  # simulation helpers
└── __init__.py    # stable public imports
```

The public `from fastCGC import GcStar, FastGcStar` interface is unchanged.
The former `fastCGC.fast_causalised_GC` module remains as a compatibility
forwarder; new code should import estimator internals from `fastCGC.core`.

Cross-method experiment notebooks use a dataset-first layout under
`notebooks/`:

```text
notebooks/
├── simulations/
│   ├── sparseVAR/
│   ├── lorenz96/
│   ├── autocorrelatedSCM/
│   └── dream3/
└── empirical/
    ├── causalChamber/
    └── causalRivers/
```

Each dataset directory contains one shared-data preparation notebook and one
notebook for each of fastC-GC, fastC-GC*, conventional c-GC/c-GC*, PCMCI+,
Bagged PCMCI+, and CUTS+. Simulation directories are named for their
data-generating processes; link functions and noise families are recorded as
experimental factors instead of being split into linear and nonlinear trees.
See `notebooks/README.md` for the full matrix, dataset-sharing, regeneration,
and pause/resume conventions.

```python
from fastCGC import FastGcStar

model = FastGcStar(
    n_pasts=2,
    method="cgc",  # use "fcgc" for c-GC*
    dependence_profile="randomized_kernel",
    random_state=7,
)
model.fit(data, alpha=0.01, beta=0.001)
graph = model.get_connectivity_matrix()
```

Available profiles are:

| Profile | Marginal test | Conditional test |
| --- | --- | --- |
| `linear` | Pearson | ridge-residual partial correlation |
| `kernel` | exact HSIC | guarded KCIT |
| `randomized_kernel_rcit` | RFF-HSIC | RCIT |
| `randomized_kernel` | RFF-HSIC | RCoT |
| `information` | kNN mutual information | kNN conditional mutual information |

The marginal and conditional tests can also be selected independently:

```python
model = FastGcStar(
    n_pasts=2,
    method="fcgc",
    dependence_profile="linear",
    marginal_test="rff_hsic",
    conditional_test="cmi_knn",
    evaluation_policy="full",
    backend_options={
        "rff_hsic": {"n_features": 128},
        "cmi_knn": {
            "n_neighbors": 5,
            "n_permutations": 64,
            "local_permutation_neighbors": 5,
        },
    },
)
```

`evaluation_policy="auto"` evaluates every pair for the linear profile and
short-circuits nonlinear conditional tests when the marginal test fails.
Use `"full"` when every unconditional and conditional test must be evaluated.
Exact HSIC and KCIT enforce sample and memory guards; randomized-kernel modes
are the practical default for larger datasets.

For the linear profile, `screen_alpha` is a legacy compatibility parameter and
does not select conditional candidates. The estimator exposes the analytic
precision-derived quantities through `analytic_score_` and
`analytic_pvalue_`; `screen_score_` and `screen_p_` remain compatibility
aliases. Set `evaluation_policy="short_circuit"` to skip conditional tests for
pairs that fail the fitted marginal threshold; this policy preserves the final
conjunction at those fitted thresholds.

Run the checks with:

```bash
uv sync --extra dev
uv run --extra dev python -m compileall src/fastCGC experiments tests
uv run --extra dev pytest
uv run python experiments/simulations/nonlinear/validate_nonlinear_backends.py
uv run python experiments/simulations/nonlinear/benchmark_dependence_backends.py
```

## Cross-method causal-structure comparison

The prespecified method list, graph convention, controlled recovery metrics,
empirical graph-visualization protocol, statistical comparisons, and failure
policy are documented in
`experiments/benchmarks/CAUSAL_COMPARISON_PROTOCOL.md`.

The recent-literature shortlist uses PCMCI+, Bagged PCMCI+, and CUTS+ as the
headline external CSL comparators. CUTS+ is evaluated on aggregate any-lag
graphs in a separate frozen PyTorch environment. The 2026 Stationarity-Aware
Discovery method is recorded as pending because its advertised official source
repository was empty when checked; DYNOTEARS is historical-only rather than a
headline baseline.

Comparison algorithms are not implemented in this repository. Each is run from
a pinned checkout of its authors' repository, using that repository's own
environment and native entry point.

The conventional c-GC and c-GC* notebooks use the authors' repository directly:

```bash
git clone https://github.com/adedayoas91/causalized-GC.git external/causalized-GC
git -C external/causalized-GC checkout 575ac1dc67a0282a5ca88bc11a2804df052f674d
```

The local `external/` directory is ignored by Git. If the checkout lives
elsewhere, set `CAUSALIZED_GC_ROOT` before starting a notebook. Run the
conventional implementation in an isolated environment created from its own
repository, and record that environment with the resulting experiment:

```bash
export CAUSALIZED_GC_ROOT=/absolute/path/to/causalized-GC
git -C "$CAUSALIZED_GC_ROOT" rev-parse HEAD
```

The PCMCI+ family and CUTS+ notebooks use similarly pinned author checkouts:

```bash
git clone https://github.com/jakobrunge/tigramite.git external/tigramite
git -C external/tigramite checkout ff3ff13e1481073b8c5833a6fde1c304627a208e

git clone https://github.com/jarrycyx/UNN.git external/UNN
git -C external/UNN checkout 2212724cce8a7752db564089a7537fecedb0e36d

git clone \
  https://github.com/EyringMLClimateGroup/debeire24clear_Bagged_TimeSeries_Causality.git \
  external/debeire24clear_Bagged_TimeSeries_Causality
git -C external/debeire24clear_Bagged_TimeSeries_Causality checkout \
  8e16fb89b8f3b4db47ff57f2bd09dcd5c9f7e4c9

git clone \
  https://github.com/juangamella/causal-chamber-package.git \
  external/causal-chamber-package
git -C external/causal-chamber-package checkout \
  9fb5d82e391bb91a89a64f2e1c9b6ae8e7aed6d6

git clone https://github.com/CausalRivers/causalrivers.git external/causalrivers
git -C external/causalrivers checkout \
  a30154fbb9a2e03bae133c643d7e978df63b3766
```

Set `TIGRAMITE_ROOT`, `UNN_ROOT`, `BAGGED_PCMCI_DATA_ROOT`,
`CAUSAL_CHAMBER_PACKAGE_ROOT`, or `CAUSAL_RIVERS_ROOT` when a checkout is stored
elsewhere. Each notebook verifies the remote URL and exact commit before
importing a native entry point, data generator, reference graph, or sampler.

The notebook loader verifies both the remote URL and the recorded commit before
importing `GcStar`. The upstream estimator remains unchanged.

List all selected methods and their execution tracks:

```bash
uv run --extra dev python experiments/benchmarks/benchmark_causal_methods.py --list-methods
```

Generate the common datasets and run only the two study methods locally:

```bash
uv run --extra dev python experiments/benchmarks/benchmark_causal_methods.py \
  --replicates 50
```

The runner writes method-neutral `.npz` bundles under
`data/outputs/causal-comparison/datasets/` and evaluates fastC-GC and
fastC-GC* in isolated subprocesses. Run each cloned comparison repository
against those bundles and preserve its native outputs before converting graph
orientation for shared scoring. The study compares graph-recovery efficacy; it
does not collect elapsed time, memory use, or hardware rankings. The exact
repository policy is in the comparison protocol.

For the dataset-first notebooks, `00-prepare-data.ipynb` writes the frozen
bundles under `data/shared/causal-comparison/<dataset>/datasets/`. Each method
notebook reads those exact files and checkpoints one completed dataset fit at a
time. For simulation and in-silico datasets,
`90-collect-and-score.ipynb` creates hash-verified run-level metrics and mean
(sample SD) summaries with explicit missing/failure counts. For the two
empirical datasets, `90-collect-and-visualize.ipynb` exports edge lists and
directed graph panels without requiring recovery metrics. Regenerate the
notebook matrix with:

```bash
uv run --extra dev python experiments/benchmarks/sync_comparison_notebooks.py
```

Every simulation DGP has the same seven method notebooks. Each completed
dataset fit has an atomic checkpoint and an unmodified native-output archive.
Set `MAX_NEW_FITS` in a method notebook for a bounded session, or interrupt the
current fit and rerun the cell; completed matching checkpoints are skipped. The
upstream estimators have no common mid-fit checkpoint interface, so an
interrupted fit itself must restart.

DREAM3 is grouped with the simulations because its Size 100 networks and time
series are generated in silico. The two measured-data case studies are Causal
Chambers wind-tunnel recordings and CausalRivers discharge recordings. Their
published physical or constructed networks are retained as contextual graph
panels. The empirical collector does not treat those references as mandatory,
exhaustive lag-resolved truth and can also visualize datasets that provide no
reference graph.

The notebooks set and record an algorithm seed, but the upstream estimator uses
NumPy randomness inside concurrent Numba code and exposes no random-state
argument. Input datasets are therefore guaranteed identical by hash; estimator
repeatability must be checked empirically before it is claimed.

The autocorrelated-SCM study calls the official generator used by Debeire et
al. (2024) from its pinned repository. Its reference cell and one-factor stress
cells vary sample count, variable count, autocorrelation, link functions, and
innovation families; a cross-link-free cell measures false positives. The
matched comparison disables contemporaneous links and records this departure
from the source study, because the seven methods do not share a common
contemporaneous graph output.
