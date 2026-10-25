# fastCGC

`FastGcStar` implements fastC-GC and fastC-GC*. The `fastC-` prefix identifies
these vectorized algorithms; it does not identify a dependence model. They use
the RCCP decision rule and the c-GC/c-GC* conditioning sets described by
Adedayo (2026), but standardization, ridge residualization, and complete-shift
calibration make them distinct estimators. The software supports linear,
kernel, randomized-kernel, and information-theoretic dependence tests.
The `fastC-` names are reserved for this portable NumPy/SciPy implementation.
Separate NVIDIA/cuML implementations retain the conventional c-GC and c-GC*
names and are outside this repository.

The project targets CPython 3.14. Comparison software is installed in the
active virtual environment when an official package is available. Source
checkouts are used only for repository-only methods and generators; they are
not committed here.

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

Install the comparison dependencies into `fastC-GC/.venv` with:

```bash
uv sync --extra dev --extra notebooks --extra comparison --extra conventional-cgc
```

The `notebooks` extra installs the complete IPython kernel environment,
including Matplotlib's Jupyter backend and IPython's traceback dependencies.
The remaining extras install the pinned `tigramite==5.2.10.1` distribution used
by PCMCI+ and Bagged PCMCI+, plus `causalchamber==0.2.8` for the Causal Chambers
reference graph. They also install conventional c-GC/c-GC* directly from commit
`575ac1dc67a0282a5ca88bc11a2804df052f674d` of the authors' Git repository and
supply the runtime dependencies omitted from that repository's package
metadata, including `torch==2.14.1`, which CDT imports but does not declare.
The notebooks verify package versions and Git provenance from the active kernel
environment. The conventional c-GC extra targets Apple Silicon, where Numba
and PyTorch publish CPython 3.14 wheels.

Register the synchronized environment once, then select this kernel in Jupyter
or VS Code and restart it after dependency changes:

```bash
.venv/bin/python -m ipykernel install --user --name fastcgc-py314-arm64 \
  --display-name "fastcgc (Python 3.14, Apple Silicon)"
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

Comparison algorithms are not implemented in this repository. PCMCI+ and
Bagged PCMCI+ use the pinned official Tigramite distribution installed in the
notebook environment. Conventional c-GC/c-GC* use the pinned Git distribution
installed in the same environment. Repository-only methods use their authors'
unchanged source at a pinned revision.

The remaining repository-only method and dataset helpers require these pinned
author checkouts:

```bash
git clone https://github.com/jarrycyx/UNN.git external/UNN
git -C external/UNN checkout 2212724cce8a7752db564089a7537fecedb0e36d

git clone \
  https://github.com/EyringMLClimateGroup/debeire24clear_Bagged_TimeSeries_Causality.git \
  external/debeire24clear_Bagged_TimeSeries_Causality
git -C external/debeire24clear_Bagged_TimeSeries_Causality checkout \
  8e16fb89b8f3b4db47ff57f2bd09dcd5c9f7e4c9

git clone https://github.com/CausalRivers/causalrivers.git external/causalrivers
git -C external/causalrivers checkout \
  a30154fbb9a2e03bae133c643d7e978df63b3766
```

Set `UNN_ROOT`, `BAGGED_PCMCI_DATA_ROOT`, or `CAUSAL_RIVERS_ROOT` when one of
these checkouts is stored elsewhere. Each repository-backed notebook verifies
the remote URL and exact commit before importing a native entry point, data
generator, or sampler.

The DREAM3 preparation notebook downloads the exact Size 100 tensors and
network files from commit `7fe4b24880a947056d7753cf51f69d5e4f0ee5fb` of the
original SRU_for_GCI authors' repository and verifies every SHA-256 digest; no
separate DREAM3 checkout is required. For CUTS+ datasets with fewer than the
32 groups in its upstream Lorenz configuration, the notebook sets the initial
group count to the number of variables. This preserves the upstream estimator
while satisfying its requirement that the initial group count not exceed the
dataset width.

The conventional notebook loader verifies the installed distribution's PEP 610
Git URL and commit before importing `GcStar`. The upstream estimator remains
unchanged.

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
fastC-GC* in isolated subprocesses. Run each comparison notebook against those
bundles and preserve its native outputs before converting graph orientation for
shared scoring. The study compares graph-recovery efficacy; it does not collect
elapsed time, memory use, or hardware rankings. The exact upstream software
policy is in the comparison protocol.

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

After all six collectors have completed, validate and collate their archived
outputs with:

```bash
uv run --extra dev --extra notebooks \
  python experiments/analyze_causal_comparison.py
```

The analysis writes publication tables, paired AUPRC contrasts,
score-agreement summaries, and figures under
`data/outputs/causal-comparison/analysis/`. It does not refit or rescore an
estimator. The report marks the archived fast-method thresholded metrics and
reconstructs Bagged PCMCI+ directed-edge frequency from its modal-mark summary
after validating the positive-lag graph alphabet. It also recovers the
lag-specific c-GC/c-GC* decisions from their archived p-value matrices and
checks that their logical OR reproduces every saved aggregate graph. The
calibration limitation of the archived fast-method thresholded metrics is documented in the
manuscript. A separately labelled thresholded comparison should first validate
`alpha=0.01, beta=0.005` on independent null or development data, then rerun
fastC-GC and fastC-GC* to archive their separate marginal and conditional
p-values. The existing c-GC and c-GC* archives already contain the two
components and can be rescored without refitting. The source-faithful
`beta=0.001` arm remains available for score-level comparison.

The separate validation workflow covers null calibration, ridge sensitivity,
and fitted-lag sensitivity without tuning on the benchmark outcomes. It saves
the marginal and conditional p-values independently, so the same full fit can
be scored at each declared beta value:

```bash
uv run --python 3.14 python \
  experiments/validation/run_statistical_validation.py --max-new-fits 25

uv run --python 3.14 python \
  experiments/validation/analyze_statistical_validation.py
```

Rerun the first command to continue from the next unfinished fit. The analysis
writes a report, statistical appendix, CSV summary, and figure catalog under
`data/outputs/statistical-validation/analysis/`. A one-seed smoke run verifies
the workflow but is not evidence for the manuscript; the configured study uses
50 independent seeds.

A separate wall-clock script is limited to the two direct method pairs rather
than timing unrelated CSL algorithms:

```bash
uv run --python 3.14 python \
  experiments/validation/run_rccp_runtime.py --max-new-cells 1
```

This timing workflow requires the `conventional-cgc` extra and is intended for
the Apple Silicon analysis host. Timing values are not included in the paper
until every declared cell has completed on one machine.

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
