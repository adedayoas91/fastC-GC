# fastCGC

`FastGcStar` implements fastC-GC and fastC-GC*. The `fastC-` prefix identifies
this vectorized implementation; it does not identify a dependence model or
change the published c-GC/c-GC* decision rule. The estimator supports linear,
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

Interactive experiment notebooks live separately under `notebooks/`:

```text
notebooks/simulations/
├── linear/
│   ├── c-GC/
│   └── c-GC-star/
└── nonlinear/
    ├── fastC-GC/
    └── fastC-GC-star/
```

See `notebooks/README.md` for the dataset-sharing, regeneration, and
pause/resume conventions.

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

The prespecified method list, graph convention, data regimes, AUROC/AUPRC and
structural metrics, statistical comparisons, and failure policy are documented
in `experiments/benchmarks/CAUSAL_COMPARISON_PROTOCOL.md`.

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
published implementation in an isolated environment created from its own
repository, and record that environment with the resulting experiment:

```bash
export CAUSALIZED_GC_ROOT=/absolute/path/to/causalized-GC
git -C "$CAUSALIZED_GC_ROOT" rev-parse HEAD
```

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

The eight conventional c-GC/c-GC* simulation notebooks under
`notebooks/simulations/linear/` follow the same rule on their scenario
suite. They create each dataset once under
`data/shared/causal-comparison-notebooks/linear-large/`, record the effective
data seed and SHA-256 digest, and reuse that exact bundle across both methods.
Each completed `(scenario, trial, n_past)` fit has an atomic checkpoint and an unmodified
native-output archive. Set `MAX_NEW_FITS` in a notebook for a bounded session,
or interrupt the current fit and rerun the cell; completed matching checkpoints
are skipped. The upstream estimator has no mid-fit checkpoint interface, so an
interrupted fit itself must restart. Python and package versions are part of the
software fingerprint. Moving a run between computers does not invalidate a
checkpoint when the dataset, method settings, source revision, and software
fingerprint still match.

The notebooks set and record an algorithm seed, but the upstream estimator uses
NumPy randomness inside concurrent Numba code and exposes no random-state
argument. Input datasets are therefore guaranteed identical by hash; estimator
repeatability must be checked empirically before it is claimed.

The two nonlinear notebooks are explicitly labeled as local fastC-GC and
fastC-GC* dependence-profile experiments, not as executions of conventional
c-GC. They use the same bundle and checkpoint contract in the separate
`nonlinear-small` suite so their smaller data regime cannot collide with the
linear comparison datasets.
