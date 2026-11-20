# Statistical validation workflows

The validation scripts save their results separately from the main benchmark
outputs.

- `run_statistical_validation.py` measures null calibration, ridge sensitivity,
  and maximum-lag sensitivity for fastC-GC and fastC-GC*.
- `analyze_statistical_validation.py` converts completed checkpoints into
  `analysis-report.md`, `stats-appendix.md`, a CSV summary, and a figure catalog.
- `run_rccp_runtime.py` measures end-to-end wall-clock time for fastC-GC versus
  c-GC, fastC-GC* versus c-GC*, or dependence-test variants within each fast method.

All runners use atomic per-fit checkpoints. With the project virtual
environment activated, each full workflow runs to completion with one command:

```bash
python experiments/validation/run_statistical_validation.py

python experiments/validation/analyze_statistical_validation.py

python experiments/validation/run_rccp_runtime.py
```

If a run is interrupted, rerun the same command; verified checkpoints are
reused. The `--max-new-fits` and `--max-new-cells` options remain available for
intentionally bounded sessions.

The component marginal and conditional p-values are archived separately. This
allows the declared beta grid to be scored from one full-evaluation fit. Do not
use short-circuit evaluation for these runs, because skipped conditional tests
cannot be recovered at a different threshold.

## Timing dependence-test variants

From the repository root, use the CPython 3.14 project environment to run

```bash
python experiments/validation/run_rccp_runtime.py \
  --dependence-variants linear,rff-hsic,rcot,rff-hsic-rcot
```

The variants use the same tests as the experiment notebooks. The linear
configuration uses Pearson correlation and a ridge residual test. `rff-hsic`
changes the marginal test, `rcot` changes the conditional test, and
`rff-hsic-rcot` changes both.

Each variant receives the same sparse-VAR inputs and repeat seeds within each
grid cell. The benchmark times full estimator fits and graph construction in
single-threaded workers, with one untimed warm-up and five timed repeats by
default.

The run saves checkpoints and `runtime-measurements.csv` under
`data/outputs/rccp-dependence-runtime/`. The accompanying `runtime-summary.csv`
reports the mean and sample standard deviation in seconds by method, variant,
and grid cell. `wall_time_ratio_vs_linear` divides each mean by the linear mean
for the same method and input. A value above 1 means the variant took longer.
Keep `linear` in the list to compute the ratios.

Use `--max-new-cells` to run the grid in batches and rerun with the same
settings to resume. Use a separate `--output-dir` when changing the variant
list, grid, or timing settings. The conventional method comparison uses
`data/outputs/rccp-runtime/`.
