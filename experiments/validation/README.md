# Statistical validation workflows

The validation scripts answer four bounded questions without modifying the
main benchmark outputs:

- `run_statistical_validation.py` measures null calibration, ridge sensitivity,
  and maximum-lag sensitivity for fastC-GC and fastC-GC*.
- `analyze_statistical_validation.py` converts completed checkpoints into
  `analysis-report.md`, `stats-appendix.md`, a CSV summary, and a figure catalog.
- `run_rccp_runtime.py` is a narrow wall-clock comparison of fastC-GC versus
  c-GC and fastC-GC* versus c-GC*. It does not time the unrelated CSL baselines.

All runners use atomic per-fit checkpoints. Start with a bounded batch and
rerun the same command to resume:

```bash
uv run --python 3.14 python experiments/validation/run_statistical_validation.py \
  --max-new-fits 25

uv run --python 3.14 python experiments/validation/analyze_statistical_validation.py

uv run --python 3.14 python experiments/validation/run_rccp_runtime.py \
  --max-new-cells 1
```

The component marginal and conditional p-values are archived separately. This
allows the declared beta grid to be scored from one full-evaluation fit. Do not
use short-circuit evaluation for these runs, because skipped conditional tests
cannot be recovered at a different threshold.
