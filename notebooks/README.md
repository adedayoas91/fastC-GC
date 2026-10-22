# Notebooks

This directory contains the runnable experiment notebooks. Reusable estimator,
dataset, metric, and checkpoint code remains in `src/fastCGC/`; command-line
experiment drivers and notebook generators remain in `experiments/`.

## Layout

```text
simulations/
├── linear/
│   ├── c-GC/
│   └── c-GC-star/
└── nonlinear/
    ├── fastC-GC/
    └── fastC-GC-star/
```

The eight linear notebooks execute conventional c-GC or c-GC* from the pinned
authors' checkout under `external/causalized-GC`. The two nonlinear notebooks
evaluate the local fastC-GC and fastC-GC* dependence profiles and are reported
as a separate experimental track.

All notebooks use deterministic, hash-verified dataset bundles. Successful
fits are checkpointed, so a stopped run can continue without repeating
completed work. Set `MAX_NEW_FITS` to a positive integer for a bounded session.

Regenerate the committed notebooks after changing their shared templates:

```bash
uv run --extra dev python experiments/benchmarks/sync_published_cgc_notebooks.py
uv run --extra dev python experiments/benchmarks/sync_nonlinear_notebooks.py
```

Notebook outputs are intentionally cleared before commit. Experimental data,
checkpoints, native estimator outputs, and figures are written under `data/`
rather than beside the notebooks.
