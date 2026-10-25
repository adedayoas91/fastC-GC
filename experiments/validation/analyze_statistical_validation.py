#!/usr/bin/env python3
"""Build a reproducible analysis bundle from validation checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "outputs" / "statistical-validation"


def _stable_seed(parts: Iterable[object]) -> int:
    encoded = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "big")


def _mean_interval(
    values: pd.Series,
    *,
    seed_parts: Iterable[object],
    bootstrap_samples: int,
) -> dict[str, float]:
    array = values.dropna().to_numpy(dtype=float)
    if array.size == 0:
        return {"n": 0.0, "mean": np.nan, "sd": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    if array.size == 1:
        return {
            "n": 1.0,
            "mean": float(array[0]),
            "sd": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }
    rng = np.random.default_rng(_stable_seed(seed_parts))
    indices = rng.integers(0, array.size, size=(bootstrap_samples, array.size))
    means = np.mean(array[indices], axis=1)
    return {
        "n": float(array.size),
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def _summarize(
    frame: pd.DataFrame,
    *,
    groups: list[str],
    metrics: list[str],
    bootstrap_samples: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(groups, dropna=False, sort=True):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        common = dict(zip(groups, key_tuple, strict=True))
        for metric in metrics:
            if metric not in group:
                continue
            summary = _mean_interval(
                group[metric],
                seed_parts=(*key_tuple, metric),
                bootstrap_samples=bootstrap_samples,
            )
            rows.append({**common, "metric": metric, **summary})
    return pd.DataFrame(rows)


def _format_estimate(row: pd.Series) -> str:
    if not np.isfinite(row["mean"]):
        return "NA"
    if np.isfinite(row["ci_low"]) and np.isfinite(row["ci_high"]):
        return f"{row['mean']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
    return f"{row['mean']:.4f}"


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "No completed rows were available."
    values = frame.loc[:, columns].astype(str)
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in values.to_numpy()]
    return "\n".join((header, rule, *body))


def _save_figure(figure: plt.Figure, root: Path, name: str) -> list[Path]:
    paths = [root / f"{name}.pdf", root / f"{name}.png"]
    figure.savefig(paths[0], bbox_inches="tight")
    figure.savefig(paths[1], dpi=300, bbox_inches="tight")
    plt.close(figure)
    return paths


def _calibration_figure(frame: pd.DataFrame, figure_dir: Path) -> list[Path]:
    subset = frame[frame["arm"] == "calibration"]
    if subset.empty:
        return []
    grouped = (
        subset.groupby(["method", "regime", "n_samples", "beta"], as_index=False)
        ["joint_false_positive_rate"]
        .mean()
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
    for axis, method in zip(axes, sorted(grouped["method"].unique()), strict=False):
        method_rows = grouped[grouped["method"] == method]
        for (regime, n_samples), regime_rows in method_rows.groupby(
            ["regime", "n_samples"]
        ):
            axis.plot(
                regime_rows["beta"],
                regime_rows["joint_false_positive_rate"],
                marker="o",
                label=f"{regime}, T={n_samples}",
            )
        axis.set_xscale("log")
        axis.set_title(method)
        axis.set_xlabel(r"Conditional threshold $\beta$")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Mean cross-edge false-positive rate")
    axes[-1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    return _save_figure(figure, figure_dir, "null-calibration")


def _lambda_figure(frame: pd.DataFrame, figure_dir: Path) -> list[Path]:
    subset = frame[(frame["arm"] == "lambda") & np.isclose(frame["beta"], 0.005)]
    if subset.empty:
        return []
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    panels = (
        ("stationary-ar-null", "joint_false_positive_rate", "Null false-positive rate"),
        ("sparse-var", "auprc", "Exact-lag AUPRC"),
    )
    for axis, (regime, metric, label) in zip(axes, panels, strict=True):
        regime_rows = subset[subset["regime"] == regime]
        if metric not in regime_rows:
            continue
        grouped = regime_rows.groupby(["method", "ridge"], as_index=False)[metric].mean()
        for method, method_rows in grouped.groupby("method"):
            axis.plot(
                method_rows["ridge"],
                method_rows[metric],
                marker="o",
                label=method,
            )
        axis.set_xscale("log")
        axis.set_xlabel(r"Ridge $\lambda$")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    return _save_figure(figure, figure_dir, "ridge-sensitivity")


def _lag_figure(frame: pd.DataFrame, figure_dir: Path) -> list[Path]:
    subset = frame[(frame["arm"] == "lag") & np.isclose(frame["beta"], 0.005)]
    if subset.empty:
        return []
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for axis, metric, label in (
        (axes[0], "auprc", "Exact-lag AUPRC"),
        (axes[1], "f1", "Exact-lag F1"),
    ):
        grouped = subset.groupby(["method", "fitted_order"], as_index=False)[metric].mean()
        for method, method_rows in grouped.groupby("method"):
            axis.plot(
                method_rows["fitted_order"],
                method_rows[metric],
                marker="o",
                label=method,
            )
        axis.set_xlabel("Fitted maximum lag")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    return _save_figure(figure, figure_dir, "maximum-lag-sensitivity")


def analyze(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir).resolve()
    metrics_path = input_dir / "validation-metrics.csv"
    protocol_path = input_dir / "validation-protocol.json"
    if not metrics_path.is_file() or not protocol_path.is_file():
        raise FileNotFoundError(
            "Run run_statistical_validation.py before building the analysis bundle."
        )
    frame = pd.read_csv(metrics_path)
    if frame.empty:
        raise ValueError("The validation metrics table is empty.")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir or input_dir / "analysis").resolve()
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    group_columns = [
        "arm",
        "method",
        "regime",
        "n_samples",
        "fitted_order",
        "ridge",
        "beta",
    ]
    candidate_metrics = [
        "joint_false_positive_rate",
        "graph_familywise_error",
        "conditional_rejection_rate",
        "f1",
        "mcc",
        "normalized_shd",
        "auroc",
        "auprc",
    ]
    summary = _summarize(
        frame,
        groups=group_columns,
        metrics=candidate_metrics,
        bootstrap_samples=args.bootstrap_samples,
    )
    summary.to_csv(output_dir / "validation-summary.csv", index=False)

    beta_rows = summary[np.isclose(summary["beta"], 0.005)].copy()
    beta_rows["estimate_95ci"] = beta_rows.apply(_format_estimate, axis=1)
    calibration = beta_rows[
        (beta_rows["arm"] == "calibration")
        & (beta_rows["metric"].isin(["joint_false_positive_rate", "graph_familywise_error"]))
    ]
    lambda_rows = beta_rows[
        (beta_rows["arm"] == "lambda")
        & (
            ((beta_rows["regime"] == "stationary-ar-null") & (beta_rows["metric"] == "joint_false_positive_rate"))
            | ((beta_rows["regime"] == "sparse-var") & (beta_rows["metric"] == "auprc"))
        )
    ]
    lag_rows = beta_rows[
        (beta_rows["arm"] == "lag")
        & (beta_rows["metric"].isin(["auprc", "f1", "normalized_shd"]))
    ]

    checkpoint_paths = sorted((input_dir / "checkpoints").glob("*.json"))
    failed = 0
    succeeded = 0
    for path in checkpoint_paths:
        status = json.loads(path.read_text(encoding="utf-8")).get("status")
        failed += int(status != "ok")
        succeeded += int(status == "ok")

    report = f"""# Statistical validation report

This bundle summarizes {succeeded} completed fits and {failed} recorded failures.
The run configuration is identified by `{protocol['configuration_hash']}`. The
tables report replicate means with seed-level percentile-bootstrap 95% confidence
intervals. Calibration, ridge sensitivity, and lag sensitivity remain separate
analyses; the same realization is reused across method settings with the same seed.

## Null calibration at beta = 0.005

{_markdown_table(calibration, ['method', 'regime', 'n_samples', 'metric', 'estimate_95ci'])}

## Ridge sensitivity at beta = 0.005

{_markdown_table(lambda_rows, ['method', 'regime', 'ridge', 'metric', 'estimate_95ci'])}

## Maximum-lag sensitivity at beta = 0.005

{_markdown_table(lag_rows, ['method', 'fitted_order', 'metric', 'estimate_95ci'])}

The sparse-VAR row for a fitted order below the true order scores only represented
lags and reports omitted true edges separately in the raw table. It must therefore
be read as an underfitting diagnostic, not as a complete-graph recovery estimate.
The drifting-null arm is an assumption-violation stress test, not an expected
type-I-error guarantee. A threshold should be frozen from calibration criteria
before the held-out benchmark datasets are rescored.
"""
    (output_dir / "analysis-report.md").write_text(report, encoding="utf-8")

    appendix = f"""# Statistical appendix

## Estimands

Null arms report the cross-variable per-edge false-positive rate and the
graph-level probability of at least one false edge. Sparse-VAR arms report
exact-lag recovery over the lags represented by the fitted model. AUPRC is the
primary ranking measure because the graph is sparse; AUROC is secondary.

## Uncertainty

Each interval is a percentile interval from {args.bootstrap_samples} resamples
of independent simulation seeds. No edge is treated as an independent
replicate. The analysis is descriptive and does not attach null-hypothesis
p-values to the sensitivity grid.

## Multiplicity and selection

The ridge and maximum-lag grids are sensitivity analyses. They do not justify
selecting a setting on the final benchmark outcomes. If beta = 0.005 is retained,
the decision must be based on the calibration arm and documented before the
existing benchmark suites are rescored.
"""
    (output_dir / "stats-appendix.md").write_text(appendix, encoding="utf-8")

    figures = [
        *_calibration_figure(frame, figure_dir),
        *_lambda_figure(frame, figure_dir),
        *_lag_figure(frame, figure_dir),
    ]
    catalog_lines = ["# Figure catalog", ""]
    descriptions = {
        "null-calibration": "Mean cross-edge null rejection across beta thresholds.",
        "ridge-sensitivity": "Null false positives and sparse-VAR AUPRC across ridge values at beta = 0.005.",
        "maximum-lag-sensitivity": "Sparse-VAR AUPRC and F1 across fitted maximum lags at beta = 0.005.",
    }
    for stem, description in descriptions.items():
        available = [path for path in figures if path.stem == stem]
        if available:
            catalog_lines.append(f"- `{stem}.pdf` / `{stem}.png`: {description}")
    if not figures:
        catalog_lines.append("No figures were produced because the required arms were incomplete.")
    (output_dir / "figure-catalog.md").write_text(
        "\n".join(catalog_lines) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote analysis bundle to {output_dir}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100.")
    return analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
