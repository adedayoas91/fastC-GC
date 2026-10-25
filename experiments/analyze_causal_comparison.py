"""Collate the completed causal-comparison experiments.

The method notebooks and dataset-level collectors remain the source of truth
for native outputs and graph-recovery metrics.  This script validates those
derived artifacts, records which metrics are interpretable, performs the
declared paired AUPRC contrasts, and creates publication-facing tables and
figures.  It deliberately does not re-fit or re-score any estimator.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "data" / "outputs" / "causal-comparison"
SHARED_ROOT = PROJECT_ROOT / "data" / "shared" / "causal-comparison"
ANALYSIS_ROOT = RESULT_ROOT / "analysis"
FIGURE_ROOT = ANALYSIS_ROOT / "figures"

SCORED_DATASETS = (
    "sparseVAR",
    "autocorrelatedSCM",
    "lorenz96",
    "dream3",
)
EMPIRICAL_DATASETS = ("causalChamber", "causalRivers")
DATASET_LABELS = {
    "sparseVAR": "Sparse VAR",
    "autocorrelatedSCM": "Autocorrelated SCM",
    "lorenz96": "Lorenz--96",
    "dream3": "DREAM3",
    "causalChamber": "Causal Chambers",
    "causalRivers": "CausalRivers",
}
METHOD_ORDER = (
    "cgc",
    "fast-cgc",
    "cgc-star",
    "fast-cgc-star",
    "pcmci-plus",
    "bagged-pcmci-plus",
    "cuts-plus",
)
METHOD_LABELS = {
    "cgc": "c-GC",
    "fast-cgc": "fastC-GC",
    "cgc-star": "c-GC*",
    "fast-cgc-star": "fastC-GC*",
    "pcmci-plus": "PCMCI+",
    "bagged-pcmci-plus": "Bagged PCMCI+",
    "cuts-plus": "CUTS+",
}
METHOD_COLORS = {
    "cgc": "#0072B2",
    "fast-cgc": "#56B4E9",
    "cgc-star": "#D55E00",
    "fast-cgc-star": "#E69F00",
    "pcmci-plus": "#009E73",
    "bagged-pcmci-plus": "#CC79A7",
    "cuts-plus": "#000000",
}

# The collector converts Tigramite's modal graph-mark frequency to directed
# edge frequency after verifying that positive-lag marks are binary (no edge or
# ``-->``). All seven methods therefore have valid ranking scores.
RANKING_VALID = frozenset(METHOD_ORDER)

# With complete circular-shift calibration, p >= 1/n_effective.  The fixed
# beta=0.001 is below that floor in every scored simulation, so the archived
# fast-method adjacencies are empty by construction.  Their rankings remain
# valid, but thresholded metrics must not be compared as efficacy estimates.
THRESHOLD_VALID = frozenset(METHOD_ORDER) - {"fast-cgc", "fast-cgc-star"}

SUMMARY_METRICS = (
    "prevalence",
    "precision",
    "recall",
    "f1",
    "mcc",
    "balanced_accuracy",
    "false_positive_rate",
    "false_discovery_rate",
    "false_positives_per_target",
    "normalized_shd",
    "auroc",
    "auprc",
)


def _write_csv(frame: pd.DataFrame, name: str) -> Path:
    path = ANALYSIS_ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _save_figure(figure: plt.Figure, stem: str) -> tuple[Path, Path]:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    pdf = FIGURE_ROOT / f"{stem}.pdf"
    png = FIGURE_ROOT / f"{stem}.png"
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return pdf, png


def load_scored_results() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for dataset in SCORED_DATASETS:
        path = RESULT_ROOT / dataset / "scores" / "run_metrics.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing scored results: {path}")
        frame = pd.read_csv(path)
        if set(frame["dataset"].unique()) != {dataset}:
            raise ValueError(f"Dataset label mismatch in {path}.")
        frames.append(frame)

    results = pd.concat(frames, ignore_index=True)
    key = ["dataset", "dataset_id", "method", "estimand"]
    if results.duplicated(key).any():
        duplicates = results.loc[results.duplicated(key, keep=False), key]
        raise ValueError(f"Duplicate score rows:\n{duplicates.to_string(index=False)}")
    if set(results["status"].unique()) != {"ok"}:
        counts = results["status"].value_counts(dropna=False).to_dict()
        raise ValueError(f"Non-success score rows are present: {counts}")
    return results


def load_empirical_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    statuses: list[pd.DataFrame] = []
    edges: list[pd.DataFrame] = []
    for dataset in EMPIRICAL_DATASETS:
        graph_root = RESULT_ROOT / dataset / "graphs"
        status_path = graph_root / "run_status.csv"
        edge_path = graph_root / "inferred_edges.csv"
        if not status_path.is_file() or not edge_path.is_file():
            raise FileNotFoundError(f"Missing empirical collector output for {dataset}.")
        statuses.append(pd.read_csv(status_path))
        edges.append(pd.read_csv(edge_path))

    status = pd.concat(statuses, ignore_index=True)
    edge = pd.concat(edges, ignore_index=True)
    key = ["dataset", "dataset_id", "method"]
    if status.duplicated(key).any():
        raise ValueError("Duplicate empirical status rows are present.")
    if set(status["status"].unique()) != {"ok"}:
        counts = status["status"].value_counts(dropna=False).to_dict()
        raise ValueError(f"Non-success empirical rows are present: {counts}")
    return status, edge


def dataset_overview(
    scored: pd.DataFrame, empirical_status: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in (*SCORED_DATASETS, *EMPIRICAL_DATASETS):
        bundle_dir = SHARED_ROOT / dataset / "datasets"
        bundles = len(list(bundle_dir.glob("*.npz")))
        if dataset in SCORED_DATASETS:
            subset = scored[scored["dataset"] == dataset]
            method_fits = subset[["dataset_id", "method"]].drop_duplicates()
            estimands = ", ".join(sorted(subset["estimand"].unique()))
        else:
            subset = empirical_status[empirical_status["dataset"] == dataset]
            method_fits = subset[["dataset_id", "method"]].drop_duplicates()
            estimands = "aggregate inferred graph"
        rows.append(
            {
                "dataset": dataset,
                "label": DATASET_LABELS[dataset],
                "bundles": bundles,
                "methods": int(method_fits["method"].nunique()),
                "completed_method_fits": int(len(method_fits)),
                "failed": 0,
                "missing": 0,
                "estimands": estimands,
            }
        )
    return pd.DataFrame(rows)


def summarize_metrics(results: pd.DataFrame, estimand: str) -> pd.DataFrame:
    subset = results[results["estimand"] == estimand].copy()
    rows: list[dict[str, object]] = []
    for (dataset, method), group in subset.groupby(["dataset", "method"], sort=False):
        row: dict[str, object] = {
            "dataset": dataset,
            "dataset_label": DATASET_LABELS[dataset],
            "method": method,
            "method_label": METHOD_LABELS[method],
            "estimand": estimand,
            "runs": int(len(group)),
            "success_rate": float((group["status"] == "ok").mean()),
            "ranking_interpretable": method in RANKING_VALID,
            "threshold_interpretable": method in THRESHOLD_VALID,
            "selected_edges_mean": float(
                (group["true_positive"] + group["false_positive"]).mean()
            ),
            "selected_edges_std": float(
                (group["true_positive"] + group["false_positive"]).std(ddof=1)
            ),
        }
        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_n"] = int(len(values))
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else np.nan
            )
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["method"] = pd.Categorical(
        summary["method"], categories=METHOD_ORDER, ordered=True
    )
    return summary.sort_values(["dataset", "method"]).reset_index(drop=True)


def _bootstrap_mean_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    means = np.empty(10_000, dtype=float)
    for draw in range(means.size):
        means[draw] = rng.choice(values, size=values.size, replace=True).mean()
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def _rank_biserial(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    nonzero = differences[differences != 0.0]
    if nonzero.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / float(ranks.sum())


def _holm_adjust(pvalues: Iterable[float]) -> np.ndarray:
    raw = np.asarray(list(pvalues), dtype=float)
    order = np.argsort(raw)
    adjusted = np.empty_like(raw)
    running = 0.0
    count = len(raw)
    for rank, index in enumerate(order):
        candidate = min(1.0, raw[index] * (count - rank))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def paired_auprc_contrasts(results: pd.DataFrame) -> pd.DataFrame:
    contrasts = {
        "fast-cgc": (
            "cgc",
            "pcmci-plus",
            "bagged-pcmci-plus",
            "cuts-plus",
        ),
        "fast-cgc-star": (
            "cgc-star",
            "pcmci-plus",
            "bagged-pcmci-plus",
            "cuts-plus",
        ),
    }
    rows: list[dict[str, object]] = []
    aggregate = results[results["estimand"] == "aggregate-any-lag"]
    for dataset in SCORED_DATASETS:
        dataset_rows = aggregate[aggregate["dataset"] == dataset]
        for method_a, comparators in contrasts.items():
            left = dataset_rows[dataset_rows["method"] == method_a][
                ["dataset_id", "seed", "auprc"]
            ].rename(columns={"auprc": "metric_a"})
            for method_b in comparators:
                right = dataset_rows[dataset_rows["method"] == method_b][
                    ["dataset_id", "auprc"]
                ].rename(columns={"auprc": "metric_b"})
                paired = left.merge(right, on="dataset_id", validate="one_to_one")
                paired = paired.dropna(subset=["metric_a", "metric_b"])
                repeated_seed = paired["seed"].nunique() < len(paired)
                if repeated_seed:
                    paired = paired.groupby("seed", as_index=False)[
                        ["metric_a", "metric_b"]
                    ].mean()
                    unit = "seed macro-average across factorial cells"
                else:
                    unit = "stored graph instance"
                difference = (paired["metric_a"] - paired["metric_b"]).to_numpy()
                if difference.size == 0:
                    raise ValueError(f"No paired AUPRC values for {dataset}: {method_a}/{method_b}")
                if difference.size >= 3 and np.ptp(difference) > 0:
                    shapiro_p = float(stats.shapiro(difference).pvalue)
                else:
                    shapiro_p = np.nan
                try:
                    wilcoxon = stats.wilcoxon(
                        difference,
                        zero_method="wilcox",
                        alternative="two-sided",
                        method="auto",
                    )
                    statistic = float(wilcoxon.statistic)
                    pvalue = float(wilcoxon.pvalue)
                except ValueError:
                    statistic = 0.0
                    pvalue = 1.0
                token = f"{dataset}:{method_a}:{method_b}".encode("utf-8")
                bootstrap_seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
                ci_low, ci_high = _bootstrap_mean_ci(difference, bootstrap_seed)
                rows.append(
                    {
                        "dataset": dataset,
                        "method_a": method_a,
                        "method_b": method_b,
                        "comparison_unit": unit,
                        "n": int(difference.size),
                        "mean_difference": float(difference.mean()),
                        "std_difference": (
                            float(difference.std(ddof=1))
                            if difference.size > 1
                            else np.nan
                        ),
                        "bootstrap_95_ci_low": ci_low,
                        "bootstrap_95_ci_high": ci_high,
                        "shapiro_p": shapiro_p,
                        "test": "two-sided Wilcoxon signed-rank",
                        "statistic": statistic,
                        "p_raw": pvalue,
                        "rank_biserial": _rank_biserial(difference),
                    }
                )
    result = pd.DataFrame(rows)
    result["p_holm"] = np.nan
    for _dataset, indices in result.groupby("dataset").groups.items():
        result.loc[indices, "p_holm"] = _holm_adjust(result.loc[indices, "p_raw"])
    result["significant_holm_0_05"] = result["p_holm"] < 0.05
    return result


def score_agreement(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    aggregate = results[results["estimand"] == "aggregate-any-lag"]
    for dataset in SCORED_DATASETS:
        dataset_rows = aggregate[aggregate["dataset"] == dataset]
        for fast_method, conventional in (
            ("fast-cgc", "cgc"),
            ("fast-cgc-star", "cgc-star"),
        ):
            left = dataset_rows[dataset_rows["method"] == fast_method][
                ["dataset_id", "auprc", "auroc"]
            ]
            right = dataset_rows[dataset_rows["method"] == conventional][
                ["dataset_id", "auprc", "auroc"]
            ]
            paired = left.merge(
                right,
                on="dataset_id",
                suffixes=("_fast", "_conventional"),
                validate="one_to_one",
            )
            for metric in ("auprc", "auroc"):
                valid = paired.dropna(
                    subset=[f"{metric}_fast", f"{metric}_conventional"]
                )
                fast_values = valid[f"{metric}_fast"].to_numpy()
                conventional_values = valid[f"{metric}_conventional"].to_numpy()
                rows.append(
                    {
                        "dataset": dataset,
                        "fast_method": fast_method,
                        "conventional_method": conventional,
                        "metric": metric,
                        "n": int(len(valid)),
                        "pearson_r": float(stats.pearsonr(fast_values, conventional_values).statistic),
                        "spearman_rho": float(stats.spearmanr(fast_values, conventional_values).statistic),
                        "mean_absolute_difference": float(
                            np.mean(np.abs(fast_values - conventional_values))
                        ),
                    }
                )
    return pd.DataFrame(rows)


def factor_summaries(results: pd.DataFrame) -> pd.DataFrame:
    aggregate = results[
        (results["estimand"] == "aggregate-any-lag")
        & results["method"].isin(RANKING_VALID)
    ]
    rows: list[dict[str, object]] = []
    factor_map = {
        "sparseVAR": ("n_variables", "n_samples", "fit_order", "noise"),
        "autocorrelatedSCM": ("regime",),
        "lorenz96": ("n_variables", "n_samples"),
    }
    for dataset, factors in factor_map.items():
        dataset_rows = aggregate[aggregate["dataset"] == dataset]
        for factor in factors:
            for (method, level), group in dataset_rows.groupby(
                ["method", factor], dropna=False
            ):
                values = group["auprc"].dropna()
                rows.append(
                    {
                        "dataset": dataset,
                        "factor": factor,
                        "level": level,
                        "method": method,
                        "n": int(len(values)),
                        "auprc_mean": float(values.mean()),
                        "auprc_std": float(values.std(ddof=1)),
                    }
                )
    return pd.DataFrame(rows)


def empirical_summaries(
    status: pd.DataFrame, edges: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = (
        status.groupby(["dataset", "method"])["edge_count"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={"count": "runs"})
    )
    summary["method_label"] = summary["method"].map(METHOD_LABELS)

    edge_sets: dict[tuple[str, str, str], set[tuple[int, int]]] = defaultdict(set)
    for row in edges.itertuples(index=False):
        edge_sets[(row.dataset, row.dataset_id, row.method)].add(
            (int(row.source_index), int(row.target_index))
        )

    agreement_rows: list[dict[str, object]] = []
    pairs = (("fast-cgc", "cgc"), ("fast-cgc-star", "cgc-star"))
    for row in status[["dataset", "dataset_id"]].drop_duplicates().itertuples(index=False):
        for fast_method, conventional in pairs:
            fast_edges = edge_sets[(row.dataset, row.dataset_id, fast_method)]
            conventional_edges = edge_sets[(row.dataset, row.dataset_id, conventional)]
            union = fast_edges | conventional_edges
            intersection = fast_edges & conventional_edges
            agreement_rows.append(
                {
                    "dataset": row.dataset,
                    "dataset_id": row.dataset_id,
                    "fast_method": fast_method,
                    "conventional_method": conventional,
                    "fast_edges": len(fast_edges),
                    "conventional_edges": len(conventional_edges),
                    "intersection": len(intersection),
                    "symmetric_difference": len(fast_edges ^ conventional_edges),
                    "jaccard": len(intersection) / len(union) if union else 1.0,
                }
            )
    agreement = pd.DataFrame(agreement_rows)
    agreement_summary = (
        agreement.groupby(
            ["dataset", "fast_method", "conventional_method"], as_index=False
        )
        .agg(
            runs=("dataset_id", "count"),
            jaccard_mean=("jaccard", "mean"),
            jaccard_std=("jaccard", "std"),
            symmetric_difference_mean=("symmetric_difference", "mean"),
            symmetric_difference_std=("symmetric_difference", "std"),
        )
    )
    return summary, agreement, agreement_summary


def source_revisions() -> pd.DataFrame:
    revisions: dict[str, set[str]] = defaultdict(set)
    for path in RESULT_ROOT.glob("*/*/checkpoints/*.json"):
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        revisions[str(checkpoint["method"])].add(
            str(checkpoint.get("upstream_commit", ""))
        )
    rows = [
        {"method": method, "source_revision": revision}
        for method, values in sorted(revisions.items())
        for revision in sorted(values)
    ]
    return pd.DataFrame(rows)


def plot_aggregate_auprc(summary: pd.DataFrame) -> tuple[Path, Path]:
    methods = [method for method in METHOD_ORDER if method in RANKING_VALID]
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharey=True)
    for axis, dataset in zip(axes.flat, SCORED_DATASETS, strict=True):
        subset = summary[summary["dataset"] == dataset].set_index("method")
        x = np.arange(len(methods))
        means = np.array([subset.loc[method, "auprc_mean"] for method in methods])
        errors = np.array([subset.loc[method, "auprc_std"] for method in methods])
        for index, method in enumerate(methods):
            axis.errorbar(
                x[index],
                means[index],
                yerr=errors[index],
                marker="o",
                markersize=4.5,
                capsize=2.5,
                color=METHOD_COLORS[method],
                linewidth=1.2,
            )
        prevalence = float(subset["prevalence_mean"].iloc[0])
        axis.axhline(prevalence, color="#777777", linestyle="--", linewidth=1)
        axis.set_title(DATASET_LABELS[dataset], fontsize=9)
        axis.set_xticks(x, [METHOD_LABELS[method] for method in methods], rotation=40, ha="right")
        axis.set_ylim(0.0, 1.03)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(labelsize=7)
    axes[0, 0].set_ylabel(r"AUPRC, mean $\pm$ SD")
    axes[1, 0].set_ylabel(r"AUPRC, mean $\pm$ SD")
    figure.tight_layout()
    return _save_figure(figure, "aggregate-auprc")


def plot_threshold_f1(summary: pd.DataFrame) -> tuple[Path, Path]:
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharey=True)
    for axis, dataset in zip(axes.flat, SCORED_DATASETS, strict=True):
        subset = summary[summary["dataset"] == dataset].set_index("method")
        x = np.arange(len(METHOD_ORDER))
        for index, method in enumerate(METHOD_ORDER):
            if method not in THRESHOLD_VALID:
                axis.scatter(index, 0.0, marker="x", s=28, color="#777777", zorder=3)
                continue
            axis.errorbar(
                index,
                subset.loc[method, "f1_mean"],
                yerr=subset.loc[method, "f1_std"],
                marker="o",
                markersize=4.5,
                capsize=2.5,
                color=METHOD_COLORS[method],
                linewidth=1.2,
            )
        axis.set_title(DATASET_LABELS[dataset], fontsize=9)
        axis.set_xticks(
            x,
            [METHOD_LABELS[method] for method in METHOD_ORDER],
            rotation=40,
            ha="right",
        )
        axis.set_ylim(-0.03, 1.03)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(labelsize=7)
    axes[0, 0].set_ylabel(r"F1, mean $\pm$ SD")
    axes[1, 0].set_ylabel(r"F1, mean $\pm$ SD")
    figure.tight_layout()
    return _save_figure(figure, "aggregate-threshold-f1")


def plot_exact_lag_auprc(summary: pd.DataFrame) -> tuple[Path, Path]:
    methods = (
        "cgc",
        "fast-cgc",
        "cgc-star",
        "fast-cgc-star",
        "pcmci-plus",
        "bagged-pcmci-plus",
    )
    datasets = ("sparseVAR", "autocorrelatedSCM")
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for axis, dataset in zip(axes, datasets, strict=True):
        subset = summary[summary["dataset"] == dataset].set_index("method")
        x = np.arange(len(methods))
        for index, method in enumerate(methods):
            axis.errorbar(
                index,
                subset.loc[method, "auprc_mean"],
                yerr=subset.loc[method, "auprc_std"],
                marker="o",
                markersize=5,
                capsize=3,
                color=METHOD_COLORS[method],
                linewidth=1.2,
            )
        prevalence = float(subset["prevalence_mean"].iloc[0])
        axis.axhline(prevalence, color="#777777", linestyle="--", linewidth=1)
        axis.set_title(DATASET_LABELS[dataset], fontsize=9)
        axis.set_xticks(
            x,
            [METHOD_LABELS[method] for method in methods],
            rotation=25,
            ha="right",
        )
        axis.set_ylim(0.0, 1.03)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(labelsize=8)
    axes[0].set_ylabel(r"Exact-lag AUPRC, mean $\pm$ SD")
    figure.tight_layout()
    return _save_figure(figure, "exact-lag-auprc")


def plot_empirical_edge_counts(summary: pd.DataFrame) -> tuple[Path, Path]:
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for axis, dataset in zip(axes, EMPIRICAL_DATASETS, strict=True):
        subset = summary[summary["dataset"] == dataset].set_index("method")
        x = np.arange(len(METHOD_ORDER))
        means = np.array([subset.loc[method, "mean"] for method in METHOD_ORDER])
        errors = np.array([subset.loc[method, "std"] for method in METHOD_ORDER])
        errors = np.nan_to_num(errors, nan=0.0)
        for index, method in enumerate(METHOD_ORDER):
            axis.errorbar(
                index,
                means[index],
                yerr=errors[index],
                marker="o",
                markersize=4.5,
                capsize=2.5,
                color=METHOD_COLORS[method],
                linewidth=1.2,
            )
        axis.set_title(DATASET_LABELS[dataset], fontsize=9)
        axis.set_xticks(
            x,
            [METHOD_LABELS[method] for method in METHOD_ORDER],
            rotation=40,
            ha="right",
        )
        axis.set_ylabel("Inferred cross-variable edges")
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(labelsize=7)
    figure.tight_layout()
    return _save_figure(figure, "empirical-edge-counts")


def write_reports(
    overview: pd.DataFrame,
    aggregate: pd.DataFrame,
    contrasts: pd.DataFrame,
    agreement: pd.DataFrame,
    empirical: pd.DataFrame,
) -> None:
    total_bundles = int(overview["bundles"].sum())
    total_fits = int(overview["completed_method_fits"].sum())
    top_lines: list[str] = []
    for dataset in SCORED_DATASETS:
        subset = aggregate[
            (aggregate["dataset"] == dataset)
            & aggregate["ranking_interpretable"]
        ]
        best = subset.loc[subset["auprc_mean"].idxmax()]
        top_lines.append(
            f"- {DATASET_LABELS[dataset]}: {best['method_label']} had the highest "
            f"mean AUPRC, {best['auprc_mean']:.4f} (SD {best['auprc_std']:.4f})."
        )
    empirical_lines: list[str] = []
    for dataset in EMPIRICAL_DATASETS:
        subset = empirical[empirical["dataset"] == dataset]
        lowest = subset.loc[subset["mean"].idxmin()]
        highest = subset.loc[subset["mean"].idxmax()]
        empirical_lines.append(
            f"- {DATASET_LABELS[dataset]}: mean edge counts ranged from "
            f"{lowest['mean']:.1f} ({lowest['method_label']}) to "
            f"{highest['mean']:.1f} ({highest['method_label']})."
        )

    report = f"""# Causal-comparison analysis report

## Analysis question

The analysis asks whether the continuous scores of fastC-GC and fastC-GC*
retain the graph-discrimination behavior of conventional c-GC and c-GC*, how
all seven methods compare on common controlled datasets, and how their inferred
graphs differ on the two measured-data case studies. Runtime is not analyzed.

## Artifact validation

- {total_bundles:,} stored datasets were found across six suites.
- {total_fits:,} method--dataset fits were complete; none were failed or missing.
- Score rows were unique by dataset, method, and estimand.
- Exact-lag and aggregate-any-lag results were kept separate.

## Main ranking results

{chr(10).join(top_lines)}

For Bagged PCMCI+, the collector converts Tigramite's modal-mark frequency to
directed-edge frequency. Every archived positive-lag graph contains only the
no-edge and directed-edge marks, so the edge frequency is the modal frequency
when the edge is modal and its complement when no edge is modal.

## Calibration finding

All archived fastC-GC and fastC-GC* simulation adjacencies are empty. The
complete-shift p-value has minimum 1/n_effective, whereas the fitted conditional
threshold is beta=0.001. Every simulated or in-silico series has fewer than
1,000 effective observations after lag expansion, so beta is unattainable.
The thresholded fast-method F1, MCC, and SHD values are therefore calibration
artifacts and are not used for efficacy ranking. AUROC and AUPRC remain valid
because their stored score, 1-max(p_marginal,p_conditional), varies across
candidate edges and remains rankable.

## Score-level agreement

Across the controlled datasets, per-graph AUROC correlations between each fast
method and its conventional counterpart ranged from
{agreement[agreement.metric == 'auroc'].pearson_r.min():.4f} to
{agreement[agreement.metric == 'auroc'].pearson_r.max():.4f}. This supports
agreement in edge ranking, not parity of thresholded graphs.

## Measured-data graphs

{chr(10).join(empirical_lines)}

These counts describe graph density only. The supplied physical and river
networks are contextual references rather than exhaustive lag-resolved truth,
so no empirical recovery ranking is reported.

## Interpretation

The archived outputs support the continuous-score comparison and the
thresholded results of methods whose decision rules were attainable. A
thresholded comparison of all four c-GC variants requires a separate,
prespecified calibration arm. For these datasets, beta=0.005 is above the
largest complete-shift p-value floor (1/245 = 0.00408) and below alpha=0.01.
It must be validated on separate null or development data, frozen, and then
used to rerun all four variants on the same test datasets. The source-faithful
beta=0.001 arm remains available for score-level comparison.
"""
    (ANALYSIS_ROOT / "analysis-report.md").write_text(report, encoding="utf-8")

    table_rows = []
    for row in contrasts.itertuples(index=False):
        table_rows.append(
            f"| {DATASET_LABELS[row.dataset]} | {METHOD_LABELS[row.method_a]} | "
            f"{METHOD_LABELS[row.method_b]} | {row.n} | {row.mean_difference:.4f} "
            f"[{row.bootstrap_95_ci_low:.4f}, {row.bootstrap_95_ci_high:.4f}] | "
            f"{row.statistic:.1f} | {row.p_holm:.4g} | {row.rank_biserial:.3f} |"
        )
    stats_report = f"""# Statistical appendix

## Comparison unit and test

AUPRC is the primary ranking metric. Comparisons are paired by stored dataset.
For sparse VAR and Lorenz--96, the same seed indexes several factorial cells;
metrics were averaged across those cells before testing, leaving 50 paired seed
macro-averages. Autocorrelated-SCM runs use distinct stored graph instances,
and the 50 empty-truth graphs were excluded because AUPRC is undefined. DREAM3
uses its five distinct networks as the paired units.

The analysis uses two-sided Wilcoxon signed-rank tests because AUPRC is bounded
and several paired-difference distributions reject normality. Rank-biserial
correlation is the effect size. Mean differences have deterministic 10,000-draw
paired bootstrap 95% intervals. Holm correction is applied over the eight
declared contrasts within each dataset suite.

| Dataset | Method A | Method B | n | Mean A-B [95% CI] | W | Holm p | Rank-biserial |
|---|---|---|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

## Boundaries

- Bagged PCMCI+ directed-edge frequency is reconstructed only after verifying
  that the archived positive-lag mark alphabet is binary.
- Thresholded fast-method contrasts are excluded because beta=0.001 is below
  the attainable complete-shift p-value floor.
- DREAM3 has only five heterogeneous networks; its intervals and tests have
  low power and are descriptive support rather than population inference.
- No individual edge is treated as an independent replicate.
"""
    (ANALYSIS_ROOT / "stats-appendix.md").write_text(
        stats_report, encoding="utf-8"
    )

    figure_catalog = """# Figure catalog

## aggregate-auprc.pdf

- Purpose: compare aggregate directed-edge ranking across the four datasets
  with graph truth.
- Data: aggregate-any-lag run metrics; points are means and error bars are
  sample standard deviations across stored graphs. Dashed lines show mean edge
  prevalence.
- Observation: c-GC and fastC-GC variants are close on sparse VAR and the
  autocorrelated SCM; CUTS+ is strongest on Lorenz--96.
- Caveat: DREAM3 contains five heterogeneous networks rather than repeated
  seeds.

## aggregate-threshold-f1.pdf

- Purpose: compare thresholded graph recovery where the archived decision rule
  is interpretable.
- Data: aggregate-any-lag F1, mean plus or minus sample standard deviation.
- Observation: gray crosses mark fast-method outputs affected by the
  unattainable beta threshold; they are not efficacy estimates.
- Caveat: threshold choices differ by native method and DREAM3 has five
  networks rather than repeated seeds.

## exact-lag-auprc.pdf

- Purpose: compare lag localization for methods with valid continuous
  lag-resolved scores.
- Data: sparse VAR and autocorrelated-SCM exact-lag AUPRC.
- Observation: both c-GC pairs, PCMCI+, and Bagged PCMCI+ retain strong
  exact-lag ranking; performance is lower for the autocorrelated nonlinear
  process.
- Caveat: CUTS+ does not expose a lag-resolved native output.

## empirical-edge-counts.pdf

- Purpose: show differences in inferred graph density without implying
  empirical recovery accuracy.
- Data: one Causal Chambers graph and ten CausalRivers subgraphs; error bars on
  CausalRivers denote sample standard deviation across subgraphs.
- Observation: CUTS+ is much denser than the other methods in Causal Chambers;
  CausalRivers densities also vary by method.
- Caveat: contextual networks are not treated as exhaustive causal truth.
"""
    (ANALYSIS_ROOT / "figure-catalog.md").write_text(
        figure_catalog, encoding="utf-8"
    )


def main() -> None:
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    scored = load_scored_results()
    empirical_status, empirical_edges = load_empirical_results()
    overview = dataset_overview(scored, empirical_status)
    aggregate = summarize_metrics(scored, "aggregate-any-lag")
    exact = summarize_metrics(scored, "exact-lag")
    contrasts = paired_auprc_contrasts(scored)
    agreement = score_agreement(scored)
    factors = factor_summaries(scored)
    empirical, empirical_agreement, empirical_agreement_summary = empirical_summaries(
        empirical_status, empirical_edges
    )

    _write_csv(overview, "dataset-overview.csv")
    _write_csv(aggregate, "aggregate-summary.csv")
    _write_csv(exact, "exact-lag-summary.csv")
    _write_csv(contrasts, "paired-auprc-contrasts.csv")
    _write_csv(agreement, "score-agreement.csv")
    _write_csv(factors, "factor-summary.csv")
    _write_csv(source_revisions(), "source-revisions.csv")
    _write_csv(empirical, "empirical-edge-count-summary.csv")
    _write_csv(empirical_agreement, "empirical-edge-agreement.csv")
    _write_csv(
        empirical_agreement_summary,
        "empirical-edge-agreement-summary.csv",
    )
    dream = scored[
        (scored["dataset"] == "dream3")
        & (scored["estimand"] == "aggregate-any-lag")
    ]
    _write_csv(dream, "dream3-network-metrics.csv")
    null_rows = scored[
        (scored["dataset"] == "autocorrelatedSCM")
        & (scored["estimand"] == "aggregate-any-lag")
        & (scored["regime"] == "cross-link-null")
    ]
    _write_csv(null_rows, "autocorrelated-null-metrics.csv")

    plot_aggregate_auprc(aggregate)
    plot_threshold_f1(aggregate)
    plot_exact_lag_auprc(exact)
    plot_empirical_edge_counts(empirical)
    write_reports(
        overview,
        aggregate,
        contrasts,
        agreement,
        empirical,
    )
    print(f"Wrote causal-comparison analysis to {ANALYSIS_ROOT}")


if __name__ == "__main__":
    main()
