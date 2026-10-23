"""Collect and score native causal-comparison notebook outputs.

The method notebooks intentionally save arrays in each upstream package's
native representation.  This module is the single conversion boundary from
those representations to the study convention ``[lag, source, target]``.  It
also keeps exact-lag and aggregate-any-lag results in separate rows.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .causal_benchmark import aggregate_lags, graph_metrics
from .notebook_checkpoint import file_sha256, read_checkpoint


METHODS = (
    "fast-cgc",
    "fast-cgc-star",
    "cgc",
    "cgc-star",
    "pcmci-plus",
    "bagged-pcmci-plus",
    "cuts-plus",
)

EXACT_LAG_METHODS = frozenset(
    {
        "fast-cgc",
        "fast-cgc-star",
        "pcmci-plus",
        "bagged-pcmci-plus",
    }
)

METRIC_NAMES = (
    "candidates",
    "positives",
    "prevalence",
    "true_positive",
    "false_positive",
    "false_negative",
    "true_negative",
    "precision",
    "recall",
    "specificity",
    "negative_predictive_value",
    "f1",
    "mcc",
    "balanced_accuracy",
    "false_positive_rate",
    "false_discovery_rate",
    "false_positives_per_target",
    "shd",
    "normalized_shd",
    "auroc",
    "auprc",
)

GROUP_COLUMNS = (
    "dataset",
    "method",
    "estimand",
    "n_variables",
    "n_samples",
    "fit_order",
    "noise",
    "regime",
)


@dataclass(frozen=True, slots=True)
class NormalizedOutput:
    """One native result converted to a declared common estimand."""

    estimand: str
    selected: np.ndarray
    scores: np.ndarray | None


@dataclass(frozen=True, slots=True)
class ScoreArtifacts:
    """In-memory score tables and their persisted CSV paths."""

    run_metrics: pd.DataFrame
    summary: pd.DataFrame
    run_metrics_path: Path
    summary_path: Path


@dataclass(frozen=True, slots=True)
class GraphArtifacts:
    """Empirical inferred-edge tables and their persisted CSV paths."""

    edge_table: pd.DataFrame
    status_table: pd.DataFrame
    reference_edges: pd.DataFrame
    edge_table_path: Path
    status_table_path: Path
    reference_edges_path: Path


EDGE_COLUMNS = (
    "dataset",
    "dataset_id",
    "method",
    "source_index",
    "target_index",
    "source",
    "target",
    "score",
)

STATUS_COLUMNS = (
    "dataset",
    "dataset_id",
    "method",
    "status",
    "n_variables",
    "n_samples",
    "fit_order",
    "regime",
    "edge_count",
    "error",
)

REFERENCE_EDGE_COLUMNS = (
    "dataset",
    "dataset_id",
    "source_index",
    "target_index",
    "source",
    "target",
)


def _require_shape(
    value: Any,
    shape: tuple[int, ...],
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}; expected {shape}.")
    return array


def _lagged_tigramite_output(
    graph: Any,
    scores: Any,
    *,
    fit_order: int,
    n_variables: int,
    graph_name: str,
    score_name: str,
) -> list[NormalizedOutput]:
    native_shape = (n_variables, n_variables, fit_order + 1)
    graph_array = _require_shape(graph, native_shape, name=graph_name)
    score_array = _require_shape(scores, native_shape, name=score_name).astype(
        float,
        copy=False,
    )

    # Tigramite stores graph[source, target, lag].  Index zero is the
    # contemporaneous slice and is intentionally excluded from this study.
    exact_selected = np.transpose(
        graph_array[:, :, 1 : fit_order + 1] == "-->",
        (2, 0, 1),
    )
    exact_scores = np.transpose(
        score_array[:, :, 1 : fit_order + 1],
        (2, 0, 1),
    )
    return [
        NormalizedOutput("exact-lag", exact_selected, exact_scores),
        NormalizedOutput(
            "aggregate-any-lag",
            aggregate_lags(exact_selected, binary=True),
            aggregate_lags(exact_scores, binary=False),
        ),
    ]


def normalize_native_output(
    method: str,
    native: Mapping[str, Any],
    *,
    fit_order: int,
    n_variables: int,
) -> list[NormalizedOutput]:
    """Convert one saved native output without changing its estimand.

    Exact-lag rows are returned only for methods whose native result identifies
    positive lags.  Conventional c-GC/c-GC* and CUTS+ remain aggregate-only; the
    collector never invents a lag assignment for them.
    """

    if method not in METHODS:
        raise ValueError(f"Unknown comparison method: {method!r}.")
    if fit_order < 1 or n_variables < 2:
        raise ValueError("fit_order must be positive and n_variables at least two.")

    lag_shape = (fit_order, n_variables, n_variables)
    aggregate_shape = (n_variables, n_variables)

    if method in {"fast-cgc", "fast-cgc-star"}:
        selected = _require_shape(
            native["lag_adjacency"],
            lag_shape,
            name="lag_adjacency",
        ).astype(bool, copy=False)
        scores = _require_shape(
            native["lag_scores"],
            lag_shape,
            name="lag_scores",
        ).astype(float, copy=False)
        return [
            NormalizedOutput("exact-lag", selected, scores),
            NormalizedOutput(
                "aggregate-any-lag",
                aggregate_lags(selected, binary=True),
                aggregate_lags(scores, binary=False),
            ),
        ]

    if method in {"cgc", "cgc-star"}:
        selected = _require_shape(
            native["conn_mat"],
            aggregate_shape,
            name="conn_mat",
        ).astype(bool, copy=False)
        native_pvalue_shape = (
            (fit_order + 1) * n_variables,
            n_variables,
        )
        marginal_pvalues = _require_shape(
            native["pvalue_corr"],
            native_pvalue_shape,
            name="pvalue_corr",
        ).astype(float, copy=False)
        conditional_pvalues = _require_shape(
            native["pvalue_conditional"],
            native_pvalue_shape,
            name="pvalue_conditional",
        ).astype(float, copy=False)
        combined_pvalues = np.maximum(
            marginal_pvalues,
            conditional_pvalues,
        ).reshape(fit_order + 1, n_variables, n_variables)[1:]
        # Both tests must support an edge.  Thus 1 - max(p_marginal,
        # p_conditional) is a monotone conjunction score, with larger values
        # consistently denoting stronger edge evidence.
        scores = aggregate_lags(1.0 - combined_pvalues, binary=False)
        return [NormalizedOutput("aggregate-any-lag", selected, scores)]

    if method == "pcmci-plus":
        pvalues = np.asarray(native["p_matrix"], dtype=float)
        return _lagged_tigramite_output(
            native["graph"],
            1.0 - pvalues,
            fit_order=fit_order,
            n_variables=n_variables,
            graph_name="graph",
            score_name="p_matrix",
        )

    if method == "bagged-pcmci-plus":
        return _lagged_tigramite_output(
            native["most_frequent_links"],
            native["link_frequency"],
            fit_order=fit_order,
            n_variables=n_variables,
            graph_name="most_frequent_links",
            score_name="link_frequency",
        )

    graph = _require_shape(native["graph"], aggregate_shape, name="graph").astype(
        float,
        copy=False,
    )
    return [
        NormalizedOutput(
            "aggregate-any-lag",
            graph >= 0.5,
            graph,
        )
    ]


def _series_digest(series: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(series, dtype=np.float64)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _expected_estimands(method: str, *, exact_truth: bool) -> tuple[str, ...]:
    if exact_truth and method in EXACT_LAG_METHODS:
        return ("exact-lag", "aggregate-any-lag")
    return ("aggregate-any-lag",)


def _empty_metric_values() -> dict[str, float]:
    return {name: np.nan for name in METRIC_NAMES}


def _base_row(
    *,
    dataset_key: str,
    dataset_id: str,
    method: str,
    estimand: str,
    status: str,
    n_variables: int,
    n_samples: int,
    fit_order: int,
    noise: str,
    regime: str,
    seed: int | str,
    error: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset_key,
        "dataset_id": dataset_id,
        "method": method,
        "estimand": estimand,
        "status": status,
        "n_variables": n_variables,
        "n_samples": n_samples,
        "fit_order": fit_order,
        "noise": noise,
        "regime": regime,
        "seed": seed,
        "error": error,
    }


def _load_native_output(
    project_root: Path,
    checkpoint: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    recorded_path = checkpoint.get("native_output")
    if not isinstance(recorded_path, str) or not recorded_path:
        raise ValueError("Successful checkpoint has no native_output path.")
    native_path = Path(recorded_path)
    if not native_path.is_absolute():
        native_path = project_root / native_path
    if not native_path.is_file():
        raise FileNotFoundError(f"Native output is missing: {native_path}.")
    recorded_digest = checkpoint.get("native_output_sha256")
    observed_digest = file_sha256(native_path)
    if observed_digest != recorded_digest:
        raise ValueError(f"Native-output digest mismatch for {native_path}.")
    with np.load(native_path, allow_pickle=False) as stored:
        return {name: np.array(stored[name], copy=True) for name in stored.files}


def _validate_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    dataset_key: str,
    dataset_id: str,
    method: str,
    dataset_digest: str,
    fit_order: int,
) -> None:
    expected = {
        "dataset": dataset_key,
        "dataset_id": dataset_id,
        "method": method,
        "dataset_digest": dataset_digest,
        "fit_order": fit_order,
    }
    mismatches = {
        key: (checkpoint.get(key), value)
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: recorded={recorded!r}, expected={expected_value!r}"
            for key, (recorded, expected_value) in mismatches.items()
        )
        raise ValueError(f"Checkpoint provenance mismatch: {details}.")


def _summarize(run_metrics: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    grouped = run_metrics.groupby(list(GROUP_COLUMNS), dropna=False, sort=True)
    for group_values, group in grouped:
        row = dict(zip(GROUP_COLUMNS, group_values, strict=True))
        expected = int(len(group))
        missing = int((group["status"] == "missing").sum())
        attempted = expected - missing
        completed = int((group["status"] == "ok").sum())
        row.update(
            {
                "expected": expected,
                "missing": missing,
                "attempted": attempted,
                "completed": completed,
                "failed": attempted - completed,
                "success_rate": completed / attempted if attempted else np.nan,
            }
        )

        for metric in (*METRIC_NAMES, "failure_aware_f1"):
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            count = int(len(values))
            mean = float(values.mean()) if count else np.nan
            standard_deviation = (
                float(values.std(ddof=1)) if count > 1 else np.nan
            )
            row[f"{metric}_n"] = count
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = standard_deviation
            if np.isfinite(mean) and np.isfinite(standard_deviation):
                formatted = f"{mean:.4f} ({standard_deviation:.4f})"
            elif np.isfinite(mean):
                formatted = f"{mean:.4f} (NA)"
            else:
                formatted = "NA"
            row[f"{metric}_mean_std"] = formatted
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(temporary, path)


def _named_edges(
    selected: np.ndarray,
    *,
    variable_names: Sequence[str],
    scores: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Return cross-variable directed edges in source-target orientation."""

    n_variables = len(variable_names)
    adjacency = _require_shape(
        selected,
        (n_variables, n_variables),
        name="selected graph",
    ).astype(bool, copy=False)
    score_array = None
    if scores is not None:
        score_array = _require_shape(
            scores,
            adjacency.shape,
            name="edge scores",
        ).astype(float, copy=False)

    rows: list[dict[str, Any]] = []
    for source_index, target_index in np.argwhere(adjacency):
        source = int(source_index)
        target = int(target_index)
        if source == target:
            continue
        rows.append(
            {
                "source_index": source,
                "target_index": target,
                "source": str(variable_names[source]),
                "target": str(variable_names[target]),
                "score": (
                    float(score_array[source, target])
                    if score_array is not None
                    else np.nan
                ),
            }
        )
    return rows


def collect_dataset_graphs(
    project_root: Path,
    dataset_key: str,
    *,
    methods: Sequence[str] = METHODS,
) -> GraphArtifacts:
    """Collect empirical inferred graphs without requiring a truth matrix.

    The collector validates the same dataset and native-output provenance used
    by the simulation scorer. It exports aggregate cross-variable edges for
    visualization and keeps missing or failed fits in a separate status table.
    A bundle may include a ``reference_graph`` for a contextual panel, but that
    graph is never scored by this function.
    """

    project_root = Path(project_root).resolve()
    requested_methods = tuple(methods)
    unknown = sorted(set(requested_methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown comparison methods: {unknown}.")
    if len(requested_methods) != len(set(requested_methods)):
        raise ValueError("methods must not contain duplicates.")

    dataset_dir = (
        project_root
        / "data"
        / "shared"
        / "causal-comparison"
        / dataset_key
        / "datasets"
    )
    bundle_paths = sorted(dataset_dir.glob("*.npz"))
    if not bundle_paths:
        raise FileNotFoundError(
            f"No frozen {dataset_key} bundles found in {dataset_dir}."
        )

    edge_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    for bundle_path in bundle_paths:
        metadata_path = bundle_path.with_suffix(".json")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing metadata for {bundle_path}.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(bundle_path, allow_pickle=False) as stored:
            series = np.asarray(stored["series"], dtype=float)
            if series.ndim != 2:
                raise ValueError(f"series must be two-dimensional in {bundle_path}.")
            reference_graph = (
                np.asarray(stored["reference_graph"], dtype=bool)
                if "reference_graph" in stored.files
                else None
            )

        dataset_id = bundle_path.stem
        if metadata.get("dataset_id") != dataset_id:
            raise ValueError(f"Dataset identifier mismatch for {bundle_path}.")
        observed_digest = _series_digest(series)
        recorded_digest = metadata.get("data_digest", metadata.get("digest"))
        if observed_digest != recorded_digest:
            raise ValueError(f"Dataset digest mismatch for {bundle_path}.")

        n_variables, n_samples = map(int, series.shape)
        names_value = metadata.get("variable_names")
        if names_value is None:
            variable_names = tuple(f"X{index}" for index in range(n_variables))
        elif (
            isinstance(names_value, list)
            and len(names_value) == n_variables
            and all(isinstance(name, (str, int)) for name in names_value)
        ):
            variable_names = tuple(str(name) for name in names_value)
        else:
            raise ValueError(
                f"variable_names must contain {n_variables} strings in {metadata_path}."
            )
        fit_order = int(metadata.get("fit_order", 1))
        regime = str(metadata.get("regime", ""))

        if reference_graph is not None:
            for row in _named_edges(
                reference_graph,
                variable_names=variable_names,
            ):
                reference_rows.append(
                    {
                        "dataset": dataset_key,
                        "dataset_id": dataset_id,
                        **{key: row[key] for key in REFERENCE_EDGE_COLUMNS[2:]},
                    }
                )

        for method in requested_methods:
            checkpoint_path = (
                project_root
                / "data"
                / "outputs"
                / "causal-comparison"
                / dataset_key
                / method
                / "checkpoints"
                / f"{dataset_id}.json"
            )
            status = "missing"
            error = "checkpoint not found"
            edge_count = 0
            if checkpoint_path.is_file():
                checkpoint = read_checkpoint(checkpoint_path)
                _validate_checkpoint(
                    checkpoint,
                    dataset_key=dataset_key,
                    dataset_id=dataset_id,
                    method=method,
                    dataset_digest=observed_digest,
                    fit_order=fit_order,
                )
                if checkpoint.get("status") == "ok":
                    native = _load_native_output(project_root, checkpoint)
                    normalized = normalize_native_output(
                        method,
                        native,
                        fit_order=fit_order,
                        n_variables=n_variables,
                    )
                    aggregate = next(
                        (
                            output
                            for output in normalized
                            if output.estimand == "aggregate-any-lag"
                        ),
                        None,
                    )
                    if aggregate is None:
                        raise ValueError(
                            f"{method} did not return an aggregate graph for {dataset_id}."
                        )
                    named_edges = _named_edges(
                        aggregate.selected,
                        variable_names=variable_names,
                        scores=aggregate.scores,
                    )
                    edge_count = len(named_edges)
                    for row in named_edges:
                        edge_rows.append(
                            {
                                "dataset": dataset_key,
                                "dataset_id": dataset_id,
                                "method": method,
                                **row,
                            }
                        )
                    status = "ok"
                    error = ""
                else:
                    status = str(checkpoint.get("status", "failed"))
                    error = str(checkpoint.get("error", "fit did not complete"))

            status_rows.append(
                {
                    "dataset": dataset_key,
                    "dataset_id": dataset_id,
                    "method": method,
                    "status": status,
                    "n_variables": n_variables,
                    "n_samples": n_samples,
                    "fit_order": fit_order,
                    "regime": regime,
                    "edge_count": edge_count,
                    "error": error,
                }
            )

    edge_table = pd.DataFrame(edge_rows, columns=EDGE_COLUMNS)
    status_table = pd.DataFrame(status_rows, columns=STATUS_COLUMNS).sort_values(
        ["dataset_id", "method"],
        kind="stable",
        ignore_index=True,
    )
    reference_edges = pd.DataFrame(
        reference_rows,
        columns=REFERENCE_EDGE_COLUMNS,
    )
    if not edge_table.empty:
        edge_table = edge_table.sort_values(
            ["dataset_id", "method", "source_index", "target_index"],
            kind="stable",
            ignore_index=True,
        )
    if not reference_edges.empty:
        reference_edges = reference_edges.sort_values(
            ["dataset_id", "source_index", "target_index"],
            kind="stable",
            ignore_index=True,
        )

    graph_dir = (
        project_root
        / "data"
        / "outputs"
        / "causal-comparison"
        / dataset_key
        / "graphs"
    )
    edge_table_path = graph_dir / "inferred_edges.csv"
    status_table_path = graph_dir / "run_status.csv"
    reference_edges_path = graph_dir / "reference_edges.csv"
    _atomic_write_csv(edge_table, edge_table_path)
    _atomic_write_csv(status_table, status_table_path)
    _atomic_write_csv(reference_edges, reference_edges_path)
    return GraphArtifacts(
        edge_table=edge_table,
        status_table=status_table,
        reference_edges=reference_edges,
        edge_table_path=edge_table_path,
        status_table_path=status_table_path,
        reference_edges_path=reference_edges_path,
    )


def score_dataset_outputs(
    project_root: Path,
    dataset_key: str,
    *,
    methods: Sequence[str] = METHODS,
) -> ScoreArtifacts:
    """Score all method checkpoints for one frozen dataset collection.

    Missing and failed fits receive explicit run rows.  Success-only means and
    sample standard deviations are accompanied by attempted/completed/failed
    counts and a failure-aware F1 in which every unsuccessful fit contributes
    zero.
    """

    project_root = Path(project_root).resolve()
    requested_methods = tuple(methods)
    unknown = sorted(set(requested_methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown comparison methods: {unknown}.")
    if len(requested_methods) != len(set(requested_methods)):
        raise ValueError("methods must not contain duplicates.")

    dataset_dir = (
        project_root
        / "data"
        / "shared"
        / "causal-comparison"
        / dataset_key
        / "datasets"
    )
    bundle_paths = sorted(dataset_dir.glob("*.npz"))
    if not bundle_paths:
        raise FileNotFoundError(
            f"No frozen {dataset_key} bundles found in {dataset_dir}."
        )

    rows: list[dict[str, Any]] = []
    for bundle_path in bundle_paths:
        metadata_path = bundle_path.with_suffix(".json")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing metadata for {bundle_path}.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(bundle_path, allow_pickle=False) as stored:
            series = np.asarray(stored["series"], dtype=float)
            if series.ndim != 2:
                raise ValueError(f"series must be two-dimensional in {bundle_path}.")
            if "lag_truth" in stored.files:
                lag_truth = np.asarray(stored["lag_truth"], dtype=bool)
                aggregate_truth = aggregate_lags(lag_truth, binary=True)
            elif "aggregate_truth" in stored.files:
                lag_truth = None
                aggregate_truth = np.asarray(stored["aggregate_truth"], dtype=bool)
            else:
                raise KeyError(f"No supported truth array in {bundle_path}.")

        dataset_id = bundle_path.stem
        if metadata.get("dataset_id") != dataset_id:
            raise ValueError(f"Dataset identifier mismatch for {bundle_path}.")
        observed_digest = _series_digest(series)
        recorded_digest = metadata.get("data_digest", metadata.get("digest"))
        if observed_digest != recorded_digest:
            raise ValueError(f"Dataset digest mismatch for {bundle_path}.")

        n_variables, n_samples = map(int, series.shape)
        if aggregate_truth.shape != (n_variables, n_variables):
            raise ValueError(f"Aggregate truth has the wrong shape in {bundle_path}.")
        default_order = 1 if lag_truth is None else int(lag_truth.shape[0])
        fit_order = int(metadata.get("fit_order", default_order))
        if lag_truth is not None and lag_truth.shape != (
            fit_order,
            n_variables,
            n_variables,
        ):
            raise ValueError(f"Lag truth has the wrong shape in {bundle_path}.")
        noise = str(metadata.get("noise", ""))
        regime = str(metadata.get("regime", ""))
        seed: int | str = metadata.get("seed", metadata.get("dataset_index", ""))

        for method in requested_methods:
            expected_estimands = _expected_estimands(
                method,
                exact_truth=lag_truth is not None,
            )
            checkpoint_path = (
                project_root
                / "data"
                / "outputs"
                / "causal-comparison"
                / dataset_key
                / method
                / "checkpoints"
                / f"{dataset_id}.json"
            )
            if not checkpoint_path.is_file():
                for estimand in expected_estimands:
                    rows.append(
                        {
                            **_base_row(
                                dataset_key=dataset_key,
                                dataset_id=dataset_id,
                                method=method,
                                estimand=estimand,
                                status="missing",
                                n_variables=n_variables,
                                n_samples=n_samples,
                                fit_order=fit_order,
                                noise=noise,
                                regime=regime,
                                seed=seed,
                                error="checkpoint not found",
                            ),
                            **_empty_metric_values(),
                            "failure_aware_f1": np.nan,
                        }
                    )
                continue

            checkpoint = read_checkpoint(checkpoint_path)
            _validate_checkpoint(
                checkpoint,
                dataset_key=dataset_key,
                dataset_id=dataset_id,
                method=method,
                dataset_digest=observed_digest,
                fit_order=fit_order,
            )
            if checkpoint.get("status") != "ok":
                status = str(checkpoint.get("status", "failed"))
                error = str(checkpoint.get("error", "fit did not complete"))
                for estimand in expected_estimands:
                    rows.append(
                        {
                            **_base_row(
                                dataset_key=dataset_key,
                                dataset_id=dataset_id,
                                method=method,
                                estimand=estimand,
                                status=status,
                                n_variables=n_variables,
                                n_samples=n_samples,
                                fit_order=fit_order,
                                noise=noise,
                                regime=regime,
                                seed=seed,
                                error=error,
                            ),
                            **_empty_metric_values(),
                            "failure_aware_f1": 0.0,
                        }
                    )
                continue

            native = _load_native_output(project_root, checkpoint)
            normalized = normalize_native_output(
                method,
                native,
                fit_order=fit_order,
                n_variables=n_variables,
            )
            by_estimand = {result.estimand: result for result in normalized}
            if set(expected_estimands) - set(by_estimand):
                raise ValueError(
                    f"{method} did not return all expected estimands for {dataset_id}."
                )
            for estimand in expected_estimands:
                result = by_estimand[estimand]
                truth = lag_truth if estimand == "exact-lag" else aggregate_truth
                if truth is None:
                    raise AssertionError("Exact-lag scoring requires lag truth.")
                metrics = graph_metrics(
                    truth,
                    result.selected,
                    scores=result.scores,
                    scope="cross",
                )
                rows.append(
                    {
                        **_base_row(
                            dataset_key=dataset_key,
                            dataset_id=dataset_id,
                            method=method,
                            estimand=estimand,
                            status="ok",
                            n_variables=n_variables,
                            n_samples=n_samples,
                            fit_order=fit_order,
                            noise=noise,
                            regime=regime,
                            seed=seed,
                            error="",
                        ),
                        **metrics,
                        "failure_aware_f1": metrics["f1"],
                    }
                )

    run_metrics = pd.DataFrame(rows).sort_values(
        ["dataset_id", "method", "estimand"],
        kind="stable",
    )
    summary = _summarize(run_metrics)
    score_dir = (
        project_root
        / "data"
        / "outputs"
        / "causal-comparison"
        / dataset_key
        / "scores"
    )
    run_metrics_path = score_dir / "run_metrics.csv"
    summary_path = score_dir / "summary_mean_std.csv"
    _atomic_write_csv(run_metrics, run_metrics_path)
    _atomic_write_csv(summary, summary_path)
    return ScoreArtifacts(
        run_metrics=run_metrics,
        summary=summary,
        run_metrics_path=run_metrics_path,
        summary_path=summary_path,
    )


__all__ = [
    "GraphArtifacts",
    "METHODS",
    "NormalizedOutput",
    "ScoreArtifacts",
    "collect_dataset_graphs",
    "normalize_native_output",
    "score_dataset_outputs",
]
