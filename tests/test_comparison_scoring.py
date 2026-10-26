"""Tests for native-output conversion and comparison summaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fastCGC.experiments.comparison_scoring import (
    collect_dataset_graphs,
    normalize_native_output,
    score_dataset_outputs,
)
from fastCGC.experiments.notebook_checkpoint import (
    configuration_hash,
    write_checkpoint,
    write_npz_checkpoint,
)


def test_fastcgc_native_output_has_exact_and_aggregate_estimands() -> None:
    lag_adjacency = np.zeros((2, 3, 3), dtype=bool)
    lag_adjacency[0, 0, 1] = True
    lag_adjacency[1, 2, 1] = True
    lag_scores = np.linspace(0.0, 1.0, lag_adjacency.size).reshape(
        lag_adjacency.shape
    )

    normalized = normalize_native_output(
        "fast-cgc",
        {
            "lag_adjacency": lag_adjacency,
            "lag_scores": lag_scores,
        },
        fit_order=2,
        n_variables=3,
    )

    assert [result.estimand for result in normalized] == [
        "exact-lag",
        "aggregate-any-lag",
    ]
    np.testing.assert_array_equal(normalized[0].selected, lag_adjacency)
    np.testing.assert_array_equal(
        normalized[1].selected,
        np.any(lag_adjacency, axis=0),
    )


def test_fastcgc_native_output_can_be_rethresholded_from_components() -> None:
    adjacency = np.zeros((1, 2, 2), dtype=bool)
    scores = np.full((1, 2, 2), 0.2)
    marginal = np.ones((1, 2, 2))
    conditional = np.ones((1, 2, 2))
    marginal[0, 0, 1] = 0.009
    conditional[0, 0, 1] = 0.004

    normalized = normalize_native_output(
        "fast-cgc",
        {
            "lag_adjacency": adjacency,
            "lag_scores": scores,
            "lag_marginal_pvalues": marginal,
            "lag_conditional_pvalues": conditional,
        },
        fit_order=1,
        n_variables=2,
        decision_alpha=0.01,
        decision_beta=0.005,
    )

    assert normalized[0].selected[0, 0, 1]
    assert normalized[1].selected[0, 1]


def test_fastcgc_threshold_override_rejects_legacy_joint_only_archive() -> None:
    with pytest.raises(ValueError, match="separately archived"):
        normalize_native_output(
            "fast-cgc",
            {
                "lag_adjacency": np.zeros((1, 2, 2), dtype=bool),
                "lag_scores": np.zeros((1, 2, 2)),
                "lag_pvalues": np.ones((1, 2, 2)),
            },
            fit_order=1,
            n_variables=2,
            decision_alpha=0.01,
            decision_beta=0.005,
        )


def test_pcmci_native_output_uses_source_target_lag_orientation() -> None:
    graph = np.full((3, 3, 3), "", dtype="<U3")
    graph[0, 1, 1] = "-->"
    graph[2, 1, 2] = "-->"
    p_matrix = np.ones((3, 3, 3), dtype=float)
    p_matrix[0, 1, 1] = 0.01
    p_matrix[2, 1, 2] = 0.02

    normalized = normalize_native_output(
        "pcmci-plus",
        {
            "graph": graph,
            "val_matrix": np.zeros_like(p_matrix),
            "p_matrix": p_matrix,
        },
        fit_order=2,
        n_variables=3,
    )

    exact = normalized[0]
    assert exact.selected.shape == (2, 3, 3)
    assert exact.selected[0, 0, 1]
    assert exact.selected[1, 2, 1]
    assert exact.scores[0, 0, 1] == 0.99


def test_bagged_pcmci_reconstructs_directed_edge_frequency() -> None:
    graph = np.full((2, 2, 2), "", dtype="<U3")
    graph[0, 1, 1] = "-->"
    frequency = np.ones((2, 2, 2), dtype=float)
    frequency[0, 1, 1] = 0.8
    frequency[1, 0, 1] = 0.8

    normalized = normalize_native_output(
        "bagged-pcmci-plus",
        {
            "most_frequent_links": graph,
            "link_frequency": frequency,
            "val_matrix_mean": np.zeros_like(frequency),
            "val_matrix_interval": np.zeros((*frequency.shape, 2)),
        },
        fit_order=1,
        n_variables=2,
    )

    assert normalized[0].selected[0, 0, 1]
    assert normalized[0].scores[0, 0, 1] == 0.8
    assert not normalized[0].selected[0, 1, 0]
    assert np.isclose(normalized[0].scores[0, 1, 0], 0.2)


def test_bagged_pcmci_rejects_nonbinary_positive_lag_marks() -> None:
    graph = np.full((2, 2, 2), "", dtype="<U3")
    graph[0, 1, 1] = "o-o"
    frequency = np.ones((2, 2, 2), dtype=float)

    with pytest.raises(ValueError, match="directed-edge frequency"):
        normalize_native_output(
            "bagged-pcmci-plus",
            {
                "most_frequent_links": graph,
                "link_frequency": frequency,
                "val_matrix_mean": np.zeros_like(frequency),
                "val_matrix_interval": np.zeros((*frequency.shape, 2)),
            },
            fit_order=1,
            n_variables=2,
        )


def test_cedar_uses_final_binary_graph_without_invalid_ranking_scores() -> None:
    graph = np.zeros((3, 3, 3), dtype=np.int8)
    graph[0, 1, 1] = 1
    graph[2, 1, 2] = 1
    # These fields are deliberately contradictory: they are archived for
    # provenance but do not represent post-pruning edge confidence.
    pvalues = np.zeros_like(graph, dtype=float)
    lag_pvalues = np.zeros_like(graph, dtype=float)

    normalized = normalize_native_output(
        "cedar",
        {
            "cg_tig": graph,
            "pvalue_matrix": pvalues,
            "lag_pvalues": lag_pvalues,
        },
        fit_order=2,
        n_variables=3,
    )

    assert [result.estimand for result in normalized] == [
        "exact-lag",
        "aggregate-any-lag",
    ]
    assert normalized[0].selected[0, 0, 1]
    assert normalized[0].selected[1, 2, 1]
    assert normalized[0].scores is None
    assert normalized[1].selected[0, 1]
    assert normalized[1].selected[2, 1]
    assert normalized[1].scores is None


def test_cedar_rejects_nonbinary_native_graph() -> None:
    graph = np.zeros((2, 2, 2), dtype=np.int8)
    graph[0, 1, 1] = 2

    with pytest.raises(ValueError, match="cg_tig must be binary"):
        normalize_native_output(
            "cedar",
            {"cg_tig": graph},
            fit_order=1,
            n_variables=2,
        )


def test_conventional_cgc_recovers_exact_lags_and_cuts_is_aggregate_only() -> None:
    n_variables = 3
    fit_order = 2
    p_corr = np.ones(((fit_order + 1) * n_variables, n_variables))
    p_cond = np.ones_like(p_corr)
    p_corr[n_variables + 0, 1] = 0.01
    p_cond[n_variables + 0, 1] = 0.001
    conn_mat = np.zeros((n_variables, n_variables))
    conn_mat[0, 1] = 0.4

    conventional = normalize_native_output(
        "cgc",
        {
            "conn_mat": conn_mat,
            "corr": np.zeros_like(p_corr),
            "pvalue_corr": p_corr,
            "conditional_corr": np.zeros_like(p_corr),
            "pvalue_conditional": p_cond,
        },
        fit_order=fit_order,
        n_variables=n_variables,
    )
    cuts = normalize_native_output(
        "cuts-plus",
        {"graph": np.array([[0.0, 0.8], [0.2, 0.0]])},
        fit_order=1,
        n_variables=2,
    )

    assert [result.estimand for result in conventional] == [
        "exact-lag",
        "aggregate-any-lag",
    ]
    assert conventional[0].selected[0, 0, 1]
    assert conventional[0].scores[0, 0, 1] == 0.99
    assert conventional[1].selected[0, 1]
    assert conventional[1].scores[0, 1] == 0.99
    assert [result.estimand for result in cuts] == ["aggregate-any-lag"]
    assert cuts[0].selected[0, 1]
    assert not cuts[0].selected[1, 0]


def _write_fastcgc_fixture(
    project_root: Path,
    *,
    seed: int,
    selected_edge: bool,
    regime: str = "reference",
) -> None:
    dataset_key = "sparseVAR"
    dataset_id = f"var3-t20-p1-gaussian-seed{seed}"
    dataset_dir = (
        project_root
        / "data"
        / "shared"
        / "causal-comparison"
        / dataset_key
        / "datasets"
    )
    dataset_dir.mkdir(parents=True, exist_ok=True)
    series = np.arange(60, dtype=float).reshape(3, 20) + seed
    lag_truth = np.zeros((1, 3, 3), dtype=bool)
    lag_truth[0, 0, 1] = True
    digest = __import__("hashlib").sha256(
        np.ascontiguousarray(series, dtype=np.float64).tobytes()
    ).hexdigest()
    np.savez_compressed(
        dataset_dir / f"{dataset_id}.npz",
        series=series,
        coefficients=np.zeros((1, 3, 3)),
        lag_truth=lag_truth,
    )
    (dataset_dir / f"{dataset_id}.json").write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "data_digest": digest,
                "seed": seed,
                "noise": "gaussian",
                "regime": regime,
                "series_layout": "variable, time",
                "truth_layout": "lag, source, target",
            }
        ),
        encoding="utf-8",
    )

    output_dir = (
        project_root
        / "data"
        / "outputs"
        / "causal-comparison"
        / dataset_key
        / "fast-cgc"
    )
    adjacency = np.zeros_like(lag_truth)
    adjacency[0, 0, 1] = selected_edge
    scores = np.full(lag_truth.shape, 0.1)
    scores[0, 0, 1] = 0.9 if selected_edge else 0.2
    native_path = output_dir / "native" / f"{dataset_id}.npz"
    native_digest = write_npz_checkpoint(
        native_path,
        {"lag_adjacency": adjacency, "lag_scores": scores},
    )
    config_hash = configuration_hash(
        {
            "dataset": dataset_key,
            "method": "fast-cgc",
            "master_seed": 42,
            "alpha": 0.01,
            "beta": 0.001,
            "n_permutations": 1000,
            "bootstrap_samples": 200,
            "truth_scope": "exact positive-lag graph",
        }
    )
    write_checkpoint(
        output_dir / "checkpoints" / f"{dataset_id}.json",
        {
            "method": "fast-cgc",
            "dataset": dataset_key,
            "dataset_id": dataset_id,
            "dataset_digest": digest,
            "fit_order": 1,
            "config_hash": config_hash,
            "upstream_commit": "local-source:test",
            "status": "ok",
            "algorithm_seed": seed,
            "native_output": str(native_path.relative_to(project_root)),
            "native_output_sha256": native_digest,
            "error": "",
        },
    )


def test_dataset_scoring_writes_run_rows_and_mean_standard_deviation(
    tmp_path,
) -> None:
    _write_fastcgc_fixture(tmp_path, seed=10, selected_edge=True)
    _write_fastcgc_fixture(tmp_path, seed=11, selected_edge=False)

    result = score_dataset_outputs(
        tmp_path,
        "sparseVAR",
        methods=("fast-cgc", "cuts-plus"),
    )

    assert result.run_metrics_path.is_file()
    assert result.summary_path.is_file()
    exact = result.summary[
        (result.summary["method"] == "fast-cgc")
        & (result.summary["estimand"] == "exact-lag")
    ].iloc[0]
    assert exact["attempted"] == 2
    assert exact["completed"] == 2
    assert exact["f1_mean"] == 0.5
    assert np.isclose(exact["f1_std"], np.sqrt(0.5))
    assert exact["f1_mean_std"] == "0.5000 (0.7071)"
    assert exact["regime"] == "reference"

    missing = result.summary[
        (result.summary["method"] == "cuts-plus")
        & (result.summary["estimand"] == "aggregate-any-lag")
    ].iloc[0]
    assert missing["expected"] == 2
    assert missing["missing"] == 2
    assert missing["attempted"] == 0
    assert missing["completed"] == 0
    assert missing["failed"] == 0
    assert missing["failure_aware_f1_n"] == 0


def test_dataset_scoring_keeps_simulation_regimes_separate(tmp_path) -> None:
    _write_fastcgc_fixture(
        tmp_path,
        seed=20,
        selected_edge=True,
        regime="reference",
    )
    _write_fastcgc_fixture(
        tmp_path,
        seed=21,
        selected_edge=True,
        regime="autocorrelation-0.95",
    )

    result = score_dataset_outputs(
        tmp_path,
        "sparseVAR",
        methods=("fast-cgc",),
    )
    exact = result.summary[result.summary["estimand"] == "exact-lag"]

    assert set(exact["regime"]) == {
        "reference",
        "autocorrelation-0.95",
    }
    assert set(exact["expected"]) == {1}


def _write_empirical_fixture(
    project_root: Path,
    *,
    include_reference: bool,
) -> None:
    dataset_key = "measurements"
    dataset_id = "measured-system-1"
    dataset_dir = (
        project_root
        / "data"
        / "shared"
        / "causal-comparison"
        / dataset_key
        / "datasets"
    )
    dataset_dir.mkdir(parents=True, exist_ok=True)
    series = np.arange(60, dtype=float).reshape(3, 20)
    digest = __import__("hashlib").sha256(
        np.ascontiguousarray(series, dtype=np.float64).tobytes()
    ).hexdigest()
    arrays = {"series": series}
    if include_reference:
        reference = np.zeros((3, 3), dtype=bool)
        reference[2, 1] = True
        arrays["reference_graph"] = reference
    np.savez_compressed(dataset_dir / f"{dataset_id}.npz", **arrays)
    (dataset_dir / f"{dataset_id}.json").write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "data_digest": digest,
                "fit_order": 1,
                "regime": "measured",
                "variable_names": ["upstream", "middle", "downstream"],
                "series_layout": "variable, time",
            }
        ),
        encoding="utf-8",
    )

    output_dir = (
        project_root
        / "data"
        / "outputs"
        / "causal-comparison"
        / dataset_key
        / "fast-cgc"
    )
    adjacency = np.zeros((1, 3, 3), dtype=bool)
    adjacency[0, 0, 1] = True
    scores = np.zeros((1, 3, 3), dtype=float)
    scores[0, 0, 1] = 0.75
    native_path = output_dir / "native" / f"{dataset_id}.npz"
    native_digest = write_npz_checkpoint(
        native_path,
        {"lag_adjacency": adjacency, "lag_scores": scores},
    )
    write_checkpoint(
        output_dir / "checkpoints" / f"{dataset_id}.json",
        {
            "method": "fast-cgc",
            "dataset": dataset_key,
            "dataset_id": dataset_id,
            "dataset_digest": digest,
            "fit_order": 1,
            "status": "ok",
            "algorithm_seed": 42,
            "upstream_commit": "local-source:test",
            "native_output": str(native_path.relative_to(project_root)),
            "native_output_sha256": native_digest,
            "error": "",
        },
    )


def test_empirical_collector_needs_no_ground_truth(tmp_path) -> None:
    _write_empirical_fixture(tmp_path, include_reference=False)

    result = collect_dataset_graphs(
        tmp_path,
        "measurements",
        methods=("fast-cgc", "cuts-plus"),
    )

    assert result.edge_table_path.is_file()
    assert result.status_table_path.is_file()
    assert result.reference_edges_path.is_file()
    assert set(result.status_table["status"]) == {"ok", "missing"}
    edge = result.edge_table.iloc[0]
    assert edge["method"] == "fast-cgc"
    assert edge["source"] == "upstream"
    assert edge["target"] == "middle"
    assert edge["score"] == 0.75
    assert result.reference_edges.empty
    assert not {
        "auroc",
        "auprc",
        "f1",
        "precision",
        "recall",
    }.intersection(result.edge_table.columns)


def test_empirical_collector_exports_reference_network_separately(tmp_path) -> None:
    _write_empirical_fixture(tmp_path, include_reference=True)

    result = collect_dataset_graphs(
        tmp_path,
        "measurements",
        methods=("fast-cgc",),
    )

    reference = result.reference_edges.iloc[0]
    assert reference["source"] == "downstream"
    assert reference["target"] == "middle"
    assert "method" not in result.reference_edges.columns
