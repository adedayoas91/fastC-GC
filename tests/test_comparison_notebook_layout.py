"""Tests for the dataset-first causal-comparison notebook matrix."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks"

SIMULATION_ROOTS = {
    "sparseVAR": NOTEBOOK_ROOT / "simulations" / "sparseVAR",
    "lorenz96": NOTEBOOK_ROOT / "simulations" / "lorenz96",
    "autocorrelatedSCM": (
        NOTEBOOK_ROOT / "simulations" / "autocorrelatedSCM"
    ),
    "dream3": NOTEBOOK_ROOT / "simulations" / "dream3",
}
EMPIRICAL_ROOTS = {
    "causalChamber": NOTEBOOK_ROOT / "empirical" / "causalChamber",
    "causalRivers": NOTEBOOK_ROOT / "empirical" / "causalRivers",
}
DATASET_ROOTS = {**SIMULATION_ROOTS, **EMPIRICAL_ROOTS}
METHOD_NOTEBOOKS = {
    "fastC-GC.ipynb": "fast-cgc",
    "fastC-GC-star.ipynb": "fast-cgc-star",
    "c-GC.ipynb": "cgc",
    "c-GC-star.ipynb": "cgc-star",
    "PCMCI-plus.ipynb": "pcmci-plus",
    "Bagged-PCMCI-plus.ipynb": "bagged-pcmci-plus",
    "CUTS-plus.ipynb": "cuts-plus",
}


def _source(notebook: dict[str, object]) -> str:
    cells = notebook["cells"]
    assert isinstance(cells, list)
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in cells
        if isinstance(cell, dict)
    )


def test_comparison_notebooks_are_grouped_by_dataset() -> None:
    for dataset_root in SIMULATION_ROOTS.values():
        assert (dataset_root / "00-prepare-data.ipynb").is_file()
        assert {path.name for path in dataset_root.glob("*.ipynb")} == {
            "00-prepare-data.ipynb",
            "90-collect-and-score.ipynb",
            *METHOD_NOTEBOOKS,
        }
    for dataset_root in EMPIRICAL_ROOTS.values():
        assert (dataset_root / "00-prepare-data.ipynb").is_file()
        assert {path.name for path in dataset_root.glob("*.ipynb")} == {
            "00-prepare-data.ipynb",
            "90-collect-and-visualize.ipynb",
            *METHOD_NOTEBOOKS,
        }


def test_each_method_notebook_has_dataset_and_resume_contract() -> None:
    for dataset, dataset_root in DATASET_ROOTS.items():
        for filename, method in METHOD_NOTEBOOKS.items():
            path = dataset_root / filename
            notebook = json.loads(path.read_text(encoding="utf-8"))
            source = _source(notebook)

            assert f'DATASET_KEY = "{dataset}"' in source
            assert f'METHOD_KEY = "{method}"' in source
            assert "RESUME = True" in source
            assert "MAX_NEW_FITS: int | None = None" in source
            assert "write_checkpoint" in source
            assert "write_npz_checkpoint" in source
            assert notebook["metadata"]["language_info"]["version"] == "3.14"
            assert all(
                not cell.get("outputs")
                for cell in notebook["cells"]
                if cell.get("cell_type") == "code"
            )


def test_repository_only_notebooks_call_pinned_author_checkouts() -> None:
    expectations = {
        "CUTS-plus.ipynb": "cuts_plus.main",
    }
    for dataset_root in DATASET_ROOTS.values():
        for filename, native_entry_point in expectations.items():
            notebook = json.loads(
                (dataset_root / filename).read_text(encoding="utf-8")
            )
            source = _source(notebook)
            assert native_entry_point in source
            assert "verify_upstream_revision" in source


def test_conventional_cgc_notebooks_use_installed_git_distribution() -> None:
    for dataset_root in DATASET_ROOTS.values():
        for filename in ("c-GC.ipynb", "c-GC-star.ipynb"):
            notebook = json.loads(
                (dataset_root / filename).read_text(encoding="utf-8")
            )
            source = _source(notebook)
            assert "load_installed_conventional_cgc" in source
            assert "CAUSALIZED_GC_ROOT" not in source
            assert "load_conventional_cgc(" not in source


def test_tigramite_notebooks_use_installed_distribution() -> None:
    expectations = {
        "PCMCI-plus.ipynb": "run_pcmciplus",
        "Bagged-PCMCI-plus.ipynb": "run_bootstrap_of",
    }
    for dataset_root in DATASET_ROOTS.values():
        for filename, native_entry_point in expectations.items():
            notebook = json.loads(
                (dataset_root / filename).read_text(encoding="utf-8")
            )
            source = _source(notebook)
            assert native_entry_point in source
            assert 'verify_installed_distribution(\n    "tigramite"' in source
            assert "TIGRAMITE_ROOT" not in source
            assert "verify_upstream_revision" not in source


def test_pending_mssd_notebook_is_not_fabricated() -> None:
    assert not list(NOTEBOOK_ROOT.glob("**/MSSD*.ipynb"))


def test_scoring_notebooks_use_shared_tested_scoring_code() -> None:
    for dataset, dataset_root in SIMULATION_ROOTS.items():
        path = dataset_root / "90-collect-and-score.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = _source(notebook)

        assert f'DATASET_KEY = "{dataset}"' in source
        assert "score_dataset_outputs" in source
        assert "mean_std" in source
        assert '"regime"' in source


def test_empirical_notebooks_collect_and_visualize_inferred_graphs() -> None:
    for dataset, dataset_root in EMPIRICAL_ROOTS.items():
        path = dataset_root / "90-collect-and-visualize.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = _source(notebook)

        assert f'DATASET_KEY = "{dataset}"' in source
        assert "collect_dataset_graphs" in source
        assert "plot_directed_graph" in source
        assert "edge_table" in source
        assert "reference_edges" in source
        assert "score_dataset_outputs" not in source
        assert "AUROC" not in source
        assert "AUPRC" not in source


def test_simulation_layout_is_dgp_first() -> None:
    simulation_root = NOTEBOOK_ROOT / "simulations"

    assert not (simulation_root / "linear").exists()
    assert not (simulation_root / "nonlinear").exists()
    assert {path.name for path in simulation_root.iterdir() if path.is_dir()} == {
        "sparseVAR",
        "lorenz96",
        "autocorrelatedSCM",
        "dream3",
    }


def test_empirical_layout_contains_two_real_measurement_studies() -> None:
    empirical_root = NOTEBOOK_ROOT / "empirical"

    assert {path.name for path in empirical_root.iterdir() if path.is_dir()} == {
        "causalChamber",
        "causalRivers",
    }


def test_autocorrelated_scm_uses_pinned_official_generator() -> None:
    path = SIMULATION_ROOTS["autocorrelatedSCM"] / "00-prepare-data.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = _source(notebook)

    assert "EyringMLClimateGroup/debeire24clear_Bagged_TimeSeries_Causality" in source
    assert "8e16fb89b8f3b4db47ff57f2bd09dcd5c9f7e4c9" in source
    assert "generate_random_contemp_model" in source
    assert "generate_nonlinear_contemp_timeseries" in source
    assert "contemp_fraction=0.0" in source
    assert "REPLICATES = 50" in source
    assert "MAX_NEW_DATASETS: int | None = None" in source
    assert '"cross-link-null"' in source
    assert '"regime"' in source


def test_empirical_preparation_uses_verified_official_sources() -> None:
    chamber = json.loads(
        (EMPIRICAL_ROOTS["causalChamber"] / "00-prepare-data.ipynb").read_text(
            encoding="utf-8"
        )
    )
    chamber_source = _source(chamber)
    assert "wt_walks_v1.zip" in chamber_source
    assert "1ef92d39fff7645a9e6e8cd41311e708362606491590a1534c6a5135d207736e" in chamber_source
    assert 'verify_installed_distribution(\n    "causalchamber"' in chamber_source
    assert "from causalchamber.ground_truth import graph" in chamber_source
    assert "CAUSAL_CHAMBER_PACKAGE_ROOT" not in chamber_source
    assert "reference_graph" in chamber_source

    rivers = json.loads(
        (EMPIRICAL_ROOTS["causalRivers"] / "00-prepare-data.ipynb").read_text(
            encoding="utf-8"
        )
    )
    rivers_source = _source(rivers)
    assert "product.zip" in rivers_source
    assert "4f779fa9222653af092046eacba39a955cc65638bfb70fb7e31a32d9d8373b55" in rivers_source
    assert "a30154fbb9a2e03bae133c643d7e978df63b3766" in rivers_source
    assert "get_all_subgraphs" in rivers_source
    assert "reference_graph" in rivers_source


def test_generated_notebook_code_cells_are_valid_python() -> None:
    for dataset_root in DATASET_ROOTS.values():
        for path in sorted(dataset_root.glob("*.ipynb")):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for index, cell in enumerate(notebook["cells"]):
                if cell.get("cell_type") != "code":
                    continue
                compile(
                    "".join(cell.get("source", [])),
                    f"{path.name}:cell-{index}",
                    "exec",
                )
