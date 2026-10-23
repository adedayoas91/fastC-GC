"""Tests for the dataset-first causal-comparison notebook matrix."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks"

DATASETS = ("sparseVAR", "lorenz96", "dream3")
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
    for dataset in DATASETS:
        dataset_root = NOTEBOOK_ROOT / dataset
        assert (dataset_root / "00-prepare-data.ipynb").is_file()
        assert {path.name for path in dataset_root.glob("*.ipynb")} == {
            "00-prepare-data.ipynb",
            "90-collect-and-score.ipynb",
            *METHOD_NOTEBOOKS,
        }


def test_each_method_notebook_has_dataset_and_resume_contract() -> None:
    for dataset in DATASETS:
        for filename, method in METHOD_NOTEBOOKS.items():
            path = NOTEBOOK_ROOT / dataset / filename
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


def test_external_notebooks_call_pinned_author_checkouts() -> None:
    expectations = {
        "c-GC.ipynb": "load_conventional_cgc",
        "c-GC-star.ipynb": "load_conventional_cgc",
        "PCMCI-plus.ipynb": "run_pcmciplus",
        "Bagged-PCMCI-plus.ipynb": "run_bootstrap_of",
        "CUTS-plus.ipynb": "cuts_plus.main",
    }
    for dataset in DATASETS:
        for filename, native_entry_point in expectations.items():
            notebook = json.loads(
                (NOTEBOOK_ROOT / dataset / filename).read_text(encoding="utf-8")
            )
            source = _source(notebook)
            assert native_entry_point in source
            assert "verify_upstream_revision" in source or (
                native_entry_point == "load_conventional_cgc"
            )


def test_pending_mssd_notebook_is_not_fabricated() -> None:
    assert not list(NOTEBOOK_ROOT.glob("*/MSSD*.ipynb"))


def test_scoring_notebooks_use_shared_tested_scoring_code() -> None:
    for dataset in DATASETS:
        path = NOTEBOOK_ROOT / dataset / "90-collect-and-score.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = _source(notebook)

        assert f'DATASET_KEY = "{dataset}"' in source
        assert "score_dataset_outputs" in source
        assert "mean_std" in source


def test_generated_notebook_code_cells_are_valid_python() -> None:
    for dataset in DATASETS:
        for path in sorted((NOTEBOOK_ROOT / dataset).glob("*.ipynb")):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for index, cell in enumerate(notebook["cells"]):
                if cell.get("cell_type") != "code":
                    continue
                compile(
                    "".join(cell.get("source", [])),
                    f"{path.name}:cell-{index}",
                    "exec",
                )
