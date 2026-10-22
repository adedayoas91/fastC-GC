from __future__ import annotations

import json
from pathlib import Path
import tomllib

import numpy as np
import pytest

from fastCGC.experiments.notebook_checkpoint import (
    checkpoint_path,
    configuration_hash,
    derive_algorithm_seed,
    file_sha256,
    metric_rows,
    read_checkpoints,
    reusable_success,
    write_checkpoint,
    write_npz_checkpoint,
)
from fastCGC.experiments.upstream import (
    CAUSALIZED_GC_COMMIT,
    CAUSALIZED_GC_URL,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMULATION_NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks" / "simulations"
NOTEBOOK_ROOT = SIMULATION_NOTEBOOK_ROOT / "linear"


def _record(status: str = "ok") -> dict[str, object]:
    return {
        "method": "published-cgc",
        "scenario": "singleLag-Markovian",
        "trial_id": 0,
        "n_past": 1,
        "dataset_digest": "data-digest",
        "config_hash": "config-digest",
        "upstream_commit": CAUSALIZED_GC_COMMIT,
        "status": status,
        "data_seed": 42,
        "algorithm_seed": 123,
        "metrics": {"f1": 0.75},
        "native_output": "native/result.npz",
        "native_output_sha256": "native-digest",
        "error": "" if status == "ok" else "failure",
    }


def test_checkpoint_resume_requires_matching_success(tmp_path) -> None:
    path = checkpoint_path(tmp_path, trial_id=0, n_past=1)
    record = _record()
    write_checkpoint(path, record)

    expected = {
        key: record[key]
        for key in (
            "method",
            "scenario",
            "trial_id",
            "n_past",
            "dataset_digest",
            "config_hash",
            "upstream_commit",
        )
    }
    assert reusable_success(path, expected=expected) is not None

    stale = {**expected, "dataset_digest": "different"}
    with pytest.raises(ValueError, match="Stale checkpoint"):
        reusable_success(path, expected=stale)

    write_checkpoint(path, _record(status="failed"))
    assert reusable_success(path, expected=expected) is None


def test_checkpoint_collection_and_native_output_digest(tmp_path) -> None:
    path = checkpoint_path(tmp_path / "checkpoints", trial_id=0, n_past=1)
    write_checkpoint(path, _record())
    native = tmp_path / "native" / "result.npz"
    digest = write_npz_checkpoint(native, {"graph": np.eye(3)})

    assert digest == file_sha256(native)
    records = read_checkpoints(path.parent)
    rows = metric_rows(
        records,
        method="published-cgc",
        scenario="singleLag-Markovian",
        config_hash="config-digest",
        upstream_commit=CAUSALIZED_GC_COMMIT,
    )
    assert rows[0]["f1"] == 0.75
    assert "fit_seconds" not in rows[0]


def test_hashes_and_algorithm_seeds_are_stable() -> None:
    config = {"alpha": 0.01, "n_pasts": [1, 2]}

    assert configuration_hash(config) == configuration_hash(dict(reversed(
        list(config.items())
    )))
    assert derive_algorithm_seed(42, "cgc", 0, 1) == derive_algorithm_seed(
        42, "cgc", 0, 1
    )
    assert derive_algorithm_seed(42, "cgc", 0, 1) != derive_algorithm_seed(
        42, "cgc", 0, 2
    )


def test_conventional_notebooks_use_only_the_pinned_upstream_estimator() -> None:
    paths = sorted((NOTEBOOK_ROOT / "c-GC").glob("*.ipynb"))
    paths += sorted((NOTEBOOK_ROOT / "c-GC-star").glob("*.ipynb"))
    assert len(paths) == 8

    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
        )
        assert "FastGcStar as GcStar" not in source
        assert "load_published_cgc" in source
        assert "load_or_generate_trial_data" in source
        assert "reusable_success" in source
        assert CAUSALIZED_GC_URL.removesuffix(".git") in source
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert cell["outputs"] == []
                compile(
                    "".join(cell["source"]),
                    f"{path.name}:cell-{index}",
                    "exec",
                )


def test_all_simulation_notebooks_share_data_and_resume_contract() -> None:
    paths = sorted(SIMULATION_NOTEBOOK_ROOT.glob("**/*.ipynb"))
    assert len(paths) == 10

    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
        )
        assert "load_or_generate_trial_data" in source
        assert "dataset_digest" in source
        assert "reusable_success" in source
        assert "MAX_NEW_FITS" in source
        assert 'EXPECTED_PYTHON = (3, 14)' in source
        assert '"software_fingerprint": SOFTWARE_FINGERPRINT' in source
        assert "fit_seconds" not in source
        assert "runtime_fingerprint" not in source
        assert "runtime_provenance" not in source
        assert "EXPECTED_MACHINE" not in source
        assert "THREAD_LIMITS" not in source
        assert "timing.png" not in source
        assert notebook["metadata"]["kernelspec"] == {
            "display_name": "fastcgc (Python 3.14, Apple Silicon)",
            "language": "python",
            "name": "fastcgc-py314-arm64",
        }
        assert notebook["metadata"]["language_info"]["version"] == "3.14"
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert cell["outputs"] == []
                compile(
                    "".join(cell["source"]),
                    f"{path.name}:cell-{index}",
                    "exec",
                )


def test_notebook_loader_pins_upstream_revision() -> None:
    assert CAUSALIZED_GC_URL == (
        "https://github.com/adedayoas91/causalized-GC.git"
    )
    assert CAUSALIZED_GC_COMMIT == (
        "575ac1dc67a0282a5ca88bc11a2804df052f674d"
    )


def test_project_requires_python_314() -> None:
    project = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert (PROJECT_ROOT / ".python-version").read_text().strip() == "3.14"
    assert project["project"]["requires-python"] == ">=3.14,<3.15"
