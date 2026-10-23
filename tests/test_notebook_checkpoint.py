from __future__ import annotations

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


def _record(status: str = "ok") -> dict[str, object]:
    return {
        "method": "cgc",
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
        method="cgc",
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
