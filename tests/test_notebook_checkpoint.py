from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tomllib

import numpy as np
import pytest

from fastCGC.experiments import upstream as upstream_module
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
    CAUSALTS_COMMIT,
    CAUSALTS_URL,
    CAUSALTS_VERSION,
    CAUSALIZED_GC_COMMIT,
    CAUSALIZED_GC_URL,
    load_installed_cedar,
    verify_installed_distribution,
    verify_installed_vcs_distribution,
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
    assert CAUSALTS_URL == "https://github.com/bloomberg/causal-ts.git"
    assert CAUSALTS_COMMIT == "0d1fae061b2445a51d0c8b75fc565d897dd54fb7"
    assert CAUSALTS_VERSION == "0.27.0"


def test_project_requires_python_314() -> None:
    project = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert (PROJECT_ROOT / ".python-version").read_text().strip() == "3.14"
    assert project["project"]["requires-python"] == ">=3.14,<3.15"


def test_comparison_extra_pins_installable_upstream_packages() -> None:
    project = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert set(project["project"]["optional-dependencies"]["comparison"]) == {
        "causalchamber==0.2.8",
        "tigramite==5.2.10.1",
    }
    assert set(
        project["project"]["optional-dependencies"]["conventional-cgc"]
    ) == {
        "cdt==0.6.0",
        "gc-extension @ git+https://github.com/adedayoas91/causalized-GC.git@575ac1dc67a0282a5ca88bc11a2804df052f674d",
        "numba==0.68.0",
        "torch==2.14.1",
    }
    assert set(project["project"]["optional-dependencies"]["cedar"]) == {
        "causalts @ git+https://github.com/bloomberg/causal-ts.git@0d1fae061b2445a51d0c8b75fc565d897dd54fb7",
    }


def test_notebook_extra_provides_the_jupyter_kernel() -> None:
    project = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert set(project["project"]["optional-dependencies"]["notebooks"]) == {
        "ipykernel==7.4.0",
    }


def test_installed_distribution_revision_is_versioned() -> None:
    assert verify_installed_distribution(
        "pytest",
        expected_version=pytest.__version__,
    ) == f"pytest=={pytest.__version__}"

    with pytest.raises(RuntimeError, match="expected 0.0.0"):
        verify_installed_distribution("pytest", expected_version="0.0.0")


def test_installed_git_distribution_verifies_pep610_provenance(
    tmp_path,
    monkeypatch,
) -> None:
    class FakeDistribution:
        def read_text(self, filename: str) -> str | None:
            assert filename == "direct_url.json"
            return json.dumps(
                {
                    "url": CAUSALIZED_GC_URL,
                    "vcs_info": {
                        "vcs": "git",
                        "commit_id": CAUSALIZED_GC_COMMIT,
                    },
                }
            )

        def locate_file(self, path: str) -> Path:
            assert path == ""
            return tmp_path

    monkeypatch.setattr(
        upstream_module,
        "installed_distribution",
        lambda name: FakeDistribution(),
    )
    revision = verify_installed_vcs_distribution(
        "gc-extension",
        expected_url=CAUSALIZED_GC_URL,
        expected_commit=CAUSALIZED_GC_COMMIT,
        install_extra="conventional-cgc",
    )

    assert revision.root == tmp_path
    assert revision.remote_url == CAUSALIZED_GC_URL
    assert revision.commit == CAUSALIZED_GC_COMMIT


def test_cedar_loader_verifies_import_root_and_returns_native_api(
    tmp_path,
    monkeypatch,
) -> None:
    class FakeDistribution:
        def read_text(self, filename: str) -> str | None:
            assert filename == "direct_url.json"
            return json.dumps(
                {
                    "url": CAUSALTS_URL,
                    "vcs_info": {
                        "vcs": "git",
                        "commit_id": CAUSALTS_COMMIT,
                    },
                }
            )

        def locate_file(self, path: str) -> Path:
            assert path == ""
            return tmp_path

    run_cedar = object()
    parcorr_gpu = type("ParCorrGPU", (), {})
    modules = {
        "causalts.cedar.discovery": SimpleNamespace(
            __file__=tmp_path / "causalts" / "cedar" / "discovery.py",
            run_cedar=run_cedar,
        ),
        "causalts.ci_tests": SimpleNamespace(ParCorrGPU=parcorr_gpu),
    }
    monkeypatch.setattr(
        upstream_module,
        "installed_distribution",
        lambda name: FakeDistribution(),
    )
    monkeypatch.setattr(
        upstream_module,
        "version",
        lambda name: CAUSALTS_VERSION,
    )
    monkeypatch.setattr(
        upstream_module,
        "import_module",
        lambda name: modules[name],
    )

    observed_run, observed_ci, revision = load_installed_cedar()

    assert observed_run is run_cedar
    assert observed_ci is parcorr_gpu
    assert revision.commit == CAUSALTS_COMMIT
