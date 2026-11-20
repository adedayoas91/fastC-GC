"""Tests for the runtime CLI's dependence-variant comparison."""

from __future__ import annotations

from argparse import Namespace
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest


@pytest.fixture
def runtime_cli() -> ModuleType:
    script = Path(__file__).resolve().parents[1] / "experiments/validation/run_rccp_runtime.py"
    spec = importlib.util.spec_from_file_location("rccp_runtime_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("variants", (None, "linear,rff-hsic-rcot"))
def test_runtime_modes_select_scoped_outputs_and_paired_jobs(
    runtime_cli: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    variants: str | None,
) -> None:
    jobs_by_seed = {}
    argv = ["runtime", "--variables", "5,10", "--samples", "50", "--orders", "1"]
    if variants is not None:
        argv.extend(("--dependence-variants", variants))

    def check_jobs(args: Namespace) -> int:
        expected_methods = ("cgc", "fast-cgc", "cgc-star", "fast-cgc-star")
        expected_directory = "rccp-runtime"
        expected_variants = [None]
        if variants is not None:
            expected_methods = ("fast-cgc", "fast-cgc-star")
            expected_directory = "rccp-dependence-runtime"
            expected_variants = variants.split(",")

        assert args.methods == expected_methods
        assert args.output_dir.name == expected_directory
        for job in runtime_cli._jobs(args):
            variant = job.get("dependence_variant")
            jobs_by_seed.setdefault(job["seed"], []).append((job["method"], variant))
            command = runtime_cli._worker_command(args, job, tmp_path / "worker.json")
            if variant is None:
                assert "--dependence-variant" not in command
            else:
                assert command[command.index("--dependence-variant") + 1] == variant

        expected_jobs = [(method, variant) for method in expected_methods for variant in expected_variants]
        assert jobs_by_seed == {40_000: expected_jobs, 40_001: expected_jobs}
        return 0

    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(runtime_cli, "_run", check_jobs)
    assert runtime_cli.main() == 0


def test_dependence_timings_use_only_matched_linear_baselines(runtime_cli: ModuleType) -> None:
    cell = {
        "n_variables": 5,
        "n_samples": 50,
        "order": 1,
        "seed": 40_000,
        "method": "fast-cgc",
        "data_digest": "matched-input",
    }
    timings = [
        ({**cell, "dependence_variant": "linear"}, (1.0, 3.0)),
        ({**cell, "dependence_variant": "rff-hsic-rcot"}, (4.0, 8.0)),
        ({**cell, "dependence_variant": "rcot", "data_digest": "different-input"}, (2.0, 6.0)),
        ({**cell, "dependence_variant": "rcot", "method": "fast-cgc-star"}, (3.0, 5.0)),
    ]
    rows = [
        {**identity, "wall_seconds": elapsed}
        for identity, repeats in timings
        for elapsed in repeats
    ]
    summary = runtime_cli._summarize_dependence_rows(rows)
    assert len(summary) == 4
    by_cell = {(row["dependence_variant"], row["method"], row["data_digest"]): row for row in summary}
    linear = by_cell["linear", "fast-cgc", "matched-input"]
    nonlinear = by_cell["rff-hsic-rcot", "fast-cgc", "matched-input"]

    assert linear["wall_time_ratio_vs_linear"] == 1.0
    assert nonlinear["repeats"] == 2
    assert nonlinear["mean_wall_seconds"] == 6.0
    assert nonlinear["sd_wall_seconds"] == pytest.approx(8.0**0.5)
    assert nonlinear["wall_time_ratio_vs_linear"] == 3.0
    assert (nonlinear["marginal_test"], nonlinear["conditional_test"]) == ("rff_hsic", "rcot")
    assert by_cell["rcot", "fast-cgc", "different-input"]["wall_time_ratio_vs_linear"] is None
    assert by_cell["rcot", "fast-cgc-star", "matched-input"]["wall_time_ratio_vs_linear"] is None
