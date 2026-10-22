"""Tests for the installable package layout and public import surface."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_package_uses_standard_src_layout() -> None:
    package_root = PROJECT_ROOT / "src" / "fastCGC"

    assert (package_root / "__init__.py").is_file()
    assert (package_root / "core" / "__init__.py").is_file()
    assert (package_root / "core" / "gc_star.py").is_file()
    assert (package_root / "core" / "fast_gc_star.py").is_file()
    assert (package_root / "dependence" / "__init__.py").is_file()
    assert (package_root / "experiments" / "__init__.py").is_file()
    assert not (PROJECT_ROOT / "src" / "__init__.py").exists()
    assert not (PROJECT_ROOT / "src" / "dependence").exists()
    assert not (PROJECT_ROOT / "src" / "experiments").exists()
    assert not (package_root / "causalised-GC.py").exists()

    project = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    package_discovery = project["tool"]["setuptools"]["packages"]["find"]
    assert package_discovery == {
        "where": ["src"],
        "include": ["fastCGC*"],
    }


def test_notebooks_have_a_dedicated_top_level_directory() -> None:
    notebook_root = PROJECT_ROOT / "notebooks" / "simulations"

    assert (notebook_root / "linear").is_dir()
    assert (notebook_root / "nonlinear").is_dir()
    assert len(list(notebook_root.glob("**/*.ipynb"))) == 10
    assert not list((PROJECT_ROOT / "experiments").glob("**/*.ipynb"))


def test_core_and_compatibility_imports_share_estimator_classes() -> None:
    package = import_module("fastCGC")
    core = import_module("fastCGC.core")
    compatibility = import_module("fastCGC.fast_causalised_GC")
    simulation = import_module("fastCGC.simulation")
    compatibility_utils = import_module("fastCGC.utils")

    assert package.GcStar is core.GcStar
    assert package.FastGcStar is core.FastGcStar
    assert compatibility.FastGcStar is core.FastGcStar
    assert core.GcStar.__module__ == "fastCGC.core.gc_star"
    assert core.FastGcStar.__module__ == "fastCGC.core.fast_gc_star"
    assert compatibility_utils.adj_mtx is simulation.adj_mtx
    assert (
        compatibility_utils.continuous_noise_fun
        is simulation.continuous_noise_fun
    )
