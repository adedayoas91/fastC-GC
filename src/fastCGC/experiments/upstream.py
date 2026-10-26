"""Load comparison estimators from verified packages or source checkouts."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import (
    PackageNotFoundError,
    distribution as installed_distribution,
    version,
)
import json
from pathlib import Path
import subprocess
from typing import Any


CAUSALIZED_GC_URL = "https://github.com/adedayoas91/causalized-GC.git"
CAUSALIZED_GC_COMMIT = "575ac1dc67a0282a5ca88bc11a2804df052f674d"
CAUSALIZED_GC_DISTRIBUTION = "gc-extension"
CAUSALTS_URL = "https://github.com/bloomberg/causal-ts.git"
CAUSALTS_COMMIT = "0d1fae061b2445a51d0c8b75fc565d897dd54fb7"
CAUSALTS_DISTRIBUTION = "causalts"
CAUSALTS_VERSION = "0.27.0"


@dataclass(frozen=True, slots=True)
class UpstreamRevision:
    """Verified source identity for one external method implementation."""

    root: Path
    remote_url: str
    commit: str


def verify_installed_distribution(
    distribution: str,
    *,
    expected_version: str,
) -> str:
    """Return a reproducible identifier for a pinned installed package."""

    try:
        observed_version = version(distribution)
    except PackageNotFoundError as error:
        raise ModuleNotFoundError(
            f"Install {distribution}=={expected_version} in the active "
            "environment before running this notebook. From the fastC-GC "
            "root, run `uv sync --extra comparison`."
        ) from error
    if observed_version != expected_version:
        raise RuntimeError(
            f"Installed {distribution} version is {observed_version}; "
            f"expected {expected_version}. Run `uv sync --extra comparison`."
        )
    return f"{distribution}=={observed_version}"


def _normalized_repository_url(url: str) -> str:
    return url.removeprefix("git+").rstrip("/").removesuffix(".git")


def verify_installed_vcs_distribution(
    distribution: str,
    *,
    expected_url: str,
    expected_commit: str,
    install_extra: str,
) -> UpstreamRevision:
    """Verify the PEP 610 provenance of an installed Git distribution."""

    try:
        package = installed_distribution(distribution)
    except PackageNotFoundError as error:
        raise ModuleNotFoundError(
            f"Install the pinned {distribution} Git distribution in the "
            "active environment. From the fastC-GC root, run "
            f"`uv sync --extra {install_extra}`."
        ) from error
    direct_url_text = package.read_text("direct_url.json")
    if direct_url_text is None:
        raise RuntimeError(
            f"Installed {distribution} has no PEP 610 source record; run "
            f"`uv sync --extra {install_extra}`."
        )
    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Installed {distribution} has an invalid PEP 610 source record."
        ) from error
    if not isinstance(direct_url, dict):
        raise RuntimeError(
            f"Installed {distribution} has an invalid PEP 610 source record."
        )
    remote_url = str(direct_url.get("url", ""))
    vcs_info = direct_url.get("vcs_info", {})
    commit = vcs_info.get("commit_id") if isinstance(vcs_info, dict) else None
    vcs = vcs_info.get("vcs") if isinstance(vcs_info, dict) else None
    if vcs != "git":
        raise RuntimeError(
            f"Installed {distribution} does not record a Git source."
        )
    if _normalized_repository_url(remote_url) != _normalized_repository_url(
        expected_url
    ):
        raise RuntimeError(
            f"Installed {distribution} came from {remote_url!r}; expected "
            f"{expected_url!r}."
        )
    if commit != expected_commit:
        raise RuntimeError(
            f"Installed {distribution} is at {commit!r}; expected "
            f"{expected_commit}."
        )
    root = Path(package.locate_file("")).resolve()
    return UpstreamRevision(root, remote_url, commit)


def _git_value(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        error = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Could not inspect upstream checkout {root}: {error}")
    return completed.stdout.strip()


def verify_upstream_revision(
    root: Path,
    *,
    expected_url: str,
    expected_commit: str,
) -> UpstreamRevision:
    """Reject a missing checkout or a repository at the wrong revision."""

    root = Path(root).expanduser().resolve()
    if not (root / ".git").is_dir():
        raise FileNotFoundError(
            f"Clone {expected_url} into {root} before running this notebook."
        )
    remote_url = _git_value(root, "remote", "get-url", "origin")
    commit = _git_value(root, "rev-parse", "HEAD")
    normalized_remote = remote_url.removesuffix("/")
    normalized_expected = expected_url.removesuffix("/")
    if normalized_remote != normalized_expected:
        raise RuntimeError(
            f"Unexpected upstream remote {remote_url!r}; expected {expected_url!r}."
        )
    if commit != expected_commit:
        raise RuntimeError(
            f"Upstream checkout is at {commit}; expected {expected_commit}."
        )
    return UpstreamRevision(root, remote_url, commit)


def load_installed_conventional_cgc() -> tuple[type[Any], UpstreamRevision]:
    """Load conventional c-GC from its verified installed Git distribution."""

    revision = verify_installed_vcs_distribution(
        CAUSALIZED_GC_DISTRIBUTION,
        expected_url=CAUSALIZED_GC_URL,
        expected_commit=CAUSALIZED_GC_COMMIT,
        install_extra="conventional-cgc",
    )
    module = import_module("d_CSL")
    gcstar = getattr(module, "GcStar", None)
    if gcstar is None:
        raise ImportError("Installed d_CSL module does not define GcStar.")
    return gcstar, revision


def load_installed_cedar() -> tuple[Any, type[Any], UpstreamRevision]:
    """Load CEDAR from the verified official CausalTS Git distribution."""

    revision = verify_installed_vcs_distribution(
        CAUSALTS_DISTRIBUTION,
        expected_url=CAUSALTS_URL,
        expected_commit=CAUSALTS_COMMIT,
        install_extra="cedar",
    )
    observed_version = version(CAUSALTS_DISTRIBUTION)
    if observed_version != CAUSALTS_VERSION:
        raise RuntimeError(
            f"Installed {CAUSALTS_DISTRIBUTION} version is {observed_version}; "
            f"expected {CAUSALTS_VERSION}. Run `uv sync --extra cedar`."
        )

    discovery = import_module("causalts.cedar.discovery")
    ci_tests = import_module("causalts.ci_tests")
    module_path = Path(discovery.__file__).resolve()
    expected_package_root = (revision.root / "causalts").resolve()
    if not module_path.is_relative_to(expected_package_root):
        raise ImportError(
            "Imported causalts is not the verified installed distribution: "
            f"{module_path}."
        )
    run_cedar = getattr(discovery, "run_cedar", None)
    parcorr_gpu = getattr(ci_tests, "ParCorrGPU", None)
    if run_cedar is None or parcorr_gpu is None:
        raise ImportError(
            "Installed causalts does not expose run_cedar and ParCorrGPU."
        )
    return run_cedar, parcorr_gpu, revision


__all__ = [
    "CAUSALTS_COMMIT",
    "CAUSALTS_DISTRIBUTION",
    "CAUSALTS_URL",
    "CAUSALTS_VERSION",
    "CAUSALIZED_GC_COMMIT",
    "CAUSALIZED_GC_DISTRIBUTION",
    "CAUSALIZED_GC_URL",
    "UpstreamRevision",
    "load_installed_cedar",
    "load_installed_conventional_cgc",
    "verify_installed_distribution",
    "verify_installed_vcs_distribution",
    "verify_upstream_revision",
]
