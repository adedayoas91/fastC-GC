"""Load comparison estimators from verified upstream checkouts."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import subprocess
import sys
from typing import Any


CAUSALIZED_GC_URL = "https://github.com/adedayoas91/causalized-GC.git"
CAUSALIZED_GC_COMMIT = "575ac1dc67a0282a5ca88bc11a2804df052f674d"


@dataclass(frozen=True, slots=True)
class UpstreamRevision:
    """Verified source identity for one external method checkout."""

    root: Path
    remote_url: str
    commit: str


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


def load_conventional_cgc(
    root: Path,
) -> tuple[type[Any], UpstreamRevision]:
    """Load the authors' GcStar class without copying estimator code."""

    revision = verify_upstream_revision(
        root,
        expected_url=CAUSALIZED_GC_URL,
        expected_commit=CAUSALIZED_GC_COMMIT,
    )
    source_path = revision.root / "src" / "d_CSL.py"
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing upstream estimator source: {source_path}.")

    module_name = (
        "fastcgc_external_causalized_gc_"
        f"{revision.commit[:12]}"
    )
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, source_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load upstream module from {source_path}.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise

    gcstar = getattr(module, "GcStar", None)
    if gcstar is None:
        raise ImportError(f"Upstream module {source_path} does not define GcStar.")
    return gcstar, revision


__all__ = [
    "CAUSALIZED_GC_COMMIT",
    "CAUSALIZED_GC_URL",
    "UpstreamRevision",
    "load_conventional_cgc",
    "verify_upstream_revision",
]
