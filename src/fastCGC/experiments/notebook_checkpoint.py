"""Crash-safe checkpoints for long-running comparison notebooks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


CHECKPOINT_FORMAT_VERSION = 1


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def configuration_hash(configuration: Mapping[str, Any]) -> str:
    """Hash the settings that determine one method fit."""

    encoded = json.dumps(
        _json_ready(configuration),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def derive_algorithm_seed(*components: Any) -> int:
    """Derive a stable uint32 seed from explicit run identifiers."""

    encoded = json.dumps(
        _json_ready(list(components)),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "big")


def checkpoint_path(
    root: Path,
    *,
    trial_id: int,
    n_past: int,
) -> Path:
    """Return one checkpoint path at the smallest resumable fit boundary."""

    if trial_id < 0 or n_past < 1:
        raise ValueError("trial_id must be non-negative and n_past positive.")
    return Path(root) / f"trial-{trial_id:05d}-past-{n_past:02d}.json"


def write_checkpoint(path: Path, record: Mapping[str, Any]) -> None:
    """Atomically replace a checkpoint after a fit or explicit failure."""

    payload = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        **_json_ready(record),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_npz_checkpoint(
    path: Path,
    arrays: Mapping[str, np.ndarray],
) -> str:
    """Atomically save native arrays and return the file SHA-256 digest."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        np.savez_compressed(
            handle,
            **{name: np.asarray(value) for name, value in arrays.items()},
        )
        temporary = Path(handle.name)
    os.replace(temporary, path)
    return file_sha256(path)


def file_sha256(path: Path) -> str:
    """Hash a saved artifact without loading it into memory at once."""

    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_checkpoint(path: Path) -> dict[str, Any]:
    """Read and validate the checkpoint envelope."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format in {path}.")
    return payload


def reusable_success(
    path: Path,
    *,
    expected: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return a matching successful checkpoint or request another fit.

    Failed checkpoints are deliberately not reusable. A successful checkpoint
    with different provenance is rejected instead of silently mixed into a new
    experiment.
    """

    path = Path(path)
    if not path.exists():
        return None
    payload = read_checkpoint(path)
    if payload.get("status") != "ok":
        return None
    mismatches = {
        key: (payload.get(key), _json_ready(value))
        for key, value in expected.items()
        if payload.get(key) != _json_ready(value)
    }
    if mismatches:
        formatted = ", ".join(
            f"{key}: recorded={recorded!r}, expected={wanted!r}"
            for key, (recorded, wanted) in mismatches.items()
        )
        raise ValueError(f"Stale checkpoint {path}: {formatted}")
    return payload


def read_checkpoints(root: Path) -> list[dict[str, Any]]:
    """Read all checkpoints in deterministic filename order."""

    root = Path(root)
    if not root.exists():
        return []
    return [read_checkpoint(path) for path in sorted(root.rglob("*.json"))]


def metric_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    method: str,
    scenario: str,
    config_hash: str,
    upstream_commit: str,
) -> list[dict[str, Any]]:
    """Flatten records from one current run configuration for tabulation."""

    rows: list[dict[str, Any]] = []
    for record in records:
        if (
            record.get("method") != method
            or record.get("scenario") != scenario
            or record.get("config_hash") != config_hash
            or record.get("upstream_commit") != upstream_commit
        ):
            continue
        row = {
            key: record.get(key)
            for key in (
                "trial_id",
                "n_past",
                "profile",
                "status",
                "dataset_digest",
                "data_seed",
                "algorithm_seed",
                "native_output",
                "native_output_sha256",
                "error",
            )
        }
        for name, value in record.get("metrics", {}).items():
            row[str(name)] = value
        rows.append(row)
    return sorted(rows, key=lambda row: (row["trial_id"], row["n_past"]))


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "checkpoint_path",
    "configuration_hash",
    "derive_algorithm_seed",
    "file_sha256",
    "metric_rows",
    "read_checkpoint",
    "read_checkpoints",
    "reusable_success",
    "write_checkpoint",
    "write_npz_checkpoint",
]
