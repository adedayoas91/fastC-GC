"""Render print-sized empirical graph panels from the saved collector tables.

Run from any directory with the project's Python environment. This script only
reads completed outputs and writes figure PDFs; it never fits or scores a method.
Node keys follow the saved variable order, and all methods share one layout.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = PROJECT_ROOT.parent / "manuscript" / "figures"
DATASETS = {
    "causalChamber": "causal-chamber-wt-actuators-random-walk-1",
    "causalRivers": "causal-rivers-bavaria-n5-sample01",
}
METHODS = (
    "fast-cgc", "fast-cgc-star", "cgc", "cgc-star",
    "pcmci-plus", "bagged-pcmci-plus", "cedar", "cuts-plus",
)


def graph_layout(bundle: Path, node_count: int) -> np.ndarray:
    """Use saved geographic coordinates, or a deterministic circular layout."""
    with np.load(bundle, allow_pickle=False) as stored:
        positions = (
            np.asarray(stored["node_positions"], dtype=float)
            if "node_positions" in stored.files else None
        )
    if positions is None:
        angles = np.linspace(0, 2 * np.pi, node_count, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
    if positions.shape != (node_count, 2) or not np.isfinite(positions).all():
        raise ValueError(f"Invalid node positions in {bundle}")
    # CausalRivers stores latitude, longitude; its own map renderer reverses
    # these columns. Put longitude on the horizontal axis and north upwards.
    positions = positions[:, [1, 0]]
    positions = positions - positions.mean(axis=0)
    scale = np.linalg.norm(positions, axis=1).max()
    if scale <= 0:
        raise ValueError(f"Coincident node positions in {bundle}")
    return positions / scale


def edge_pairs(frame: pd.DataFrame, node_count: int) -> set[tuple[int, int]]:
    """Reject duplicate, self, and out-of-range edges before plotting."""
    pairs = [(int(row.source_index), int(row.target_index))
             for row in frame.itertuples(index=False)]
    if len(set(pairs)) != len(pairs):
        raise ValueError("Duplicate directed edges in collector output")
    if any(source == target or not (0 <= source < node_count)
           or not (0 <= target < node_count) for source, target in pairs):
        raise ValueError("Unexpected self-edge or node index in collector output")
    return set(pairs)


def render_panel(
    pairs: set[tuple[int, int]],
    positions: np.ndarray,
    labels: list[str],
    destination: Path,
    *,
    geographic: bool,
    reference_pairs: set[tuple[int, int]] | None = None,
) -> None:
    """Draw equal-weight edges, optionally marking absence from context in red."""
    figure, axis = plt.subplots(figsize=(2.15, 1.3 if geographic else 2.15))
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    for source, target in sorted(pairs):
        reciprocal = (target, source) in pairs
        color = ("#c62828" if reference_pairs is not None
                 and (source, target) not in reference_pairs else "#2f5597")
        axis.add_patch(FancyArrowPatch(
            positions[source], positions[target], arrowstyle="-|>",
            mutation_scale=8 if geographic else 9.5, linewidth=0.85, color=color,
            connectionstyle=f"arc3,rad={0.17 if reciprocal else 0.0}",
            shrinkA=7.5 if geographic else 8.2,
            shrinkB=7.5 if geographic else 8.2, zorder=1,
        ))
    axis.scatter(
        positions[:, 0], positions[:, 1], s=155 if geographic else 190,
        facecolor="#f9d6a9",
        edgecolor="#715124", linewidth=0.65, zorder=2,
    )
    for label, position in zip(labels, positions, strict=True):
        axis.text(*position, label, ha="center", va="center",
                  fontsize=10.5,
                  color="#171717", zorder=3)
    if geographic:
        axis.annotate("N", xy=(0.92, 0.96), xytext=(0.92, 0.74),
                      xycoords="axes fraction", ha="center", va="center",
                      fontsize=10, arrowprops={"arrowstyle": "->", "lw": 0.7})
    axis.set(xlim=(-1.2, 1.2),
             ylim=(-0.68, 0.68) if geographic else (-1.2, 1.2), aspect="equal")
    axis.axis("off")
    figure.savefig(destination, metadata={"Title": destination.stem})
    plt.close(figure)


def main() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    for dataset, dataset_id in DATASETS.items():
        shared = PROJECT_ROOT / "data/shared/causal-comparison" / dataset
        bundle = shared / "datasets" / f"{dataset_id}.npz"
        metadata = json.loads(bundle.with_suffix(".json").read_text())
        node_count = len(metadata["variable_names"])
        positions = graph_layout(bundle, node_count)
        geographic = dataset == "causalRivers"
        labels = ([chr(ord("A") + index) for index in range(node_count)]
                  if geographic else [str(index + 1) for index in range(node_count)])
        graph_root = PROJECT_ROOT / "data/outputs/causal-comparison" / dataset / "graphs"
        status = pd.read_csv(graph_root / "run_status.csv")
        status = status[status["dataset_id"] == dataset_id]
        if (status["method"].duplicated().any()
                or set(status["method"]) != set(METHODS)
                or set(status["status"]) != {"ok"}):
            raise ValueError(f"Expected one completed fit per method for {dataset_id}")
        inferred = pd.read_csv(graph_root / "inferred_edges.csv")
        inferred = inferred[inferred["dataset_id"] == dataset_id]
        reference = pd.read_csv(graph_root / "reference_edges.csv")
        reference = reference[reference["dataset_id"] == dataset_id]
        panels = {"context": reference}
        panels.update({method: inferred[inferred["method"] == method]
                       for method in METHODS})
        prefix = "chamber" if dataset == "causalChamber" else "rivers"
        reference_pairs = edge_pairs(reference, node_count)
        for method, frame in panels.items():
            pairs = edge_pairs(frame, node_count)
            destination = FIGURE_ROOT / f"{prefix}-{method}.pdf"
            render_panel(
                pairs, positions, labels, destination, geographic=geographic,
                reference_pairs=reference_pairs if method != "context" else None,
            )
            print(f"{destination.name}: {len(pairs)} edges, {node_count} nodes")


if __name__ == "__main__":
    main()
