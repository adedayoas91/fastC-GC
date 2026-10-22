"""Shared data and scoring contracts for causal-structure benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Literal

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


EdgeScope = Literal["cross", "self", "all"]


@dataclass(frozen=True, slots=True)
class CausalBenchmarkData:
    """One stable VAR realization with an exact lag-resolved ground truth.

    ``coefficients[lag, target, source]`` follows the regression convention,
    whereas ``lag_truth[lag, source, target]`` follows the benchmark's graph
    convention.  All method adapters return the latter orientation.
    """

    series: np.ndarray
    coefficients: np.ndarray
    lag_truth: np.ndarray
    seed: int
    noise: str
    spectral_radius: float

    @property
    def digest(self) -> str:
        """Return a stable digest used to verify paired method inputs."""

        contiguous = np.ascontiguousarray(self.series, dtype=np.float64)
        return hashlib.sha256(contiguous.tobytes()).hexdigest()


def companion_spectral_radius(coefficients: np.ndarray) -> float:
    """Return the spectral radius of a VAR companion matrix."""

    matrices = np.asarray(coefficients, dtype=float)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("coefficients must have shape (order, target, source).")
    order, n_variables, _ = matrices.shape
    companion = np.zeros((n_variables * order, n_variables * order))
    companion[:n_variables] = np.hstack(list(matrices))
    if order > 1:
        companion[n_variables:, :-n_variables] = np.eye(
            n_variables * (order - 1)
        )
    return float(np.max(np.abs(np.linalg.eigvals(companion))))


def simulate_sparse_var(
    *,
    n_variables: int,
    n_samples: int,
    order: int,
    seed: int,
    edge_probability: float = 0.12,
    noise: Literal["gaussian", "laplace"] = "gaussian",
    max_radius: float = 0.8,
    burn_in: int = 500,
) -> CausalBenchmarkData:
    """Simulate a stable sparse VAR with signed, lag-specific coefficients."""

    if n_variables < 2:
        raise ValueError("n_variables must be at least 2.")
    if n_samples <= order:
        raise ValueError("n_samples must be larger than order.")
    if order < 1:
        raise ValueError("order must be at least 1.")
    if not 0.0 < edge_probability < 1.0:
        raise ValueError("edge_probability must lie strictly between 0 and 1.")
    if noise not in {"gaussian", "laplace"}:
        raise ValueError("noise must be 'gaussian' or 'laplace'.")
    if not 0.0 < max_radius < 1.0:
        raise ValueError("max_radius must lie strictly between 0 and 1.")

    rng = np.random.default_rng(seed)
    coefficients = np.zeros((order, n_variables, n_variables), dtype=float)

    # A modest lag-1 self term stabilizes the marginal dynamics.  Self-lags are
    # scored separately from cross-variable recovery.
    coefficients[0, np.arange(n_variables), np.arange(n_variables)] = rng.uniform(
        0.20, 0.40, size=n_variables
    )

    support = rng.random((order, n_variables, n_variables)) < edge_probability
    for lag in range(order):
        np.fill_diagonal(support[lag], False)

    # Ensure that every target has at least one cross-variable parent so recall
    # is defined in small benchmark cells.
    for target in range(n_variables):
        if not np.any(support[:, target, :]):
            choices = np.delete(np.arange(n_variables), target)
            source = int(rng.choice(choices))
            lag = int(rng.integers(0, order))
            support[lag, target, source] = True

    magnitudes = rng.uniform(0.12, 0.35, size=support.shape)
    signs = rng.choice(np.array([-1.0, 1.0]), size=support.shape)
    coefficients[support] = (magnitudes * signs)[support]

    radius = companion_spectral_radius(coefficients)
    if radius > max_radius:
        coefficients *= max_radius / radius
        radius = companion_spectral_radius(coefficients)

    total = n_samples + burn_in
    series = np.zeros((n_variables, total), dtype=float)
    series[:, :order] = rng.normal(scale=0.5, size=(n_variables, order))
    innovation_scale = 1.0 if noise == "gaussian" else 1.0 / np.sqrt(2.0)
    for time_index in range(order, total):
        if noise == "gaussian":
            value = rng.normal(scale=innovation_scale, size=n_variables)
        else:
            value = rng.laplace(scale=innovation_scale, size=n_variables)
        for lag, matrix in enumerate(coefficients, start=1):
            value += matrix @ series[:, time_index - lag]
        series[:, time_index] = value

    lag_truth = np.transpose(coefficients != 0.0, (0, 2, 1))
    return CausalBenchmarkData(
        series=series[:, burn_in:],
        coefficients=coefficients,
        lag_truth=lag_truth,
        seed=seed,
        noise=noise,
        spectral_radius=radius,
    )


def aggregate_lags(values: np.ndarray, *, binary: bool) -> np.ndarray:
    """Collapse the lag dimension using OR for decisions or max for scores."""

    array = np.asarray(values)
    if array.ndim != 3:
        raise ValueError("lag-resolved values must have shape (lag, source, target).")
    if binary:
        return np.any(array.astype(bool), axis=0)
    return np.max(array.astype(float), axis=0)


def _candidate_mask(shape: tuple[int, ...], scope: EdgeScope) -> np.ndarray:
    if len(shape) not in {2, 3} or shape[-1] != shape[-2]:
        raise ValueError("graphs must be square 2D or lag-resolved 3D arrays.")
    n_variables = shape[-1]
    diagonal = np.eye(n_variables, dtype=bool)
    base = {
        "cross": ~diagonal,
        "self": diagonal,
        "all": np.ones_like(diagonal),
    }[scope]
    if len(shape) == 3:
        return np.broadcast_to(base, shape)
    return base


def graph_metrics(
    truth: np.ndarray,
    selected: np.ndarray,
    *,
    scores: np.ndarray | None = None,
    scope: EdgeScope = "cross",
) -> dict[str, float]:
    """Compute binary and ranking metrics on a declared edge universe.

    AUROC and AUPRC are returned only when a non-binary score is available and
    both classes occur in the selected edge universe.  A single binary graph is
    not treated as a ranking curve.
    """

    truth_array = np.asarray(truth, dtype=bool)
    selected_array = np.asarray(selected, dtype=bool)
    if truth_array.shape != selected_array.shape:
        raise ValueError("truth and selected must have identical shapes.")
    mask = _candidate_mask(truth_array.shape, scope)
    y_true = truth_array[mask]
    y_pred = selected_array[mask]

    true_positive = int(np.count_nonzero(y_true & y_pred))
    false_positive = int(np.count_nonzero(~y_true & y_pred))
    false_negative = int(np.count_nonzero(y_true & ~y_pred))
    true_negative = int(np.count_nonzero(~y_true & ~y_pred))
    candidates = y_true.size

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    specificity = true_negative / max(true_negative + false_positive, 1)
    negative_predictive_value = true_negative / max(
        true_negative + false_negative, 1
    )
    mcc_denominator = np.sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )
    mcc = (
        (true_positive * true_negative - false_positive * false_negative)
        / mcc_denominator
        if mcc_denominator > 0.0
        else 0.0
    )

    auroc = np.nan
    auprc = np.nan
    if scores is not None:
        score_array = np.asarray(scores, dtype=float)
        if score_array.shape != truth_array.shape:
            raise ValueError("scores must have the same shape as truth.")
        y_score = score_array[mask]
        finite = np.isfinite(y_score)
        unique_scores = np.unique(y_score[finite])
        if (
            np.all(finite)
            and np.unique(y_true).size == 2
            and unique_scores.size > 2
        ):
            auroc = float(roc_auc_score(y_true, y_score))
            auprc = float(average_precision_score(y_true, y_score))

    return {
        "candidates": float(candidates),
        "positives": float(np.count_nonzero(y_true)),
        "prevalence": float(np.mean(y_true)) if candidates else np.nan,
        "true_positive": float(true_positive),
        "false_positive": float(false_positive),
        "false_negative": float(false_negative),
        "true_negative": float(true_negative),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "negative_predictive_value": negative_predictive_value,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-15),
        "mcc": float(mcc),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "false_positive_rate": false_positive
        / max(false_positive + true_negative, 1),
        "false_discovery_rate": false_positive
        / max(true_positive + false_positive, 1),
        "false_positives_per_target": false_positive / truth_array.shape[-1],
        "shd": float(false_positive + false_negative),
        "normalized_shd": (false_positive + false_negative) / max(candidates, 1),
        "auroc": auroc,
        "auprc": auprc,
    }


def comparison_metrics(
    truth: np.ndarray,
    selected: np.ndarray,
    scores: np.ndarray | None,
) -> dict[str, float]:
    """Return exact-lag, aggregate, and self-lag benchmark metrics."""

    truth_array = np.asarray(truth, dtype=bool)
    selected_array = np.asarray(selected, dtype=bool)
    score_array = None if scores is None else np.asarray(scores, dtype=float)
    exact = graph_metrics(
        truth_array,
        selected_array,
        scores=score_array,
        scope="cross",
    )
    aggregate_truth = aggregate_lags(truth_array, binary=True)
    aggregate_selected = aggregate_lags(selected_array, binary=True)
    aggregate_scores = (
        None if score_array is None else aggregate_lags(score_array, binary=False)
    )
    aggregate = graph_metrics(
        aggregate_truth,
        aggregate_selected,
        scores=aggregate_scores,
        scope="cross",
    )
    self_lag = graph_metrics(
        truth_array,
        selected_array,
        scores=score_array,
        scope="self",
    )
    return {
        **{f"exact_{key}": value for key, value in exact.items()},
        **{f"aggregate_{key}": value for key, value in aggregate.items()},
        **{f"self_{key}": value for key, value in self_lag.items()},
    }


__all__ = [
    "CausalBenchmarkData",
    "aggregate_lags",
    "companion_spectral_radius",
    "comparison_metrics",
    "graph_metrics",
    "simulate_sparse_var",
]
