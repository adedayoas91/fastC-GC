"""Statistical validation helpers for the correlation-profile algorithms."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Literal

import numpy as np

from .causal_benchmark import graph_metrics


NullRegime = Literal["iid-null", "stationary-ar-null", "drifting-null"]


@dataclass(frozen=True, slots=True)
class ValidationData:
    """One deterministic validation series and its optional lag truth."""

    series: np.ndarray
    regime: str
    seed: int
    lag_truth: np.ndarray | None = None

    @property
    def digest(self) -> str:
        contiguous = np.ascontiguousarray(self.series, dtype=np.float64)
        return hashlib.sha256(contiguous.tobytes()).hexdigest()


def simulate_null_series(
    *,
    regime: NullRegime,
    n_variables: int,
    n_samples: int,
    seed: int,
    ar_coefficient: float = 0.8,
    drift_strength: float = 1.5,
    burn_in: int = 500,
) -> ValidationData:
    """Generate mutually independent series under three temporal regimes.

    ``iid-null`` is the exchangeable primitive. ``stationary-ar-null`` checks
    calibration with autocorrelation and finite-window boundaries.
    ``drifting-null`` deliberately violates stationarity by adding independent
    deterministic drifts with variable-specific signs and magnitudes.
    """

    if regime not in {"iid-null", "stationary-ar-null", "drifting-null"}:
        raise ValueError(f"Unknown null regime {regime!r}.")
    if n_variables < 2:
        raise ValueError("n_variables must be at least 2.")
    if n_samples < 10:
        raise ValueError("n_samples must be at least 10.")
    if not 0.0 <= abs(ar_coefficient) < 1.0:
        raise ValueError("abs(ar_coefficient) must be smaller than 1.")
    if drift_strength < 0.0:
        raise ValueError("drift_strength must be non-negative.")

    rng = np.random.default_rng(seed)
    if regime == "iid-null":
        series = rng.normal(size=(n_variables, n_samples))
    else:
        total = n_samples + burn_in
        innovations = rng.normal(size=(n_variables, total))
        series_full = np.zeros_like(innovations)
        stationary_scale = 1.0 / np.sqrt(1.0 - ar_coefficient**2)
        series_full[:, 0] = rng.normal(
            scale=stationary_scale,
            size=n_variables,
        )
        for time_index in range(1, total):
            series_full[:, time_index] = (
                ar_coefficient * series_full[:, time_index - 1]
                + innovations[:, time_index]
            )
        series = series_full[:, burn_in:]
        if regime == "drifting-null":
            time = np.linspace(-1.0, 1.0, n_samples)
            signs = rng.choice(np.array([-1.0, 1.0]), size=n_variables)
            magnitudes = rng.uniform(0.5, 1.0, size=n_variables)
            series = series + drift_strength * (signs * magnitudes)[:, None] * time

    return ValidationData(
        series=np.asarray(series, dtype=float),
        regime=regime,
        seed=seed,
    )


def threshold_adjacency(
    marginal_pvalues: np.ndarray,
    conditional_pvalues: np.ndarray,
    *,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Apply the two-test conjunction to stored component p-values."""

    marginal = np.asarray(marginal_pvalues, dtype=float)
    conditional = np.asarray(conditional_pvalues, dtype=float)
    if marginal.shape != conditional.shape:
        raise ValueError("Marginal and conditional p-values must have equal shapes.")
    if marginal.ndim != 3 or marginal.shape[-1] != marginal.shape[-2]:
        raise ValueError("P-values must have shape (lag, source, target).")
    if not 0.0 < alpha <= 1.0 or not 0.0 < beta <= 1.0:
        raise ValueError("alpha and beta must lie in (0, 1].")
    return (marginal <= alpha) & (conditional <= beta)


def truth_for_fitted_order(
    lag_truth: np.ndarray,
    fitted_order: int,
) -> tuple[np.ndarray, int]:
    """Align true lags to a fitted order and count omitted positive edges."""

    truth = np.asarray(lag_truth, dtype=bool)
    if truth.ndim != 3 or truth.shape[-1] != truth.shape[-2]:
        raise ValueError("lag_truth must have shape (lag, source, target).")
    if fitted_order < 1:
        raise ValueError("fitted_order must be positive.")
    true_order, n_variables, _ = truth.shape
    omitted = int(np.count_nonzero(truth[fitted_order:]))
    if fitted_order <= true_order:
        return truth[:fitted_order].copy(), omitted
    padding = np.zeros(
        (fitted_order - true_order, n_variables, n_variables),
        dtype=bool,
    )
    return np.concatenate((truth, padding), axis=0), 0


def _cross_mask(shape: tuple[int, int, int]) -> np.ndarray:
    n_variables = shape[-1]
    return np.broadcast_to(~np.eye(n_variables, dtype=bool), shape)


def null_calibration_metrics(
    marginal_pvalues: np.ndarray,
    conditional_pvalues: np.ndarray,
    *,
    alpha: float,
    beta: float,
) -> dict[str, float]:
    """Summarize cross-variable false rejections for one null realization."""

    selected = threshold_adjacency(
        marginal_pvalues,
        conditional_pvalues,
        alpha=alpha,
        beta=beta,
    )
    marginal = np.asarray(marginal_pvalues, dtype=float)
    conditional = np.asarray(conditional_pvalues, dtype=float)
    mask = _cross_mask(selected.shape)
    candidate_count = int(np.count_nonzero(mask))
    joint_count = int(np.count_nonzero(selected[mask]))
    return {
        "candidate_count": float(candidate_count),
        "marginal_rejection_rate": float(np.mean(marginal[mask] <= alpha)),
        "conditional_rejection_rate": float(np.mean(conditional[mask] <= beta)),
        "joint_false_positive_rate": joint_count / max(candidate_count, 1),
        "graph_familywise_error": float(joint_count > 0),
        "selected_cross_edges": float(joint_count),
        "mean_marginal_pvalue": float(np.mean(marginal[mask])),
        "mean_conditional_pvalue": float(np.mean(conditional[mask])),
    }


def recovery_metrics(
    lag_truth: np.ndarray,
    marginal_pvalues: np.ndarray,
    conditional_pvalues: np.ndarray,
    *,
    alpha: float,
    beta: float,
) -> dict[str, float]:
    """Compute represented-lag cross-edge recovery from stored p-values."""

    truth = np.asarray(lag_truth, dtype=bool)
    selected = threshold_adjacency(
        marginal_pvalues,
        conditional_pvalues,
        alpha=alpha,
        beta=beta,
    )
    if truth.shape != selected.shape:
        raise ValueError("lag_truth and p-values must have identical shapes.")
    scores = 1.0 - np.maximum(marginal_pvalues, conditional_pvalues)
    return graph_metrics(truth, selected, scores=scores, scope="cross")


__all__ = [
    "NullRegime",
    "ValidationData",
    "null_calibration_metrics",
    "recovery_metrics",
    "simulate_null_series",
    "threshold_adjacency",
    "truth_for_fitted_order",
]
