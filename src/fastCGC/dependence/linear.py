from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BackendConfig
from .conditioning import ConditioningPlan
from .context import FitContext
from .registry import register_conditional, register_marginal
from .results import DependenceResult


def _fft_cross_null(
    x_rows: np.ndarray, y_rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    x_rows = np.asarray(x_rows, dtype=float)
    y_rows = np.asarray(y_rows, dtype=float)
    if x_rows.shape != y_rows.shape:
        raise ValueError("x_rows and y_rows must share the same shape.")
    if x_rows.ndim != 2:
        raise ValueError("x_rows and y_rows must be two-dimensional.")

    n_pairs, n_samples = x_rows.shape
    if n_pairs == 0:
        return np.zeros(0), np.ones(0)
    if n_samples < 2:
        return np.zeros(n_pairs), np.ones(n_pairs)

    x_centered = x_rows - x_rows.mean(axis=1, keepdims=True)
    y_centered = y_rows - y_rows.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(x_centered, axis=1) * np.linalg.norm(
        y_centered, axis=1
    )
    valid = denominator > 0

    x_fft = np.fft.rfft(x_centered, axis=1)
    y_fft = np.fft.rfft(y_centered, axis=1)
    cross = np.fft.irfft(x_fft * np.conj(y_fft), n=n_samples, axis=1)
    correlations = cross / np.where(valid, denominator, 1.0)[:, None]
    observed = np.abs(correlations[:, 0])
    null = np.abs(correlations[:, 1:])
    count = (null >= observed[:, None] - 1e-12).sum(axis=1)
    pvalues = (count + 1.0) / n_samples
    return np.where(valid, observed, 0.0), np.where(valid, pvalues, 1.0)



def _absolute_row_correlation(
    x_rows: np.ndarray, y_rows: np.ndarray
) -> np.ndarray:
    x_centered = x_rows - x_rows.mean(axis=1, keepdims=True)
    y_centered = y_rows - y_rows.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(x_centered, axis=1) * np.linalg.norm(
        y_centered, axis=1
    )
    numerator = np.einsum("ij,ij->i", x_centered, y_centered)
    return np.divide(
        np.abs(numerator),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )



def _analytic_partial_correlation_pvalues(
    corr: np.ndarray, n_samples: int, dof_used: int
) -> np.ndarray:
    degrees_of_freedom = max(n_samples - dof_used - 2, 1)
    correlation = np.clip(np.abs(corr), 0.0, 1.0 - 1e-9)
    statistic = correlation * np.sqrt(
        degrees_of_freedom / np.clip(1.0 - correlation**2, 1e-12, None)
    )
    try:
        from scipy.stats import t as student_t

        return np.asarray(
            2.0 * student_t.sf(statistic, degrees_of_freedom),
            dtype=float,
        )
    except ImportError:  # pragma: no cover
        from math import erfc, sqrt

        return np.fromiter(
            (erfc(float(value) / sqrt(2.0)) for value in statistic.flat),
            dtype=float,
            count=statistic.size,
        ).reshape(statistic.shape)



def _symmetric_inverse(matrix: np.ndarray) -> np.ndarray:
    identity = np.eye(matrix.shape[0], dtype=matrix.dtype)
    try:
        factor = np.linalg.cholesky(matrix)
        inverse = np.linalg.solve(factor.T, np.linalg.solve(factor, identity))
    except np.linalg.LinAlgError:
        inverse = np.linalg.pinv(matrix)
    return (inverse + inverse.T) * 0.5



def _schur_precision(
    precision: np.ndarray, excluded: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    n_rows = precision.shape[0]
    excluded = np.unique(np.asarray(excluded, dtype=int))
    if excluded.size == 0:
        return np.arange(n_rows), precision

    keep_mask = np.ones(n_rows, dtype=bool)
    keep_mask[excluded] = False
    retained = np.flatnonzero(keep_mask)
    q_aa = precision[np.ix_(retained, retained)]
    q_af = precision[np.ix_(retained, excluded)]
    q_ff = precision[np.ix_(excluded, excluded)]
    q_fa = precision[np.ix_(excluded, retained)]
    downdated = q_aa - q_af @ np.linalg.solve(q_ff, q_fa)
    return retained, (downdated + downdated.T) * 0.5


@dataclass(slots=True)
class _BackendBase:
    config: BackendConfig

    @property
    def options(self) -> dict[str, object]:
        return dict(self.config.options)

    @property
    def estimator_config(self) -> dict[str, object]:
        return dict(self.config.estimator_config)


@register_marginal("pearson")
@dataclass(slots=True)
class PearsonMarginalBackend(_BackendBase):
    name: str = "pearson"

    def evaluate(
        self,
        context: FitContext,
        *,
        candidate_mask: np.ndarray | None = None,
    ) -> DependenceResult:
        del candidate_mask
        standardized = context.standardized
        rows, n_samples = standardized.shape
        n_targets = context.n_variables
        norms = np.linalg.norm(standardized, axis=1)
        safe_norms = np.where(norms > 0.0, norms, 1.0)
        transformed = np.fft.rfft(standardized, axis=1)

        correlations = np.zeros((rows, n_targets))
        pvalues = np.ones((rows, n_targets))
        for target in range(n_targets):
            cross = np.fft.irfft(
                transformed * np.conj(transformed[target]),
                n=n_samples,
                axis=1,
            )
            shifted_correlations = cross / (
                safe_norms * safe_norms[target]
            )[:, None]
            observed = np.abs(shifted_correlations[:, 0])
            null = np.abs(shifted_correlations[:, 1:])
            count = (null >= observed[:, None] - 1e-12).sum(axis=1)
            valid = context.valid_rows & context.valid_rows[target]
            correlations[:, target] = np.where(valid, observed, 0.0)
            pvalues[:, target] = np.where(valid, (count + 1.0) / n_samples, 1.0)

        correlations[~context.allowed_pairs] = 0.0
        pvalues[~context.allowed_pairs] = 1.0
        pvalue_source = np.where(context.allowed_pairs, "shift", "excluded")
        return DependenceResult(
            statistic=correlations.copy(),
            effect=correlations,
            pvalue=pvalues,
            evaluated=context.allowed_pairs.copy(),
            pvalue_source=np.asarray(pvalue_source, dtype="<U8"),
            backend_name=self.name,
            backend_metadata={"null_method": "complete_circular_shift"},
        )


@register_conditional("ridge_residual")
@dataclass(slots=True)
class RidgeResidualConditionalBackend(_BackendBase):
    name: str = "ridge_residual"

    def _global_precision(self, context: FitContext) -> np.ndarray:
        covariance = context.standardized @ context.standardized.T / context.n_samples
        covariance.flat[:: covariance.shape[0] + 1] += float(
            self.estimator_config["ridge"]
        )
        return _symmetric_inverse(covariance)

    def _evaluate_residual_group(
        self,
        precision: np.ndarray,
        transformed_rows: np.ndarray,
        source_position: int,
        target_positions: np.ndarray,
        output_rows: np.ndarray,
        output_targets: np.ndarray,
        conditional_score: np.ndarray,
        shift_p: np.ndarray,
        pvalues: np.ndarray,
        pvalue_source: np.ndarray,
    ) -> list[float]:
        residual_variances: list[float] = []
        batch_size = int(self.estimator_config["candidate_batch_size"])
        for start in range(0, target_positions.size, batch_size):
            stop = min(start + batch_size, target_positions.size)
            targets = target_positions[start:stop]
            source_rows = np.full(targets.size, source_position, dtype=int)
            pair_precision = np.empty((targets.size, 2, 2))
            pair_precision[:, 0, 0] = precision[source_position, source_position]
            pair_precision[:, 0, 1] = precision[source_position, targets]
            pair_precision[:, 1, 0] = precision[targets, source_position]
            pair_precision[:, 1, 1] = precision[targets, targets]

            rhs = np.empty((targets.size, 2, transformed_rows.shape[1]))
            rhs[:, 0, :] = transformed_rows[source_rows]
            rhs[:, 1, :] = transformed_rows[targets]
            residuals = np.linalg.solve(pair_precision, rhs)
            source_residuals = residuals[:, 0, :]
            target_residuals = residuals[:, 1, :]
            rows = output_rows[start:stop]
            cols = output_targets[start:stop]
            scores = _absolute_row_correlation(source_residuals, target_residuals)
            conditional_score[rows, cols] = scores
            residual_variances.extend(np.mean(source_residuals**2, axis=1).tolist())

            observed, exact_p = _fft_cross_null(source_residuals, target_residuals)
            conditional_score[rows, cols] = observed
            shift_p[rows, cols] = exact_p
            pvalues[rows, cols] = exact_p
            pvalue_source[rows, cols] = "shift"
        return residual_variances

    def _fcgc_result(
        self,
        context: FitContext,
        *,
        candidate_mask: np.ndarray | None,
    ) -> DependenceResult:
        rows, n_samples = context.standardized.shape
        n_targets = context.n_variables
        precision = self._global_precision(context)
        diagonal = np.sqrt(np.clip(np.diag(precision), 1e-18, None))
        analytic_score = np.abs(
            precision[:, :n_targets] / np.outer(diagonal, diagonal[:n_targets])
        )
        analytic_p = _analytic_partial_correlation_pvalues(
            analytic_score,
            n_samples,
            dof_used=rows - 2,
        )

        allowed = context.allowed_pairs.copy()
        if candidate_mask is not None:
            evaluated = allowed & candidate_mask
        else:
            evaluated = allowed.copy()
        analytic_score[~allowed] = 0.0
        analytic_p[~allowed] = 1.0

        conditional_score = np.where(allowed, analytic_score, 0.0)
        shift_p = np.full((rows, n_targets), np.nan)
        pvalues = analytic_p.copy()
        pvalue_source = np.full((rows, n_targets), "excluded", dtype="<U8")
        pvalue_source[allowed] = "analytic"
        pvalue_source[allowed & ~evaluated] = "skipped"
        pvalues[allowed & ~evaluated] = 1.0

        variances: list[float] = []
        if bool(self.estimator_config["exact_permutation"]):
            transformed = precision @ context.standardized
            pair_rows, pair_targets = np.nonzero(evaluated)
            for start in range(0, pair_rows.size, int(self.estimator_config["candidate_batch_size"])):
                stop = min(start + int(self.estimator_config["candidate_batch_size"]), pair_rows.size)
                sources = pair_rows[start:stop]
                targets = pair_targets[start:stop]
                pair_precision = np.empty((sources.size, 2, 2))
                pair_precision[:, 0, 0] = precision[sources, sources]
                pair_precision[:, 0, 1] = precision[sources, targets]
                pair_precision[:, 1, 0] = precision[targets, sources]
                pair_precision[:, 1, 1] = precision[targets, targets]
                rhs = np.stack((transformed[sources], transformed[targets]), axis=1)
                residuals = np.linalg.solve(pair_precision, rhs)
                source_residuals = residuals[:, 0]
                target_residuals = residuals[:, 1]
                observed, exact_p = _fft_cross_null(source_residuals, target_residuals)
                conditional_score[sources, targets] = observed
                shift_p[sources, targets] = exact_p
                pvalues[sources, targets] = exact_p
                pvalue_source[sources, targets] = "shift"
                variances.extend(np.mean(source_residuals**2, axis=1).tolist())

        pvalues[~allowed] = 1.0
        diagnostics: dict[str, object] = {
            "analytic_score": analytic_score,
            "analytic_p": analytic_p,
            # Compatibility aliases retained for downstream notebooks.
            "screen_score": analytic_score,
            "screen_p": analytic_p,
            "shift_p": shift_p,
            "precision": precision,
        }
        if variances:
            diagnostics["residual_variance"] = float(np.mean(variances))
        return DependenceResult(
            statistic=conditional_score.copy(),
            effect=conditional_score,
            pvalue=pvalues,
            evaluated=evaluated,
            pvalue_source=pvalue_source,
            backend_name=self.name,
            backend_metadata={
                "null_method": (
                    "ridge_residual_complete_circular_shift"
                    if bool(self.estimator_config["exact_permutation"])
                    else "analytic_regularized_partial_correlation"
                )
            },
            diagnostics=diagnostics,
        )

    def _cgc_result(
        self,
        context: FitContext,
        plan: ConditioningPlan,
        *,
        candidate_mask: np.ndarray | None,
    ) -> DependenceResult:
        rows, n_samples = context.standardized.shape
        n_targets = context.n_variables
        global_precision = self._global_precision(context)
        allowed = context.allowed_pairs.copy()
        evaluated = allowed.copy() if candidate_mask is None else allowed & candidate_mask

        conditional_score = np.zeros((rows, n_targets))
        analytic_score = np.zeros((rows, n_targets))
        analytic_p = np.ones((rows, n_targets))
        shift_p = np.full((rows, n_targets), np.nan)
        pvalues = np.ones((rows, n_targets))
        pvalue_source = np.full((rows, n_targets), "excluded", dtype="<U8")
        variance_values: list[float] = []
        precision_cache: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        def retained_precision(
            excluded: tuple[int, ...],
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            cached = precision_cache.get(excluded)
            if cached is None:
                retained, precision = _schur_precision(global_precision, np.asarray(excluded, dtype=int))
                positions = np.full(rows, -1, dtype=int)
                positions[retained] = np.arange(retained.size)
                cached = (retained, positions, precision)
                precision_cache[excluded] = cached
            return cached

        for source_row, groups in plan.groups_by_source.items():
            for group in groups:
                retained, positions, precision = retained_precision(group.excluded_rows)
                source_position = int(positions[source_row])
                target_positions = positions[group.targets]
                diagonal = np.sqrt(np.clip(np.diag(precision), 1e-18, None))
                scores = np.abs(
                    precision[source_position, target_positions]
                    / (diagonal[source_position] * diagonal[target_positions])
                )
                analytic_score[source_row, group.targets] = scores
                conditional_score[source_row, group.targets] = scores

                dof_used = retained.size - 2
                group_p = _analytic_partial_correlation_pvalues(
                    scores,
                    n_samples,
                    dof_used,
                )
                analytic_p[source_row, group.targets] = group_p
                pvalues[source_row, group.targets] = group_p
                pvalue_source[source_row, group.targets] = "analytic"

                skipped_targets = group.targets[
                    ~evaluated[source_row, group.targets]
                ]
                if skipped_targets.size:
                    pvalue_source[source_row, skipped_targets] = "skipped"
                    pvalues[source_row, skipped_targets] = 1.0

                if bool(self.estimator_config["exact_permutation"]):
                    candidate_targets = group.targets[evaluated[source_row, group.targets]]
                    if candidate_targets.size == 0:
                        continue
                    candidate_positions = positions[candidate_targets]
                    required_positions = np.concatenate(([source_position], candidate_positions))
                    transformed_rows = precision[required_positions] @ context.standardized[retained]
                    reduced_precision = precision[np.ix_(required_positions, required_positions)]
                    variance_values.extend(
                        self._evaluate_residual_group(
                            reduced_precision,
                            transformed_rows,
                            0,
                            np.arange(1, candidate_targets.size + 1),
                            np.full(candidate_targets.size, source_row, dtype=int),
                            candidate_targets,
                            conditional_score,
                            shift_p,
                            pvalues,
                            pvalue_source,
                        )
                    )

        conditional_score[~allowed] = 0.0
        analytic_score[~allowed] = 0.0
        analytic_p[~allowed] = 1.0
        pvalues[~allowed] = 1.0
        diagnostics: dict[str, object] = {
            "analytic_score": analytic_score,
            "analytic_p": analytic_p,
            # Compatibility aliases retained for downstream notebooks.
            "screen_score": analytic_score,
            "screen_p": analytic_p,
            "shift_p": shift_p,
            "precision": global_precision,
        }
        if variance_values:
            diagnostics["residual_variance"] = float(np.mean(variance_values))
        return DependenceResult(
            statistic=conditional_score.copy(),
            effect=conditional_score,
            pvalue=pvalues,
            evaluated=evaluated,
            pvalue_source=pvalue_source,
            backend_name=self.name,
            backend_metadata={
                "null_method": (
                    "ridge_residual_complete_circular_shift"
                    if bool(self.estimator_config["exact_permutation"])
                    else "analytic_regularized_partial_correlation"
                )
            },
            diagnostics=diagnostics,
        )

    def evaluate(
        self,
        context: FitContext,
        plan: ConditioningPlan,
        *,
        marginal_result: DependenceResult | None = None,
        candidate_mask: np.ndarray | None = None,
    ) -> DependenceResult:
        del marginal_result
        if plan.method == "fcgc":
            return self._fcgc_result(context, candidate_mask=candidate_mask)
        return self._cgc_result(context, plan, candidate_mask=candidate_mask)
