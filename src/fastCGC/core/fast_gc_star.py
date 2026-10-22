"""Backend-selectable causalised-GC estimator orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Mapping, Optional

import numpy as np

from .gc_star import GcStar
from ..dependence import (
    BackendConfig,
    ConditioningPlan,
    build_fit_context,
    create_conditional_backend,
    create_marginal_backend,
    normalize_backend_options,
    resolve_dependence_selection,
    resolve_evaluation_policy,
    validate_evaluation_policy,
)
from ..dependence.linear import _fft_cross_null, _schur_precision
from ..dependence.results import DependenceResult

logger = logging.getLogger(__name__)

__all__ = ["FastGcStar", "_fft_cross_null", "_schur_precision"]


@dataclass
class FastGcStar(GcStar):
    """fastC-GC/fastC-GC* estimator with selectable dependence backends.

    Parameters
    ----------
    ridge:
        Dimensionless ridge added after row standardization.
    screen_alpha:
        Legacy compatibility parameter. It is validated and stored but does
        not select candidates in the current linear backend.
    exact_permutation:
        If true, use complete circular-shift p-values for evaluated linear
        conditional pairs. If false, use the analytic diagnostic values as a
        parametric approximation.
    candidate_batch_size:
        Maximum number of residual pairs transformed by one FFT batch.
    dependence_profile:
        Named marginal/conditional backend profile.
    marginal_test:
        Optional marginal backend override.
    conditional_test:
        Optional conditional backend override.
    backend_options:
        Per-backend option dictionary.
    evaluation_policy:
        One of ``"auto"``, ``"full"``, or ``"short_circuit"``.
    random_state:
        Stable backend seed.
    memory_budget_mb:
        Soft memory budget passed to backends.
    """

    ridge: float = 1e-6
    screen_alpha: float = 0.2
    exact_permutation: bool = True
    candidate_batch_size: int = 256
    dependence_profile: str = "linear"
    marginal_test: str | None = None
    conditional_test: str | None = None
    backend_options: Mapping[str, Mapping[str, object]] | None = None
    evaluation_policy: str = "auto"
    random_state: int | None = 0
    memory_budget_mb: int = 1024

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.ridge <= 0.0:
            raise ValueError("ridge must be positive.")
        if not 0.0 <= self.screen_alpha <= 1.0:
            raise ValueError("screen_alpha must lie in [0, 1].")
        if self.candidate_batch_size < 1:
            raise ValueError("candidate_batch_size must be positive.")
        if self.memory_budget_mb < 1:
            raise ValueError("memory_budget_mb must be positive.")
        validate_evaluation_policy(self.evaluation_policy)
        self._backend_options = normalize_backend_options(self.backend_options)
        self._fit_context = None
        self._context_data_id = None
        self._resolved_evaluation_policy = None
        self._fit_alpha = None
        self._fit_beta = None

    def _prepare_context(self, data: np.ndarray) -> None:
        context = build_fit_context(data, self.n_pasts)
        self._fit_context = context
        self.shifted_data = context.shifted
        self.standardized_shifted_ = context.standardized
        self.row_mean_ = context.row_mean
        self.row_scale_ = context.row_scale
        self.valid_row_mask_ = context.valid_rows
        self.n_neur = context.n_variables
        self._context_data_id = id(data)

    def _ensure_context(self, data: np.ndarray) -> None:
        if self._context_data_id != id(data):
            self._prepare_context(data)

    def _allowed_pair_mask(self) -> np.ndarray:
        if self._fit_context is None:
            raise RuntimeError("fit context is unavailable.")
        return self._fit_context.allowed_pairs

    def _resolve_selection(self):
        selection = resolve_dependence_selection(
            dependence_profile=self.dependence_profile,
            marginal_test=self.marginal_test,
            conditional_test=self.conditional_test,
        )
        policy = resolve_evaluation_policy(
            requested=self.evaluation_policy,
            selection=selection,
        )
        return selection, policy

    def _backend_config(self, backend_name: str) -> BackendConfig:
        return BackendConfig(
            options=self._backend_options.get(backend_name, {}),
            random_state=self.random_state,
            memory_budget_mb=self.memory_budget_mb,
            estimator_config={
                "ridge": self.ridge,
                "exact_permutation": self.exact_permutation,
                "candidate_batch_size": self.candidate_batch_size,
                "method": self.method,
            },
        )

    def _publish_result(self, marginal: DependenceResult, conditional: DependenceResult) -> None:
        self.marginal_statistic_ = marginal.statistic
        self.marginal_effect_ = marginal.effect
        self.marginal_pvalue_ = marginal.pvalue
        self.conditional_statistic_ = conditional.statistic
        self.conditional_effect_ = conditional.effect
        self.conditional_pvalue_ = conditional.pvalue
        self.conditional_evaluated_ = conditional.evaluated
        self.pvalue_source_ = conditional.pvalue_source
        self.backend_metadata_ = {
            "dependence_profile": self.dependence_profile,
            "marginal_backend": marginal.backend_name,
            "conditional_backend": conditional.backend_name,
            "marginal": marginal.backend_metadata,
            "conditional": conditional.backend_metadata,
            "evaluation_policy": self._resolved_evaluation_policy,
            "random_state": self.random_state,
            "memory_budget_mb": self.memory_budget_mb,
            "fit_alpha": self._fit_alpha,
            "fit_beta": self._fit_beta,
            "backend_options": self._backend_options,
        }
        self.corr_ = self.marginal_effect_
        self.pVal_corr_ = self.marginal_pvalue_
        self.inv_corr_ = self.conditional_effect_
        self.pVal_inv_corr_ = self.conditional_pvalue_

        analytic_score = conditional.diagnostics.get("analytic_score")
        analytic_p = conditional.diagnostics.get("analytic_p")
        screen_score = conditional.diagnostics.get("screen_score")
        screen_p = conditional.diagnostics.get("screen_p")
        shift_p = conditional.diagnostics.get("shift_p")
        if analytic_score is not None:
            self.analytic_score_ = analytic_score
        if analytic_p is not None:
            self.analytic_pvalue_ = analytic_p
        if screen_score is not None:
            self.screen_score_ = screen_score
        if screen_p is not None:
            self.screen_p_ = screen_p
        if shift_p is not None:
            self.shift_p_ = shift_p
            self.shift_tested_ = np.isfinite(shift_p)
        else:
            self.shift_p_ = np.full_like(self.conditional_pvalue_, np.nan, dtype=float)
            self.shift_tested_ = np.zeros_like(self.conditional_evaluated_, dtype=bool)
        precision = conditional.diagnostics.get("precision")
        if precision is not None:
            self.precision_ = precision
        residual_variance = conditional.diagnostics.get("residual_variance")
        if residual_variance is not None:
            self.residual_variance_ = float(residual_variance)

    def _evaluate_backends(
        self,
        data: np.ndarray,
        *,
        alpha: float,
        beta: float,
    ) -> tuple[DependenceResult, DependenceResult]:
        del beta
        self._ensure_context(data)
        if self._fit_context is None:
            raise RuntimeError("fit context is unavailable.")
        selection, policy = self._resolve_selection()
        self._resolved_evaluation_policy = policy
        marginal_backend = create_marginal_backend(
            selection.marginal_test,
            config=self._backend_config(selection.marginal_test),
        )
        conditional_backend = create_conditional_backend(
            selection.conditional_test,
            config=self._backend_config(selection.conditional_test),
        )
        plan = ConditioningPlan.build(self._fit_context, self.method)

        candidate_mask = None
        if self.parallel and policy == "full":
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                marginal_future = executor.submit(marginal_backend.evaluate, self._fit_context)
                conditional_future = executor.submit(
                    conditional_backend.evaluate,
                    self._fit_context,
                    plan,
                )
                marginal_result = marginal_future.result()
                conditional_result = conditional_future.result()
        else:
            marginal_result = marginal_backend.evaluate(self._fit_context)
            if policy == "short_circuit":
                candidate_mask = self._fit_context.allowed_pairs & (marginal_result.pvalue <= alpha)
            conditional_result = conditional_backend.evaluate(
                self._fit_context,
                plan,
                marginal_result=marginal_result,
                candidate_mask=candidate_mask,
            )
        return marginal_result, conditional_result

    def fit(
        self,
        data: np.ndarray,
        verbose: int = 0,
        *,
        alpha: float = 0.01,
        beta: float = 0.001,
    ) -> "FastGcStar":
        self.data = np.asarray(data, dtype=float).copy()
        self._configure_logging(verbose)
        self._prepare_context(self.data)
        self._fit_alpha = float(alpha)
        self._fit_beta = float(beta)
        marginal_result, conditional_result = self._evaluate_backends(
            self.data,
            alpha=self._fit_alpha,
            beta=self._fit_beta,
        )
        self._publish_result(marginal_result, conditional_result)
        return self

    def correlation_func(
        self, data: np.ndarray, shifts: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        del shifts
        marginal_result, _ = self._evaluate_backends(
            data,
            alpha=self._fit_alpha or 0.01,
            beta=self._fit_beta or 0.001,
        )
        return marginal_result.effect, marginal_result.pvalue

    def inv_correlation_func(
        self, data: np.ndarray, shifts: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        del shifts
        marginal_result, conditional_result = self._evaluate_backends(
            data,
            alpha=self._fit_alpha or 0.01,
            beta=self._fit_beta or 0.001,
        )
        self._publish_result(marginal_result, conditional_result)
        return conditional_result.effect, conditional_result.pvalue

    def get_connectivity_matrix(
        self,
        *,
        simulation: bool = True,
        alpha: float | None = None,
        beta: float | None = None,
    ) -> np.ndarray:
        if self.corr_ is None or self.inv_corr_ is None:
            raise RuntimeError("fit must be called before get_connectivity_matrix.")
        if self.pVal_corr_ is None or self.pVal_inv_corr_ is None:
            raise RuntimeError("dependence p-values are unavailable.")

        alpha = self._fit_alpha if alpha is None else float(alpha)
        beta = self._fit_beta if beta is None else float(beta)
        if alpha is None or beta is None:
            raise RuntimeError("fit thresholds are unavailable.")
        if (
            self._resolved_evaluation_policy == "short_circuit"
            and (alpha != self._fit_alpha or beta != self._fit_beta)
        ):
            raise ValueError(
                "Short-circuit fits require get_connectivity_matrix() to use the fitted thresholds. Refit for different thresholds."
            )

        marginal_pvalue = getattr(self, "marginal_pvalue_", self.pVal_corr_)
        conditional_pvalue = getattr(self, "conditional_pvalue_", self.pVal_inv_corr_)
        conditional_effect = getattr(self, "conditional_effect_", self.inv_corr_)
        inferred = np.logical_and(
            marginal_pvalue <= alpha,
            conditional_pvalue <= beta,
        )
        n_variables = inferred.shape[1]
        expected_rows = (self.n_pasts + 1) * n_variables
        if inferred.shape[0] != expected_rows:
            raise RuntimeError("dependence arrays do not match n_pasts.")

        np.fill_diagonal(inferred[:n_variables], False)
        self.decision_mask_ = inferred
        self.lag_adjacency_ = inferred.reshape(self.n_pasts + 1, n_variables, n_variables)
        self.lag_score_ = conditional_effect.reshape(
            self.n_pasts + 1, n_variables, n_variables
        )

        if simulation:
            selected_lags = list(range(1, min(self.n_lags, self.n_pasts) + 1))
            if not selected_lags:
                selected_lags = [0]
        else:
            selected_lags = list(range(0, min(self.n_lags, self.n_pasts) + 1))

        selected_adjacency = self.lag_adjacency_[selected_lags]
        selected_scores = self.lag_score_[selected_lags]
        retained_scores = np.where(selected_adjacency, selected_scores, -np.inf)
        best_index = np.argmax(retained_scores, axis=0)
        any_retained = np.any(selected_adjacency, axis=0)
        best_score = np.take_along_axis(
            retained_scores, best_index[None, :, :], axis=0
        )[0]
        self.conn_mat = np.where(any_retained, best_score, 0.0)
        selected_lag_array = np.asarray(selected_lags, dtype=int)
        self.edge_lag_ = np.where(any_retained, selected_lag_array[best_index], -1)
        self.edge_score_ = self.conn_mat.copy()
        return self.conn_mat
