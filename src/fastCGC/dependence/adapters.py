"""FastGcStar adapters for nonlinear marginal and conditional tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .base import BackendConfig
from .conditioning import ConditioningPlan
from .context import FitContext
from .hsic import PairwiseTestResult, RffHsicBackend, hsic_test
from .information import (
    conditional_mutual_information_test,
    mutual_information_test,
)
from .kcit import kcit_test
from .random_features import stable_seed
from .rcit import RandomizedConditionalBackend
from .registry import register_conditional, register_marginal
from .results import DependenceResult


def _validated_options(
    backend_name: str,
    options: Mapping[str, Any],
    allowed: set[str],
) -> dict[str, Any]:
    unknown = set(options) - allowed
    if unknown:
        choices = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown option(s) for {backend_name}: {choices}.")
    return dict(options)


def _empty_result(
    context: FitContext,
    *,
    backend_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = (context.standardized.shape[0], context.n_variables)
    return (
        np.zeros(shape, dtype=float),
        np.zeros(shape, dtype=float),
        np.ones(shape, dtype=float),
        np.zeros(shape, dtype=bool),
        np.full(shape, "excluded", dtype="<U24"),
    )


def _pair_values(result: Any) -> tuple[float, float, float, bool, dict[str, Any]]:
    statistic = float(np.asarray(result.statistic).reshape(-1)[0])
    pvalue = float(np.asarray(result.pvalue).reshape(-1)[0])
    raw_effect = getattr(result, "effect", statistic)
    effect = float(np.asarray(raw_effect).reshape(-1)[0])
    raw_evaluated = getattr(result, "evaluated", True)
    evaluated = bool(np.asarray(raw_evaluated).reshape(-1)[0])
    metadata = dict(getattr(result, "metadata", {}))
    null = getattr(result, "null_distribution", None)
    if null is not None:
        metadata["null_draws"] = int(np.asarray(null).size)
    return statistic, effect, pvalue, evaluated, metadata


class _MarginalAdapter:
    name = ""
    pvalue_label = "permutation"

    def __init__(self, *, config: BackendConfig) -> None:
        self.config = config

    def _evaluate_pair(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        source_row: int,
        target: int,
    ) -> Any:
        raise NotImplementedError

    def evaluate(
        self,
        context: FitContext,
        *,
        candidate_mask: np.ndarray | None = None,
    ) -> DependenceResult:
        statistic, effect, pvalue, evaluated, pvalue_source = _empty_result(
            context,
            backend_name=self.name,
        )
        selected = context.allowed_pairs.copy()
        if candidate_mask is not None:
            if candidate_mask.shape != selected.shape:
                raise ValueError("candidate_mask must match the allowed-pair shape.")
            selected &= candidate_mask
            pvalue_source[context.allowed_pairs & ~selected] = "skipped"

        first_metadata: dict[str, Any] = {}
        for source_row, target in zip(*np.nonzero(selected), strict=True):
            result = self._evaluate_pair(
                context.standardized[source_row, :, None],
                context.standardized[target, :, None],
                source_row=int(source_row),
                target=int(target),
            )
            pair = _pair_values(result)
            statistic[source_row, target] = pair[0]
            effect[source_row, target] = pair[1]
            pvalue[source_row, target] = pair[2]
            evaluated[source_row, target] = pair[3]
            pvalue_source[source_row, target] = self.pvalue_label
            if not first_metadata:
                first_metadata = pair[4]

        return DependenceResult(
            statistic=statistic,
            effect=effect,
            pvalue=pvalue,
            evaluated=evaluated,
            pvalue_source=pvalue_source,
            backend_name=self.name,
            backend_metadata={
                "evaluated_pairs": int(np.count_nonzero(evaluated)),
                "options": dict(self.config.options),
                **first_metadata,
            },
        )


@register_marginal("hsic")
class HsicMarginalBackend(_MarginalAdapter):
    name = "hsic"
    pvalue_label = "shift"

    def __init__(self, *, config: BackendConfig) -> None:
        super().__init__(config=config)
        self.options = _validated_options(
            self.name,
            config.options,
            {
                "bandwidth_x",
                "bandwidth_y",
                "block_size",
                "n_shifts",
                "max_samples",
            },
        )

    def _evaluate_pair(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        source_row: int,
        target: int,
    ) -> PairwiseTestResult:
        return hsic_test(
            x,
            y,
            random_state=stable_seed(
                self.config.random_state, self.name, source_row, target
            ),
            memory_budget_mb=self.config.memory_budget_mb,
            **self.options,
        )


@register_marginal("rff_hsic")
class RffHsicMarginalBackend(_MarginalAdapter):
    name = "rff_hsic"
    pvalue_label = "shift"

    def __init__(self, *, config: BackendConfig) -> None:
        super().__init__(config=config)
        options = _validated_options(
            self.name,
            config.options,
            {
                "n_features",
                "bandwidth",
                "feature_pair_batch_size",
                "cache_max_bytes",
            },
        )
        options.setdefault(
            "cache_max_bytes",
            min(config.memory_budget_mb * 1024**2, 128 * 1024**2),
        )
        self.backend = RffHsicBackend(
            random_state=config.random_state,
            **options,
        )

    def _evaluate_pair(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        source_row: int,
        target: int,
    ) -> PairwiseTestResult:
        return self.backend.evaluate_pairs(
            x[:, 0][None, :],
            y[:, 0][None, :],
            row_keys=[(("source", source_row), ("target", target))],
        )


@register_marginal("mi_knn")
class MutualInformationMarginalBackend(_MarginalAdapter):
    name = "mi_knn"
    pvalue_label = "permutation"

    def __init__(self, *, config: BackendConfig) -> None:
        super().__init__(config=config)
        self.options = _validated_options(
            self.name,
            config.options,
            {"n_neighbors", "n_permutations"},
        )

    def _evaluate_pair(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        source_row: int,
        target: int,
    ) -> Any:
        return mutual_information_test(
            x,
            y,
            random_state=stable_seed(
                self.config.random_state, self.name, source_row, target
            ),
            **self.options,
        )


class _ConditionalAdapter:
    name = ""
    pvalue_label = "conditional"

    def __init__(self, *, config: BackendConfig) -> None:
        self.config = config

    def _evaluate_pair(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        *,
        source_row: int,
        target: int,
    ) -> Any:
        raise NotImplementedError

    @staticmethod
    def _conditioning_matrix(
        context: FitContext,
        plan: ConditioningPlan,
        source_row: int,
        target: int,
    ) -> np.ndarray:
        excluded = set(plan.excluded_rows_for_pair(source_row, target))
        excluded.update((source_row, target))
        controls = [
            row
            for row in range(context.standardized.shape[0])
            if row not in excluded and context.valid_rows[row]
        ]
        if not controls:
            return np.zeros((context.n_samples, 1), dtype=float)
        return context.standardized[np.asarray(controls, dtype=int)].T

    def evaluate(
        self,
        context: FitContext,
        plan: ConditioningPlan,
        *,
        marginal_result: DependenceResult | None = None,
        candidate_mask: np.ndarray | None = None,
    ) -> DependenceResult:
        del marginal_result
        statistic, effect, pvalue, evaluated, pvalue_source = _empty_result(
            context,
            backend_name=self.name,
        )
        selected = context.allowed_pairs.copy()
        if candidate_mask is not None:
            if candidate_mask.shape != selected.shape:
                raise ValueError("candidate_mask must match the allowed-pair shape.")
            selected &= candidate_mask
            pvalue_source[context.allowed_pairs & ~selected] = "skipped"

        conditioning_size = np.zeros(selected.shape, dtype=int)
        first_metadata: dict[str, Any] = {}
        for source_row, target in zip(*np.nonzero(selected), strict=True):
            z = self._conditioning_matrix(
                context,
                plan,
                int(source_row),
                int(target),
            )
            conditioning_size[source_row, target] = z.shape[1]
            result = self._evaluate_pair(
                context.standardized[source_row, :, None],
                context.standardized[target, :, None],
                z,
                source_row=int(source_row),
                target=int(target),
            )
            pair = _pair_values(result)
            statistic[source_row, target] = pair[0]
            effect[source_row, target] = pair[1]
            pvalue[source_row, target] = pair[2]
            evaluated[source_row, target] = pair[3]
            pvalue_source[source_row, target] = self.pvalue_label
            if not first_metadata:
                first_metadata = pair[4]

        return DependenceResult(
            statistic=statistic,
            effect=effect,
            pvalue=pvalue,
            evaluated=evaluated,
            pvalue_source=pvalue_source,
            backend_name=self.name,
            backend_metadata={
                "evaluated_pairs": int(np.count_nonzero(evaluated)),
                "options": dict(self.config.options),
                **first_metadata,
            },
            diagnostics={"conditioning_size": conditioning_size},
        )


@register_conditional("kcit")
class KcitConditionalBackend(_ConditionalAdapter):
    name = "kcit"
    pvalue_label = "local_permutation"

    def __init__(self, *, config: BackendConfig) -> None:
        super().__init__(config=config)
        self.options = _validated_options(
            self.name,
            config.options,
            {
                "bandwidth_x",
                "bandwidth_y",
                "bandwidth_z",
                "ridge",
                "max_samples",
                "local_permutation_neighbors",
                "n_permutations",
            },
        )
        self.options.setdefault(
            "ridge",
            float(config.estimator_config.get("ridge", 1e-6)),
        )

    def _evaluate_pair(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        *,
        source_row: int,
        target: int,
    ) -> Any:
        return kcit_test(
            x,
            y,
            z,
            random_state=stable_seed(
                self.config.random_state, self.name, source_row, target
            ),
            memory_budget_mb=self.config.memory_budget_mb,
            **self.options,
        )


class _RandomizedConditionalAdapter(_ConditionalAdapter):
    mode = "rcot"
    pvalue_label = "analytic_gamma"

    def __init__(self, *, config: BackendConfig) -> None:
        super().__init__(config=config)
        options = _validated_options(
            self.name,
            config.options,
            {
                "n_features_x",
                "n_features_y",
                "n_features_z",
                "n_features_joint",
                "ridge",
                "cache_max_bytes",
            },
        )
        options.setdefault(
            "ridge",
            float(config.estimator_config.get("ridge", 1e-6)),
        )
        options.setdefault(
            "cache_max_bytes",
            min(config.memory_budget_mb * 1024**2, 128 * 1024**2),
        )
        self.backend = RandomizedConditionalBackend(
            mode=self.mode,
            random_state=config.random_state,
            **options,
        )

    def _evaluate_pair(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        *,
        source_row: int,
        target: int,
    ) -> PairwiseTestResult:
        return self.backend.evaluate(
            x,
            y,
            z,
            key=(source_row, target),
        )


@register_conditional("rcit")
class RcitConditionalBackend(_RandomizedConditionalAdapter):
    name = "rcit"
    mode = "rcit"


@register_conditional("rcot")
class RcotConditionalBackend(_RandomizedConditionalAdapter):
    name = "rcot"
    mode = "rcot"


@register_conditional("cmi_knn")
class ConditionalMutualInformationBackend(_ConditionalAdapter):
    name = "cmi_knn"
    pvalue_label = "local_permutation"

    def __init__(self, *, config: BackendConfig) -> None:
        super().__init__(config=config)
        self.options = _validated_options(
            self.name,
            config.options,
            {
                "n_neighbors",
                "local_permutation_neighbors",
                "n_permutations",
            },
        )

    def _evaluate_pair(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        *,
        source_row: int,
        target: int,
    ) -> Any:
        return conditional_mutual_information_test(
            x,
            y,
            z,
            random_state=stable_seed(
                self.config.random_state, self.name, source_row, target
            ),
            **self.options,
        )
