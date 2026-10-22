from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from .conditioning import ConditioningPlan
    from .context import FitContext
    from .results import DependenceResult


@dataclass(slots=True)
class BackendConfig:
    options: Mapping[str, Any] = field(default_factory=dict)
    random_state: int | None = 0
    memory_budget_mb: int = 1024
    estimator_config: Mapping[str, Any] = field(default_factory=dict)


class MarginalBackend(Protocol):
    name: str

    def evaluate(
        self,
        context: FitContext,
        *,
        candidate_mask: np.ndarray | None = None,
    ) -> DependenceResult: ...


class ConditionalBackend(Protocol):
    name: str

    def evaluate(
        self,
        context: FitContext,
        plan: ConditioningPlan,
        *,
        marginal_result: DependenceResult | None = None,
        candidate_mask: np.ndarray | None = None,
    ) -> DependenceResult: ...
