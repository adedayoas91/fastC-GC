from __future__ import annotations

from .base import BackendConfig, ConditionalBackend, MarginalBackend
from .conditioning import ConditioningGroup, ConditioningPlan
from .config import (
    DependenceSelection,
    PROFILE_BACKENDS,
    normalize_backend_options,
    resolve_dependence_selection,
    resolve_evaluation_policy,
    supported_profiles,
    validate_evaluation_policy,
)
from .context import FitContext, build_fit_context
from .linear import (  # noqa: F401
    PearsonMarginalBackend,
    RidgeResidualConditionalBackend,
    _fft_cross_null,
    _schur_precision,
)
from .registry import (
    CONDITIONAL_BACKENDS,
    MARGINAL_BACKENDS,
    create_conditional_backend,
    create_marginal_backend,
    register_conditional,
    register_marginal,
)
from .adapters import (  # noqa: F401
    ConditionalMutualInformationBackend,
    HsicMarginalBackend,
    KcitConditionalBackend,
    MutualInformationMarginalBackend,
    RcitConditionalBackend,
    RcotConditionalBackend,
    RffHsicMarginalBackend,
)
from .results import DependenceResult

__all__ = [
    "BackendConfig",
    "ConditionalBackend",
    "ConditionalMutualInformationBackend",
    "ConditioningGroup",
    "ConditioningPlan",
    "CONDITIONAL_BACKENDS",
    "DependenceResult",
    "DependenceSelection",
    "FitContext",
    "HsicMarginalBackend",
    "KcitConditionalBackend",
    "MARGINAL_BACKENDS",
    "MarginalBackend",
    "MutualInformationMarginalBackend",
    "PROFILE_BACKENDS",
    "PearsonMarginalBackend",
    "RidgeResidualConditionalBackend",
    "RcitConditionalBackend",
    "RcotConditionalBackend",
    "RffHsicMarginalBackend",
    "build_fit_context",
    "create_conditional_backend",
    "create_marginal_backend",
    "normalize_backend_options",
    "register_conditional",
    "register_marginal",
    "resolve_dependence_selection",
    "resolve_evaluation_policy",
    "supported_profiles",
    "validate_evaluation_policy",
    "_fft_cross_null",
    "_schur_precision",
]
