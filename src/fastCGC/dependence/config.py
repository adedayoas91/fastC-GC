from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

PROFILE_BACKENDS: dict[str, tuple[str, str]] = {
    "linear": ("pearson", "ridge_residual"),
    "kernel": ("hsic", "kcit"),
    "randomized_kernel_rcit": ("rff_hsic", "rcit"),
    "randomized_kernel": ("rff_hsic", "rcot"),
    "information": ("mi_knn", "cmi_knn"),
}
VALID_EVALUATION_POLICIES = {"auto", "full", "short_circuit"}


@dataclass(frozen=True, slots=True)
class DependenceSelection:
    dependence_profile: str
    marginal_test: str
    conditional_test: str


@dataclass(frozen=True, slots=True)
class BackendSelection:
    marginal: str
    conditional: str


def supported_profiles() -> tuple[str, ...]:
    return tuple(PROFILE_BACKENDS)


def normalize_backend_options(
    backend_options: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, dict[str, object]]:
    if backend_options is None:
        return {}
    return {
        str(name): dict(options)
        for name, options in backend_options.items()
    }


def validate_evaluation_policy(policy: str) -> str:
    if policy not in VALID_EVALUATION_POLICIES:
        choices = ", ".join(sorted(VALID_EVALUATION_POLICIES))
        raise ValueError(
            f"evaluation_policy must be one of {{{choices}}}."
        )
    return policy


def resolve_dependence_selection(
    *,
    dependence_profile: str,
    marginal_test: str | None,
    conditional_test: str | None,
) -> DependenceSelection:
    if dependence_profile not in PROFILE_BACKENDS:
        choices = ", ".join(supported_profiles())
        raise ValueError(
            f"Unknown dependence_profile {dependence_profile!r}. "
            f"Valid profiles: {choices}."
        )
    profile_marginal, profile_conditional = PROFILE_BACKENDS[dependence_profile]
    return DependenceSelection(
        dependence_profile=dependence_profile,
        marginal_test=marginal_test or profile_marginal,
        conditional_test=conditional_test or profile_conditional,
    )


def resolve_evaluation_policy(
    *,
    requested: str,
    selection: DependenceSelection,
) -> str:
    validate_evaluation_policy(requested)
    if requested != "auto":
        return requested
    if selection.dependence_profile == "linear":
        return "full"
    return "short_circuit"
