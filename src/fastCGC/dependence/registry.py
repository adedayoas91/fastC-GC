from __future__ import annotations

from typing import Callable

from .base import BackendConfig, ConditionalBackend, MarginalBackend

MARGINAL_BACKENDS: dict[str, type] = {}
CONDITIONAL_BACKENDS: dict[str, type] = {}


def _register(
    registry: dict[str, type],
    kind: str,
    name: str,
) -> Callable[[type], type]:
    def decorator(cls: type) -> type:
        if name in registry:
            raise ValueError(
                f"Duplicate {kind} backend registration for {name!r}."
            )
        registry[name] = cls
        return cls

    return decorator



def register_marginal(name: str) -> Callable[[type], type]:
    return _register(MARGINAL_BACKENDS, "marginal", name)



def register_conditional(name: str) -> Callable[[type], type]:
    return _register(CONDITIONAL_BACKENDS, "conditional", name)



def _create_backend(
    registry: dict[str, type],
    kind: str,
    name: str,
    *,
    config: BackendConfig,
):
    try:
        cls = registry[name]
    except KeyError as exc:
        choices = ", ".join(sorted(registry)) or "<none>"
        raise ValueError(
            f"Unknown {kind} backend {name!r}. Available {kind} backends: {choices}."
        ) from exc
    return cls(config=config)



def create_marginal_backend(
    name: str,
    *,
    config: BackendConfig,
) -> MarginalBackend:
    return _create_backend(MARGINAL_BACKENDS, "marginal", name, config=config)



def create_conditional_backend(
    name: str,
    *,
    config: BackendConfig,
) -> ConditionalBackend:
    return _create_backend(CONDITIONAL_BACKENDS, "conditional", name, config=config)
