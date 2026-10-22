from __future__ import annotations

import pytest

from fastCGC.dependence import (
    BackendConfig,
    create_conditional_backend,
    create_marginal_backend,
    register_conditional,
    register_marginal,
)


def test_unknown_backend_factory_lists_available_names() -> None:
    with pytest.raises(
        ValueError,
        match=r"Available marginal backends: .*pearson",
    ):
        create_marginal_backend("missing", config=BackendConfig())
    with pytest.raises(
        ValueError,
        match=r"Available conditional backends: .*ridge_residual",
    ):
        create_conditional_backend("missing", config=BackendConfig())


def test_duplicate_registration_is_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate marginal backend"):

        @register_marginal("pearson")
        class _DuplicateMarginal:
            pass

    with pytest.raises(ValueError, match="Duplicate conditional backend"):

        @register_conditional("ridge_residual")
        class _DuplicateConditional:
            pass
