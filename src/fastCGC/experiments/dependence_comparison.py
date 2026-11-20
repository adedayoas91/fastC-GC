"""Matched dependence-test variants for the fast c-GC study methods."""

from __future__ import annotations

import numpy as np

from ..core import FastGcStar
from .causal_methods import MethodResult, _two_test_result, fit_method


DEPENDENCE_TESTS = {
    "linear": ("pearson", "ridge_residual"),
    "rff-hsic": ("rff_hsic", "ridge_residual"),
    "rcot": ("pearson", "rcot"),
    "rff-hsic-rcot": ("rff_hsic", "rcot"),
}

SCM_REGIMES = ("reference", "mixed-link-functions", "cross-link-null")


def fit_dependence_variant(
    method: str,
    data: np.ndarray,
    *,
    order: int,
    alpha: float,
    beta: float,
    seed: int,
    variant: str,
) -> MethodResult:
    """Fit a fast study method with the selected pair of dependence tests."""

    if method not in {"fast-cgc", "fast-cgc-star"}:
        raise ValueError(f"Unsupported fast method {method!r}.")
    marginal, conditional = DEPENDENCE_TESTS[variant]
    if variant == "linear":
        return fit_method(method, data, order=order, alpha=alpha, beta=beta, seed=seed)

    estimator = FastGcStar(
        n_pasts=order,
        n_lags=order,
        method="fcgc" if method == "fast-cgc-star" else "cgc",
        parallel=False,
        exact_permutation=True,
        evaluation_policy="full",
        ridge=1e-6,
        dependence_profile="linear",
        marginal_test=marginal,
        conditional_test=conditional,
        random_state=seed,
    )
    return _two_test_result(
        method_name=method,
        estimator=estimator,
        data=data,
        order=order,
        alpha=alpha,
        beta=beta,
    )
