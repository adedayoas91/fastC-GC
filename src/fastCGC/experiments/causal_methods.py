"""Study-method runners and the comparison-method registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..core import FastGcStar


@dataclass(frozen=True, slots=True)
class MethodResult:
    """Normalized positive-lag result in ``[lag, source, target]`` order."""

    method: str
    lag_adjacency: np.ndarray
    lag_scores: np.ndarray
    lag_pvalues: np.ndarray | None = None
    lag_marginal_pvalues: np.ndarray | None = None
    lag_conditional_pvalues: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        adjacency = np.asarray(self.lag_adjacency)
        scores = np.asarray(self.lag_scores)
        if adjacency.ndim != 3 or adjacency.shape[-1] != adjacency.shape[-2]:
            raise ValueError(
                "lag_adjacency must have shape (lag, source, target)."
            )
        if scores.shape != adjacency.shape:
            raise ValueError("lag_scores must match lag_adjacency.")
        if self.lag_pvalues is not None:
            pvalues = np.asarray(self.lag_pvalues)
            if pvalues.shape != adjacency.shape:
                raise ValueError("lag_pvalues must match lag_adjacency.")
        for name, values in (
            ("lag_marginal_pvalues", self.lag_marginal_pvalues),
            ("lag_conditional_pvalues", self.lag_conditional_pvalues),
        ):
            if values is not None and np.asarray(values).shape != adjacency.shape:
                raise ValueError(f"{name} must match lag_adjacency.")


@dataclass(frozen=True, slots=True)
class MethodSpec:
    key: str
    display_name: str
    family: str
    track: str
    runnable: bool
    dependency: str
    note: str


METHOD_SPECS: tuple[MethodSpec, ...] = (
    MethodSpec(
        "cgc",
        "c-GC",
        "two-test constraint-based",
        "installed-vcs-aggregate",
        False,
        "gc-extension @ git+https://github.com/adedayoas91/causalized-GC.git@575ac1dc67a0282a5ca88bc11a2804df052f674d",
        "Run the pinned Git distribution; orders above one produce an aggregate graph.",
    ),
    MethodSpec(
        "cgc-star",
        "c-GC*",
        "two-test constraint-based",
        "installed-vcs-aggregate",
        False,
        "gc-extension @ git+https://github.com/adedayoas91/causalized-GC.git@575ac1dc67a0282a5ca88bc11a2804df052f674d",
        "Run native fcgc from the pinned Git distribution; orders above one are aggregate.",
    ),
    MethodSpec(
        "fast-cgc",
        "fastC-GC",
        "two-test constraint-based",
        "local-study-method",
        True,
        "fastCGC",
        "Vectorized c-GC implementation using the correlation profile.",
    ),
    MethodSpec(
        "fast-cgc-star",
        "fastC-GC*",
        "two-test constraint-based",
        "local-study-method",
        True,
        "fastCGC",
        "Vectorized full-conditioning implementation using the correlation profile.",
    ),
    MethodSpec(
        "var-gc",
        "Full VAR-GC coefficient test",
        "regression test",
        "external-repository",
        False,
        "official statsmodels source",
        "Use an established implementation from a pinned upstream revision.",
    ),
    MethodSpec(
        "lasso-var",
        "Lasso-VAR",
        "penalized score-based regression",
        "external-repository",
        False,
        "pinned upstream implementation",
        "Optional classical control; no local fitting implementation.",
    ),
    MethodSpec(
        "pcmci-plus",
        "PCMCI+ (ParCorr)",
        "constraint-based",
        "installed-package",
        False,
        "tigramite==5.2.10.1",
        "Run PCMCI+ from Tigramite in the comparison notebook.",
    ),
    MethodSpec(
        "bagged-pcmci-plus",
        "Bagged PCMCI+ (ParCorr)",
        "bootstrap-aggregated constraint-based",
        "installed-package",
        False,
        "tigramite==5.2.10.1",
        "Run Tigramite's bootstrap aggregation in the comparison notebook.",
    ),
    MethodSpec(
        "cuts-plus",
        "CUTS+",
        "neural Granger / graph neural network",
        "external-repository-aggregate",
        False,
        "github.com/jarrycyx/UNN",
        "Run the authors' CUTS+ code unchanged; score its aggregate variable graph separately.",
    ),
    MethodSpec(
        "mssd",
        "Stationarity-Aware Discovery (MSSD)",
        "constraint-based minimal separating sets",
        "pending-upstream",
        False,
        "authors' StationarityAwareDiscovery repository",
        "AISTATS 2026 method; the advertised official repository was empty when checked on 2026-10-20.",
    ),
    MethodSpec(
        "var-lingam",
        "VAR-LiNGAM",
        "non-Gaussian functional model",
        "external-repository",
        False,
        "github.com/cdt15/lingam",
        "Run the pinned upstream implementation in the non-Gaussian track.",
    ),
    MethodSpec(
        "dynotears",
        "DYNOTEARS",
        "continuous score-based",
        "historical-only",
        False,
        "causalnex (discontinued upstream)",
        "Not a headline comparator; retain only as an explicitly labeled historical sensitivity analysis.",
    ),
    MethodSpec(
        "lpcmci",
        "LPCMCI (ParCorr)",
        "constraint-based latent-variable",
        "pag",
        False,
        "tigramite",
        "Evaluate PAG endpoints separately; not in the directed-DAG leaderboard.",
    ),
)


def method_spec(key: str) -> MethodSpec:
    for spec in METHOD_SPECS:
        if spec.key == key:
            return spec
    raise KeyError(f"Unknown method {key!r}.")


def _two_test_result(
    *,
    method_name: str,
    estimator: FastGcStar,
    data: np.ndarray,
    order: int,
    alpha: float,
    beta: float,
) -> MethodResult:
    estimator.fit(data, alpha=alpha, beta=beta)

    marginal_p = np.asarray(estimator.pVal_corr_, dtype=float)
    conditional_p = np.asarray(estimator.pVal_inv_corr_, dtype=float)
    joint_p = np.maximum(marginal_p, conditional_p)
    selected = (marginal_p <= alpha) & (conditional_p <= beta)
    n_variables = data.shape[0]
    expected = (order + 1) * n_variables
    if selected.shape != (expected, n_variables):
        raise RuntimeError("c-GC output does not match the requested lag order.")

    lag_adjacency = selected.reshape(order + 1, n_variables, n_variables)[1:]
    lag_marginal_pvalues = marginal_p.reshape(
        order + 1, n_variables, n_variables
    )[1:]
    lag_conditional_pvalues = conditional_p.reshape(
        order + 1, n_variables, n_variables
    )[1:]
    lag_pvalues = joint_p.reshape(order + 1, n_variables, n_variables)[1:]
    # The conjunction is stronger when its larger component p-value is smaller.
    lag_scores = 1.0 - lag_pvalues
    return MethodResult(
        method=method_name,
        lag_adjacency=lag_adjacency,
        lag_scores=lag_scores,
        lag_pvalues=lag_pvalues,
        lag_marginal_pvalues=lag_marginal_pvalues,
        lag_conditional_pvalues=lag_conditional_pvalues,
        metadata={
            "alpha": alpha,
            "beta": beta,
            "ridge": estimator.ridge,
            "evaluation_policy": estimator.evaluation_policy,
            "score": "1 - max(marginal_p, conditional_p)",
        },
    )


def fit_fast_cgc(
    data: np.ndarray,
    *,
    order: int,
    alpha: float,
    beta: float,
    seed: int,
    star: bool,
    ridge: float = 1e-6,
    evaluation_policy: str = "full",
    **_: Any,
) -> MethodResult:
    method_name = "fast-cgc-star" if star else "fast-cgc"
    estimator = FastGcStar(
        n_pasts=order,
        n_lags=order,
        method="fcgc" if star else "cgc",
        parallel=False,
        exact_permutation=True,
        evaluation_policy=evaluation_policy,
        dependence_profile="linear",
        ridge=ridge,
        random_state=seed,
    )
    return _two_test_result(
        method_name=method_name,
        estimator=estimator,
        data=data,
        order=order,
        alpha=alpha,
        beta=beta,
    )


FitFunction = Callable[..., MethodResult]


def fit_method(
    method: str,
    data: np.ndarray,
    **kwargs: Any,
) -> MethodResult:
    """Fit an in-repository study method and return its normalized lag graph."""

    runners: dict[str, FitFunction] = {
        "fast-cgc": lambda values, **options: fit_fast_cgc(
            values, star=False, **options
        ),
        "fast-cgc-star": lambda values, **options: fit_fast_cgc(
            values, star=True, **options
        ),
    }
    if method not in runners:
        spec = method_spec(method)
        raise NotImplementedError(f"{spec.display_name}: {spec.note}")
    return runners[method](np.asarray(data, dtype=float), **kwargs)


__all__ = [
    "METHOD_SPECS",
    "MethodResult",
    "MethodSpec",
    "fit_method",
    "method_spec",
]
