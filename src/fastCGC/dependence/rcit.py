"""Randomized conditional independence utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Literal

import numpy as np
from scipy.stats import gamma as gamma_dist

from .hsic import PairwiseTestResult
from .kernels import ByteBudgetCache, _as_2d_float, median_bandwidth
from .random_features import (
    feature_cross_covariance_norm,
    random_fourier_features,
)

Array = np.ndarray


def residualize_matrix(
    targets: Array,
    conditioning: Array,
    *,
    ridge: float = 1e-6,
) -> Array:
    """Residualize one feature matrix against another with one batched solve."""

    values = np.asarray(targets, dtype=float)
    controls = np.asarray(conditioning, dtype=float)
    if values.ndim != 2 or controls.ndim != 2:
        raise ValueError("targets and conditioning must be two-dimensional")
    if values.shape[0] != controls.shape[0]:
        raise ValueError("targets and conditioning must share the sample axis")
    if ridge <= 0.0:
        raise ValueError("ridge must be positive")
    centered_values = values - values.mean(axis=0, keepdims=True)
    if controls.shape[1] == 0:
        return centered_values
    centered_controls = controls - controls.mean(axis=0, keepdims=True)
    n_samples = values.shape[0]
    gram = centered_controls.T @ centered_controls / n_samples
    gram.flat[:: gram.shape[0] + 1] += ridge
    rhs = centered_controls.T @ centered_values / n_samples
    try:
        factor = np.linalg.cholesky(gram)
        coefficients = np.linalg.solve(factor.T, np.linalg.solve(factor, rhs))
    except np.linalg.LinAlgError:
        coefficients = np.linalg.solve(gram, rhs)
    residuals = centered_values - centered_controls @ coefficients
    return residuals - residuals.mean(axis=0, keepdims=True)


def analytic_null_pvalue(
    statistic: float,
    residual_x: Array,
    residual_y: Array,
) -> float:
    """Approximate a null p-value by moment matching."""

    n_samples = residual_x.shape[0]
    cov_x = residual_x.T @ residual_x / n_samples
    cov_y = residual_y.T @ residual_y / n_samples
    mean = float(np.trace(cov_x) * np.trace(cov_y) / n_samples)
    variance = float(
        2.0
        * np.trace(cov_x @ cov_x)
        * np.trace(cov_y @ cov_y)
        / (n_samples * n_samples)
    )
    if not np.isfinite(mean) or not np.isfinite(variance) or mean <= 0.0 or variance <= 0.0:
        return 1.0 if statistic <= 0.0 else 0.0
    shape = mean * mean / variance
    scale = variance / mean
    return float(gamma_dist.sf(statistic, a=shape, scale=scale))


def _joint_random_features(
    left: Array,
    right: Array,
    *,
    n_features: int,
    bandwidth: float,
    random_state: int | None,
    key: object | None,
) -> Array:
    return random_fourier_features(
        np.hstack([left, right]),
        n_features=n_features,
        bandwidth=bandwidth,
        random_state=random_state,
        key=key,
    )


def rcot_test(
    x: Array,
    y: Array,
    z: Array,
    *,
    n_features_x: int = 32,
    n_features_y: int = 32,
    n_features_z: int = 64,
    bandwidth_x: float | None = None,
    bandwidth_y: float | None = None,
    bandwidth_z: float | None = None,
    ridge: float = 1e-6,
    random_state: int | None = 0,
    key: object | None = None,
) -> PairwiseTestResult:
    """Randomized conditional correlation test."""

    left = _as_2d_float(x)
    right = _as_2d_float(y)
    conditioning = _as_2d_float(z)
    if not (left.shape[0] == right.shape[0] == conditioning.shape[0]):
        raise ValueError("x, y, and z must share the same sample axis")

    x_bandwidth = bandwidth_x or median_bandwidth(left, random_state=random_state)
    y_bandwidth = bandwidth_y or median_bandwidth(right, random_state=random_state)
    z_bandwidth = bandwidth_z or median_bandwidth(conditioning, random_state=random_state)
    features_x = random_fourier_features(
        left,
        n_features=n_features_x,
        bandwidth=x_bandwidth,
        random_state=random_state,
        key=(key, "x"),
    )
    features_y = random_fourier_features(
        right,
        n_features=n_features_y,
        bandwidth=y_bandwidth,
        random_state=random_state,
        key=(key, "y"),
    )
    features_z = random_fourier_features(
        conditioning,
        n_features=n_features_z,
        bandwidth=z_bandwidth,
        random_state=random_state,
        key=(key, "z"),
    )
    residual_x = residualize_matrix(features_x, features_z, ridge=ridge)
    residual_y = residualize_matrix(features_y, features_z, ridge=ridge)
    statistic = feature_cross_covariance_norm(residual_x, residual_y)
    pvalue = analytic_null_pvalue(statistic, residual_x, residual_y)
    return PairwiseTestResult(
        statistic=np.array([statistic]),
        effect=np.array([np.sqrt(max(statistic, 0.0))]),
        pvalue=np.array([pvalue]),
        evaluated=np.array([True]),
        metadata={
            "backend": "rcot",
            "null_method": "analytic_gamma",
            "n_features_x": n_features_x,
            "n_features_y": n_features_y,
            "n_features_z": n_features_z,
            "bandwidth_x": x_bandwidth,
            "bandwidth_y": y_bandwidth,
            "bandwidth_z": z_bandwidth,
        },
    )


def rcit_test(
    x: Array,
    y: Array,
    z: Array,
    *,
    n_features_x: int = 32,
    n_features_y: int = 32,
    n_features_z: int = 64,
    n_features_joint: int = 64,
    bandwidth_x: float | None = None,
    bandwidth_y: float | None = None,
    bandwidth_z: float | None = None,
    bandwidth_joint: float | None = None,
    ridge: float = 1e-6,
    random_state: int | None = 0,
    key: object | None = None,
) -> PairwiseTestResult:
    """Randomized conditional independence test with joint endpoint features."""

    left = _as_2d_float(x)
    right = _as_2d_float(y)
    conditioning = _as_2d_float(z)
    if not (left.shape[0] == right.shape[0] == conditioning.shape[0]):
        raise ValueError("x, y, and z must share the same sample axis")

    x_bandwidth = bandwidth_x or median_bandwidth(left, random_state=random_state)
    y_bandwidth = bandwidth_y or median_bandwidth(right, random_state=random_state)
    z_bandwidth = bandwidth_z or median_bandwidth(conditioning, random_state=random_state)
    joint_bandwidth = bandwidth_joint or median_bandwidth(
        np.hstack([left, right, conditioning]),
        random_state=random_state,
    )
    features_z = random_fourier_features(
        conditioning,
        n_features=n_features_z,
        bandwidth=z_bandwidth,
        random_state=random_state,
        key=(key, "z"),
    )
    joint_left = _joint_random_features(
        left,
        conditioning,
        n_features=n_features_joint,
        bandwidth=joint_bandwidth,
        random_state=random_state,
        key=(key, "joint-left"),
    )
    joint_right = _joint_random_features(
        right,
        conditioning,
        n_features=n_features_joint,
        bandwidth=joint_bandwidth,
        random_state=random_state,
        key=(key, "joint-right"),
    )
    # Populate endpoint-specific feature draws to keep integration hooks aligned
    # with the planned backend configuration.
    _ = random_fourier_features(
        left,
        n_features=n_features_x,
        bandwidth=x_bandwidth,
        random_state=random_state,
        key=(key, "x"),
    )
    _ = random_fourier_features(
        right,
        n_features=n_features_y,
        bandwidth=y_bandwidth,
        random_state=random_state,
        key=(key, "y"),
    )
    residual_left = residualize_matrix(joint_left, features_z, ridge=ridge)
    residual_right = residualize_matrix(joint_right, features_z, ridge=ridge)
    statistic = feature_cross_covariance_norm(residual_left, residual_right)
    pvalue = analytic_null_pvalue(statistic, residual_left, residual_right)
    return PairwiseTestResult(
        statistic=np.array([statistic]),
        effect=np.array([np.sqrt(max(statistic, 0.0))]),
        pvalue=np.array([pvalue]),
        evaluated=np.array([True]),
        metadata={
            "backend": "rcit",
            "null_method": "analytic_gamma",
            "n_features_x": n_features_x,
            "n_features_y": n_features_y,
            "n_features_z": n_features_z,
            "n_features_joint": n_features_joint,
            "bandwidth_x": x_bandwidth,
            "bandwidth_y": y_bandwidth,
            "bandwidth_z": z_bandwidth,
            "bandwidth_joint": joint_bandwidth,
        },
    )


@dataclass
class RandomizedConditionalBackend:
    """Reusable RCIT/RCoT evaluator with deterministic feature caching."""

    mode: Literal["rcit", "rcot"] = "rcot"
    n_features_x: int = 32
    n_features_y: int = 32
    n_features_z: int = 64
    n_features_joint: int = 64
    ridge: float = 1e-6
    random_state: int | None = 0
    cache_max_bytes: int = 128 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.mode not in {"rcit", "rcot"}:
            raise ValueError("mode must be 'rcit' or 'rcot'")
        self._cache = ByteBudgetCache[Array](self.cache_max_bytes)

    def _cached_features(
        self,
        samples: Array,
        *,
        n_features: int,
        bandwidth: float,
        key: Hashable,
    ) -> Array:
        cache_key = (self.mode, key, n_features, round(float(bandwidth), 12))
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        features = random_fourier_features(
            samples,
            n_features=n_features,
            bandwidth=bandwidth,
            random_state=self.random_state,
            key=key,
        )
        return self._cache.set(cache_key, features)

    def evaluate(
        self,
        x: Array,
        y: Array,
        z: Array,
        *,
        key: object | None = None,
    ) -> PairwiseTestResult:
        """Evaluate one conditional candidate and annotate cache stats."""

        left = _as_2d_float(x)
        right = _as_2d_float(y)
        conditioning = _as_2d_float(z)
        x_bandwidth = median_bandwidth(left, random_state=self.random_state)
        y_bandwidth = median_bandwidth(right, random_state=self.random_state)
        z_bandwidth = median_bandwidth(conditioning, random_state=self.random_state)
        if isinstance(key, tuple) and len(key) == 2:
            x_cache_key = (key[0], "x")
            y_cache_key = (key[1], "y")
        else:
            x_cache_key = (key, "x")
            y_cache_key = (key, "y")
        features_x = self._cached_features(
            left,
            n_features=self.n_features_x,
            bandwidth=x_bandwidth,
            key=x_cache_key,
        )
        features_y = self._cached_features(
            right,
            n_features=self.n_features_y,
            bandwidth=y_bandwidth,
            key=y_cache_key,
        )
        features_z = self._cached_features(
            conditioning,
            n_features=self.n_features_z,
            bandwidth=z_bandwidth,
            key=(key, "z"),
        )
        if self.mode == "rcit":
            result = rcit_test(
                left,
                right,
                conditioning,
                n_features_x=self.n_features_x,
                n_features_y=self.n_features_y,
                n_features_z=self.n_features_z,
                n_features_joint=self.n_features_joint,
                ridge=self.ridge,
                random_state=self.random_state,
                key=key,
            )
        else:
            residual_x = residualize_matrix(
                features_x,
                features_z,
                ridge=self.ridge,
            )
            residual_y = residualize_matrix(
                features_y,
                features_z,
                ridge=self.ridge,
            )
            statistic = feature_cross_covariance_norm(residual_x, residual_y)
            result = PairwiseTestResult(
                statistic=np.array([statistic]),
                effect=np.array([np.sqrt(max(statistic, 0.0))]),
                pvalue=np.array(
                    [analytic_null_pvalue(statistic, residual_x, residual_y)]
                ),
                evaluated=np.array([True]),
                metadata={
                    "backend": "rcot",
                    "null_method": "analytic_gamma",
                    "n_features_x": self.n_features_x,
                    "n_features_y": self.n_features_y,
                    "n_features_z": self.n_features_z,
                    "bandwidth_x": x_bandwidth,
                    "bandwidth_y": y_bandwidth,
                    "bandwidth_z": z_bandwidth,
                },
            )
        metadata = dict(result.metadata)
        metadata["cache"] = self._cache.stats()
        return PairwiseTestResult(
            statistic=result.statistic,
            effect=result.effect,
            pvalue=result.pvalue,
            evaluated=result.evaluated,
            metadata=metadata,
        )
