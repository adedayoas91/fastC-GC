"""Random-feature primitives for fast nonlinear dependence backends."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np

from .kernels import _as_2d_float

Array = np.ndarray


def stable_seed(random_state: int | None, *parts: object) -> int:
    """Derive a deterministic seed from a base seed plus semantic keys."""

    digest = hashlib.blake2b(digest_size=8)
    digest.update(str(0 if random_state is None else int(random_state)).encode())
    for part in parts:
        digest.update(b"\0")
        digest.update(repr(part).encode())
    return int.from_bytes(digest.digest(), "little") % (2**32)


@dataclass(frozen=True)
class RandomFeatureParameters:
    """Random Fourier feature parameters."""

    weights: Array
    phases: Array
    bandwidth: float


def sample_rff_parameters(
    n_input_features: int,
    n_features: int,
    *,
    bandwidth: float,
    random_state: int | None = 0,
    key: object | None = None,
) -> RandomFeatureParameters:
    """Sample deterministic Gaussian random-feature parameters."""

    if n_input_features < 1 or n_features < 1:
        raise ValueError("feature counts must be positive")
    if bandwidth <= 0.0 or not np.isfinite(bandwidth):
        raise ValueError("bandwidth must be finite and positive")
    rng = np.random.default_rng(
        stable_seed(random_state, "rff-parameters", key, n_input_features, n_features)
    )
    weights = rng.normal(
        loc=0.0,
        scale=1.0 / bandwidth,
        size=(n_input_features, n_features),
    )
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_features)
    return RandomFeatureParameters(weights=weights, phases=phases, bandwidth=bandwidth)


def center_features(features: Array) -> Array:
    """Center feature columns."""

    values = np.asarray(features, dtype=float)
    if values.ndim != 2:
        raise ValueError("features must be two-dimensional")
    return values - values.mean(axis=0, keepdims=True)


def random_fourier_features(
    samples: Array,
    *,
    n_features: int,
    bandwidth: float,
    random_state: int | None = 0,
    key: object | None = None,
    dtype: np.dtype | type[np.floating] = np.float64,
    center: bool = True,
) -> Array:
    """Map samples to deterministic, optionally centered RFF features."""

    data = _as_2d_float(samples)
    parameters = sample_rff_parameters(
        data.shape[1],
        n_features,
        bandwidth=bandwidth,
        random_state=random_state,
        key=key,
    )
    projection = data @ parameters.weights + parameters.phases
    features = np.sqrt(2.0 / n_features) * np.cos(projection)
    if center:
        features = center_features(features)
    return features.astype(dtype, copy=False)


def feature_cross_covariance(x_features: Array, y_features: Array) -> Array:
    """Return the feature cross-covariance matrix."""

    left = np.asarray(x_features, dtype=float)
    right = np.asarray(y_features, dtype=float)
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("features must be two-dimensional")
    if left.shape[0] != right.shape[0]:
        raise ValueError("features must share the same sample axis")
    n_samples = left.shape[0]
    if n_samples == 0:
        raise ValueError("features must contain at least one sample")
    return left.T @ right / n_samples


def feature_cross_covariance_norm(x_features: Array, y_features: Array) -> float:
    """Return the squared Frobenius norm of feature cross-covariance."""

    covariance = feature_cross_covariance(x_features, y_features)
    return float(np.sum(covariance * covariance))


def complete_shift_feature_statistics(
    x_features: Array,
    y_features: Array,
    *,
    feature_pair_batch_size: int | None = None,
) -> Array:
    """Evaluate all circular-shift feature cross-covariance norms."""

    left = center_features(np.asarray(x_features, dtype=float))
    right = center_features(np.asarray(y_features, dtype=float))
    if left.shape[0] != right.shape[0]:
        raise ValueError("x_features and y_features must share the same sample axis")
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("features must be two-dimensional")
    n_samples = left.shape[0]
    if n_samples < 1:
        raise ValueError("features must contain at least one sample")

    x_fft = np.fft.rfft(left, axis=0)
    y_fft = np.fft.rfft(right, axis=0)
    x_block = feature_pair_batch_size or left.shape[1]
    y_block = feature_pair_batch_size or right.shape[1]
    statistics = np.zeros(n_samples, dtype=float)
    for x_start in range(0, left.shape[1], x_block):
        x_stop = min(x_start + x_block, left.shape[1])
        x_fft_block = x_fft[:, x_start:x_stop]
        for y_start in range(0, right.shape[1], y_block):
            y_stop = min(y_start + y_block, right.shape[1])
            y_fft_block = y_fft[:, y_start:y_stop]
            cross = np.fft.irfft(
                x_fft_block[:, :, None] * np.conj(y_fft_block[:, None, :]),
                n=n_samples,
                axis=0,
            )
            statistics += np.sum(cross * cross, axis=(1, 2))
    statistics /= n_samples**2
    if n_samples > 1:
        statistics = np.concatenate([statistics[:1], statistics[:0:-1]])
    return statistics
