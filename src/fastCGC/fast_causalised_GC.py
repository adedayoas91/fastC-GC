"""Compatibility imports for the former estimator module path."""

from __future__ import annotations

from .core.fast_gc_star import FastGcStar, _fft_cross_null, _schur_precision

__all__ = ["FastGcStar", "_fft_cross_null", "_schur_precision"]
