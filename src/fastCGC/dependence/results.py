from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class DependenceResult:
    statistic: np.ndarray
    effect: np.ndarray
    pvalue: np.ndarray
    evaluated: np.ndarray
    pvalue_source: np.ndarray
    backend_name: str
    backend_metadata: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
