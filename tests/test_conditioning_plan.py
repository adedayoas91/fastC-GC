from __future__ import annotations

import numpy as np
import pytest

from fastCGC.dependence import ConditioningPlan, build_fit_context


@pytest.mark.parametrize("method", ["cgc", "fcgc"])
def test_conditioning_plan_matches_direct_pair_exclusions(method: str) -> None:
    rng = np.random.default_rng(5)
    data = rng.normal(size=(4, 32))
    context = build_fit_context(data, n_pasts=3)
    plan = ConditioningPlan.build(context, method)

    for source_row in range(context.standardized.shape[0]):
        source_variable = source_row % context.n_variables
        lag = source_row // context.n_variables
        for target in range(context.n_variables):
            if not context.allowed_pairs[source_row, target]:
                continue
            if method == "fcgc":
                expected = ()
            else:
                excluded = tuple(
                    source_variable + block * context.n_variables
                    for block in range(lag)
                )
                if target == source_variable and lag > 0:
                    expected = tuple(row for row in excluded if row != source_variable)
                else:
                    expected = excluded
            assert plan.excluded_rows_for_pair(source_row, target) == expected
