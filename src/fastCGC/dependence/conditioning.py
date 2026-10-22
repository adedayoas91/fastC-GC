from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .context import FitContext


@dataclass(frozen=True, slots=True)
class ConditioningGroup:
    source_row: int
    excluded_rows: tuple[int, ...]
    targets: np.ndarray


@dataclass(frozen=True, slots=True)
class ConditioningPlan:
    method: str
    groups_by_source: dict[int, tuple[ConditioningGroup, ...]]
    pair_exclusions: dict[tuple[int, int], tuple[int, ...]]

    @classmethod
    def build(cls, context: FitContext, method: str) -> "ConditioningPlan":
        if method not in {"cgc", "fcgc"}:
            raise ValueError("method must be 'cgc' or 'fcgc'.")
        groups_by_source: dict[int, tuple[ConditioningGroup, ...]] = {}
        pair_exclusions: dict[tuple[int, int], tuple[int, ...]] = {}

        for source_row in range(context.standardized.shape[0]):
            if not context.valid_rows[source_row]:
                continue
            source_variable = int(context.row_variable[source_row])
            lag = int(context.row_lag[source_row])
            excluded_history = (
                tuple(source_variable + block * context.n_variables for block in range(lag))
                if method == "cgc"
                else ()
            )
            groups: list[ConditioningGroup] = []
            ordinary_targets = np.array(
                [
                    target
                    for target in range(context.n_variables)
                    if target != source_variable and context.allowed_pairs[source_row, target]
                ],
                dtype=int,
            )
            if ordinary_targets.size:
                groups.append(
                    ConditioningGroup(
                        source_row=source_row,
                        excluded_rows=excluded_history,
                        targets=ordinary_targets,
                    )
                )
                for target in ordinary_targets.tolist():
                    pair_exclusions[(source_row, int(target))] = excluded_history

            if lag > 0 and context.allowed_pairs[source_row, source_variable]:
                add_back = tuple(row for row in excluded_history if row != source_variable)
                self_targets = np.array([source_variable], dtype=int)
                groups.append(
                    ConditioningGroup(
                        source_row=source_row,
                        excluded_rows=add_back,
                        targets=self_targets,
                    )
                )
                pair_exclusions[(source_row, source_variable)] = add_back

            if groups:
                groups_by_source[source_row] = tuple(groups)

        return cls(
            method=method,
            groups_by_source=groups_by_source,
            pair_exclusions=pair_exclusions,
        )

    def groups_for_source(self, source_row: int) -> tuple[ConditioningGroup, ...]:
        return self.groups_by_source.get(source_row, ())

    def excluded_rows_for_pair(self, source_row: int, target: int) -> tuple[int, ...]:
        if (source_row, target) not in self.pair_exclusions:
            raise KeyError((source_row, target))
        return self.pair_exclusions[(source_row, target)]
