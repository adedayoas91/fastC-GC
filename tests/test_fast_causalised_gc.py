"""Numerical regression tests for :class:`FastGcStar`."""

from __future__ import annotations

import numpy as np
import pytest

from fastCGC import FastGcStar
from fastCGC.core.fast_gc_star import (
    _fft_cross_null,
    _schur_precision,
)


def _brute_shift_test(
    x_rows: np.ndarray, y_rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    observed = np.zeros(x_rows.shape[0])
    pvalues = np.ones(x_rows.shape[0])
    for row, (x, y) in enumerate(zip(x_rows, y_rows)):
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        denominator = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
        if denominator == 0:
            continue
        correlations = np.array(
            [
                np.dot(np.roll(x_centered, shift), y_centered)
                / denominator
                for shift in range(x.size)
            ]
        )
        observed[row] = abs(correlations[0])
        pvalues[row] = (
            1
            + np.count_nonzero(
                np.abs(correlations[1:]) >= observed[row] - 1e-12
            )
        ) / x.size
    return observed, pvalues


def _ridge_residuals(
    standardized: np.ndarray,
    source: int,
    target: int,
    excluded: tuple[int, ...],
    ridge: float,
) -> np.ndarray:
    retained = np.array(
        [
            row
            for row in range(standardized.shape[0])
            if row not in excluded
        ],
        dtype=int,
    )
    controls = retained[(retained != source) & (retained != target)]
    pair = standardized[[source, target]]
    if controls.size == 0:
        return pair
    control_rows = standardized[controls]
    n_samples = standardized.shape[1]
    covariance_ab = pair @ control_rows.T / n_samples
    covariance_bb = control_rows @ control_rows.T / n_samples
    covariance_bb.flat[:: controls.size + 1] += ridge
    coefficients = np.linalg.solve(
        covariance_bb, covariance_ab.T
    ).T
    return pair - coefficients @ control_rows


def _random_data(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(3, 48))
    data[1, 1:] += 0.4 * data[0, :-1]
    data[2, 2:] -= 0.3 * data[1, :-2]
    return data


def test_real_fft_matches_brute_force_complete_shift_orbit() -> None:
    rng = np.random.default_rng(11)
    x_rows = rng.normal(size=(5, 17))
    y_rows = rng.normal(size=(5, 17))
    x_rows[-1] = 2.0

    observed, pvalues = _fft_cross_null(x_rows, y_rows)
    expected_observed, expected_pvalues = _brute_shift_test(x_rows, y_rows)

    np.testing.assert_allclose(observed, expected_observed, atol=1e-12)
    np.testing.assert_allclose(pvalues, expected_pvalues, atol=1e-12)


def test_schur_precision_matches_direct_principal_inverse() -> None:
    rng = np.random.default_rng(3)
    design = rng.normal(size=(9, 9))
    covariance = design @ design.T + 0.2 * np.eye(9)
    precision = np.linalg.inv(covariance)
    excluded = np.array([1, 4, 7])

    retained, schur = _schur_precision(precision, excluded)
    direct = np.linalg.inv(covariance[np.ix_(retained, retained)])

    np.testing.assert_allclose(schur, direct, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("method", ["cgc", "fcgc"])
def test_conditional_scores_and_shift_statistics_match_direct_ridge_residuals(
    method: str,
) -> None:
    model = FastGcStar(
        n_pasts=2,
        n_lags=2,
        method=method,
        ridge=2e-3,
        screen_alpha=1.0,
        candidate_batch_size=3,
    ).fit(_random_data(7))
    standardized = model.standardized_shifted_
    n_variables = model.n_neur

    pairs = [(1, 2), (n_variables, 1), (2 * n_variables, 0)]
    for source, target in pairs:
        lag = source // n_variables
        if method == "cgc":
            excluded = tuple(
                row
                for row in (
                    source % n_variables + block * n_variables
                    for block in range(lag)
                )
                if row != target
            )
        else:
            excluded = ()
        residuals = _ridge_residuals(
            standardized,
            source,
            target,
            excluded,
            model.ridge,
        )
        expected_score, expected_p = _fft_cross_null(
            residuals[[0]], residuals[[1]]
        )

        np.testing.assert_allclose(
            model.inv_corr_[source, target],
            expected_score[0],
            rtol=1e-9,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            model.shift_p_[source, target],
            expected_p[0],
            atol=1e-12,
        )
        assert model.pvalue_source_[source, target] == "shift"
        assert model.pVal_inv_corr_[source, target] == expected_p[0]


def test_ridge_residual_recovery_identity() -> None:
    model = FastGcStar(
        n_pasts=1,
        method="fcgc",
        ridge=5e-3,
        exact_permutation=False,
    ).fit(_random_data(13))
    standardized = model.standardized_shifted_
    precision = model.precision_
    source, target = 3, 1
    pair = np.array([source, target])

    recovered = np.linalg.solve(
        precision[np.ix_(pair, pair)],
        (precision @ standardized)[pair],
    )
    direct = _ridge_residuals(
        standardized, source, target, (), model.ridge
    )

    np.testing.assert_allclose(recovered, direct, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("method", ["cgc", "fcgc"])
def test_scale_invariance_after_standardization(method: str) -> None:
    data = _random_data(19)
    scales = np.array([1e-3, -25.0, 4e4])[:, None]
    first = FastGcStar(
        n_pasts=2,
        method=method,
        ridge=1e-3,
        screen_alpha=1.0,
        candidate_batch_size=5,
    ).fit(data)
    second = FastGcStar(
        n_pasts=2,
        method=method,
        ridge=1e-3,
        screen_alpha=1.0,
        candidate_batch_size=5,
    ).fit(data * scales)

    np.testing.assert_allclose(first.corr_, second.corr_, atol=1e-10)
    np.testing.assert_allclose(first.inv_corr_, second.inv_corr_, atol=1e-9)
    np.testing.assert_allclose(first.pVal_corr_, second.pVal_corr_, atol=1e-12)
    np.testing.assert_allclose(
        first.pVal_inv_corr_, second.pVal_inv_corr_, atol=1e-12
    )


@pytest.mark.parametrize("method", ["cgc", "fcgc"])
def test_constant_rows_are_deterministically_excluded(method: str) -> None:
    data = _random_data(23)
    data[1] = 7.0
    model = FastGcStar(
        n_pasts=2,
        method=method,
        ridge=1e-3,
        screen_alpha=1.0,
    ).fit(data)

    constant_rows = np.array([1, 4, 7])
    assert not np.any(model.valid_row_mask_[constant_rows])
    np.testing.assert_array_equal(model.corr_[constant_rows], 0.0)
    np.testing.assert_array_equal(model.inv_corr_[constant_rows], 0.0)
    np.testing.assert_array_equal(model.pVal_corr_[constant_rows], 1.0)
    np.testing.assert_array_equal(model.pVal_inv_corr_[constant_rows], 1.0)
    np.testing.assert_array_equal(model.corr_[:, 1], 0.0)
    np.testing.assert_array_equal(model.inv_corr_[:, 1], 0.0)


@pytest.mark.parametrize("method", ["cgc", "fcgc"])
def test_zero_lag_self_edges_are_masked_but_lagged_self_edges_are_allowed(
    method: str,
) -> None:
    model = FastGcStar(
        n_pasts=2,
        method=method,
        ridge=1e-3,
        screen_alpha=1.0,
    ).fit(_random_data(29))
    n_variables = model.n_neur
    diagonal = np.arange(n_variables)

    np.testing.assert_array_equal(model.corr_[diagonal, diagonal], 0.0)
    np.testing.assert_array_equal(model.inv_corr_[diagonal, diagonal], 0.0)
    np.testing.assert_array_equal(
        model.pVal_corr_[diagonal, diagonal], 1.0
    )
    np.testing.assert_array_equal(
        model.pVal_inv_corr_[diagonal, diagonal], 1.0
    )
    for variable in range(n_variables):
        lagged_row = n_variables + variable
        assert model.pvalue_source_[lagged_row, variable] != "excluded"


@pytest.mark.parametrize("method", ["cgc", "fcgc"])
def test_fit_shapes_and_separate_diagnostic_shift_outputs(method: str) -> None:
    model = FastGcStar(
        n_pasts=2,
        method=method,
        ridge=1e-3,
        screen_alpha=0.2,
    ).fit(_random_data(31))
    expected_shape = (9, 3)

    assert model.corr_.shape == expected_shape
    assert model.pVal_corr_.shape == expected_shape
    assert model.inv_corr_.shape == expected_shape
    assert model.pVal_inv_corr_.shape == expected_shape
    assert model.screen_score_.shape == expected_shape
    assert model.screen_p_.shape == expected_shape
    np.testing.assert_array_equal(model.analytic_score_, model.screen_score_)
    np.testing.assert_array_equal(model.analytic_pvalue_, model.screen_p_)
    assert model.shift_p_.shape == expected_shape
    np.testing.assert_array_equal(
        model.shift_tested_, np.isfinite(model.shift_p_)
    )
    assert np.all(np.isnan(model.shift_p_[model.pvalue_source_ != "shift"]))
    allowed = model._allowed_pair_mask()
    np.testing.assert_array_equal(model.conditional_evaluated_, allowed)
    assert np.all(model.pvalue_source_[allowed] == "shift")
    assert np.all(np.isfinite(model.shift_p_[allowed]))


@pytest.mark.parametrize("method", ["cgc", "fcgc"])
def test_screen_alpha_is_a_legacy_noop_for_linear_profile(method: str) -> None:
    data = _random_data(32)
    disabled = FastGcStar(
        n_pasts=2,
        method=method,
        screen_alpha=0.0,
    ).fit(data)
    permissive = FastGcStar(
        n_pasts=2,
        method=method,
        screen_alpha=1.0,
    ).fit(data)

    np.testing.assert_allclose(
        disabled.conditional_effect_, permissive.conditional_effect_
    )
    np.testing.assert_allclose(
        disabled.conditional_pvalue_, permissive.conditional_pvalue_
    )
    np.testing.assert_array_equal(
        disabled.conditional_evaluated_, permissive.conditional_evaluated_
    )
    np.testing.assert_array_equal(
        disabled.pvalue_source_, permissive.pvalue_source_
    )


@pytest.mark.parametrize("method", ["cgc", "fcgc"])
@pytest.mark.parametrize("exact_permutation", [True, False])
def test_short_circuit_matches_full_graph_at_fitted_thresholds(
    method: str,
    exact_permutation: bool,
) -> None:
    data = _random_data(33)
    alpha = 0.05
    beta = 0.05
    full = FastGcStar(
        n_pasts=2,
        n_lags=2,
        method=method,
        evaluation_policy="full",
        exact_permutation=exact_permutation,
    ).fit(data, alpha=alpha, beta=beta)
    short = FastGcStar(
        n_pasts=2,
        n_lags=2,
        method=method,
        evaluation_policy="short_circuit",
        exact_permutation=exact_permutation,
    ).fit(data, alpha=alpha, beta=beta)

    full_graph = full.get_connectivity_matrix(alpha=alpha, beta=beta)
    short_graph = short.get_connectivity_matrix(alpha=alpha, beta=beta)

    np.testing.assert_allclose(short_graph, full_graph)
    np.testing.assert_allclose(short.marginal_pvalue_, full.marginal_pvalue_)
    assert np.count_nonzero(short.conditional_evaluated_) < np.count_nonzero(
        full.conditional_evaluated_
    )
    assert np.all(
        short.pvalue_source_[
            full._allowed_pair_mask() & ~short.conditional_evaluated_
        ]
        == "skipped"
    )
    expected_source = "shift" if exact_permutation else "analytic"
    assert np.all(
        short.pvalue_source_[short.conditional_evaluated_]
        == expected_source
    )
    np.testing.assert_array_equal(
        short.conditional_pvalue_[
            full._allowed_pair_mask() & ~short.conditional_evaluated_
        ],
        1.0,
    )
    with pytest.raises(ValueError, match="fitted thresholds"):
        short.get_connectivity_matrix(alpha=0.01, beta=beta)


def test_fixed_ridge_residuals_can_correlate_under_gaussian_ci_moments() -> None:
    z = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=float)
    epsilon_x = np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=float)
    epsilon_y = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=float)
    x = (z + epsilon_x) / np.sqrt(2.0)
    y = (z + epsilon_y) / np.sqrt(2.0)
    standardized = np.vstack((x, y, z))

    for ridge in (1e-3, 0.1, 1.0):
        residuals = _ridge_residuals(
            standardized,
            source=0,
            target=1,
            excluded=(),
            ridge=ridge,
        )
        observed = np.corrcoef(residuals)[0, 1]
        expected = ridge**2 / ((1.0 + ridge) ** 2 + ridge**2)
        assert observed > 0.0
        np.testing.assert_allclose(observed, expected, rtol=1e-10, atol=1e-15)


def test_graph_uses_two_tests_and_a_retained_lag_score() -> None:
    model = FastGcStar(n_pasts=2, n_lags=2, method="fcgc")
    shape = (6, 2)
    model.corr_ = np.full(shape, 0.9)
    model.inv_corr_ = np.zeros(shape)
    model.pVal_corr_ = np.ones(shape)
    model.pVal_inv_corr_ = np.ones(shape)

    # This zero-lag self-pair must be discarded despite both small p-values.
    model.pVal_corr_[0, 0] = 1e-6
    model.pVal_inv_corr_[0, 0] = 1e-6
    model.inv_corr_[0, 0] = 0.99
    # Two retained lags for 0 -> 1; lag two has the larger final score.
    model.pVal_corr_[2, 1] = 1e-6
    model.pVal_inv_corr_[2, 1] = 1e-6
    model.inv_corr_[2, 1] = 0.4
    model.pVal_corr_[4, 1] = 1e-6
    model.pVal_inv_corr_[4, 1] = 1e-6
    model.inv_corr_[4, 1] = 0.8
    # A positive-lag autoregressive edge is permitted.
    model.pVal_corr_[3, 1] = 1e-6
    model.pVal_inv_corr_[3, 1] = 1e-6
    model.inv_corr_[3, 1] = 0.6
    # Conditional significance alone must not create an edge.
    model.pVal_inv_corr_[5, 0] = 1e-6
    model.inv_corr_[5, 0] = 0.95

    graph = model.get_connectivity_matrix(
        simulation=True, alpha=0.01, beta=0.01
    )

    assert graph[0, 0] == 0.0
    assert graph[0, 1] == 0.8
    assert model.edge_lag_[0, 1] == 2
    assert graph[1, 1] == 0.6
    assert graph[1, 0] == 0.0
    assert model.lag_adjacency_.shape == (3, 2, 2)
    assert model.lag_score_.shape == (3, 2, 2)


def test_linear_profile_publishes_canonical_results_and_aliases() -> None:
    model = FastGcStar(n_pasts=2, method="cgc").fit(_random_data(37))

    np.testing.assert_allclose(model.marginal_effect_, model.corr_)
    np.testing.assert_allclose(model.marginal_pvalue_, model.pVal_corr_)
    np.testing.assert_allclose(model.conditional_effect_, model.inv_corr_)
    np.testing.assert_allclose(model.conditional_pvalue_, model.pVal_inv_corr_)
    np.testing.assert_array_equal(model.conditional_evaluated_, model._allowed_pair_mask())
    assert model.backend_metadata_["marginal_backend"] == "pearson"
    assert model.backend_metadata_["conditional_backend"] == "ridge_residual"


def test_invalid_dependence_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown dependence_profile"):
        FastGcStar(dependence_profile="unknown").fit(_random_data(41))
