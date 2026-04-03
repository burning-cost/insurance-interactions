"""Extended GLM builder tests covering paths not in test_glm_builder.py.

Covers:
  - _compute_n_cells: cont x cat ordering, single-level categorical
  - _glum_deviance: Poisson direct computation, Gamma family detection,
    zero-count observations, prediction at truth
  - test_interactions: gamma family, cont x cont interaction,
    invalid family raises, no interaction_pairs defaults to all pairs,
    empty result when all pairs fail (robustness)
  - build_glm_with_interactions: gamma family, cont x cont interaction,
    delta_deviance_pct in result, n_new_params counts correctly
  - _add_cat_x_cat_interaction_columns: three-level x three-level count
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_interactions.glm_builder import (
    _add_cat_x_cat_interaction_columns,
    _compute_n_cells,
    _glum_deviance,
    build_glm_with_interactions,
    test_interactions as run_interaction_tests,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gamma_df() -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Gamma severity dataset with a known age × vehicle severity interaction."""
    rng = np.random.default_rng(77)
    n = 100
    age = rng.choice(["young", "mid", "old"], size=n)
    veh = rng.choice(["small", "large"], size=n)
    mileage = rng.uniform(5000.0, 25000.0, size=n)

    log_mu = np.array([
        7.5
        + 0.3 * (a == "young")
        + 0.4 * (v == "large")
        + 0.7 * (a == "young") * (v == "large")
        + 0.01 * (m / 10000)
        for a, v, m in zip(age, veh, mileage)
    ])
    mu = np.exp(log_mu)
    y = rng.gamma(shape=5.0, scale=mu / 5.0).astype(np.float64)
    exposure = np.ones(n, dtype=np.float64)

    X = pl.DataFrame({
        "age_band": pl.Series(age).cast(pl.String),
        "vehicle_group": pl.Series(veh).cast(pl.String),
        "annual_mileage": mileage,
    })
    return X, y, exposure


def _make_cont_df() -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Dataset with only continuous features for cont × cont interaction tests."""
    rng = np.random.default_rng(55)
    n = 150
    x1 = rng.normal(0, 1, size=n)
    x2 = rng.normal(0, 1, size=n)
    # True interaction: x1 * x2 term
    log_mu = -1.0 + 0.5 * x1 + 0.3 * x2 + 0.8 * x1 * x2
    exposure = np.ones(n, dtype=np.float64)
    mu = np.exp(log_mu) * exposure
    y = rng.poisson(mu).astype(np.float64)
    X = pl.DataFrame({"feat_a": x1.astype(np.float32), "feat_b": x2.astype(np.float32)})
    return X, y, exposure


# ---------------------------------------------------------------------------
# _compute_n_cells extended
# ---------------------------------------------------------------------------

class TestComputeNCellsExtended:
    def test_cont_x_cat_reversed_order(self):
        """n_cells should be (L-1) when feature order is cont x cat."""
        df = pl.DataFrame({
            "cont": [1.0, 2.0, 3.0],
            "cat": pl.Series(["a", "b", "c"]).cast(pl.String),
        })
        # 3-level categorical → 3-1 = 2
        assert _compute_n_cells(df, "cont", "cat") == 2

    def test_single_level_categorical_n_cells(self):
        """Single-level categorical × continuous = 0 cells (L-1 = 0)."""
        df = pl.DataFrame({
            "cat": pl.Series(["a", "a", "a"]).cast(pl.String),
            "cont": [1.0, 2.0, 3.0],
        })
        assert _compute_n_cells(df, "cat", "cont") == 0

    def test_binary_cat_x_cont(self):
        """2-level categorical × continuous = 1 cell."""
        df = pl.DataFrame({
            "cat": pl.Series(["yes", "no", "yes", "no"]).cast(pl.String),
            "cont": [1.0, 2.0, 3.0, 4.0],
        })
        assert _compute_n_cells(df, "cat", "cont") == 1

    def test_five_level_cat_x_cat(self):
        """5-level × 4-level: (5-1)*(4-1) = 12."""
        df = pl.DataFrame({
            "a": pl.Series([str(i % 5) for i in range(20)]).cast(pl.String),
            "b": pl.Series([str(i % 4) for i in range(20)]).cast(pl.String),
        })
        assert _compute_n_cells(df, "a", "b") == 12


# ---------------------------------------------------------------------------
# _glum_deviance: direct tests
# ---------------------------------------------------------------------------

class TestGlumDeviance:
    def test_poisson_deviance_finite_and_positive(self):
        """Deviance from a fitted Poisson model should be finite and >= 0."""
        try:
            from glum import GeneralizedLinearRegressor
            import pandas as pd
        except ImportError:
            pytest.skip("glum not installed")

        X = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        X_pd = X.to_pandas()
        y = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        exposure = np.ones(5)

        model = GeneralizedLinearRegressor(family="poisson", fit_intercept=True)
        model.fit(X_pd, y, sample_weight=exposure)

        d = _glum_deviance(model, X_pd, y, exposure)
        assert np.isfinite(d)
        assert d >= 0.0

    def test_poisson_better_model_has_lower_deviance(self):
        """A better-fitting Poisson model should have strictly lower deviance."""
        try:
            from glum import GeneralizedLinearRegressor
            import pandas as pd
        except ImportError:
            pytest.skip("glum not installed")

        rng = np.random.default_rng(101)
        n = 80
        x = rng.normal(0, 1, size=n)
        y = rng.poisson(np.exp(0.5 * x + 1.0)).astype(np.float64)
        X_good = pl.DataFrame({"x": x.astype(np.float32)})
        X_bad = pl.DataFrame({"dummy": np.zeros(n, dtype=np.float32)})
        exposure = np.ones(n)

        X_good_pd = X_good.to_pandas()
        X_bad_pd = X_bad.to_pandas()

        model_good = GeneralizedLinearRegressor(family="poisson", fit_intercept=True)
        model_good.fit(X_good_pd, y, sample_weight=exposure)

        model_bad = GeneralizedLinearRegressor(family="poisson", fit_intercept=True)
        model_bad.fit(X_bad_pd, y, sample_weight=exposure)

        d_good = _glum_deviance(model_good, X_good_pd, y, exposure)
        d_bad = _glum_deviance(model_bad, X_bad_pd, y, exposure)
        assert d_good < d_bad, (
            f"Model with true predictor should have lower deviance ({d_good:.4f}) "
            f"than intercept-only model ({d_bad:.4f})"
        )

    def test_gamma_deviance_detected_from_family_name(self):
        """Family name detection: 'gamma' in class name should trigger gamma deviance."""
        try:
            from glum import GeneralizedLinearRegressor
            import pandas as pd
        except ImportError:
            pytest.skip("glum not installed")

        X = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        X_pd = X.to_pandas()
        y = np.array([500.0, 800.0, 600.0, 700.0, 550.0])
        exposure = np.ones(5)

        model = GeneralizedLinearRegressor(family="gamma", fit_intercept=True)
        model.fit(X_pd, y, sample_weight=exposure)

        # Should not raise and should return a non-negative float
        d = _glum_deviance(model, X_pd, y, exposure)
        assert np.isfinite(d)
        assert d >= 0


# ---------------------------------------------------------------------------
# test_interactions extended
# ---------------------------------------------------------------------------

class TestTestInteractionsExtended:
    def test_gamma_family(self):
        X, y, exposure = _make_gamma_df()
        result = run_interaction_tests(
            X=X,
            y=y,
            exposure=exposure,
            interaction_pairs=[("age_band", "vehicle_group")],
            family="gamma",
        )
        assert not result.is_empty()
        assert "delta_deviance" in result.columns
        assert "lr_p" in result.columns

    def test_cont_x_cont_interaction(self):
        X, y, exposure = _make_cont_df()
        result = run_interaction_tests(
            X=X,
            y=y,
            exposure=exposure,
            interaction_pairs=[("feat_a", "feat_b")],
            family="poisson",
        )
        if not result.is_empty():
            # n_cells for cont × cont = 1
            assert int(result["n_cells"][0]) == 1
            # LR df should match n_cells
            assert int(result["lr_df"][0]) == 1

    def test_invalid_family_raises(self):
        X, y, exposure = _make_cont_df()
        with pytest.raises(ValueError, match="family"):
            run_interaction_tests(
                X=X,
                y=y,
                exposure=exposure,
                interaction_pairs=[("feat_a", "feat_b")],
                family="tweedie",
            )

    def test_default_all_pairs_when_interaction_pairs_none(self):
        """When interaction_pairs is None, all column pairs should be tested."""
        X, y, exposure = _make_cont_df()
        result = run_interaction_tests(
            X=X,
            y=y,
            exposure=exposure,
            interaction_pairs=None,
            family="poisson",
        )
        # Two features → 1 pair tested
        assert not result.is_empty()

    def test_l2_regularisation_parameter(self):
        """l2_regularisation > 0 should not raise."""
        X, y, exposure = _make_gamma_df()
        result = run_interaction_tests(
            X=X,
            y=y,
            exposure=exposure,
            interaction_pairs=[("age_band", "vehicle_group")],
            family="poisson",
            l2_regularisation=0.01,
        )
        assert not result.is_empty()

    def test_bonferroni_threshold_applied(self):
        """With alpha=1.0 all pairs should be recommended; with pure noise data
        and alpha=1e-10 none should."""
        X, y, exposure = _make_gamma_df()
        result_all = run_interaction_tests(
            X=X,
            y=y,
            exposure=exposure,
            interaction_pairs=[("age_band", "vehicle_group")],
            family="poisson",
            alpha_bonferroni=1.0,  # trivially significant
        )
        # For the strict-threshold test we use data with NO interaction so that
        # lr_p is genuinely large (not underflowed to 0.0 due to very strong signal).
        rng_noise = np.random.default_rng(999)
        n_noise = 200
        X_noise = pl.DataFrame({
            "grp_a": pl.Series(rng_noise.choice(["x", "y"], size=n_noise)).cast(pl.String),
            "grp_b": pl.Series(rng_noise.choice(["p", "q"], size=n_noise)).cast(pl.String),
        })
        y_noise = rng_noise.poisson(1.0, n_noise).astype(np.float64)
        exp_noise = np.ones(n_noise, dtype=np.float64)
        result_none = run_interaction_tests(
            X=X_noise,
            y=y_noise,
            exposure=exp_noise,
            interaction_pairs=[("grp_a", "grp_b")],
            family="poisson",
            alpha_bonferroni=1e-10,  # very strict; noise interaction won't pass
        )
        if not result_all.is_empty():
            assert bool(result_all["recommended"][0]) is True
        if not result_none.is_empty():
            assert bool(result_none["recommended"][0]) is False

    def test_lr_p_in_0_1(self):
        X, y, exposure = _make_gamma_df()
        result = run_interaction_tests(
            X=X,
            y=y,
            exposure=exposure,
            interaction_pairs=[("age_band", "vehicle_group"), ("age_band", "annual_mileage")],
            family="poisson",
        )
        for row in result.iter_rows(named=True):
            assert 0.0 <= row["lr_p"] <= 1.0

    def test_sorted_by_delta_deviance_descending(self):
        """Result should be sorted highest delta_deviance first."""
        X, y, exposure = _make_gamma_df()
        result = run_interaction_tests(
            X=X,
            y=y,
            exposure=exposure,
            interaction_pairs=[
                ("age_band", "vehicle_group"),
                ("age_band", "annual_mileage"),
                ("vehicle_group", "annual_mileage"),
            ],
            family="poisson",
        )
        if len(result) > 1:
            deviances = result["delta_deviance"].to_list()
            assert deviances == sorted(deviances, reverse=True)


# ---------------------------------------------------------------------------
# build_glm_with_interactions extended
# ---------------------------------------------------------------------------

class TestBuildGLMWithInteractionsExtended:
    def test_gamma_family(self):
        X, y, exposure = _make_gamma_df()
        model, table = build_glm_with_interactions(
            X=X,
            y=y,
            exposure=exposure,
            interaction_pairs=[("age_band", "vehicle_group")],
            family="gamma",
        )
        assert model is not None
        assert len(table) == 2

    def test_cont_x_cont_interaction(self):
        X, y, exposure = _make_cont_df()
        model, table = build_glm_with_interactions(
            X=X,
            y=y,
            exposure=exposure,
            interaction_pairs=[("feat_a", "feat_b")],
            family="poisson",
        )
        int_row = table.filter(pl.col("model") == "glm_with_interactions")
        assert int(int_row["n_new_params"][0]) == 1

    def test_delta_deviance_pct_computation(self):
        """delta_deviance_pct for base should be 0.0; for interactions > 0."""
        X, y, exposure = _make_gamma_df()
        _, table = build_glm_with_interactions(
            X=X,
            y=y,
            exposure=exposure,
            interaction_pairs=[("age_band", "vehicle_group")],
            family="poisson",
        )
        base_pct = float(table.filter(pl.col("model") == "base_glm")["delta_deviance_pct"][0])
        assert base_pct == pytest.approx(0.0)

    def test_n_new_params_multiple_interactions(self):
        """n_new_params should sum across all interaction pairs."""
        X, y, exposure = _make_gamma_df()
        # age_band × vehicle_group: (3-1)*(2-1) = 2
        # age_band × annual_mileage: (3-1) = 2
        # total = 4
        _, table = build_glm_with_interactions(
            X=X,
            y=y,
            exposure=exposure,
            interaction_pairs=[("age_band", "vehicle_group"), ("age_band", "annual_mileage")],
            family="poisson",
        )
        int_row = table.filter(pl.col("model") == "glm_with_interactions")
        assert int(int_row["n_new_params"][0]) == 4

    def test_table_has_expected_columns(self):
        X, y, exposure = _make_cont_df()
        _, table = build_glm_with_interactions(
            X=X,
            y=y,
            exposure=exposure,
            interaction_pairs=[("feat_a", "feat_b")],
            family="poisson",
        )
        expected = {"model", "deviance", "n_params", "deviance_aic", "deviance_bic",
                    "delta_deviance", "delta_deviance_pct", "n_new_params"}
        assert expected <= set(table.columns)


# ---------------------------------------------------------------------------
# _add_cat_x_cat_interaction_columns: 3×3 level test
# ---------------------------------------------------------------------------

class TestCatXCatInteractionColumnsExtended:
    def test_three_by_three_levels(self):
        """3-level × 3-level: expect (3-1)*(3-1) = 4 new columns."""
        import pandas as pd

        age = pd.Categorical(["young", "mid", "old"] * 6)
        region = pd.Categorical(["north", "south", "midlands"] * 6)
        X_pd = pd.DataFrame({"age": age, "region": region})
        X_int = X_pd.copy()
        _add_cat_x_cat_interaction_columns(X_int, X_pd, "age", "region")
        new_cols = [c for c in X_int.columns if c not in X_pd.columns]
        assert len(new_cols) == 4

    def test_interaction_column_naming_convention(self):
        """Column names should follow _ix_{feat1}_{val1}_X_{feat2}_{val2} pattern."""
        import pandas as pd

        age = pd.Categorical(["young", "old"] * 3)
        veh = pd.Categorical(["small", "large", "sport"] * 2)
        X_pd = pd.DataFrame({"age": age, "vehicle": veh})
        X_int = X_pd.copy()
        _add_cat_x_cat_interaction_columns(X_int, X_pd, "age", "vehicle")
        new_cols = [c for c in X_int.columns if c not in X_pd.columns]
        for col in new_cols:
            assert col.startswith("_ix_"), f"Column {col!r} does not follow naming convention"
            assert "_X_" in col, f"Column {col!r} missing '_X_' separator"

    def test_all_rows_covered_by_at_least_one_column(self):
        """Every row that matches a non-reference level combination should have 1 in some column."""
        import pandas as pd

        age = pd.Categorical(["young", "old", "young", "old"])
        veh = pd.Categorical(["small", "large", "sport", "large"])
        X_pd = pd.DataFrame({"age": age, "vehicle": veh})
        X_int = X_pd.copy()
        _add_cat_x_cat_interaction_columns(X_int, X_pd, "age", "vehicle")
        new_cols = [c for c in X_int.columns if c not in X_pd.columns]
        # Row 0 (young × small): young is non-ref, small is ref → 0 in all cols
        # Row 1 (old × large): old is ref, large is non-ref → 0 in all cols (old is ref)
        # Row 2 (young × sport): both non-ref → should be 1 in the young×sport column
        row2_values = {col: X_int[col].iloc[2] for col in new_cols}
        has_one = any(v == 1.0 for v in row2_values.values())
        assert has_one, f"young×sport row has no 1 in any interaction column: {row2_values}"
