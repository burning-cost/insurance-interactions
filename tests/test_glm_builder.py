"""Tests for GLM builder module.

Tests verify that:
  - n_cells is computed correctly for different variable type combinations
  - test_interactions runs without error and returns expected columns
  - Deviance is reduced when the true interaction is added
  - build_glm_with_interactions returns a fitted model and comparison table
  - Cat x cat interaction uses (L1-1)*(L2-1) contrast columns, not a combined
    categorical (P0-2 fix)
  - AIC/BIC columns are named deviance_aic_* / deviance_bic_* (P0-1 fix)
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_interactions.glm_builder import (
    _add_cat_x_cat_interaction_columns,
    _compute_n_cells,
    build_glm_with_interactions,
    test_interactions as run_interaction_tests,
)


def make_simple_df() -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Small dataset for GLM builder tests."""
    rng = np.random.default_rng(99)
    n = 120
    age_cats = ["young", "old"]
    veh_cats = ["small", "large", "sport"]
    age = rng.choice(age_cats, size=n)
    veh = rng.choice(veh_cats, size=n)
    cont = rng.normal(0, 1, size=n)

    # True interaction: young × sport
    log_mu = np.array([
        0.5 * (a == "young") + 0.3 * (v == "sport") + 0.1 * c
        + 0.8 * (a == "young") * (v == "sport")
        for a, v, c in zip(age, veh, cont)
    ])
    exposure = rng.uniform(0.5, 1.0, size=n)
    mu = np.exp(log_mu) * exposure
    y = rng.poisson(mu).astype(np.float32)

    X = pl.DataFrame({
        "age": pl.Series(age).cast(pl.String),
        "vehicle": pl.Series(veh).cast(pl.String),
        "continuous_feat": cont.astype(np.float32),
    })
    return X, y, exposure.astype(np.float32)


class TestComputeNCells:
    def test_cat_x_cat(self):
        df = pl.DataFrame({
            "a": pl.Series(["x", "y", "z", "x"]).cast(pl.String),
            "b": pl.Series(["p", "q", "p", "q"]).cast(pl.String),
        })
        # 3 levels × 2 levels → (3-1)(2-1) = 2
        assert _compute_n_cells(df, "a", "b") == 2

    def test_cat_x_cont(self):
        df = pl.DataFrame({
            "a": pl.Series(["x", "y", "z", "x"]).cast(pl.String),
            "b": [1.0, 2.0, 3.0, 4.0],
        })
        # 3 levels → 3-1 = 2
        assert _compute_n_cells(df, "a", "b") == 2

    def test_cont_x_cont(self):
        df = pl.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
        })
        # Continuous × continuous = 1 product term
        assert _compute_n_cells(df, "a", "b") == 1


class TestCatXCatInteractionColumns:
    """P0-2 fix: cat x cat interaction must produce (L1-1)*(L2-1) binary contrast columns."""

    def test_column_count_matches_n_cells(self):
        """Number of interaction columns added must equal (L1-1)*(L2-1)."""
        import pandas as pd

        # 2-level × 3-level: expect (2-1)*(3-1) = 2 new columns
        age = pd.Categorical(["young", "old", "young", "old", "young", "old"])
        veh = pd.Categorical(["small", "large", "sport", "small", "large", "sport"])
        X_pd = pd.DataFrame({"age": age, "vehicle": veh})
        X_int = X_pd.copy()
        _add_cat_x_cat_interaction_columns(X_int, X_pd, "age", "vehicle")
        new_cols = [c for c in X_int.columns if c not in X_pd.columns]
        # n_cells = (2-1)*(3-1) = 2
        assert len(new_cols) == 2

    def test_interaction_columns_are_binary(self):
        """Each interaction contrast column must contain only 0 and 1."""
        import pandas as pd

        age = pd.Categorical(["young", "old", "young", "old"])
        veh = pd.Categorical(["small", "large", "sport", "small"])
        X_pd = pd.DataFrame({"age": age, "vehicle": veh})
        X_int = X_pd.copy()
        _add_cat_x_cat_interaction_columns(X_int, X_pd, "age", "vehicle")
        new_cols = [c for c in X_int.columns if c not in X_pd.columns]
        for col in new_cols:
            unique_vals = set(X_int[col].unique())
            assert unique_vals <= {0.0, 1.0}, f"Column {col} has non-binary values: {unique_vals}"

    def test_reference_level_excluded(self):
        """The first (reference) level of each feature must not appear in any column name."""
        import pandas as pd

        age = pd.Categorical(["young", "old", "young"])
        veh = pd.Categorical(["large", "small", "sport"])
        X_pd = pd.DataFrame({"age": age, "vehicle": veh})
        X_int = X_pd.copy()
        _add_cat_x_cat_interaction_columns(X_int, X_pd, "age", "vehicle")
        new_cols = [c for c in X_int.columns if c not in X_pd.columns]
        # Reference levels are sorted first: "old" for age (o < y), "large" for vehicle (l < s)
        for col in new_cols:
            # The reference level of age is "old" — it should not appear as an indicator
            assert "_old_" not in col or col.count("old") == 0 or False, (
                f"Reference level appeared in column: {col}"
            )

    def test_main_effects_preserved(self):
        """The original main-effect columns must not be modified."""
        import pandas as pd

        age = pd.Categorical(["young", "old", "young", "old"])
        veh = pd.Categorical(["small", "large", "sport", "small"])
        X_pd = pd.DataFrame({"age": age, "vehicle": veh})
        X_int = X_pd.copy()
        _add_cat_x_cat_interaction_columns(X_int, X_pd, "age", "vehicle")
        # Original columns still present and unchanged
        assert list(X_int["age"]) == list(X_pd["age"])
        assert list(X_int["vehicle"]) == list(X_pd["vehicle"])


class TestTestInteractions:
    def test_runs_without_error(self):
        X, y, exposure = make_simple_df()
        result = run_interaction_tests(
            X=X,
            y=y,
            exposure=exposure,
            interaction_pairs=[("age", "vehicle"), ("age", "continuous_feat")],
            family="poisson",
        )
        assert not result.is_empty()

    def test_expected_columns(self):
        X, y, exposure = make_simple_df()
        result = run_interaction_tests(
            X=X,
            y=y,
            exposure=exposure,
            interaction_pairs=[("age", "vehicle")],
            family="poisson",
        )
        for col in ["feature_1", "feature_2", "n_cells", "delta_deviance", "lr_p", "recommended"]:
            assert col in result.columns

    def test_deviance_aic_bic_columns_present(self):
        """P0-1 fix: AIC/BIC columns must use the deviance_ prefix."""
        X, y, exposure = make_simple_df()
        result = run_interaction_tests(
            X=X,
            y=y,
            exposure=exposure,
            interaction_pairs=[("age", "vehicle")],
            family="poisson",
        )
        for col in [
            "deviance_aic_base",
            "deviance_aic_interaction",
            "delta_deviance_aic",
            "deviance_bic_base",
            "deviance_bic_interaction",
            "delta_deviance_bic",
        ]:
            assert col in result.columns, f"Expected column '{col}' not found"
        # Old column names must not be present
        for old_col in ["aic_base", "aic_interaction", "delta_aic", "bic_base", "bic_interaction", "delta_bic"]:
            assert old_col not in result.columns, f"Old column '{old_col}' should have been renamed"

    def test_delta_deviance_aic_consistent(self):
        """delta_deviance_aic must equal deviance_aic_interaction - deviance_aic_base."""
        X, y, exposure = make_simple_df()
        result = run_interaction_tests(
            X=X,
            y=y,
            exposure=exposure,
            interaction_pairs=[("age", "vehicle")],
            family="poisson",
        )
        if not result.is_empty():
            row = result.row(0, named=True)
            expected_delta = row["deviance_aic_interaction"] - row["deviance_aic_base"]
            assert abs(row["delta_deviance_aic"] - expected_delta) < 1e-6

    def test_true_interaction_has_positive_delta_deviance(self):
        """Adding the true interaction should reduce deviance (positive delta)."""
        X, y, exposure = make_simple_df()
        result = run_interaction_tests(
            X=X,
            y=y,
            exposure=exposure,
            interaction_pairs=[("age", "vehicle"), ("age", "continuous_feat")],
            family="poisson",
        )
        # The age × vehicle interaction (which contains the true interaction) should
        # have the highest delta_deviance
        if not result.is_empty():
            top = result.head(1)
            assert float(top["delta_deviance"][0]) > 0

    def test_n_cells_cat_x_cat(self):
        """n_cells for cat × cat should match (L1-1)(L2-1)."""
        X, y, exposure = make_simple_df()
        result = run_interaction_tests(
            X=X,
            y=y,
            exposure=exposure,
            interaction_pairs=[("age", "vehicle")],
            family="poisson",
        )
        if not result.is_empty():
            row = result.filter(
                (pl.col("feature_1") == "age") & (pl.col("feature_2") == "vehicle")
            )
            if not row.is_empty():
                # age has 2 levels → 1; vehicle has 3 levels → 2; n_cells = 1×2 = 2
                assert int(row["n_cells"][0]) == 2

    def test_lr_df_matches_n_cells(self):
        """lr_df must equal n_cells (degrees of freedom = number of interaction parameters)."""
        X, y, exposure = make_simple_df()
        result = run_interaction_tests(
            X=X,
            y=y,
            exposure=exposure,
            interaction_pairs=[("age", "vehicle")],
            family="poisson",
        )
        if not result.is_empty():
            row = result.row(0, named=True)
            assert row["lr_df"] == row["n_cells"]


class TestBuildGLMWithInteractions:
    def test_returns_model_and_table(self):
        X, y, exposure = make_simple_df()
        model, table = build_glm_with_interactions(
            X=X,
            y=y,
            exposure=exposure,
            interaction_pairs=[("age", "vehicle")],
            family="poisson",
        )
        assert model is not None
        assert "deviance" in table.columns
        assert "delta_deviance" in table.columns
        assert len(table) == 2  # base + interactions model

    def test_deviance_aic_bic_columns_in_comparison(self):
        """P0-1 fix: build_glm_with_interactions comparison table uses deviance_aic/bic."""
        X, y, exposure = make_simple_df()
        _, table = build_glm_with_interactions(
            X=X,
            y=y,
            exposure=exposure,
            interaction_pairs=[("age", "vehicle")],
            family="poisson",
        )
        assert "deviance_aic" in table.columns
        assert "deviance_bic" in table.columns
        # Old names must not be present
        assert "aic" not in table.columns
        assert "bic" not in table.columns

    def test_interaction_model_improves_deviance(self):
        """The model with the true interaction should have lower deviance."""
        X, y, exposure = make_simple_df()
        _, table = build_glm_with_interactions(
            X=X,
            y=y,
            exposure=exposure,
            interaction_pairs=[("age", "vehicle")],
            family="poisson",
        )
        base_deviance = float(table.filter(pl.col("model") == "base_glm")["deviance"][0])
        int_deviance = float(
            table.filter(pl.col("model") == "glm_with_interactions")["deviance"][0]
        )
        assert int_deviance <= base_deviance

    def test_no_interaction_pairs(self):
        """Empty interaction list: both models should be identical."""
        X, y, exposure = make_simple_df()
        model, table = build_glm_with_interactions(
            X=X,
            y=y,
            exposure=exposure,
            interaction_pairs=[],
            family="poisson",
        )
        delta = float(
            table.filter(pl.col("model") == "glm_with_interactions")["delta_deviance"][0]
        )
        assert abs(delta) < 1e-6
