"""Tests for the shap_interactions module.

shap_interactions.py has zero existing test coverage. These tests cover:
  - ShapInteractionScore NamedTuple construction and field access
  - shap_to_dataframe: empty list, single pair, multiple pairs, column names,
    normalisation invariant
  - _require_shapiq / _require_catboost: ImportError paths (mocked)
  - compute_shap_interactions: subsampling logic, feature_names default,
    phi_matrix aggregation (mocked shapiq to avoid heavy dependency)
  - fit_catboost: parameter forwarding, family selection (mocked catboost)
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from insurance_interactions.shap_interactions import (
    ShapInteractionScore,
    _require_catboost,
    _require_shapiq,
    shap_to_dataframe,
)


# ---------------------------------------------------------------------------
# ShapInteractionScore
# ---------------------------------------------------------------------------

class TestShapInteractionScore:
    def test_construction(self):
        s = ShapInteractionScore(
            feature_1="age_band",
            feature_2="vehicle_group",
            shap_score=0.42,
            shap_score_normalised=1.0,
        )
        assert s.feature_1 == "age_band"
        assert s.feature_2 == "vehicle_group"
        assert s.shap_score == pytest.approx(0.42)
        assert s.shap_score_normalised == pytest.approx(1.0)

    def test_is_namedtuple(self):
        s = ShapInteractionScore("a", "b", 0.5, 0.5)
        assert isinstance(s, tuple)
        assert len(s) == 4

    def test_field_access_by_index(self):
        s = ShapInteractionScore("a", "b", 0.3, 0.6)
        assert s[0] == "a"
        assert s[1] == "b"
        assert s[2] == pytest.approx(0.3)
        assert s[3] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# shap_to_dataframe
# ---------------------------------------------------------------------------

class TestShapToDataframe:
    def test_empty_returns_empty_dataframe(self):
        df = shap_to_dataframe([])
        assert df.is_empty()

    def test_single_pair_shape(self):
        scores = [ShapInteractionScore("a", "b", 0.5, 1.0)]
        df = shap_to_dataframe(scores)
        assert len(df) == 1

    def test_expected_columns(self):
        scores = [
            ShapInteractionScore("ncd", "age_band", 0.3, 0.6),
            ShapInteractionScore("region", "vehicle_group", 0.5, 1.0),
        ]
        df = shap_to_dataframe(scores)
        assert set(df.columns) == {"feature_1", "feature_2", "shap_score", "shap_score_normalised"}

    def test_scores_preserved_correctly(self):
        scores = [
            ShapInteractionScore("a", "b", 0.8, 1.0),
            ShapInteractionScore("a", "c", 0.4, 0.5),
        ]
        df = shap_to_dataframe(scores)
        assert float(df["shap_score"][0]) == pytest.approx(0.8)
        assert float(df["shap_score_normalised"][1]) == pytest.approx(0.5)

    def test_feature_names_preserved(self):
        scores = [ShapInteractionScore("annual_mileage", "ncd_years", 0.1, 0.2)]
        df = shap_to_dataframe(scores)
        assert df["feature_1"][0] == "annual_mileage"
        assert df["feature_2"][0] == "ncd_years"

    @pytest.mark.parametrize("n_pairs", [1, 5, 15])
    def test_row_count_matches_input(self, n_pairs: int):
        scores = [
            ShapInteractionScore(f"f{i}", f"f{i+1}", float(i), float(i) / n_pairs)
            for i in range(n_pairs)
        ]
        df = shap_to_dataframe(scores)
        assert len(df) == n_pairs


# ---------------------------------------------------------------------------
# _require_shapiq and _require_catboost: ImportError paths
# ---------------------------------------------------------------------------

class TestRequireImports:
    def test_require_shapiq_raises_import_error_when_missing(self):
        with patch.dict(sys.modules, {"shapiq": None}):
            with pytest.raises(ImportError, match="shapiq"):
                _require_shapiq()

    def test_require_catboost_raises_import_error_when_missing(self):
        with patch.dict(sys.modules, {"catboost": None}):
            with pytest.raises(ImportError, match="catboost"):
                _require_catboost()

    def test_require_shapiq_returns_module_when_present(self):
        mock_shapiq = MagicMock()
        with patch.dict(sys.modules, {"shapiq": mock_shapiq}):
            result = _require_shapiq()
            assert result is mock_shapiq

    def test_require_catboost_returns_module_when_present(self):
        mock_catboost = MagicMock()
        with patch.dict(sys.modules, {"catboost": mock_catboost}):
            result = _require_catboost()
            assert result is mock_catboost


# ---------------------------------------------------------------------------
# compute_shap_interactions: mocked shapiq
# ---------------------------------------------------------------------------

class TestComputeShapInteractions:
    """Tests for compute_shap_interactions using a mocked shapiq explainer.

    We mock shapiq entirely to avoid the heavy dependency and focus on testing
    the aggregation logic, subsampling, and output format.
    """

    def _make_mock_shapiq(self, n_features: int, score_value: float = 0.5) -> MagicMock:
        """Build a mock shapiq module whose TreeExplainer returns controlled values."""
        # Build a fake InteractionValues object
        iv = MagicMock()
        # dict() returns {(i, j): score} for all pairs plus some (i,) main effects
        iv_dict = {}
        for i in range(n_features):
            iv_dict[(i,)] = 0.1  # main effect — should be ignored
            for j in range(i + 1, n_features):
                iv_dict[(i, j)] = score_value
        iv.dict.return_value = iv_dict
        iv.values = True  # so hasattr(iv, "values") is True

        mock_shapiq = MagicMock()
        explainer = MagicMock()
        explainer.explain_all.return_value = iv
        mock_shapiq.TreeExplainer.return_value = explainer

        return mock_shapiq

    def test_returns_list_of_scores(self):
        from insurance_interactions.shap_interactions import compute_shap_interactions

        n_features = 4
        X = pl.DataFrame({f"f{i}": np.random.rand(20).astype(np.float32) for i in range(n_features)})
        mock_model = MagicMock()
        mock_shapiq = self._make_mock_shapiq(n_features)

        with patch.dict(sys.modules, {"shapiq": mock_shapiq}):
            scores = compute_shap_interactions(mock_model, X)

        assert isinstance(scores, list)
        assert len(scores) == n_features * (n_features - 1) // 2

    def test_scores_are_namedtuples(self):
        from insurance_interactions.shap_interactions import compute_shap_interactions

        n_features = 3
        X = pl.DataFrame({f"feat_{i}": np.ones(10, dtype=np.float32) for i in range(n_features)})
        mock_model = MagicMock()
        mock_shapiq = self._make_mock_shapiq(n_features)

        with patch.dict(sys.modules, {"shapiq": mock_shapiq}):
            scores = compute_shap_interactions(mock_model, X)

        for s in scores:
            assert isinstance(s, ShapInteractionScore)
            assert isinstance(s.feature_1, str)
            assert isinstance(s.feature_2, str)

    def test_normalised_scores_bounded(self):
        from insurance_interactions.shap_interactions import compute_shap_interactions

        n_features = 4
        X = pl.DataFrame({f"f{i}": np.random.rand(30).astype(np.float32) for i in range(n_features)})
        mock_model = MagicMock()
        mock_shapiq = self._make_mock_shapiq(n_features, score_value=1.0)

        with patch.dict(sys.modules, {"shapiq": mock_shapiq}):
            scores = compute_shap_interactions(mock_model, X)

        for s in scores:
            assert 0.0 <= s.shap_score_normalised <= 1.0 + 1e-8

    def test_scores_ranked_descending(self):
        """Scores should be returned in descending order of shap_score."""
        from insurance_interactions.shap_interactions import compute_shap_interactions

        n_features = 4
        X = pl.DataFrame({f"f{i}": np.random.rand(20).astype(np.float32) for i in range(n_features)})
        mock_model = MagicMock()

        # Give each pair a different score to ensure ordering is testable
        iv = MagicMock()
        score_counter = [0.0]
        iv_dict = {}
        for i in range(n_features):
            for j in range(i + 1, n_features):
                score_counter[0] += 0.1
                iv_dict[(i, j)] = score_counter[0]
        iv.dict.return_value = iv_dict
        iv.values = True

        mock_shapiq = MagicMock()
        explainer = MagicMock()
        explainer.explain_all.return_value = iv
        mock_shapiq.TreeExplainer.return_value = explainer

        with patch.dict(sys.modules, {"shapiq": mock_shapiq}):
            scores = compute_shap_interactions(mock_model, X)

        raw = [s.shap_score for s in scores]
        assert raw == sorted(raw, reverse=True)

    def test_subsampling_occurs_above_max_rows(self):
        """When len(X) > max_rows, the explainer should receive a subset."""
        from insurance_interactions.shap_interactions import compute_shap_interactions

        n_features = 2
        n_rows = 200
        max_rows = 50
        X = pl.DataFrame({f"f{i}": np.random.rand(n_rows).astype(np.float32) for i in range(n_features)})
        mock_model = MagicMock()
        mock_shapiq = self._make_mock_shapiq(n_features)

        with patch.dict(sys.modules, {"shapiq": mock_shapiq}):
            compute_shap_interactions(mock_model, X, max_rows=max_rows)

        # explain_all is called once
        call_args = mock_shapiq.TreeExplainer.return_value.explain_all.call_args
        X_np_passed = call_args[0][0]
        assert X_np_passed.shape[0] == max_rows

    def test_no_subsampling_when_below_max_rows(self):
        """When len(X) <= max_rows, all rows should be used."""
        from insurance_interactions.shap_interactions import compute_shap_interactions

        n_features = 2
        n_rows = 30
        X = pl.DataFrame({f"f{i}": np.random.rand(n_rows).astype(np.float32) for i in range(n_features)})
        mock_model = MagicMock()
        mock_shapiq = self._make_mock_shapiq(n_features)

        with patch.dict(sys.modules, {"shapiq": mock_shapiq}):
            compute_shap_interactions(mock_model, X, max_rows=100)

        call_args = mock_shapiq.TreeExplainer.return_value.explain_all.call_args
        X_np_passed = call_args[0][0]
        assert X_np_passed.shape[0] == n_rows

    def test_custom_feature_names_used(self):
        """feature_names parameter should be reflected in output."""
        from insurance_interactions.shap_interactions import compute_shap_interactions

        n_features = 2
        X = pl.DataFrame({f"col_{i}": np.ones(10, dtype=np.float32) for i in range(n_features)})
        custom_names = ["age_band", "vehicle_group"]
        mock_model = MagicMock()
        mock_shapiq = self._make_mock_shapiq(n_features)

        with patch.dict(sys.modules, {"shapiq": mock_shapiq}):
            scores = compute_shap_interactions(mock_model, X, feature_names=custom_names)

        assert scores[0].feature_1 in custom_names
        assert scores[0].feature_2 in custom_names

    def test_explainer_raises_wrapped_as_runtime_error(self):
        """If shapiq raises, we should get a RuntimeError with a helpful message."""
        from insurance_interactions.shap_interactions import compute_shap_interactions

        X = pl.DataFrame({"a": np.ones(10, dtype=np.float32), "b": np.ones(10, dtype=np.float32)})
        mock_model = MagicMock()

        mock_shapiq = MagicMock()
        explainer = MagicMock()
        explainer.explain_all.side_effect = ValueError("shapiq internal error")
        mock_shapiq.TreeExplainer.return_value = explainer

        with patch.dict(sys.modules, {"shapiq": mock_shapiq}):
            with pytest.raises(RuntimeError, match="shapiq TreeExplainer failed"):
                compute_shap_interactions(mock_model, X)

    def test_main_effects_in_dict_ignored(self):
        """Keys of length 1 (main effects) in the interaction dict must be ignored."""
        from insurance_interactions.shap_interactions import compute_shap_interactions

        n_features = 3
        X = pl.DataFrame({f"f{i}": np.random.rand(20).astype(np.float32) for i in range(n_features)})
        mock_model = MagicMock()

        iv = MagicMock()
        # Include a very large main-effect value that should be ignored
        iv_dict = {
            (0,): 999.0,   # main effect — must not inflate any pairwise score
            (1,): 999.0,
            (0, 1): 0.3,
            (0, 2): 0.2,
            (1, 2): 0.1,
        }
        iv.dict.return_value = iv_dict
        iv.values = True

        mock_shapiq = MagicMock()
        explainer = MagicMock()
        explainer.explain_all.return_value = iv
        mock_shapiq.TreeExplainer.return_value = explainer

        with patch.dict(sys.modules, {"shapiq": mock_shapiq}):
            scores = compute_shap_interactions(mock_model, X)

        # Max normalised score should be 1.0 (corresponding to pair (0,1) = 0.3)
        assert scores[0].shap_score_normalised == pytest.approx(1.0)
        # All normalised scores should be <= 1.0
        for s in scores:
            assert s.shap_score_normalised <= 1.0 + 1e-8


# ---------------------------------------------------------------------------
# fit_catboost: parameter forwarding via mocked catboost
# ---------------------------------------------------------------------------

class TestFitCatboost:
    """Tests for fit_catboost using mocked catboost to avoid heavy dependency."""

    def _make_mock_catboost(self) -> MagicMock:
        mock_cb = MagicMock()
        mock_model = MagicMock()
        mock_cb.CatBoostRegressor.return_value = mock_model
        mock_cb.Pool.return_value = MagicMock()
        return mock_cb

    def test_poisson_family_uses_poisson_loss(self):
        from insurance_interactions.shap_interactions import fit_catboost

        X = pl.DataFrame({
            "age_band": pl.Series(["young", "old", "young", "mid"]).cast(pl.String),
            "annual_mileage": pl.Series([10000.0, 15000.0, 8000.0, 12000.0]),
        })
        y = np.array([1.0, 0.0, 2.0, 1.0], dtype=np.float32)
        mock_cb = self._make_mock_catboost()

        with patch.dict(sys.modules, {"catboost": mock_cb}):
            fit_catboost(X, y, family="poisson")

        call_kwargs = mock_cb.CatBoostRegressor.call_args[1]
        assert call_kwargs["loss_function"] == "Poisson"

    def test_gamma_family_uses_rmse_loss(self):
        from insurance_interactions.shap_interactions import fit_catboost

        X = pl.DataFrame({
            "vehicle_group": pl.Series(["small", "large", "small"]).cast(pl.String),
            "ncd": pl.Series([3.0, 5.0, 0.0]),
        })
        y = np.array([1500.0, 2000.0, 800.0], dtype=np.float32)
        mock_cb = self._make_mock_catboost()

        with patch.dict(sys.modules, {"catboost": mock_cb}):
            fit_catboost(X, y, family="gamma")

        call_kwargs = mock_cb.CatBoostRegressor.call_args[1]
        assert call_kwargs["loss_function"] == "RMSE"

    def test_cat_features_inferred_from_polars_dtype(self):
        """String/Categorical columns should be auto-detected as cat_features."""
        from insurance_interactions.shap_interactions import fit_catboost

        X = pl.DataFrame({
            "age_band": pl.Series(["young", "mid", "old"]).cast(pl.String),
            "vehicle_group": pl.Series(["small", "large", "small"]).cast(pl.String),
            "annual_mileage": pl.Series([10000.0, 15000.0, 8000.0]),
        })
        y = np.array([0.5, 1.0, 0.3], dtype=np.float32)
        mock_cb = self._make_mock_catboost()

        with patch.dict(sys.modules, {"catboost": mock_cb}):
            fit_catboost(X, y, family="poisson")

        pool_call = mock_cb.Pool.call_args
        cat_features = pool_call[1]["cat_features"]
        assert "age_band" in cat_features
        assert "vehicle_group" in cat_features
        assert "annual_mileage" not in cat_features

    def test_explicit_cat_features_respected(self):
        """Explicit cat_features overrides dtype inference."""
        from insurance_interactions.shap_interactions import fit_catboost

        X = pl.DataFrame({
            "region": pl.Series(["north", "south", "north"]).cast(pl.String),
            "cont": pl.Series([1.0, 2.0, 3.0]),
        })
        y = np.array([1.0, 2.0, 0.5], dtype=np.float32)
        mock_cb = self._make_mock_catboost()

        with patch.dict(sys.modules, {"catboost": mock_cb}):
            fit_catboost(X, y, family="poisson", cat_features=["region"])

        pool_call = mock_cb.Pool.call_args
        assert pool_call[1]["cat_features"] == ["region"]

    def test_exposure_none_defaults_to_ones(self):
        """When exposure is not provided, sample_weight should be all ones."""
        from insurance_interactions.shap_interactions import fit_catboost

        X = pl.DataFrame({"cont": pl.Series([1.0, 2.0, 3.0])})
        y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        mock_cb = self._make_mock_catboost()

        with patch.dict(sys.modules, {"catboost": mock_cb}):
            fit_catboost(X, y, family="poisson", exposure=None)

        pool_call = mock_cb.Pool.call_args
        weight = pool_call[1]["weight"]
        np.testing.assert_allclose(weight, np.ones(3, dtype=np.float32))

    def test_returns_fitted_model(self):
        from insurance_interactions.shap_interactions import fit_catboost

        X = pl.DataFrame({"cont": pl.Series([1.0, 2.0, 3.0])})
        y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        mock_cb = self._make_mock_catboost()
        expected_model = mock_cb.CatBoostRegressor.return_value

        with patch.dict(sys.modules, {"catboost": mock_cb}):
            result = fit_catboost(X, y, family="poisson")

        assert result is expected_model

    def test_hyperparameter_forwarding(self):
        """iterations, depth, learning_rate and seed should be forwarded."""
        from insurance_interactions.shap_interactions import fit_catboost

        X = pl.DataFrame({"cont": pl.Series([1.0, 2.0])})
        y = np.array([1.0, 2.0], dtype=np.float32)
        mock_cb = self._make_mock_catboost()

        with patch.dict(sys.modules, {"catboost": mock_cb}):
            fit_catboost(X, y, family="poisson", iterations=100, depth=4, learning_rate=0.1, seed=7)

        call_kwargs = mock_cb.CatBoostRegressor.call_args[1]
        assert call_kwargs["iterations"] == 100
        assert call_kwargs["depth"] == 4
        assert call_kwargs["learning_rate"] == pytest.approx(0.1)
        assert call_kwargs["random_seed"] == 7
