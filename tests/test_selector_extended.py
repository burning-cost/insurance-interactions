"""Extended InteractionDetector (selector) tests.

Covers paths not in test_selector.py:
  - nid_table() method
  - glm_test_table() before fit raises RuntimeError
  - suggest_interactions with no significant results
  - suggest_interactions with require_significant=False
  - three-way NID config (nid_max_order=3)
  - SHAP warning path: bad shap_model triggers UserWarning, not exception
  - MLP-M config flows through to CANN
  - detector.nid_scores property access after fit
  - interaction_table returns a Polars DataFrame
  - DetectorConfig defaults are sane (validate field types)
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

pytest.importorskip("torch", reason="torch not installed — InteractionDetector requires torch")

from insurance_interactions import DetectorConfig, InteractionDetector


# ---------------------------------------------------------------------------
# Shared minimal config for fast tests
# ---------------------------------------------------------------------------

def _fast_cfg(**kwargs) -> DetectorConfig:
    defaults = dict(
        cann_n_epochs=5,
        cann_n_ensemble=1,
        cann_patience=3,
        cann_hidden_dims=[8, 4],
        top_k_nid=5,
    )
    defaults.update(kwargs)
    return DetectorConfig(**defaults)


# ---------------------------------------------------------------------------
# nid_table() method
# ---------------------------------------------------------------------------

class TestNidTable:
    def test_nid_table_returns_dataframe(self, synthetic_poisson_data):
        data = synthetic_poisson_data
        detector = InteractionDetector(family="poisson", config=_fast_cfg())
        detector.fit(
            X=data["X"],
            y=data["y"],
            glm_predictions=data["glm_predictions"],
            exposure=data["exposure"],
        )
        df = detector.nid_table()
        assert isinstance(df, pl.DataFrame)

    def test_nid_table_has_pairwise_columns(self, synthetic_poisson_data):
        data = synthetic_poisson_data
        detector = InteractionDetector(family="poisson", config=_fast_cfg())
        detector.fit(
            X=data["X"],
            y=data["y"],
            glm_predictions=data["glm_predictions"],
            exposure=data["exposure"],
        )
        df = detector.nid_table()
        assert "feature_1" in df.columns
        assert "feature_2" in df.columns
        assert "nid_score" in df.columns

    def test_nid_table_not_empty_after_fit(self, synthetic_poisson_data):
        data = synthetic_poisson_data
        detector = InteractionDetector(family="poisson", config=_fast_cfg())
        detector.fit(
            X=data["X"],
            y=data["y"],
            glm_predictions=data["glm_predictions"],
            exposure=data["exposure"],
        )
        assert not detector.nid_table().is_empty()

    def test_nid_table_pairwise_only(self, synthetic_poisson_data):
        """nid_table() should only contain pairwise (order-2) scores."""
        data = synthetic_poisson_data
        cfg = _fast_cfg(nid_max_order=3)
        detector = InteractionDetector(family="poisson", config=cfg)
        detector.fit(
            X=data["X"],
            y=data["y"],
            glm_predictions=data["glm_predictions"],
            exposure=data["exposure"],
        )
        df = detector.nid_table()
        # Should have feature_1 and feature_2 columns (pairwise layout)
        assert "feature_1" in df.columns
        assert "feature_2" in df.columns


# ---------------------------------------------------------------------------
# glm_test_table() error path
# ---------------------------------------------------------------------------

class TestGlmTestTableErrorPath:
    def test_glm_test_table_before_fit_raises(self):
        detector = InteractionDetector(family="poisson")
        with pytest.raises(RuntimeError, match="fit"):
            detector.glm_test_table()


# ---------------------------------------------------------------------------
# suggest_interactions edge cases
# ---------------------------------------------------------------------------

class TestSuggestInteractionsEdgeCases:
    def test_require_significant_false_returns_top_k(self, synthetic_poisson_data):
        """require_significant=False should return pairs regardless of p-value."""
        data = synthetic_poisson_data
        detector = InteractionDetector(family="poisson", config=_fast_cfg(top_k_final=3))
        detector.fit(
            X=data["X"],
            y=data["y"],
            glm_predictions=data["glm_predictions"],
            exposure=data["exposure"],
        )
        suggestions = detector.suggest_interactions(top_k=3, require_significant=False)
        assert len(suggestions) <= 3
        # All returned items are 2-tuples of strings
        for pair in suggestions:
            assert len(pair) == 2
            assert all(isinstance(f, str) for f in pair)

    def test_suggest_interactions_default_top_k(self, synthetic_poisson_data):
        """When top_k=None, should use config.top_k_final."""
        data = synthetic_poisson_data
        cfg = _fast_cfg(top_k_final=2)
        detector = InteractionDetector(family="poisson", config=cfg)
        detector.fit(
            X=data["X"],
            y=data["y"],
            glm_predictions=data["glm_predictions"],
            exposure=data["exposure"],
        )
        suggestions = detector.suggest_interactions(require_significant=False)
        assert len(suggestions) <= cfg.top_k_final

    def test_suggest_interactions_require_significant_filters(self, synthetic_poisson_data):
        """require_significant=True should only return pairs with recommended==True."""
        data = synthetic_poisson_data
        detector = InteractionDetector(family="poisson", config=_fast_cfg())
        detector.fit(
            X=data["X"],
            y=data["y"],
            glm_predictions=data["glm_predictions"],
            exposure=data["exposure"],
        )
        table = detector.interaction_table()
        suggestions = detector.suggest_interactions(require_significant=True)

        # All suggested pairs should have recommended=True in the table (if the col exists)
        if "recommended" in table.columns and suggestions:
            for f1, f2 in suggestions:
                row = table.filter(
                    (pl.col("feature_1") == f1) & (pl.col("feature_2") == f2)
                )
                if not row.is_empty():
                    assert bool(row["recommended"][0]) is True


# ---------------------------------------------------------------------------
# three-way NID config
# ---------------------------------------------------------------------------

class TestThreeWayNIDConfig:
    def test_nid_max_order_3_pipeline_runs(self, synthetic_poisson_data):
        """nid_max_order=3 should run without error."""
        data = synthetic_poisson_data
        cfg = _fast_cfg(nid_max_order=3)
        detector = InteractionDetector(family="poisson", config=cfg)
        detector.fit(
            X=data["X"],
            y=data["y"],
            glm_predictions=data["glm_predictions"],
            exposure=data["exposure"],
        )
        assert detector.cann is not None

    def test_nid_scores_include_three_way_after_max_order_3(self, synthetic_poisson_data):
        data = synthetic_poisson_data
        cfg = _fast_cfg(nid_max_order=3)
        detector = InteractionDetector(family="poisson", config=cfg)
        detector.fit(
            X=data["X"],
            y=data["y"],
            glm_predictions=data["glm_predictions"],
            exposure=data["exposure"],
        )
        three_way = [s for s in detector.nid_scores if len(s.features) == 3]
        assert len(three_way) > 0


# ---------------------------------------------------------------------------
# SHAP warning path
# ---------------------------------------------------------------------------

class TestSHAPWarningPath:
    def test_bad_shap_model_triggers_warning_not_exception(self, synthetic_poisson_data):
        """Passing an object that fails the SHAP computation should warn, not raise."""
        data = synthetic_poisson_data
        detector = InteractionDetector(family="poisson", config=_fast_cfg())

        # A plain object that will cause AttributeError / RuntimeError inside the SHAP path
        bad_model = object()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            detector.fit(
                X=data["X"],
                y=data["y"],
                glm_predictions=data["glm_predictions"],
                exposure=data["exposure"],
                shap_model=bad_model,
            )

        # Should complete without raising, and emit at least one UserWarning
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) >= 1, (
            "Expected a UserWarning when SHAP computation fails, got none"
        )

    def test_pipeline_complete_after_shap_failure(self, synthetic_poisson_data):
        """Even after SHAP failure, interaction_table() should be populated."""
        data = synthetic_poisson_data
        detector = InteractionDetector(family="poisson", config=_fast_cfg())

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            detector.fit(
                X=data["X"],
                y=data["y"],
                glm_predictions=data["glm_predictions"],
                exposure=data["exposure"],
                shap_model=object(),
            )

        table = detector.interaction_table()
        assert not table.is_empty()


# ---------------------------------------------------------------------------
# MLP-M config flows through
# ---------------------------------------------------------------------------

class TestMLPMConfig:
    def test_mlp_m_pipeline_runs(self, synthetic_poisson_data):
        data = synthetic_poisson_data
        cfg = _fast_cfg(mlp_m=True)
        detector = InteractionDetector(family="poisson", config=cfg)
        detector.fit(
            X=data["X"],
            y=data["y"],
            glm_predictions=data["glm_predictions"],
            exposure=data["exposure"],
        )
        assert detector.cann is not None

    def test_mlp_m_cann_has_univariate_nets(self, synthetic_poisson_data):
        data = synthetic_poisson_data
        cfg = _fast_cfg(mlp_m=True)
        detector = InteractionDetector(family="poisson", config=cfg)
        detector.fit(
            X=data["X"],
            y=data["y"],
            glm_predictions=data["glm_predictions"],
            exposure=data["exposure"],
        )
        # The internal CANN model should have been built with mlp_m=True
        assert detector.cann is not None
        assert len(detector.cann._models) > 0
        first_model = detector.cann._models[0]
        assert first_model.mlp_m is True


# ---------------------------------------------------------------------------
# detector.nid_scores property
# ---------------------------------------------------------------------------

class TestNIDScoresProperty:
    def test_nid_scores_is_list_after_fit(self, synthetic_poisson_data):
        data = synthetic_poisson_data
        detector = InteractionDetector(family="poisson", config=_fast_cfg())
        detector.fit(
            X=data["X"],
            y=data["y"],
            glm_predictions=data["glm_predictions"],
            exposure=data["exposure"],
        )
        assert isinstance(detector.nid_scores, list)
        assert len(detector.nid_scores) > 0

    def test_nid_scores_are_interaction_scores(self, synthetic_poisson_data):
        from insurance_interactions.nid import InteractionScore
        data = synthetic_poisson_data
        detector = InteractionDetector(family="poisson", config=_fast_cfg())
        detector.fit(
            X=data["X"],
            y=data["y"],
            glm_predictions=data["glm_predictions"],
            exposure=data["exposure"],
        )
        for s in detector.nid_scores:
            assert isinstance(s, InteractionScore)


# ---------------------------------------------------------------------------
# interaction_table returns Polars DataFrame
# ---------------------------------------------------------------------------

class TestInteractionTableType:
    def test_interaction_table_is_polars_dataframe(self, synthetic_poisson_data):
        data = synthetic_poisson_data
        detector = InteractionDetector(family="poisson", config=_fast_cfg())
        detector.fit(
            X=data["X"],
            y=data["y"],
            glm_predictions=data["glm_predictions"],
            exposure=data["exposure"],
        )
        table = detector.interaction_table()
        assert isinstance(table, pl.DataFrame)

    def test_interaction_table_has_consensus_score(self, synthetic_poisson_data):
        """consensus_score column must be present (even without SHAP)."""
        data = synthetic_poisson_data
        detector = InteractionDetector(family="poisson", config=_fast_cfg())
        detector.fit(
            X=data["X"],
            y=data["y"],
            glm_predictions=data["glm_predictions"],
            exposure=data["exposure"],
        )
        table = detector.interaction_table()
        assert "consensus_score" in table.columns

    def test_interaction_table_sorted_by_consensus_score(self, synthetic_poisson_data):
        """Table should be sorted by consensus_score ascending (best first)."""
        data = synthetic_poisson_data
        detector = InteractionDetector(family="poisson", config=_fast_cfg())
        detector.fit(
            X=data["X"],
            y=data["y"],
            glm_predictions=data["glm_predictions"],
            exposure=data["exposure"],
        )
        table = detector.interaction_table()
        scores = table["consensus_score"].to_list()
        assert scores == sorted(scores)


# ---------------------------------------------------------------------------
# DetectorConfig defaults
# ---------------------------------------------------------------------------

class TestDetectorConfigDefaults:
    @pytest.mark.parametrize("field,expected_type", [
        ("cann_hidden_dims", list),
        ("cann_n_epochs", int),
        ("cann_batch_size", int),
        ("cann_learning_rate", float),
        ("cann_n_ensemble", int),
        ("top_k_nid", int),
        ("top_k_final", int),
        ("alpha_bonferroni", float),
    ])
    def test_default_field_types(self, field: str, expected_type: type):
        cfg = DetectorConfig()
        assert isinstance(getattr(cfg, field), expected_type)

    def test_default_activation_valid(self):
        cfg = DetectorConfig()
        assert cfg.cann_activation in ("tanh", "relu")

    def test_default_nid_max_order_is_2(self):
        cfg = DetectorConfig()
        assert cfg.nid_max_order == 2

    def test_validation_fraction_in_range(self):
        cfg = DetectorConfig()
        assert 0 < cfg.cann_validation_fraction < 1
