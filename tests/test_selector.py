"""Tests for InteractionDetector end-to-end pipeline."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from insurance_interactions import DetectorConfig, InteractionDetector


class TestInteractionDetectorEndToEnd:
    def test_fit_returns_self(self, synthetic_poisson_data):
        data = synthetic_poisson_data
        cfg = DetectorConfig(
            cann_n_epochs=10,
            cann_n_ensemble=1,
            cann_patience=5,
            cann_hidden_dims=[8, 4],
            top_k_nid=5,
        )
        detector = InteractionDetector(family="poisson", config=cfg)
        result = detector.fit(
            X=data["X"],
            y=data["y"],
            glm_predictions=data["glm_predictions"],
            exposure=data["exposure"],
        )
        assert result is detector

    def test_interaction_table_has_expected_columns(self, synthetic_poisson_data):
        data = synthetic_poisson_data
        cfg = DetectorConfig(
            cann_n_epochs=10,
            cann_n_ensemble=1,
            cann_patience=5,
            cann_hidden_dims=[8, 4],
            top_k_nid=5,
        )
        detector = InteractionDetector(family="poisson", config=cfg)
        detector.fit(
            X=data["X"],
            y=data["y"],
            glm_predictions=data["glm_predictions"],
            exposure=data["exposure"],
        )
        table = detector.interaction_table()
        assert "feature_1" in table.columns
        assert "feature_2" in table.columns
        assert "nid_score" in table.columns
        assert "nid_score_normalised" in table.columns

    def test_suggest_interactions_returns_tuples(self, synthetic_poisson_data):
        data = synthetic_poisson_data
        cfg = DetectorConfig(
            cann_n_epochs=10,
            cann_n_ensemble=1,
            cann_patience=5,
            cann_hidden_dims=[8, 4],
            top_k_nid=5,
            top_k_final=3,
        )
        detector = InteractionDetector(family="poisson", config=cfg)
        detector.fit(
            X=data["X"],
            y=data["y"],
            glm_predictions=data["glm_predictions"],
            exposure=data["exposure"],
        )
        suggestions = detector.suggest_interactions(top_k=3, require_significant=False)
        assert isinstance(suggestions, list)
        for pair in suggestions:
            assert isinstance(pair, tuple)
            assert len(pair) == 2
            assert all(isinstance(f, str) for f in pair)

    def test_glm_test_table_has_n_cells(self, synthetic_poisson_data):
        data = synthetic_poisson_data
        cfg = DetectorConfig(
            cann_n_epochs=10,
            cann_n_ensemble=1,
            cann_patience=5,
            cann_hidden_dims=[8, 4],
            top_k_nid=3,
        )
        detector = InteractionDetector(family="poisson", config=cfg)
        detector.fit(
            X=data["X"],
            y=data["y"],
            glm_predictions=data["glm_predictions"],
            exposure=data["exposure"],
        )
        glm_table = detector.glm_test_table()
        if not glm_table.is_empty():
            assert "n_cells" in glm_table.columns
            assert "delta_deviance" in glm_table.columns
            assert "lr_p" in glm_table.columns
            assert "recommended" in glm_table.columns

    def test_cann_accessible_after_fit(self, synthetic_poisson_data):
        data = synthetic_poisson_data
        cfg = DetectorConfig(
            cann_n_epochs=5,
            cann_n_ensemble=1,
            cann_patience=3,
            cann_hidden_dims=[8],
            top_k_nid=2,
        )
        detector = InteractionDetector(family="poisson", config=cfg)
        detector.fit(
            X=data["X"],
            y=data["y"],
            glm_predictions=data["glm_predictions"],
            exposure=data["exposure"],
        )
        assert detector.cann is not None
        assert detector.nid_scores

    def test_gamma_family_pipeline(self, synthetic_gamma_data):
        data = synthetic_gamma_data
        cfg = DetectorConfig(
            cann_n_epochs=10,
            cann_n_ensemble=1,
            cann_patience=5,
            cann_hidden_dims=[8, 4],
            top_k_nid=3,
        )
        detector = InteractionDetector(family="gamma", config=cfg)
        detector.fit(
            X=data["X"],
            y=data["y"],
            glm_predictions=data["glm_predictions"],
            exposure=data["exposure"],
        )
        table = detector.interaction_table()
        assert not table.is_empty()

    def test_fit_without_exposure(self, synthetic_poisson_data):
        """Exposure defaults to ones when not supplied."""
        data = synthetic_poisson_data
        cfg = DetectorConfig(
            cann_n_epochs=5,
            cann_n_ensemble=1,
            cann_patience=3,
            cann_hidden_dims=[8],
            top_k_nid=2,
        )
        detector = InteractionDetector(family="poisson", config=cfg)
        # Should not raise
        detector.fit(
            X=data["X"],
            y=data["y"],
            glm_predictions=data["glm_predictions"],
            # exposure omitted
        )
        assert detector.cann is not None

    def test_error_before_fit(self):
        detector = InteractionDetector(family="poisson")
        with pytest.raises(RuntimeError, match="fit"):
            detector.interaction_table()
