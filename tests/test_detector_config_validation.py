"""
Regression tests for P1 bug fixes in DetectorConfig (batch 3 audit).

Covers:
- DetectorConfig validates field values on construction
- Invalid values raise ValueError immediately
"""

from __future__ import annotations

import pytest

from insurance_interactions import DetectorConfig


class TestDetectorConfigValidation:
    """DetectorConfig must reject invalid field values via __post_init__."""

    def test_valid_config_default_constructs(self):
        """Default config should construct without error."""
        cfg = DetectorConfig()
        assert cfg.top_k_nid > 0
        assert cfg.cann_activation in ("tanh", "relu")

    def test_invalid_activation_raises(self):
        with pytest.raises(ValueError, match="cann_activation"):
            DetectorConfig(cann_activation="sigmoid")

    def test_zero_top_k_nid_raises(self):
        with pytest.raises(ValueError, match="top_k_nid"):
            DetectorConfig(top_k_nid=0)

    def test_negative_top_k_nid_raises(self):
        with pytest.raises(ValueError, match="top_k_nid"):
            DetectorConfig(top_k_nid=-5)

    def test_zero_top_k_final_raises(self):
        with pytest.raises(ValueError, match="top_k_final"):
            DetectorConfig(top_k_final=0)

    def test_validation_fraction_zero_raises(self):
        with pytest.raises(ValueError, match="cann_validation_fraction"):
            DetectorConfig(cann_validation_fraction=0.0)

    def test_validation_fraction_one_raises(self):
        with pytest.raises(ValueError, match="cann_validation_fraction"):
            DetectorConfig(cann_validation_fraction=1.0)

    def test_zero_ensemble_raises(self):
        with pytest.raises(ValueError, match="cann_n_ensemble"):
            DetectorConfig(cann_n_ensemble=0)

    def test_invalid_nid_max_order_raises(self):
        with pytest.raises(ValueError, match="nid_max_order"):
            DetectorConfig(nid_max_order=4)

    def test_zero_epochs_raises(self):
        with pytest.raises(ValueError, match="cann_n_epochs"):
            DetectorConfig(cann_n_epochs=0)

    def test_valid_non_default_config(self):
        """Non-default but valid config should construct."""
        cfg = DetectorConfig(
            cann_activation="relu",
            top_k_nid=5,
            top_k_final=3,
            cann_validation_fraction=0.15,
            cann_n_ensemble=2,
            nid_max_order=3,
        )
        assert cfg.cann_activation == "relu"
        assert cfg.nid_max_order == 3
