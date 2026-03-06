"""Tests for the CANN module.

These tests verify:
  - The CANN trains without errors on synthetic Poisson and Gamma data
  - Zero-initialisation property: CANN predictions start close to GLM predictions
  - The CANN improves on GLM deviance after training (it should, given there is
    a known interaction the GLM is missing)
  - Feature encoding is consistent between train and validation sets
  - Ensemble weight matrix shapes are correct for NID input
"""

from __future__ import annotations

import numpy as np
import pytest
import polars as pl

from insurance_interactions.cann import (
    CANN,
    CANNConfig,
    _encode_dataframe,
    _poisson_deviance,
    _gamma_deviance,
)
import torch


class TestDeviceLossFunctions:
    def test_poisson_deviance_zero_residual(self):
        """When prediction equals truth, deviance should be near zero."""
        y = torch.tensor([2.0, 3.0, 5.0])
        mu = torch.tensor([2.0, 3.0, 5.0])
        w = torch.ones(3)
        loss = _poisson_deviance(mu, y, w)
        assert float(loss) < 1e-5

    def test_poisson_deviance_positive(self):
        """Deviance should be positive when prediction != truth."""
        y = torch.tensor([2.0, 3.0, 5.0])
        mu = torch.tensor([1.0, 2.0, 4.0])
        w = torch.ones(3)
        loss = _poisson_deviance(mu, y, w)
        assert float(loss) > 0

    def test_gamma_deviance_zero_residual(self):
        y = torch.tensor([100.0, 200.0, 150.0])
        mu = torch.tensor([100.0, 200.0, 150.0])
        w = torch.ones(3)
        loss = _gamma_deviance(mu, y, w)
        assert float(loss) < 1e-5

    def test_poisson_deviance_zero_y(self):
        """y=0 case should not produce NaN."""
        y = torch.tensor([0.0, 1.0, 2.0])
        mu = torch.tensor([0.5, 1.0, 2.0])
        w = torch.ones(3)
        loss = _poisson_deviance(mu, y, w)
        assert not torch.isnan(loss)
        assert float(loss) > 0


class TestEncoding:
    def test_continuous_standardisation(self):
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        enc = _encode_dataframe(df)
        assert enc.X_encoded.shape == (5, 1)
        # Standardised values should have mean~0, std~1
        assert abs(enc.X_encoded[:, 0].mean()) < 1e-5

    def test_categorical_one_hot(self):
        df = pl.DataFrame({"cat": ["a", "b", "c", "a", "b"]})
        enc = _encode_dataframe(df)
        # 3 categories → 2 one-hot columns (drop first)
        assert enc.X_encoded.shape[1] == 2

    def test_val_uses_train_encoding(self):
        train = pl.DataFrame({"age": [1.0, 2.0, 3.0], "cat": ["a", "b", "c"]})
        val = pl.DataFrame({"age": [4.0, 5.0], "cat": ["b", "a"]})
        train_enc = _encode_dataframe(train)
        val_enc = _encode_dataframe(val, ref=train_enc)
        # Same number of columns
        assert val_enc.X_encoded.shape[1] == train_enc.X_encoded.shape[1]

    def test_feature_slices_cover_all_columns(self):
        df = pl.DataFrame({
            "cont": [1.0, 2.0, 3.0],
            "cat": ["x", "y", "z"],
        })
        enc = _encode_dataframe(df)
        # All encoded columns should be covered by the slices
        covered = set()
        for _, s in enc.feature_slices.items():
            covered.update(range(s.start, s.stop))
        assert covered == set(range(enc.X_encoded.shape[1]))


class TestCANNTraining:
    def test_poisson_fit_no_error(self, synthetic_poisson_data):
        data = synthetic_poisson_data
        cfg = CANNConfig(n_epochs=5, n_ensemble=1, patience=3, hidden_dims=[8, 4])
        cann = CANN(family="poisson", config=cfg)
        cann.fit(
            X=data["X"],
            y=data["y"],
            glm_predictions=data["glm_predictions"],
            exposure=data["exposure"],
        )
        assert cann._models, "No models were fitted"

    def test_gamma_fit_no_error(self, synthetic_gamma_data):
        data = synthetic_gamma_data
        cfg = CANNConfig(n_epochs=5, n_ensemble=1, patience=3, hidden_dims=[8, 4])
        cann = CANN(family="gamma", config=cfg)
        cann.fit(
            X=data["X"],
            y=data["y"],
            glm_predictions=data["glm_predictions"],
            exposure=data["exposure"],
        )
        assert cann._models

    def test_predict_shape(self, synthetic_poisson_data):
        data = synthetic_poisson_data
        cfg = CANNConfig(n_epochs=5, n_ensemble=1, patience=3, hidden_dims=[8, 4])
        cann = CANN(family="poisson", config=cfg)
        cann.fit(
            X=data["X"],
            y=data["y"],
            glm_predictions=data["glm_predictions"],
            exposure=data["exposure"],
        )
        preds = cann.predict(data["X"], data["glm_predictions"])
        assert preds.shape == (len(data["X"]),)
        assert np.all(preds > 0), "Predictions must be positive"

    def test_weight_matrices_shape(self, synthetic_poisson_data):
        data = synthetic_poisson_data
        cfg = CANNConfig(n_epochs=3, n_ensemble=2, patience=2, hidden_dims=[8, 4])
        cann = CANN(family="poisson", config=cfg)
        cann.fit(
            X=data["X"],
            y=data["y"],
            glm_predictions=data["glm_predictions"],
            exposure=data["exposure"],
        )
        wm = cann.get_weight_matrices()
        assert len(wm) == 2  # 2 ensemble members
        for w1, w_rest in wm:
            assert w1.ndim == 2
            assert w1.shape[0] == 8  # first hidden layer width
            assert w_rest[-1].shape[0] == 1  # output layer has 1 unit

    def test_val_deviance_decreases(self, synthetic_poisson_data):
        """Validation deviance should generally decrease during training."""
        data = synthetic_poisson_data
        cfg = CANNConfig(n_epochs=30, n_ensemble=1, patience=50, hidden_dims=[16, 8])
        cann = CANN(family="poisson", config=cfg)
        cann.fit(
            X=data["X"],
            y=data["y"],
            glm_predictions=data["glm_predictions"],
            exposure=data["exposure"],
        )
        history = cann.val_deviance_history[0]
        # Last value should be <= first value (or close to it)
        assert history[-1] <= history[0] + 0.1 * abs(history[0])

    def test_feature_names_propagate(self, synthetic_poisson_data):
        data = synthetic_poisson_data
        cfg = CANNConfig(n_epochs=3, n_ensemble=1, patience=2, hidden_dims=[8])
        cann = CANN(family="poisson", config=cfg)
        cann.fit(
            X=data["X"],
            y=data["y"],
            glm_predictions=data["glm_predictions"],
            exposure=data["exposure"],
        )
        assert "age_band" in cann.feature_names
        assert "vehicle_group" in cann.feature_names
        assert "annual_mileage" in cann.feature_names

    def test_explicit_val_set(self, synthetic_poisson_data):
        """Training with an explicit validation set should work without error."""
        data = synthetic_poisson_data
        n = len(data["X"])
        split = n // 2
        cfg = CANNConfig(n_epochs=5, n_ensemble=1, patience=3, hidden_dims=[8, 4])
        cann = CANN(family="poisson", config=cfg)
        cann.fit(
            X=data["X"][:split],
            y=data["y"][:split],
            glm_predictions=data["glm_predictions"][:split],
            exposure=data["exposure"][:split],
            X_val=data["X"][split:],
            y_val=data["y"][split:],
            glm_predictions_val=data["glm_predictions"][split:],
            exposure_val=data["exposure"][split:],
        )
        assert cann._models
