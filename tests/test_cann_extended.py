"""Extended CANN tests covering edge cases and paths not in test_cann.py.

Covers:
  - CANNModel forward pass shape and clamping
  - CANNModel weight matrix accessors
  - _UnivariateNet zero-init property
  - MLP-M architecture: univariate nets are created and contribute
  - _encode_dataframe edge cases: single row, all-same continuous values,
    Enum dtype, unseen category in val set, zero-variance column
  - CANN error paths: predict/get_weight_matrices/feature_names before fit
  - CANN fit: explicit val set with missing optional args, relu activation
  - Loss functions: weight scaling, gamma deviance positive
  - CANN ensemble: val_deviance_history length matches n_ensemble
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
import torch.nn as nn

from insurance_interactions.cann import (
    CANN,
    CANNConfig,
    CANNModel,
    _UnivariateNet,
    _encode_dataframe,
    _gamma_deviance,
    _poisson_deviance,
)


# ---------------------------------------------------------------------------
# _UnivariateNet
# ---------------------------------------------------------------------------

class TestUnivariateNet:
    def test_output_zero_at_init(self):
        """At initialisation the output layer is zeroed, so forward() should be ~0."""
        net = _UnivariateNet(input_dim=3, hidden_dim=8)
        x = torch.randn(10, 3)
        out = net(x)
        assert torch.allclose(out, torch.zeros_like(out)), (
            "Zero-init output layer should produce all-zero outputs at init"
        )

    def test_output_shape(self):
        net = _UnivariateNet(input_dim=2, hidden_dim=4)
        x = torch.randn(5, 2)
        out = net(x)
        assert out.shape == (5, 1)

    def test_has_tanh_activation(self):
        net = _UnivariateNet(input_dim=1, hidden_dim=4)
        activations = [m for m in net.net if isinstance(m, nn.Tanh)]
        assert len(activations) >= 1


# ---------------------------------------------------------------------------
# CANNModel
# ---------------------------------------------------------------------------

class TestCANNModel:
    def test_forward_output_shape(self):
        model = CANNModel(input_dim=5, hidden_dims=[8, 4])
        x = torch.randn(12, 5)
        glm = torch.ones(12)
        out = model(x, glm)
        assert out.shape == (12,)

    def test_forward_positive_outputs(self):
        """exp(...) must always be positive."""
        model = CANNModel(input_dim=4, hidden_dims=[8])
        x = torch.randn(20, 4)
        glm = torch.zeros(20)  # GLM log pred = 0 → GLM pred = 1
        out = model(x, glm)
        assert torch.all(out > 0)

    def test_zero_init_output_equals_glm(self):
        """At initialisation, CANN output should equal exp(glm_log_pred)."""
        model = CANNModel(input_dim=4, hidden_dims=[8])
        glm_log = torch.tensor([0.5, 1.0, -0.3, 2.0])
        x = torch.randn(4, 4)
        out = model(x, glm_log)
        expected = torch.exp(glm_log)
        # Should be close (zero-init MLP correction = 0)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_clamp_prevents_overflow(self):
        """Extreme GLM values should be clamped, not overflow."""
        model = CANNModel(input_dim=2, hidden_dims=[4])
        x = torch.zeros(3, 2)
        glm_log = torch.tensor([100.0, -100.0, 20.5])  # extreme values
        out = model(x, glm_log)
        assert torch.all(torch.isfinite(out)), "Output must be finite even for extreme inputs"

    def test_get_first_layer_weights_shape(self):
        model = CANNModel(input_dim=6, hidden_dims=[16, 8])
        w1 = model.get_first_layer_weights()
        assert w1.shape == (16, 6)

    def test_get_subsequent_weight_matrices(self):
        model = CANNModel(input_dim=5, hidden_dims=[16, 8])
        w_rest = model.get_subsequent_weight_matrices()
        # [32→16], [16→1] for a [32, 16] net — here [16, 8] gives [16→8, 8→1]
        assert len(w_rest) == 2
        assert w_rest[-1].shape[0] == 1  # output layer has 1 unit

    def test_single_hidden_layer_subsequent_weights(self):
        model = CANNModel(input_dim=4, hidden_dims=[8])
        w_rest = model.get_subsequent_weight_matrices()
        assert len(w_rest) == 1
        assert w_rest[0].shape[0] == 1

    def test_relu_activation(self):
        model = CANNModel(input_dim=3, hidden_dims=[8], activation="relu")
        activations = [m for m in model.mlp if isinstance(m, nn.ReLU)]
        assert len(activations) >= 1

    def test_mlp_m_creates_univariate_nets(self):
        feature_slices = {
            "age_band": slice(0, 2),
            "vehicle_group": slice(2, 4),
            "annual_mileage": slice(4, 5),
        }
        model = CANNModel(
            input_dim=5,
            hidden_dims=[8],
            mlp_m=True,
            feature_slices=feature_slices,
        )
        assert len(model.univariate_nets) == 3

    def test_mlp_m_forward_differs_from_standard(self):
        """MLP-M should produce different outputs to standard MLP after training."""
        feature_slices = {"f0": slice(0, 1), "f1": slice(1, 2), "f2": slice(2, 3)}
        model_standard = CANNModel(input_dim=3, hidden_dims=[8], mlp_m=False)
        model_mlpm = CANNModel(input_dim=3, hidden_dims=[8], mlp_m=True, feature_slices=feature_slices)

        # Give univariate nets non-zero weights manually
        for net in model_mlpm.univariate_nets.values():
            nn.init.normal_(net.net[-1].weight, mean=0.1)
            nn.init.normal_(net.net[-1].bias, mean=0.1)

        x = torch.randn(10, 3)
        glm = torch.zeros(10)
        out_std = model_standard(x, glm)
        out_mlpm = model_mlpm(x, glm)

        # After non-zero init of univariate nets, outputs should differ
        assert not torch.allclose(out_std, out_mlpm, atol=1e-4)

    def test_forward_with_2d_glm_input(self):
        """glm_log_pred can be (batch, 1) — must be squeezed correctly."""
        model = CANNModel(input_dim=3, hidden_dims=[4])
        x = torch.randn(5, 3)
        glm = torch.zeros(5, 1)  # 2D input
        out = model(x, glm)
        assert out.shape == (5,)


# ---------------------------------------------------------------------------
# _encode_dataframe edge cases
# ---------------------------------------------------------------------------

class TestEncodingEdgeCases:
    def test_single_row(self):
        df = pl.DataFrame({"x": [3.0], "cat": ["a"]})
        enc = _encode_dataframe(df)
        assert enc.X_encoded.shape[0] == 1

    def test_zero_variance_continuous_column(self):
        """All-same continuous values → std=0, should not divide by zero."""
        df = pl.DataFrame({"x": [5.0, 5.0, 5.0, 5.0]})
        enc = _encode_dataframe(df)
        # Should not raise; std is set to 1.0 for zero-variance columns
        assert np.all(np.isfinite(enc.X_encoded))
        assert enc.col_stds["x"] == pytest.approx(1.0)

    def test_enum_dtype_treated_as_categorical(self):
        """Polars Enum dtype should be encoded like String/Categorical."""
        df = pl.DataFrame({
            "region": pl.Series(["north", "south", "midlands", "north"]).cast(pl.Enum(["north", "south", "midlands"])),
        })
        enc = _encode_dataframe(df)
        # 3 levels → 2 one-hot columns
        assert enc.X_encoded.shape[1] == 2

    def test_val_encoding_uses_train_categories(self):
        """Unseen category in val set should be encoded as all-zeros (reference)."""
        train = pl.DataFrame({
            "cat": pl.Series(["a", "b", "c", "a"]).cast(pl.String),
        })
        val = pl.DataFrame({
            "cat": pl.Series(["d", "a"]).cast(pl.String),  # "d" was not seen in training
        })
        train_enc = _encode_dataframe(train)
        val_enc = _encode_dataframe(val, ref=train_enc)
        # Same shape as training
        assert val_enc.X_encoded.shape[1] == train_enc.X_encoded.shape[1]
        # "d" row should be all zeros (falls back to reference level)
        assert np.allclose(val_enc.X_encoded[0], 0.0)

    def test_val_continuous_uses_train_stats(self):
        """Val set should use training mean/std, not its own."""
        train = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        val = pl.DataFrame({"x": [10.0, 20.0]})
        train_enc = _encode_dataframe(train)
        val_enc = _encode_dataframe(val, ref=train_enc)
        # Val mean/std should match training
        assert val_enc.col_means["x"] == pytest.approx(train_enc.col_means["x"])
        assert val_enc.col_stds["x"] == pytest.approx(train_enc.col_stds["x"])

    def test_empty_categorical_handling(self):
        """A categorical column with only one level encodes to zero columns (drop-first)."""
        df = pl.DataFrame({"cat": pl.Series(["a", "a", "a"]).cast(pl.String)})
        enc = _encode_dataframe(df)
        # 1 unique level → 0 one-hot columns after drop-first
        assert enc.X_encoded.shape[1] == 0

    def test_mixed_dtypes(self):
        """DataFrame with both continuous and categorical columns encodes correctly."""
        df = pl.DataFrame({
            "age": [25.0, 35.0, 45.0, 55.0],
            "region": pl.Series(["north", "south", "north", "midlands"]).cast(pl.String),
            "ncd": [0.0, 3.0, 6.0, 9.0],
        })
        enc = _encode_dataframe(df)
        # 2 continuous + 2 one-hot (3 region levels - 1) = 4 total cols
        assert enc.X_encoded.shape == (4, 4)

    def test_no_columns_returns_empty(self):
        """Empty DataFrame should produce zero-column encoded output."""
        df = pl.DataFrame(schema={})
        enc = _encode_dataframe(df)
        assert enc.X_encoded.shape[1] == 0


# ---------------------------------------------------------------------------
# CANN error paths
# ---------------------------------------------------------------------------

class TestCANNErrorPaths:
    def test_predict_before_fit_raises(self):
        cann = CANN(family="poisson")
        X = pl.DataFrame({"x": [1.0, 2.0]})
        with pytest.raises(RuntimeError, match="fit"):
            cann.predict(X, np.array([0.5, 0.5]))

    def test_get_weight_matrices_before_fit_raises(self):
        cann = CANN(family="poisson")
        with pytest.raises(RuntimeError, match="fit"):
            cann.get_weight_matrices()

    def test_feature_names_before_fit_raises(self):
        cann = CANN(family="poisson")
        with pytest.raises(RuntimeError, match="fit"):
            _ = cann.feature_names

    def test_encoded_feature_names_before_fit_raises(self):
        cann = CANN(family="poisson")
        with pytest.raises(RuntimeError, match="fit"):
            _ = cann.encoded_feature_names

    def test_feature_slices_before_fit_raises(self):
        cann = CANN(family="poisson")
        with pytest.raises(RuntimeError, match="fit"):
            _ = cann.feature_slices


# ---------------------------------------------------------------------------
# CANN training configurations
# ---------------------------------------------------------------------------

class TestCANNTrainingConfigs:
    def test_relu_activation_trains(self, synthetic_poisson_data):
        data = synthetic_poisson_data
        cfg = CANNConfig(n_epochs=3, n_ensemble=1, patience=2, hidden_dims=[8], activation="relu")
        cann = CANN(family="poisson", config=cfg)
        cann.fit(
            X=data["X"],
            y=data["y"],
            glm_predictions=data["glm_predictions"],
            exposure=data["exposure"],
        )
        assert cann._models

    def test_mlp_m_trains(self, synthetic_poisson_data):
        data = synthetic_poisson_data
        cfg = CANNConfig(n_epochs=3, n_ensemble=1, patience=2, hidden_dims=[8, 4], mlp_m=True)
        cann = CANN(family="poisson", config=cfg)
        cann.fit(
            X=data["X"],
            y=data["y"],
            glm_predictions=data["glm_predictions"],
            exposure=data["exposure"],
        )
        assert cann._models

    def test_ensemble_val_history_length(self, synthetic_poisson_data):
        data = synthetic_poisson_data
        n_ens = 3
        cfg = CANNConfig(n_epochs=5, n_ensemble=n_ens, patience=3, hidden_dims=[8])
        cann = CANN(family="poisson", config=cfg)
        cann.fit(
            X=data["X"],
            y=data["y"],
            glm_predictions=data["glm_predictions"],
            exposure=data["exposure"],
        )
        assert len(cann.val_deviance_history) == n_ens
        for history in cann.val_deviance_history:
            assert len(history) > 0

    def test_explicit_val_no_optional_args(self, synthetic_poisson_data):
        """Explicit val set with y_val and glm_predictions_val omitted falls back gracefully."""
        data = synthetic_poisson_data
        n = len(data["X"])
        split = n // 2
        cfg = CANNConfig(n_epochs=3, n_ensemble=1, patience=2, hidden_dims=[8])
        cann = CANN(family="poisson", config=cfg)
        # Pass X_val but omit y_val / glm_predictions_val / exposure_val
        cann.fit(
            X=data["X"][:split],
            y=data["y"][:split],
            glm_predictions=data["glm_predictions"][:split],
            X_val=data["X"][split:],
        )
        assert cann._models

    def test_gamma_predict_positive(self, synthetic_gamma_data):
        data = synthetic_gamma_data
        cfg = CANNConfig(n_epochs=3, n_ensemble=1, patience=2, hidden_dims=[8])
        cann = CANN(family="gamma", config=cfg)
        cann.fit(
            X=data["X"],
            y=data["y"],
            glm_predictions=data["glm_predictions"],
        )
        preds = cann.predict(data["X"], data["glm_predictions"])
        assert np.all(preds > 0)


# ---------------------------------------------------------------------------
# Loss function edge cases
# ---------------------------------------------------------------------------

class TestLossFunctionEdgeCases:
    def test_poisson_deviance_weight_scaling(self):
        """Doubling all weights should double the loss."""
        y = torch.tensor([1.0, 2.0, 3.0])
        mu = torch.tensor([1.2, 1.8, 3.5])
        w1 = torch.ones(3)
        w2 = torch.full((3,), 2.0)
        loss1 = _poisson_deviance(mu, y, w1)
        loss2 = _poisson_deviance(mu, y, w2)
        assert loss2 == pytest.approx(float(loss1) * 2.0, rel=1e-4)

    def test_gamma_deviance_positive_when_misfit(self):
        """Gamma deviance should be positive when prediction differs from truth."""
        y = torch.tensor([500.0, 1000.0, 800.0])
        mu = torch.tensor([600.0, 900.0, 750.0])
        w = torch.ones(3)
        loss = _gamma_deviance(mu, y, w)
        assert float(loss) > 0

    def test_gamma_deviance_weight_scaling(self):
        """Doubling all weights should double gamma deviance."""
        y = torch.tensor([200.0, 300.0])
        mu = torch.tensor([250.0, 280.0])
        w1 = torch.ones(2)
        w2 = torch.full((2,), 2.0)
        loss1 = _gamma_deviance(mu, y, w1)
        loss2 = _gamma_deviance(mu, y, w2)
        assert loss2 == pytest.approx(float(loss1) * 2.0, rel=1e-4)

    def test_poisson_deviance_finite_for_large_predictions(self):
        """Very large mu should still produce a finite loss."""
        y = torch.tensor([1.0, 0.0, 3.0])
        mu = torch.tensor([1e6, 1e6, 1e6])
        w = torch.ones(3)
        loss = _poisson_deviance(mu, y, w)
        assert torch.isfinite(loss)

    def test_gamma_deviance_near_zero_y_clamped(self):
        """Gamma deviance with y near zero should not produce NaN (clamps y)."""
        y = torch.tensor([1e-10, 1.0, 2.0])
        mu = torch.tensor([1.0, 1.0, 2.0])
        w = torch.ones(3)
        loss = _gamma_deviance(mu, y, w)
        assert torch.isfinite(loss)
