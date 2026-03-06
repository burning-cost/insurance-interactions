"""Combined Actuarial Neural Network (CANN) implementation.

Architecture from Schelldorfer and Wüthrich (2019), "Nesting Classical Actuarial
Models into Neural Networks", SSRN 3320525.

The CANN takes an existing GLM prediction as a skip connection (added in log space)
and trains a feedforward MLP to learn only the residual structure the GLM cannot
express. At initialisation, the output layer weights are zeroed so the CANN starts
from the GLM prediction exactly. This is the right architecture for interaction
detection: the deviation of the trained network from zero corresponds to structure
the GLM is missing, which NID then decodes.

Supports Poisson (frequency) and Gamma (severity) families — the two standard
choices for UK personal lines GLMs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


Family = Literal["poisson", "gamma"]


def _poisson_deviance(y_pred: torch.Tensor, y_true: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Poisson deviance loss, weighted by exposure.

    D = 2 * Σ w_i * [y_i * log(y_i / μ_i) - (y_i - μ_i)]

    Uses the mean over observations for gradient stability.
    Clamps predictions to avoid log(0).
    """
    mu = torch.clamp(y_pred, min=1e-8)
    y = torch.clamp(y_true, min=0.0)
    # Poisson deviance term: 2 * (y*log(y/mu) - (y - mu))
    # When y=0: 2*(0 - (0-mu)) = 2*mu
    safe_log = torch.where(y > 0, y * torch.log(y / mu), torch.zeros_like(y))
    deviance = 2.0 * weights * (safe_log - (y - mu))
    return deviance.mean()


def _gamma_deviance(y_pred: torch.Tensor, y_true: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Gamma deviance loss, weighted.

    D = 2 * Σ w_i * [-log(y_i / μ_i) + (y_i - μ_i) / μ_i]

    Appropriate for severity modelling. Requires y > 0 (claim amounts).
    """
    mu = torch.clamp(y_pred, min=1e-8)
    y = torch.clamp(y_true, min=1e-8)
    deviance = 2.0 * weights * (-torch.log(y / mu) + (y - mu) / mu)
    return deviance.mean()


class _UnivariateNet(nn.Module):
    """Single-feature univariate network for MLP-M architecture.

    In MLP-M, one univariate net is trained per input feature to absorb the main
    effect. This forces the main MLP to model only interactions (any deviation from
    additive main effects). The univariate nets are shallow by design.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        # Zero-initialise output so net starts at zero correction
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CANNModel(nn.Module):
    """Combined Actuarial Neural Network.

    The MLP component takes all encoded features and produces a scalar additive
    correction in log space. The GLM log-prediction is injected as a fixed offset
    (not trained), so:

        log(μ_CANN) = NN(x; θ) + log(μ_GLM)
        μ_CANN = μ_GLM * exp(NN(x; θ))

    Parameters
    ----------
    input_dim:
        Number of input features after encoding (after one-hot expansion of
        categoricals).
    hidden_dims:
        Width of each hidden layer. E.g. [32, 16] gives a 2-layer MLP.
    activation:
        Activation function for hidden layers. "tanh" is recommended per the
        original paper; "relu" also works and trains faster on large datasets.
    mlp_m:
        If True, use the MLP-M architecture variant: each feature gets its own
        univariate net for main effects. The main MLP then models only interactions.
        Reduces false positive interactions at the cost of more parameters.
    feature_slices:
        Required for MLP-M. Maps feature name to slice of the encoded input tensor
        (since categoricals expand to multiple columns after one-hot encoding).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        activation: Literal["tanh", "relu"] = "tanh",
        mlp_m: bool = False,
        feature_slices: dict[str, slice] | None = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 16]

        act_cls = nn.Tanh if activation == "tanh" else nn.ReLU

        # Main interaction MLP
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act_cls())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

        # Zero-init output layer so CANN = GLM at start
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        # MLP-M univariate nets
        self.mlp_m = mlp_m
        self.feature_slices = feature_slices or {}
        if mlp_m:
            self.univariate_nets = nn.ModuleDict(
                {
                    name.replace(".", "_"): _UnivariateNet(s.stop - s.start)
                    for name, s in self.feature_slices.items()
                }
            )
        else:
            self.univariate_nets = nn.ModuleDict()

    def forward(self, x: torch.Tensor, glm_log_pred: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Encoded input features, shape (batch, input_dim).
        glm_log_pred:
            Log of GLM prediction, shape (batch,) or (batch, 1). Treated as fixed
            offset — not trained.

        Returns
        -------
        Predicted mean on the original scale (not log).
        """
        correction = self.mlp(x).squeeze(-1)

        if self.mlp_m and self.univariate_nets:
            univariate_sum = torch.zeros_like(correction)
            for name, s in self.feature_slices.items():
                key = name.replace(".", "_")
                if key in self.univariate_nets:
                    feat = x[:, s]
                    univariate_sum = univariate_sum + self.univariate_nets[key](feat).squeeze(-1)
            correction = correction + univariate_sum

        log_mu = correction + glm_log_pred.squeeze(-1)
        return torch.exp(log_mu)

    def get_first_layer_weights(self) -> np.ndarray:
        """Return the weight matrix of the first hidden layer, shape (hidden_0, input_dim).

        This is W^(1) in the NID notation. NID operates on this matrix and the
        subsequent layers' absolute weight products.
        """
        return self.mlp[0].weight.detach().cpu().numpy()

    def get_subsequent_weight_matrices(self) -> list[np.ndarray]:
        """Return weight matrices from layer 2 onwards (layers between first hidden and output).

        In a [32, 16] MLP the structure is:
          Linear(input→32) [0], Tanh [1], Linear(32→16) [2], Tanh [3], Linear(16→1) [4]
        This returns [W^(2), W^(3)] = layers [2], [4] weight matrices.
        """
        matrices = []
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                matrices.append(module.weight.detach().cpu().numpy())
        # First element is W^(1) — skip it; return from W^(2) onward
        return matrices[1:]


@dataclass
class CANNConfig:
    """Training configuration for the CANN."""

    hidden_dims: list[int] = field(default_factory=lambda: [32, 16])
    activation: Literal["tanh", "relu"] = "tanh"
    mlp_m: bool = False
    n_epochs: int = 200
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 20
    """Early stopping patience in epochs."""
    validation_fraction: float = 0.2
    """Fraction of training data to hold out for early stopping if no explicit val set given."""
    seed: int = 42
    n_ensemble: int = 1
    """Number of CANN runs to average for more stable NID scores. 3–5 recommended."""


class _EncodedData:
    """Holds the result of encoding a Polars DataFrame for CANN input."""

    def __init__(
        self,
        X_encoded: np.ndarray,
        feature_names: list[str],
        feature_slices: dict[str, slice],
        categorical_cols: list[str],
        continuous_cols: list[str],
        col_means: dict[str, float],
        col_stds: dict[str, float],
        categories: dict[str, list[str]],
    ) -> None:
        self.X_encoded = X_encoded
        self.feature_names = feature_names
        self.feature_slices = feature_slices
        self.categorical_cols = categorical_cols
        self.continuous_cols = continuous_cols
        self.col_means = col_means
        self.col_stds = col_stds
        self.categories = categories


def _encode_dataframe(
    df: pl.DataFrame,
    ref: _EncodedData | None = None,
) -> _EncodedData:
    """One-hot encode categoricals, standardise continuous columns.

    When `ref` is provided (for validation/test sets), the encoding scheme
    (categories, means, stds) from the reference training set is applied exactly.
    This prevents data leakage and ensures consistent column order.
    """
    categorical_cols = [col for col in df.columns if df[col].dtype in (pl.Categorical, pl.String, pl.Enum)]
    continuous_cols = [col for col in df.columns if col not in categorical_cols]

    if ref is not None:
        # Apply training encoding
        categorical_cols = ref.categorical_cols
        continuous_cols = ref.continuous_cols

    parts: list[np.ndarray] = []
    feature_names: list[str] = []
    feature_slices: dict[str, slice] = {}
    col_means: dict[str, float] = {}
    col_stds: dict[str, float] = {}
    categories: dict[str, list[str]] = {}
    offset = 0

    # Continuous: standardise
    for col in continuous_cols:
        vals = df[col].cast(pl.Float64).to_numpy().astype(np.float32)
        if ref is not None:
            mean_ = ref.col_means[col]
            std_ = ref.col_stds[col]
        else:
            mean_ = float(np.mean(vals))
            std_ = float(np.std(vals)) or 1.0
        col_means[col] = mean_
        col_stds[col] = std_
        scaled = (vals - mean_) / std_
        parts.append(scaled.reshape(-1, 1))
        feature_names.append(col)
        feature_slices[col] = slice(offset, offset + 1)
        offset += 1

    # Categorical: one-hot (drop first to avoid multicollinearity)
    for col in categorical_cols:
        if ref is not None:
            cats = ref.categories[col]
        else:
            cats = sorted(df[col].drop_nulls().unique().to_list())
        categories[col] = cats
        # Encode: for each category except first, binary column
        col_vals = df[col].to_list()
        n_cols = len(cats) - 1
        ohe = np.zeros((len(df), n_cols), dtype=np.float32)
        for i, v in enumerate(col_vals):
            if v in cats[1:]:
                j = cats.index(v) - 1
                ohe[i, j] = 1.0
        parts.append(ohe)
        for k in range(n_cols):
            feature_names.append(f"{col}_{cats[k+1]}")
        feature_slices[col] = slice(offset, offset + n_cols)
        offset += n_cols

    X_encoded = np.concatenate(parts, axis=1) if parts else np.zeros((len(df), 0), dtype=np.float32)
    return _EncodedData(
        X_encoded=X_encoded,
        feature_names=feature_names,
        feature_slices=feature_slices,
        categorical_cols=categorical_cols,
        continuous_cols=continuous_cols,
        col_means=col_means,
        col_stds=col_stds,
        categories=categories,
    )


class CANN:
    """High-level CANN trainer.

    Handles data encoding, model construction, training loop with early stopping,
    and ensemble averaging. Produces trained weight matrices ready for NID.

    Parameters
    ----------
    family:
        "poisson" for claim frequency, "gamma" for claim severity.
    config:
        Training hyperparameters. Defaults work reasonably for most datasets.

    Example
    -------
    >>> cann = CANN(family="poisson")
    >>> cann.fit(X_train, y_train, glm_predictions_train, exposure=exposure_train,
    ...          X_val=X_val, y_val=y_val, glm_predictions_val=mu_val_glm,
    ...          exposure_val=exposure_val)
    >>> weights = cann.get_weight_matrices()
    """

    def __init__(self, family: Family = "poisson", config: CANNConfig | None = None) -> None:
        self.family = family
        self.config = config or CANNConfig()
        self._models: list[CANNModel] = []
        self._encoding: _EncodedData | None = None
        self._val_deviances: list[list[float]] = []

    def _loss_fn(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        if self.family == "poisson":
            return _poisson_deviance(y_pred, y_true, weights)
        else:
            return _gamma_deviance(y_pred, y_true, weights)

    def _train_one(
        self,
        X_t: torch.Tensor,
        y_t: torch.Tensor,
        glm_t: torch.Tensor,
        w_t: torch.Tensor,
        X_v: torch.Tensor,
        y_v: torch.Tensor,
        glm_v: torch.Tensor,
        w_v: torch.Tensor,
        encoding: _EncodedData,
        seed: int,
    ) -> tuple[CANNModel, list[float]]:
        torch.manual_seed(seed)
        model = CANNModel(
            input_dim=X_t.shape[1],
            hidden_dims=self.config.hidden_dims,
            activation=self.config.activation,
            mlp_m=self.config.mlp_m,
            feature_slices=encoding.feature_slices if self.config.mlp_m else {},
        )
        optimiser = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        train_ds = TensorDataset(X_t, y_t, glm_t, w_t)
        loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)

        best_val_loss = float("inf")
        best_state = None
        patience_count = 0
        val_history: list[float] = []

        for epoch in range(self.config.n_epochs):
            model.train()
            for xb, yb, gb, wb in loader:
                optimiser.zero_grad()
                pred = model(xb, gb)
                loss = self._loss_fn(pred, yb, wb)
                loss.backward()
                optimiser.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_v, glm_v)
                val_loss = self._loss_fn(val_pred, y_v, w_v).item()
            val_history.append(val_loss)

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.config.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        return model, val_history

    def fit(
        self,
        X: pl.DataFrame,
        y: np.ndarray,
        glm_predictions: np.ndarray,
        exposure: np.ndarray | None = None,
        X_val: pl.DataFrame | None = None,
        y_val: np.ndarray | None = None,
        glm_predictions_val: np.ndarray | None = None,
        exposure_val: np.ndarray | None = None,
    ) -> "CANN":
        """Fit the CANN (with optional ensemble).

        Parameters
        ----------
        X:
            Rating factors as Polars DataFrame. String/Categorical columns are
            one-hot encoded; numeric columns are standardised.
        y:
            Observed response. Claim counts for Poisson; claim amounts for Gamma.
        glm_predictions:
            Fitted values from the existing GLM (on the response scale, not log).
        exposure:
            Policy exposure (e.g. vehicle-years). Used as weights in the loss
            function. Defaults to ones if not supplied.
        X_val, y_val, glm_predictions_val, exposure_val:
            Optional explicit validation set for early stopping. If not supplied,
            ``config.validation_fraction`` of the training data is held out.
        """
        rng = np.random.default_rng(self.config.seed)

        n = len(X)
        if exposure is None:
            exposure = np.ones(n, dtype=np.float32)

        encoding = _encode_dataframe(X)
        self._encoding = encoding

        # Build validation split if no explicit val set
        if X_val is None:
            n_val = max(1, int(n * self.config.validation_fraction))
            idx = rng.permutation(n)
            val_idx, tr_idx = idx[:n_val], idx[n_val:]
            X_enc_tr = encoding.X_encoded[tr_idx]
            X_enc_v = encoding.X_encoded[val_idx]
            y_tr, y_v = y[tr_idx], y[val_idx]
            glm_tr, glm_v = glm_predictions[tr_idx], glm_predictions[val_idx]
            exp_tr, exp_v = exposure[tr_idx], exposure[val_idx]
        else:
            X_enc_tr = encoding.X_encoded
            y_tr, glm_tr, exp_tr = y, glm_predictions, exposure
            val_enc = _encode_dataframe(X_val, ref=encoding)
            X_enc_v = val_enc.X_encoded
            y_v = y_val if y_val is not None else y
            glm_v = glm_predictions_val if glm_predictions_val is not None else glm_predictions
            exp_v = exposure_val if exposure_val is not None else exposure

        def _to_t(arr: np.ndarray) -> torch.Tensor:
            return torch.tensor(arr.astype(np.float32))

        X_t = _to_t(X_enc_tr)
        y_t = _to_t(y_tr.astype(np.float32))
        glm_t = _to_t(np.log(np.clip(glm_tr, 1e-8, None)).astype(np.float32))
        w_t = _to_t(exp_tr.astype(np.float32))

        X_v_t = _to_t(X_enc_v)
        y_v_t = _to_t(y_v.astype(np.float32))
        glm_v_t = _to_t(np.log(np.clip(glm_v, 1e-8, None)).astype(np.float32))
        w_v_t = _to_t(exp_v.astype(np.float32))

        self._models = []
        self._val_deviances = []
        for i in range(self.config.n_ensemble):
            model, history = self._train_one(
                X_t, y_t, glm_t, w_t,
                X_v_t, y_v_t, glm_v_t, w_v_t,
                encoding,
                seed=self.config.seed + i,
            )
            self._models.append(model)
            self._val_deviances.append(history)

        return self

    def predict(self, X: pl.DataFrame, glm_predictions: np.ndarray) -> np.ndarray:
        """Generate predictions from the fitted CANN ensemble.

        Returns the mean prediction across ensemble members.
        """
        if self._encoding is None:
            raise RuntimeError("Call fit() before predict().")
        enc = _encode_dataframe(X, ref=self._encoding)
        X_t = torch.tensor(enc.X_encoded.astype(np.float32))
        glm_t = torch.tensor(np.log(np.clip(glm_predictions, 1e-8, None)).astype(np.float32))

        preds = []
        for model in self._models:
            model.eval()
            with torch.no_grad():
                preds.append(model(X_t, glm_t).numpy())
        return np.mean(preds, axis=0)

    def get_weight_matrices(self) -> list[tuple[np.ndarray, list[np.ndarray]]]:
        """Return (W1, subsequent_weights) for each ensemble member.

        W1 shape: (hidden_0, input_dim)
        Each element of subsequent_weights has shape: (out, in) for that layer.

        These are passed directly to the NID scorer.
        """
        if not self._models:
            raise RuntimeError("Call fit() before get_weight_matrices().")
        result = []
        for model in self._models:
            w1 = model.get_first_layer_weights()
            w_rest = model.get_subsequent_weight_matrices()
            result.append((w1, w_rest))
        return result

    @property
    def feature_names(self) -> list[str]:
        """Original feature names from the training DataFrame columns."""
        if self._encoding is None:
            raise RuntimeError("Call fit() first.")
        # Return original column names (not expanded one-hot names)
        return self._encoding.continuous_cols + self._encoding.categorical_cols

    @property
    def encoded_feature_names(self) -> list[str]:
        """Expanded feature names after one-hot encoding."""
        if self._encoding is None:
            raise RuntimeError("Call fit() first.")
        return self._encoding.feature_names

    @property
    def feature_slices(self) -> dict[str, slice]:
        """Maps original feature name to slice of encoded input columns."""
        if self._encoding is None:
            raise RuntimeError("Call fit() first.")
        return self._encoding.feature_slices

    @property
    def val_deviance_history(self) -> list[list[float]]:
        """Validation deviance per epoch for each ensemble member."""
        return self._val_deviances
