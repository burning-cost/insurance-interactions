"""InteractionDetector: the main user-facing class.

Orchestrates the full pipeline:
  1. Train CANN on GLM residuals
  2. Run NID on trained CANN weights → ranked pairwise interaction candidates
  3. Optionally validate with SHAP interaction values from a GBM
  4. Test top-K candidates via GLM likelihood-ratio tests
  5. Return a ranked, human-readable interaction table

This is designed for UK actuaries working on personal lines GLMs. The output
deliberately uses the language actuaries expect: deviance improvement, AIC/BIC,
likelihood-ratio p-values, and n_cells (parameter cost). The actuary makes the
final selection; the library provides the ranked shortlist.

Typical workflow:
  1. Fit a Poisson GLM to claim frequency
  2. Pass GLM predictions + rating factors to InteractionDetector
  3. Call .fit() to run the full pipeline
  4. Call .suggest_interactions() to get the top-K confirmed pairs
  5. Pass confirmed pairs to glm_builder.build_glm_with_interactions()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import polars as pl

from .cann import CANN, CANNConfig
from .glm_builder import _compute_n_cells, test_interactions
from .nid import InteractionScore, compute_nid_scores, nid_to_dataframe


Family = Literal["poisson", "gamma"]


@dataclass
class DetectorConfig:
    """Configuration for the full interaction detection pipeline."""

    # CANN training
    cann_hidden_dims: list[int] = field(default_factory=lambda: [32, 16])
    cann_activation: Literal["tanh", "relu"] = "tanh"
    cann_n_epochs: int = 200
    cann_batch_size: int = 512
    cann_learning_rate: float = 1e-3
    cann_weight_decay: float = 1e-4
    cann_patience: int = 20
    cann_validation_fraction: float = 0.2
    cann_n_ensemble: int = 3
    """Number of CANN training runs to average for stable NID scores."""
    cann_seed: int = 42
    mlp_m: bool = False
    """MLP-M variant: separate univariate nets for main effects. Reduces false positives."""

    # NID
    nid_max_order: int = 2
    """2 = pairwise only (default). 3 = also compute three-way interactions."""

    # GLM testing
    top_k_nid: int = 20
    """Number of top NID pairs to test with GLM likelihood-ratio tests."""
    top_k_final: int = 10
    """Number of interactions to include in suggest_interactions() output."""
    glm_l2: float = 0.0
    alpha_bonferroni: float = 0.05


class InteractionDetector:
    """Automated GLM interaction detector using CANN + NID.

    Parameters
    ----------
    family:
        "poisson" for claim frequency (Poisson GLM), "gamma" for severity (Gamma GLM).
    config:
        Pipeline configuration. Defaults are reasonable starting points.

    Example
    -------
    >>> from insurance_interactions import InteractionDetector
    >>> detector = InteractionDetector(family="poisson")
    >>> detector.fit(
    ...     X=X_train,
    ...     y=y_train,
    ...     glm_predictions=mu_glm_train,
    ...     exposure=exposure_train,
    ... )
    >>> table = detector.interaction_table()
    >>> top_pairs = detector.suggest_interactions(top_k=5)
    """

    def __init__(
        self,
        family: Family = "poisson",
        config: DetectorConfig | None = None,
    ) -> None:
        self.family = family
        self.config = config or DetectorConfig()
        self._cann: CANN | None = None
        self._nid_scores: list[InteractionScore] = []
        self._glm_test_results: pl.DataFrame | None = None
        self._combined_table: pl.DataFrame | None = None

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
        shap_model: object | None = None,
    ) -> "InteractionDetector":
        """Run the full interaction detection pipeline.

        Parameters
        ----------
        X:
            Rating factors as Polars DataFrame. Must contain all features to be
            considered for interactions.
        y:
            Observed response (claim counts for Poisson, claim amounts for Gamma).
        glm_predictions:
            Fitted values from the base GLM (on the response scale, not log).
            These enter the CANN as a fixed skip-connection offset.
        exposure:
            Policy exposure (e.g. vehicle-years). Defaults to ones.
        X_val, y_val, glm_predictions_val, exposure_val:
            Optional explicit validation set for CANN early stopping. If not
            supplied, ``config.cann_validation_fraction`` of training data is used.
        shap_model:
            Optional fitted CatBoost/tree model for SHAP interaction validation.
            If None, only NID scores are used for ranking.
        """
        cfg = self.config

        # Step 1: Train CANN
        cann_cfg = CANNConfig(
            hidden_dims=cfg.cann_hidden_dims,
            activation=cfg.cann_activation,
            n_epochs=cfg.cann_n_epochs,
            batch_size=cfg.cann_batch_size,
            learning_rate=cfg.cann_learning_rate,
            weight_decay=cfg.cann_weight_decay,
            patience=cfg.cann_patience,
            validation_fraction=cfg.cann_validation_fraction,
            seed=cfg.cann_seed,
            n_ensemble=cfg.cann_n_ensemble,
            mlp_m=cfg.mlp_m,
        )
        self._cann = CANN(family=self.family, config=cann_cfg)
        self._cann.fit(
            X=X,
            y=y,
            glm_predictions=glm_predictions,
            exposure=exposure,
            X_val=X_val,
            y_val=y_val,
            glm_predictions_val=glm_predictions_val,
            exposure_val=exposure_val,
        )

        # Step 2: NID scoring
        weight_matrices = self._cann.get_weight_matrices()
        self._nid_scores = compute_nid_scores(
            weight_matrices=weight_matrices,
            feature_slices=self._cann.feature_slices,
            max_order=cfg.nid_max_order,
            normalise=True,
        )

        # Step 3: Optional SHAP validation
        shap_results: pl.DataFrame | None = None
        if shap_model is not None:
            try:
                from .shap_interactions import compute_shap_interactions, shap_to_dataframe
                shap_scores = compute_shap_interactions(
                    model=shap_model,
                    X=X,
                    feature_names=self._cann.feature_names,
                )
                shap_results = shap_to_dataframe(shap_scores)
            except Exception as exc:
                import warnings
                warnings.warn(f"SHAP interaction computation failed: {exc}. Proceeding without SHAP scores.")

        # Step 4: GLM testing of top-K NID pairs (pairwise only)
        if cfg.nid_max_order == 2:
            top_nid_pairs = [
                (s.features[0], s.features[1])
                for s in self._nid_scores[:cfg.top_k_nid]
                if len(s.features) == 2
            ]
        else:
            # For higher-order, only test pairwise for GLM (3-way GLM terms are expensive)
            top_nid_pairs = [
                (s.features[0], s.features[1])
                for s in self._nid_scores
                if len(s.features) == 2
            ][:cfg.top_k_nid]

        self._glm_test_results = test_interactions(
            X=X,
            y=y,
            exposure=exposure,
            interaction_pairs=top_nid_pairs,
            family=self.family,
            alpha_bonferroni=cfg.alpha_bonferroni,
            l2_regularisation=cfg.glm_l2,
        )

        # Step 5: Build combined table
        self._combined_table = self._build_combined_table(shap_results)

        return self

    def _build_combined_table(
        self, shap_results: pl.DataFrame | None
    ) -> pl.DataFrame:
        """Join NID scores, SHAP scores (optional), and GLM test results."""
        # NID scores to DataFrame
        nid_df = nid_to_dataframe(
            [s for s in self._nid_scores if len(s.features) == 2]
        )
        if nid_df.is_empty():
            return pl.DataFrame()

        # Add NID rank
        nid_df = nid_df.with_row_index("nid_rank", offset=1)

        # Join with GLM test results
        if self._glm_test_results is not None and not self._glm_test_results.is_empty():
            combined = nid_df.join(
                self._glm_test_results,
                on=["feature_1", "feature_2"],
                how="left",
            )
        else:
            combined = nid_df

        # Join with SHAP scores if available
        if shap_results is not None and not shap_results.is_empty():
            shap_ranked = shap_results.with_row_index("shap_rank", offset=1)
            combined = combined.join(
                shap_ranked.select(["feature_1", "feature_2", "shap_score", "shap_score_normalised", "shap_rank"]),
                on=["feature_1", "feature_2"],
                how="left",
            )
            # Consensus rank: mean of normalised NID and SHAP ranks
            n_total = len(combined)
            combined = combined.with_columns(
                (
                    (pl.col("nid_rank") / n_total + pl.col("shap_rank").fill_null(n_total) / n_total) / 2.0
                ).alias("consensus_score")
            )
        else:
            combined = combined.with_columns(
                (pl.col("nid_rank").cast(pl.Float64) / pl.len().cast(pl.Float64)).alias("consensus_score")
            )

        # n_cells: computed from feature cardinalities already in GLM test results if present
        # If GLM tests weren't run for a pair, n_cells will be null

        # Sort by consensus score (lower = better rank)
        combined = combined.sort("consensus_score")

        return combined

    def interaction_table(self) -> pl.DataFrame:
        """Return the full ranked interaction table.

        Columns (where available):
          feature_1, feature_2, nid_score, nid_score_normalised, nid_rank,
          n_cells, delta_deviance, delta_deviance_pct,
          lr_chi2, lr_df, lr_p, recommended,
          shap_score, shap_score_normalised, shap_rank (if SHAP was run),
          consensus_score
        """
        if self._combined_table is None:
            raise RuntimeError("Call fit() before interaction_table().")
        return self._combined_table

    def suggest_interactions(
        self,
        top_k: int | None = None,
        require_significant: bool = True,
    ) -> list[tuple[str, str]]:
        """Return the top-K recommended interaction pairs.

        Parameters
        ----------
        top_k:
            Number of pairs to return. Defaults to ``config.top_k_final``.
        require_significant:
            If True (default), only return pairs where the likelihood-ratio test
            was significant after Bonferroni correction (``recommended == True``).
            Set to False to return top-K by consensus rank regardless of p-value.

        Returns
        -------
        List of (feature_1, feature_2) tuples, ranked best first.
        """
        if top_k is None:
            top_k = self.config.top_k_final
        if self._combined_table is None or self._combined_table.is_empty():
            return []

        df = self._combined_table
        if require_significant and "recommended" in df.columns:
            df = df.filter(pl.col("recommended") == True)  # noqa: E712

        return [
            (row["feature_1"], row["feature_2"])
            for row in df.head(top_k).iter_rows(named=True)
        ]

    @property
    def cann(self) -> CANN | None:
        """The fitted CANN object, for direct access to weight matrices."""
        return self._cann

    @property
    def nid_scores(self) -> list[InteractionScore]:
        """Raw NID scores before GLM testing."""
        return self._nid_scores

    def nid_table(self) -> pl.DataFrame:
        """NID scores as a Polars DataFrame, pairwise interactions only."""
        return nid_to_dataframe([s for s in self._nid_scores if len(s.features) == 2])

    def glm_test_table(self) -> pl.DataFrame:
        """GLM likelihood-ratio test results for the top NID candidates."""
        if self._glm_test_results is None:
            raise RuntimeError("Call fit() before glm_test_table().")
        return self._glm_test_results
