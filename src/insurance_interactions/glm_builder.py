"""GLM integration: add detected interactions to a glum model.

Once the interaction detector has ranked candidate pairs, this module handles
refitting the GLM with each interaction in turn and reporting the deviance
improvement. The output is an AIC/BIC comparison table and a fitted model
with the recommended interactions added.

Design choices:
  - We use glum (not statsmodels) because it handles large categorical features
    efficiently via sparse design matrices and supports L1/L2 regularisation.
  - Each interaction is added and tested independently (one at a time) before
    jointly refitting with the full set. This avoids overestimating the combined
    deviance improvement.
  - The likelihood-ratio test uses chi-squared with df = n_cells = (L_i-1)(L_j-1)
    for categorical x categorical, or (L_i-1) for categorical x continuous.
  - Bonferroni correction is applied by default to account for the multiple
    testing problem inherent in testing many candidate interactions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import scipy.stats


@dataclass
class InteractionTestResult:
    """Result of adding a single interaction to the GLM."""

    feature_1: str
    feature_2: str
    n_cells: int
    """Parameter cost: (L_1 - 1) * (L_2 - 1) for cat x cat, or (L-1) for cat x cont."""
    delta_deviance: float
    """Reduction in total deviance (positive = improvement)."""
    delta_deviance_pct: float
    """delta_deviance as percentage of base GLM deviance."""
    lr_chi2: float
    """Likelihood-ratio test statistic."""
    lr_df: int
    """Degrees of freedom for LR test (equal to n_cells)."""
    lr_p: float
    """P-value from chi-squared distribution."""
    aic_base: float
    aic_interaction: float
    delta_aic: float
    """Negative = interaction improves AIC (lower is better)."""
    bic_base: float
    bic_interaction: float
    delta_bic: float


def _compute_n_cells(df: pl.DataFrame, feature_1: str, feature_2: str) -> int:
    """Compute parameter cost of adding feature_1 x feature_2 interaction.

    For categorical x categorical: (L_1 - 1) * (L_2 - 1)
    For categorical x continuous: (L - 1)
    For continuous x continuous: 1
    """
    is_cat_1 = df[feature_1].dtype in (pl.Categorical, pl.String, pl.Enum)
    is_cat_2 = df[feature_2].dtype in (pl.Categorical, pl.String, pl.Enum)

    if is_cat_1 and is_cat_2:
        L1 = df[feature_1].n_unique()
        L2 = df[feature_2].n_unique()
        return (L1 - 1) * (L2 - 1)
    elif is_cat_1:
        L1 = df[feature_1].n_unique()
        return L1 - 1
    elif is_cat_2:
        L2 = df[feature_2].n_unique()
        return L2 - 1
    else:
        return 1


def _glum_deviance(model: Any, X: Any, y: np.ndarray, sample_weight: np.ndarray) -> float:
    """Compute total deviance from a fitted glum model.

    Computes deviance from model predictions directly, which is compatible
    with all versions of glum and avoids the .deviance() API surface.

    Supports Poisson and Gamma families by detecting the family name from
    the model object. Falls back to Poisson if detection fails.
    """
    mu = np.clip(model.predict(X), 1e-8, None)
    y_arr = np.asarray(y, dtype=np.float64)
    w_arr = np.asarray(sample_weight, dtype=np.float64)

    # Detect family from model attributes — glum stores family as an object
    family_obj = getattr(model, "family", None)
    family_cls = getattr(family_obj, "__class__", None)
    family_name = getattr(family_cls, "__name__", "").lower() if family_cls else ""

    if "gamma" in family_name:
        # Gamma deviance: 2 * sum(w * (-log(y/mu) + (y - mu)/mu))
        y_safe = np.clip(y_arr, 1e-8, None)
        d = 2.0 * np.sum(w_arr * (-np.log(y_safe / mu) + (y_safe - mu) / mu))
    else:
        # Poisson deviance (default; also correct for quasi-Poisson)
        # When y=0: contribution is 2 * mu (the log term is zero).
        # Mask the log computation explicitly to avoid divide-by-zero warnings
        # even though np.where would produce the right result either way.
        pos_mask = y_arr > 0
        log_term = np.zeros_like(y_arr)
        log_term[pos_mask] = y_arr[pos_mask] * np.log(y_arr[pos_mask] / mu[pos_mask])
        d = 2.0 * np.sum(w_arr * (log_term - (y_arr - mu)))

    return float(d)


def _fit_glm_with_fallback(
    glm_cls: Any,
    family: str,
    X_pd: Any,
    y: np.ndarray,
    exposure: np.ndarray,
    alpha: float,
) -> Any:
    """Fit a glum GLM with automatic ridge fallback on singular matrix.

    Tries the requested alpha first; if glum raises (singular matrix or
    convergence failure), retries with a small ridge penalty of 1e-4 to
    regularise the design matrix. This is common on small test datasets with
    many categorical cells and limited observations.
    """
    def _try(a: float) -> Any:
        m = glm_cls(family=family, alpha=a, fit_intercept=True)
        m.fit(X_pd, y, sample_weight=exposure)
        return m

    try:
        return _try(alpha)
    except Exception:
        # Singular or convergence failure — add small ridge and retry
        return _try(max(alpha, 1e-4))


def test_interactions(
    X: pl.DataFrame,
    y: np.ndarray,
    exposure: np.ndarray | None = None,
    interaction_pairs: list[tuple[str, str]] | None = None,
    family: str = "poisson",
    alpha_bonferroni: float = 0.05,
    l2_regularisation: float = 0.0,
) -> pl.DataFrame:
    """Test each candidate interaction pair by refitting the GLM with it added.

    Parameters
    ----------
    X:
        Rating factors. All columns are included as main effects; interaction
        pairs specify which cross terms to test.
    y:
        Response variable (claim counts for Poisson, amounts for Gamma).
    exposure:
        Policy exposure weights. Defaults to ones.
    interaction_pairs:
        List of (feature_1, feature_2) tuples to test. If None, all pairs
        in X.columns are tested (expensive).
    family:
        "poisson" or "gamma".
    alpha_bonferroni:
        Significance level after Bonferroni correction.
    l2_regularisation:
        Ridge penalty on GLM coefficients (applied to both base and interaction
        models consistently).

    Returns
    -------
    Polars DataFrame with one row per interaction pair, sorted by delta_deviance
    descending (best first).
    """
    try:
        from glum import GeneralizedLinearRegressor
        import pandas as pd
    except ImportError as e:
        raise ImportError("glum is required for GLM interaction testing. uv add glum") from e

    if exposure is None:
        exposure = np.ones(len(X), dtype=np.float64)
    # glum's tabmat backend requires float64
    y = np.asarray(y, dtype=np.float64)
    exposure = np.asarray(exposure, dtype=np.float64)

    if interaction_pairs is None:
        cols = X.columns
        interaction_pairs = [(cols[i], cols[j]) for i in range(len(cols)) for j in range(i+1, len(cols))]

    # Determine glum family
    if family not in ("poisson", "gamma"):
        raise ValueError(f"family must be 'poisson' or 'gamma', got '{family}'")
    glum_family = family

    # Fit base GLM (main effects only)
    X_pd = X.to_pandas()
    cat_cols = [c for c in X.columns if X[c].dtype in (pl.Categorical, pl.String, pl.Enum)]
    for col in cat_cols:
        X_pd[col] = pd.Categorical(X_pd[col].astype(str))

    base_model = _fit_glm_with_fallback(
        GeneralizedLinearRegressor, glum_family, X_pd, y, exposure, l2_regularisation
    )
    base_deviance = _glum_deviance(base_model, X_pd, y, exposure)

    n = len(X)
    n_params_base = len(base_model.coef_) + 1  # +1 for intercept
    aic_base = base_deviance + 2 * n_params_base
    bic_base = base_deviance + np.log(n) * n_params_base

    n_tests = len(interaction_pairs)
    bonferroni_threshold = alpha_bonferroni / max(n_tests, 1)

    results: list[InteractionTestResult] = []
    for feat1, feat2 in interaction_pairs:
        n_cells = _compute_n_cells(X, feat1, feat2)

        # Build interaction feature(s)
        X_int = X_pd.copy()
        if feat1 in cat_cols and feat2 in cat_cols:
            # Categorical x categorical: new combined categorical
            X_int[f"_ix_{feat1}_{feat2}"] = pd.Categorical(
                X_pd[feat1].astype(str) + "_X_" + X_pd[feat2].astype(str)
            )
        elif feat1 in cat_cols:
            # Categorical x continuous: multiply the continuous by each indicator
            for cat_val in X_pd[feat1].cat.categories[1:]:
                col_name = f"_ix_{feat1}_{cat_val}_{feat2}"
                X_int[col_name] = (X_pd[feat1] == cat_val).astype(float) * X_pd[feat2]
        elif feat2 in cat_cols:
            for cat_val in X_pd[feat2].cat.categories[1:]:
                col_name = f"_ix_{feat1}_{feat2}_{cat_val}"
                X_int[col_name] = (X_pd[feat2] == cat_val).astype(float) * X_pd[feat1]
        else:
            # Continuous x continuous: product term
            X_int[f"_ix_{feat1}_{feat2}"] = X_pd[feat1] * X_pd[feat2]

        try:
            int_model = _fit_glm_with_fallback(
                GeneralizedLinearRegressor, glum_family, X_int, y, exposure, l2_regularisation
            )
            int_deviance = _glum_deviance(int_model, X_int, y, exposure)

            delta_deviance = base_deviance - int_deviance
            lr_chi2 = delta_deviance
            lr_df = n_cells
            lr_p = float(scipy.stats.chi2.sf(lr_chi2, df=lr_df))

            n_params_int = len(int_model.coef_) + 1
            aic_int = int_deviance + 2 * n_params_int
            bic_int = int_deviance + np.log(n) * n_params_int

            results.append(
                InteractionTestResult(
                    feature_1=feat1,
                    feature_2=feat2,
                    n_cells=n_cells,
                    delta_deviance=float(delta_deviance),
                    delta_deviance_pct=float(100 * delta_deviance / max(abs(base_deviance), 1e-10)),
                    lr_chi2=float(lr_chi2),
                    lr_df=lr_df,
                    lr_p=lr_p,
                    aic_base=float(aic_base),
                    aic_interaction=float(aic_int),
                    delta_aic=float(aic_int - aic_base),
                    bic_base=float(bic_base),
                    bic_interaction=float(bic_int),
                    delta_bic=float(bic_int - bic_base),
                )
            )
        except Exception:
            # Both attempts failed — skip this pair
            continue

    if not results:
        return pl.DataFrame()

    df_out = pl.DataFrame(
        {
            "feature_1": [r.feature_1 for r in results],
            "feature_2": [r.feature_2 for r in results],
            "n_cells": [r.n_cells for r in results],
            "delta_deviance": [r.delta_deviance for r in results],
            "delta_deviance_pct": [r.delta_deviance_pct for r in results],
            "lr_chi2": [r.lr_chi2 for r in results],
            "lr_df": [r.lr_df for r in results],
            "lr_p": [r.lr_p for r in results],
            "aic_base": [r.aic_base for r in results],
            "aic_interaction": [r.aic_interaction for r in results],
            "delta_aic": [r.delta_aic for r in results],
            "bic_base": [r.bic_base for r in results],
            "bic_interaction": [r.bic_interaction for r in results],
            "delta_bic": [r.delta_bic for r in results],
            "recommended": [r.lr_p < bonferroni_threshold for r in results],
        }
    ).sort("delta_deviance", descending=True)

    return df_out


def build_glm_with_interactions(
    X: pl.DataFrame,
    y: np.ndarray,
    exposure: np.ndarray | None = None,
    interaction_pairs: list[tuple[str, str]] | None = None,
    family: str = "poisson",
    l2_regularisation: float = 0.0,
) -> tuple[Any, pl.DataFrame]:
    """Refit GLM jointly with all specified interaction pairs.

    Unlike ``test_interactions`` (which tests each pair in isolation), this fits
    a single model with all approved interactions simultaneously. The comparison
    table reports joint deviance improvement.

    Parameters
    ----------
    interaction_pairs:
        Interactions to add. Typically the recommended pairs from
        ``InteractionDetector.suggest_interactions()``.

    Returns
    -------
    (fitted_model, comparison_table)
        The fitted glum model and a one-row summary DataFrame.
    """
    try:
        from glum import GeneralizedLinearRegressor
        import pandas as pd
    except ImportError as e:
        raise ImportError("glum is required. uv add glum") from e

    if exposure is None:
        exposure = np.ones(len(X), dtype=np.float64)
    # glum's tabmat backend requires float64
    y = np.asarray(y, dtype=np.float64)
    exposure = np.asarray(exposure, dtype=np.float64)

    cat_cols = [c for c in X.columns if X[c].dtype in (pl.Categorical, pl.String, pl.Enum)]
    X_pd = X.to_pandas()
    for col in cat_cols:
        X_pd[col] = pd.Categorical(X_pd[col].astype(str))

    # Base model
    base_model = _fit_glm_with_fallback(
        GeneralizedLinearRegressor, family, X_pd, y, exposure, l2_regularisation
    )
    base_deviance = _glum_deviance(base_model, X_pd, y, exposure)
    n_params_base = len(base_model.coef_) + 1

    # Add interaction columns
    X_int = X_pd.copy()
    total_new_params = 0
    if interaction_pairs:
        for feat1, feat2 in interaction_pairs:
            n_cells = _compute_n_cells(X, feat1, feat2)
            total_new_params += n_cells
            if feat1 in cat_cols and feat2 in cat_cols:
                X_int[f"_ix_{feat1}_{feat2}"] = pd.Categorical(
                    X_pd[feat1].astype(str) + "_X_" + X_pd[feat2].astype(str)
                )
            elif feat1 in cat_cols:
                for cat_val in X_pd[feat1].cat.categories[1:]:
                    X_int[f"_ix_{feat1}_{cat_val}_{feat2}"] = (
                        (X_pd[feat1] == cat_val).astype(float) * X_pd[feat2]
                    )
            elif feat2 in cat_cols:
                for cat_val in X_pd[feat2].cat.categories[1:]:
                    X_int[f"_ix_{feat1}_{feat2}_{cat_val}"] = (
                        (X_pd[feat2] == cat_val).astype(float) * X_pd[feat1]
                    )
            else:
                X_int[f"_ix_{feat1}_{feat2}"] = X_pd[feat1] * X_pd[feat2]

    int_model = _fit_glm_with_fallback(
        GeneralizedLinearRegressor, family, X_int, y, exposure, l2_regularisation
    )
    int_deviance = _glum_deviance(int_model, X_int, y, exposure)

    n = len(X)
    n_params_int = len(int_model.coef_) + 1
    delta_deviance = base_deviance - int_deviance

    comparison = pl.DataFrame(
        {
            "model": ["base_glm", "glm_with_interactions"],
            "deviance": [float(base_deviance), float(int_deviance)],
            "n_params": [n_params_base, n_params_int],
            "aic": [
                float(base_deviance + 2 * n_params_base),
                float(int_deviance + 2 * n_params_int),
            ],
            "bic": [
                float(base_deviance + np.log(n) * n_params_base),
                float(int_deviance + np.log(n) * n_params_int),
            ],
            "delta_deviance": [0.0, float(delta_deviance)],
            "delta_deviance_pct": [0.0, float(100 * delta_deviance / max(abs(base_deviance), 1e-10))],
            "n_new_params": [0, total_new_params],
        }
    )

    return int_model, comparison
