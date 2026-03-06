"""SHAP interaction values via shapiq for validation/confirmation.

This module is optional — it requires ``shapiq`` and a tree model (CatBoost or
compatible). The SHAP interaction layer serves as a second opinion on the NID
ranking. When both methods flag the same pair, you can be more confident the
interaction is real rather than a CANN training artefact.

The approach:
  1. Fit a CatBoost model (or accept a pre-fitted one) as a flexible oracle
  2. Compute pairwise Shapley interaction indices via shapiq's TreeExplainer
  3. Aggregate mean(|φ_ij|) across the portfolio to get global interaction strength
  4. Return ranked pairs

Theoretical note on correlated features: SHAP interaction values assume features
can be independently perturbed (interventional conditioning). This is violated in
insurance data where age, NCD, and vehicle value are structurally correlated. The
reported scores should be treated as indicative rankings, not precise measures.
The NID scores from the CANN provide a complementary view that does not make this
independence assumption.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import polars as pl


class ShapInteractionScore(NamedTuple):
    feature_1: str
    feature_2: str
    shap_score: float
    shap_score_normalised: float


def _require_shapiq() -> Any:
    try:
        import shapiq
        return shapiq
    except ImportError as e:
        raise ImportError(
            "shapiq is required for SHAP interaction values. "
            "Install it with: uv pip install "insurance-interactions[shap]""
        ) from e


def _require_catboost() -> Any:
    try:
        import catboost
        return catboost
    except ImportError as e:
        raise ImportError(
            "catboost is required for GBM-based validation. "
            "Install it with: uv pip install "insurance-interactions[shap]""
        ) from e


def fit_catboost(
    X: pl.DataFrame,
    y: np.ndarray,
    exposure: np.ndarray | None = None,
    family: str = "poisson",
    cat_features: list[str] | None = None,
    iterations: int = 500,
    depth: int = 6,
    learning_rate: float = 0.05,
    seed: int = 42,
    verbose: bool = False,
) -> Any:
    """Fit a CatBoost model for use as the SHAP interaction oracle.

    Parameters
    ----------
    X:
        Rating factors as Polars DataFrame.
    y:
        Response variable (counts for Poisson, amounts for Gamma).
    exposure:
        Policy exposure. For Poisson frequency, this is used as a sample weight
        since CatBoost does not have a native offset for Poisson regression.
        For Gamma severity, exposure weights observations by volume.
    family:
        "poisson" uses Poisson loss; "gamma" uses RMSE on log-transformed y
        (CatBoost's Tweedie objective is not consistently available across versions).
    cat_features:
        Column names to treat as categorical. If None, inferred from Polars dtypes.
    """
    catboost = _require_catboost()

    if cat_features is None:
        cat_features = [
            col for col in X.columns
            if X[col].dtype in (pl.Categorical, pl.String, pl.Enum)
        ]

    # Convert to pandas for CatBoost (it does not have native Polars support)
    X_pd = X.to_pandas()
    # Ensure categoricals are string type
    for col in cat_features:
        X_pd[col] = X_pd[col].astype(str)

    if exposure is None:
        exposure = np.ones(len(X), dtype=np.float32)

    if family == "poisson":
        loss_function = "Poisson"
        y_cb = y.astype(np.float32)
        sample_weight = exposure.astype(np.float32)
    else:
        loss_function = "RMSE"
        y_cb = np.log(np.clip(y.astype(np.float32), 1e-8, None))
        sample_weight = exposure.astype(np.float32)

    pool = catboost.Pool(
        X_pd,
        label=y_cb,
        weight=sample_weight,
        cat_features=cat_features,
    )

    model = catboost.CatBoostRegressor(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        loss_function=loss_function,
        random_seed=seed,
        verbose=verbose,
    )
    model.fit(pool)
    return model


def compute_shap_interactions(
    model: Any,
    X: pl.DataFrame,
    feature_names: list[str] | None = None,
    max_rows: int = 5000,
    seed: int = 42,
) -> list[ShapInteractionScore]:
    """Compute pairwise SHAP interaction values for a fitted tree model.

    If X has more than ``max_rows`` rows, a random subsample is taken for
    computational feasibility. For a portfolio of 500k policies, computing
    full SHAP interaction values is expensive; 5,000 rows is typically
    sufficient for stable rankings.

    Parameters
    ----------
    model:
        Fitted CatBoost model (or any model supported by shapiq TreeExplainer).
    X:
        Rating factors as Polars DataFrame (same columns as training data).
    feature_names:
        Column names to use in the output. Defaults to X.columns.
    max_rows:
        Maximum rows to use for SHAP computation.

    Returns
    -------
    Ranked list of ShapInteractionScore, highest mean absolute interaction first.
    """
    shapiq = _require_shapiq()

    if feature_names is None:
        feature_names = X.columns

    rng = np.random.default_rng(seed)
    if len(X) > max_rows:
        idx = rng.choice(len(X), size=max_rows, replace=False)
        X_sample = X[idx.tolist()]
    else:
        X_sample = X

    X_pd = X_sample.to_pandas()
    # Encode categoricals as strings for CatBoost
    for col in X_pd.columns:
        if X_pd[col].dtype == object or str(X_pd[col].dtype) in ("category",):
            X_pd[col] = X_pd[col].astype(str)

    X_np = X_pd.values

    try:
        explainer = shapiq.TreeExplainer(model=model, max_order=2, min_order=2)
        interaction_values = explainer.explain_all(X_np)
    except Exception as e:
        raise RuntimeError(
            f"shapiq TreeExplainer failed: {e}. "
            "Ensure the model is a supported tree ensemble (CatBoost, XGBoost, LightGBM)."
        ) from e

    # interaction_values is an InteractionValues object
    # We need the pairwise (order=2) mean absolute values
    n_features = len(feature_names)
    phi_matrix = np.zeros((n_features, n_features))

    # shapiq returns aggregated interaction values when .explain_all() is used
    # Extract the interaction dict for each observation and aggregate
    if hasattr(interaction_values, "values"):
        # Single InteractionValues object returned from explain_all in some shapiq versions
        iv = interaction_values
        for (i, j), v in iv.dict().items():
            if len((i, j)) == 2:
                phi_matrix[i, j] += abs(v)
                phi_matrix[j, i] += abs(v)
    elif hasattr(interaction_values, "__iter__"):
        n_obs = 0
        for iv in interaction_values:
            n_obs += 1
            for indices, v in iv.dict().items():
                if len(indices) == 2:
                    i, j = indices
                    phi_matrix[i, j] += abs(v)
                    phi_matrix[j, i] += abs(v)
        if n_obs > 0:
            phi_matrix /= n_obs
    else:
        raise RuntimeError("Unexpected return type from shapiq.explain_all()")

    # Extract upper triangle pairs
    scores_raw: list[tuple[int, int, float]] = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            scores_raw.append((i, j, phi_matrix[i, j]))

    if not scores_raw:
        return []

    max_score = max(s for _, _, s in scores_raw) or 1.0
    results = [
        ShapInteractionScore(
            feature_1=feature_names[i],
            feature_2=feature_names[j],
            shap_score=score,
            shap_score_normalised=score / max_score,
        )
        for i, j, score in scores_raw
    ]
    results.sort(key=lambda x: x.shap_score, reverse=True)
    return results


def shap_to_dataframe(scores: list[ShapInteractionScore]) -> pl.DataFrame:
    """Convert SHAP scores to a Polars DataFrame."""
    if not scores:
        return pl.DataFrame()
    return pl.DataFrame(
        {
            "feature_1": [s.feature_1 for s in scores],
            "feature_2": [s.feature_2 for s in scores],
            "shap_score": [s.shap_score for s in scores],
            "shap_score_normalised": [s.shap_score_normalised for s in scores],
        }
    )
