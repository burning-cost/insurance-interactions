# Databricks notebook source
# MAGIC %md
# MAGIC # Automated GLM Interaction Detection: Full Pipeline Demo
# MAGIC
# MAGIC This notebook demonstrates the complete `insurance-interactions` workflow on
# MAGIC synthetic UK motor insurance data with a known interaction built into the
# MAGIC data generating process.
# MAGIC
# MAGIC **What the library does**: Takes a fitted Poisson (or Gamma) GLM, trains a
# MAGIC Combined Actuarial Neural Network (CANN) on the residual structure, applies
# MAGIC Neural Interaction Detection (NID) to the trained weights, and produces a
# MAGIC ranked table of candidate interactions with GLM likelihood-ratio test statistics.
# MAGIC
# MAGIC **The test**: We build data where `age_band × vehicle_group` is a real interaction.
# MAGIC The GLM cannot see this. The CANN learns it. NID detects it. We verify it with an LR test.

# COMMAND ----------

# MAGIC %pip install -e /Workspace/Users/pricing.frontier@gmail.com/insurance-interactions

# COMMAND ----------

# MAGIC %pip install glum scipy

# COMMAND ----------

import numpy as np
import polars as pl

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generate Synthetic Motor Insurance Data
# MAGIC
# MAGIC A Poisson frequency model with:
# MAGIC - 5 rating factors: age_band (5 levels), vehicle_group (4 levels), region (3 levels),
# MAGIC   annual_mileage (continuous), ncd (continuous)
# MAGIC - Known interaction: `age_young × vehicle_sport` adds +1.2 log-points to claim frequency

# COMMAND ----------

rng = np.random.default_rng(42)
n = 5_000

AGE_CATS = ["17-25", "26-35", "36-50", "51-65", "66+"]
VEH_CATS = ["hatchback", "saloon", "estate", "sport"]
REGION_CATS = ["london", "south_east", "midlands", "north", "scotland"]

age = rng.choice(AGE_CATS, size=n)
vehicle = rng.choice(VEH_CATS, size=n)
region = rng.choice(REGION_CATS, size=n)
mileage = rng.uniform(5_000, 40_000, size=n)
ncd = rng.integers(0, 9, size=n).astype(float)
exposure = rng.uniform(0.3, 1.0, size=n)

# Main effects (log scale)
age_effect = {"17-25": 0.9, "26-35": 0.3, "36-50": 0.0, "51-65": -0.2, "66+": -0.1}
veh_effect = {"hatchback": 0.0, "saloon": -0.1, "estate": -0.15, "sport": 0.6}
region_effect = {"london": 0.3, "south_east": 0.1, "midlands": 0.0, "north": -0.1, "scotland": -0.2}

log_mu_base = np.array([
    -2.0 + age_effect[a] + veh_effect[v] + region_effect[r]
    - 0.015 * (m / 10_000) - 0.08 * n_
    for a, v, r, m, n_ in zip(age, vehicle, region, mileage, ncd)
])

# KNOWN INTERACTION: young (17-25) × sport adds +1.2
interaction_mask = (age == "17-25") & (vehicle == "sport")
print(f"Policies with known interaction (17-25 × sport): {interaction_mask.sum()} ({100*interaction_mask.mean():.1f}%)")

log_mu_true = log_mu_base.copy()
log_mu_true[interaction_mask] += 1.2

mu_true = np.exp(log_mu_true) * exposure
y = rng.poisson(mu_true).astype(np.float32)

X = pl.DataFrame({
    "age_band": pl.Series(age).cast(pl.String),
    "vehicle_group": pl.Series(vehicle).cast(pl.String),
    "region": pl.Series(region).cast(pl.String),
    "annual_mileage": mileage.astype(np.float32),
    "ncd": ncd.astype(np.float32),
})

print(f"\nDataset: {n:,} policies, {int(y.sum()):,} claims")
print(f"Overall frequency: {y.sum() / exposure.sum():.4f}")
print(f"\nFeature summary:")
print(X.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Fit Base GLM (Main Effects Only)
# MAGIC
# MAGIC This is the starting point: a Poisson GLM with all five rating factors as main
# MAGIC effects but no interactions. The `glm_predictions` from this model are what we
# MAGIC pass to the CANN as the skip connection.

# COMMAND ----------

from glum import GeneralizedLinearRegressor
import pandas as pd

cat_cols = ["age_band", "vehicle_group", "region"]
X_pd = X.to_pandas()
for col in cat_cols:
    X_pd[col] = pd.Categorical(X_pd[col].astype(str))

glm = GeneralizedLinearRegressor(family="poisson", fit_intercept=True)
glm.fit(X_pd, y, sample_weight=exposure)

glm_predictions = glm.predict(X_pd)
base_deviance = glm.deviance(X_pd, y, sample_weight=exposure)

print(f"Base GLM deviance: {base_deviance:.2f}")
print(f"Coefficients: {len(glm.coef_)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Train CANN
# MAGIC
# MAGIC The CANN starts from the GLM prediction (zero-initialised output layer) and
# MAGIC learns only the residual structure. Any deviation from the GLM prediction after
# MAGIC training corresponds to structure the GLM cannot express - in this case, the
# MAGIC age_band × vehicle_group interaction.

# COMMAND ----------

from insurance_interactions import CANN, CANNConfig

cfg = CANNConfig(
    hidden_dims=[32, 16],
    activation="tanh",
    n_epochs=200,
    batch_size=256,
    learning_rate=1e-3,
    weight_decay=1e-4,
    patience=20,
    validation_fraction=0.2,
    n_ensemble=3,  # 3 runs; NID scores are averaged
    seed=42,
)

cann = CANN(family="poisson", config=cfg)
cann.fit(
    X=X,
    y=y.astype(np.float32),
    glm_predictions=glm_predictions.astype(np.float32),
    exposure=exposure.astype(np.float32),
)

# Compare deviances
cann_preds = cann.predict(X, glm_predictions.astype(np.float32))

# Compute Poisson deviance manually for CANN predictions
y_safe = np.clip(y.astype(np.float64), 1e-10, None)
mu_safe = np.clip(cann_preds.astype(np.float64), 1e-10, None)
cann_deviance = float(2 * np.sum(
    exposure * (y_safe * np.log(y_safe / mu_safe) - (y_safe - mu_safe))
))

print(f"Base GLM deviance: {base_deviance:,.2f}")
print(f"CANN deviance:     {cann_deviance:,.2f}")
print(f"CANN improvement:  {100 * (base_deviance - cann_deviance) / base_deviance:.2f}%")
print(f"\nValidation history (3 ensemble runs, first 10 epochs):")
for i, history in enumerate(cann.val_deviance_history):
    print(f"  Run {i+1}: initial={history[0]:.3f}, final={history[-1]:.3f}, epochs={len(history)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. NID Scoring
# MAGIC
# MAGIC Extract pairwise interaction scores from the trained CANN weights.
# MAGIC The formula is: `d(i,j) = Σ_s  z_s * min(|W1[s,i]|, |W1[s,j]|)`
# MAGIC where `z_s` is the output importance of first-layer unit `s`.

# COMMAND ----------

from insurance_interactions import compute_nid_scores, nid_to_dataframe

weight_matrices = cann.get_weight_matrices()
nid_scores = compute_nid_scores(
    weight_matrices=weight_matrices,
    feature_slices=cann.feature_slices,
    max_order=2,
    normalise=True,
)

nid_df = nid_to_dataframe(nid_scores)
print("NID Interaction Scores (ranked):")
print(nid_df)

print(f"\nTop pair: {nid_scores[0].features[0]} × {nid_scores[0].features[1]}")
print(f"Known interaction: age_band × vehicle_group")
print(f"Known pair in top-3: {frozenset(('age_band', 'vehicle_group')) in [frozenset(s.features) for s in nid_scores[:3]]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. GLM Likelihood-Ratio Tests
# MAGIC
# MAGIC For each top-5 NID candidate pair, refit the GLM with that interaction added
# MAGIC and compute the likelihood-ratio test statistic. This gives the actuary a
# MAGIC p-value and deviance improvement to judge whether the interaction is worth adding.
# MAGIC
# MAGIC The `n_cells` column is the parameter cost: for a 5-level × 4-level interaction
# MAGIC that is (5-1)(4-1) = 12 new parameters. Credibility depends on having enough
# MAGIC claims in each cell.

# COMMAND ----------

from insurance_interactions import test_interactions

top_5_pairs = [(s.features[0], s.features[1]) for s in nid_scores[:5]]
print(f"Testing interactions: {top_5_pairs}")

glm_results = test_interactions(
    X=X,
    y=y.astype(np.float32),
    exposure=exposure.astype(np.float32),
    interaction_pairs=top_5_pairs,
    family="poisson",
    alpha_bonferroni=0.05,
)

print("\nGLM Interaction Tests:")
print(glm_results.select([
    "feature_1", "feature_2", "n_cells",
    "delta_deviance", "delta_deviance_pct",
    "lr_chi2", "lr_df", "lr_p", "recommended"
]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Full Pipeline via InteractionDetector

# COMMAND ----------

from insurance_interactions import DetectorConfig, InteractionDetector

cfg_full = DetectorConfig(
    cann_hidden_dims=[32, 16],
    cann_n_epochs=150,
    cann_n_ensemble=3,
    cann_patience=20,
    top_k_nid=10,
    top_k_final=5,
    alpha_bonferroni=0.05,
)

detector = InteractionDetector(family="poisson", config=cfg_full)
detector.fit(
    X=X,
    y=y.astype(np.float32),
    glm_predictions=glm_predictions.astype(np.float32),
    exposure=exposure.astype(np.float32),
)

print("Full interaction table:")
print(detector.interaction_table())

print("\nSuggested interactions (significant at Bonferroni threshold):")
suggestions = detector.suggest_interactions(require_significant=True)
print(suggestions)

print("\nSuggested interactions (top-5 by NID rank, no p-value filter):")
suggestions_all = detector.suggest_interactions(top_k=5, require_significant=False)
print(suggestions_all)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Refit GLM with Recommended Interactions

# COMMAND ----------

from insurance_interactions import build_glm_with_interactions

if suggestions:
    final_model, comparison = build_glm_with_interactions(
        X=X,
        y=y.astype(np.float32),
        exposure=exposure.astype(np.float32),
        interaction_pairs=suggestions,
        family="poisson",
    )

    print("GLM comparison (base vs with interactions):")
    print(comparison)

    pct_improvement = float(comparison.filter(
        pl.col("model") == "glm_with_interactions"
    )["delta_deviance_pct"][0])
    print(f"\nDeviance improvement: {pct_improvement:.2f}%")
else:
    print("No interactions passed significance threshold. Check alpha or increase n_ensemble.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Run Test Suite

# COMMAND ----------

# MAGIC %pip install pytest

# COMMAND ----------

import subprocess
result = subprocess.run(
    ["python", "-m", "pytest",
     "/Workspace/Users/pricing.frontier@gmail.com/insurance-interactions/tests/",
     "-v", "--tb=short"],
    capture_output=True,
    text=True,
)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:")
    print(result.stderr)
    raise RuntimeError("Tests failed - see output above")
else:
    print("\nAll tests passed.")
