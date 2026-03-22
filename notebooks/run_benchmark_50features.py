# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-interactions: 50-Feature Benchmark
# MAGIC
# MAGIC This notebook runs the 50-feature benchmark that demonstrates the core value
# MAGIC proposition of the library: finding planted interactions when there are 1,225
# MAGIC candidate pairs and exhaustive testing is impractical.
# MAGIC
# MAGIC **Setup:**
# MAGIC - 50,000 synthetic UK motor policies
# MAGIC - 50 rating factors (2 actors in planted interactions, 48 noise/main-effects only)
# MAGIC - 2 planted interactions: age_band×veh_group (delta=0.5), cover_type×telematics_band (delta=0.3)
# MAGIC - Production CANN settings: n_ensemble=5, n_epochs=300, mlp_m=True

# COMMAND ----------

# MAGIC %pip install "insurance-interactions[torch]" glum polars --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import sys
import time
import warnings
import numpy as np
import polars as pl
import pandas as pd

warnings.filterwarnings("ignore")

from insurance_interactions import InteractionDetector, DetectorConfig, test_interactions
from glum import GeneralizedLinearRegressor

print("All imports OK")

# COMMAND ----------

# MAGIC %md ## Data-Generating Process

# COMMAND ----------

BENCHMARK_START = time.time()

RNG = np.random.default_rng(42)

N_POLICIES = 50_000
BASE_FREQ = 0.07

FACTOR_DEFS = [
    # Four actors in planted interactions
    ("age_band",        5),
    ("veh_group",       6),
    ("cover_type",      3),
    ("telematics_band", 4),
    # Genuine rating factors (main effects only, no planted interactions)
    ("ncd_years",       6),
    ("region",          8),
    ("occupation",      5),
    ("veh_age_band",    5),
    ("mileage_band",    5),
    ("payment_freq",    2),
    ("veh_fuel",        3),
    ("parking",         3),
    ("channel",         3),
    ("renewal_count",   4),
    ("excess_band",     4),
    ("licence_years",   4),
    ("urban_rural",     2),
    ("household_cars",  3),
    ("addon_count",     3),
    ("veh_seats",       3),
    # Pure noise features
    ("colour",          5),
    ("door_count",      2),
    ("right_hand",      2),
    ("imported",        2),
    ("sunroof",         2),
    ("dash_cam",        2),
    ("winter_tyres",    2),
    ("protected_ncd",   2),
    ("named_driver",    2),
    ("multi_car",       2),
    ("noise_f30",       2),
    ("noise_f31",       3),
    ("noise_f32",       2),
    ("noise_f33",       4),
    ("noise_f34",       2),
    ("noise_f35",       2),
    ("noise_f36",       3),
    ("noise_f37",       2),
    ("noise_f38",       2),
    ("noise_f39",       2),
    ("noise_f40",       3),
    ("noise_f41",       2),
    ("noise_f42",       2),
    ("noise_f43",       4),
    ("noise_f44",       2),
    ("noise_f45",       2),
    ("noise_f46",       3),
    ("noise_f47",       2),
    ("noise_f48",       2),
    ("noise_f49",       2),
]

assert len(FACTOR_DEFS) == 50

N_FEATURES = len(FACTOR_DEFS)
N_PAIRS = N_FEATURES * (N_FEATURES - 1) // 2

print(f"Features: {N_FEATURES}, Candidate pairs: {N_PAIRS:,}")
print(f"Bonferroni (exhaustive): p < {0.05/N_PAIRS:.7f}")
print(f"Bonferroni (CANN+NID top-20): p < {0.05/20:.4f}")
print(f"Multiple-testing burden ratio: {N_PAIRS/20:.0f}x")

factors = {}
for name, n_levels in FACTOR_DEFS:
    factors[name] = RNG.integers(0, n_levels, N_POLICIES)

log_mu = np.full(N_POLICIES, np.log(BASE_FREQ))
rng_effects = np.random.default_rng(99)
main_effects = {}
for i, (name, n_levels) in enumerate(FACTOR_DEFS):
    if i < 20:
        eff = rng_effects.normal(0, 0.15, n_levels)
        eff -= eff.mean()
    else:
        eff = np.zeros(n_levels)
    main_effects[name] = eff
    log_mu += eff[factors[name]]

# Planted interaction 1: age_band × veh_group (STRONG, delta=0.5)
int1_mask = (factors["age_band"] == 0) & (factors["veh_group"] == 5)
DELTA_1 = 0.50
log_mu += DELTA_1 * int1_mask

# Planted interaction 2: cover_type × telematics_band (MODERATE, delta=0.3)
int2_mask = (factors["cover_type"] == 2) & (factors["telematics_band"] == 0)
DELTA_2 = 0.30
log_mu += DELTA_2 * int2_mask

PLANTED = {
    "age_band×veh_group":         ("age_band",    "veh_group",       DELTA_1, int1_mask),
    "cover_type×telematics_band": ("cover_type",  "telematics_band", DELTA_2, int2_mask),
}

PLANTED_PAIRS = {
    (f1, f2) for _, (f1, f2, _, __) in PLANTED.items()
} | {
    (f2, f1) for _, (f1, f2, _, __) in PLANTED.items()
}

exposure = RNG.uniform(0.5, 1.5, N_POLICIES)
true_mu = np.exp(log_mu) * exposure
claim_counts = RNG.poisson(true_mu)
y = claim_counts.astype(np.float64)

for iname, (f1, f2, delta, mask) in PLANTED.items():
    print(f"  {iname}: delta={delta}, {mask.sum():,} policies ({100*mask.mean():.1f}%)")

X = pl.DataFrame({
    name: [f"{name[:4]}_{v}" for v in factors[name]]
    for name, _ in FACTOR_DEFS
})

print(f"\nClaims: {claim_counts.sum():,} total ({claim_counts.sum()/N_POLICIES:.3f} per policy)")

# COMMAND ----------

# MAGIC %md ## Baseline GLM (main effects only)

# COMMAND ----------

X_pd = pd.get_dummies(X.to_pandas(), drop_first=True)
glm_base = GeneralizedLinearRegressor(family="poisson", alpha=0, fit_intercept=True)
glm_base.fit(X_pd, y, sample_weight=exposure)
glm_preds = glm_base.predict(X_pd)

pos = y > 0
dev_terms = np.where(pos, y * np.log(y / glm_preds) - (y - glm_preds), -(y - glm_preds))
base_deviance = float(2.0 * np.sum(dev_terms))
print(f"Base GLM deviance: {base_deviance:,.0f}")

# COMMAND ----------

# MAGIC %md ## Method 1: Exhaustive Pairwise GLM Testing (1,225 pairs)

# COMMAND ----------

FEATURE_NAMES = X.columns
all_pairs = [
    (FEATURE_NAMES[i], FEATURE_NAMES[j])
    for i in range(len(FEATURE_NAMES))
    for j in range(i + 1, len(FEATURE_NAMES))
]

print(f"Testing {len(all_pairs):,} pairs...")

t0 = time.time()
exhaustive_results = test_interactions(
    X=X,
    y=y,
    exposure=exposure,
    interaction_pairs=all_pairs,
    family="poisson",
    alpha_bonferroni=0.05,
)
exhaustive_time = time.time() - t0

exhaustive_pd = exhaustive_results.to_pandas()
exhaustive_sig = exhaustive_pd[exhaustive_pd["recommended"] == True]

exhaustive_tp = sum(
    1 for _, row in exhaustive_sig.iterrows()
    if (row["feature_1"], row["feature_2"]) in PLANTED_PAIRS
)
exhaustive_fp = len(exhaustive_sig) - exhaustive_tp

print(f"\nExhaustive results:")
print(f"  Time: {exhaustive_time:.1f}s")
print(f"  Significant (Bonferroni p < {0.05/N_PAIRS:.6f}): {len(exhaustive_sig)}")
print(f"  True positives: {exhaustive_tp} / 2")
print(f"  False positives: {exhaustive_fp}")

# Show planted interaction p-values
print("\n  Planted interaction results:")
for iname, (f1, f2, delta, _) in PLANTED.items():
    rows = exhaustive_pd[
        ((exhaustive_pd["feature_1"] == f1) & (exhaustive_pd["feature_2"] == f2)) |
        ((exhaustive_pd["feature_1"] == f2) & (exhaustive_pd["feature_2"] == f1))
    ]
    if len(rows) > 0:
        row = rows.iloc[0]
        rec = row.get("recommended", False)
        lr_p = row.get("lr_p", float("nan"))
        dd = row.get("delta_deviance", float("nan"))
        print(f"    {iname}: delta_dev={dd:.1f}, p={lr_p:.6f}, "
              f"{'DETECTED' if rec else f'missed (p > {0.05/N_PAIRS:.6f})'}")

# COMMAND ----------

# MAGIC %md ## Method 2: CANN + NID (Production Settings)

# COMMAND ----------

config = DetectorConfig(
    cann_hidden_dims=[64, 32],
    cann_n_epochs=300,
    cann_n_ensemble=5,
    cann_patience=30,
    cann_batch_size=1024,
    mlp_m=True,
    top_k_nid=20,
    top_k_final=10,
    alpha_bonferroni=0.05,
)

detector = InteractionDetector(family="poisson", config=config)

print(f"Training CANN (n_ensemble=5, n_epochs=300, mlp_m=True)...")
t0 = time.time()
detector.fit(
    X=X,
    y=y,
    glm_predictions=glm_preds,
    exposure=exposure,
)
cann_time = time.time() - t0

interaction_table = detector.interaction_table()
table_pd = interaction_table.to_pandas()

sig_cann = table_pd[table_pd["recommended"] == True]

cann_tp = sum(
    1 for _, row in sig_cann.iterrows()
    if (row["feature_1"], row["feature_2"]) in PLANTED_PAIRS
)
cann_fp = len(sig_cann) - cann_tp

print(f"\nCANN+NID results:")
print(f"  Time: {cann_time:.1f}s")
print(f"  Significant (Bonferroni p < {0.05/20:.4f}): {len(sig_cann)}")
print(f"  True positives: {cann_tp} / 2")
print(f"  False positives: {cann_fp}")

# COMMAND ----------

# MAGIC %md ## Top 10 Interactions by NID Score

# COMMAND ----------

display_rows = []
for i, row in table_pd.head(10).iterrows():
    lr_p = row.get("lr_p", float("nan"))
    nid_score = row.get("nid_score_normalised", row.get("nid_score", float("nan")))
    is_planted = (row["feature_1"], row["feature_2"]) in PLANTED_PAIRS or \
                 (row["feature_2"], row["feature_1"]) in PLANTED_PAIRS
    rec = row.get("recommended", False)
    display_rows.append({
        "rank": i + 1,
        "feature_1": row["feature_1"],
        "feature_2": row["feature_2"],
        "nid_score_norm": round(nid_score, 4),
        "lr_p": round(lr_p, 6) if not np.isnan(lr_p) else None,
        "recommended": rec,
        "planted": is_planted,
    })

display(pd.DataFrame(display_rows))

# COMMAND ----------

# MAGIC %md ## NID Rank of Planted Interactions

# COMMAND ----------

print("NID rank of each planted interaction:")
for iname, (f1, f2, delta, _) in PLANTED.items():
    for rank, row in enumerate(table_pd.itertuples(), start=1):
        if ((row.feature_1 == f1 and row.feature_2 == f2) or
                (row.feature_1 == f2 and row.feature_2 == f1)):
            lr_p = getattr(row, "lr_p", float("nan"))
            nid_score = getattr(row, "nid_score_normalised",
                                getattr(row, "nid_score", float("nan")))
            rec = getattr(row, "recommended", False)
            print(f"  {iname} (delta={delta}): rank {rank}/1225, NID={nid_score:.4f}, "
                  f"p={lr_p:.6f}, {'DETECTED' if rec else 'missed'}")
            break
    else:
        print(f"  {iname}: screened out (not in top 20 by NID)")

# COMMAND ----------

# MAGIC %md ## Final Summary

# COMMAND ----------

speedup = exhaustive_time / cann_time if cann_time > 0 else float("nan")
elapsed = time.time() - BENCHMARK_START

print("=" * 70)
print("BENCHMARK RESULTS: 50 Features, C(50,2) = 1,225 Candidate Pairs")
print("=" * 70)
print()
print(f"  {'Metric':<55} {'Exhaustive':>12} {'CANN+NID':>10}")
print(f"  {'-'*55} {'-'*12} {'-'*10}")
print(f"  {'GLM fits required':<55} {'1,225':>12} {'<= 20':>10}")
print(f"  {'Bonferroni threshold':<55} {'p<0.0000408':>12} {'p<0.0025':>10}")
print(f"  {'Time (seconds)':<55} {exhaustive_time:>12.1f} {cann_time:>10.1f}")
print(f"  {'True positives (out of 2 planted)':<55} {exhaustive_tp:>12} {cann_tp:>10}")
print(f"  {'False positives':<55} {exhaustive_fp:>12} {cann_fp:>10}")
print(f"  {'Speedup':<55} {'1.0x':>12} {f'{speedup:.1f}x':>10}")
print()
print(f"  Base GLM deviance: {base_deviance:,.0f}")
print()
print(f"Total notebook runtime: {elapsed:.1f}s")

# Store results for KB entry
results_summary = {
    "n_features": N_FEATURES,
    "n_pairs": N_PAIRS,
    "n_policies": N_POLICIES,
    "exhaustive_time_s": round(exhaustive_time, 1),
    "cann_time_s": round(cann_time, 1),
    "exhaustive_tp": exhaustive_tp,
    "exhaustive_fp": exhaustive_fp,
    "cann_tp": cann_tp,
    "cann_fp": cann_fp,
    "speedup": round(speedup, 1),
    "base_deviance": round(base_deviance),
}
print("\nResults dict (for KB):")
for k, v in results_summary.items():
    print(f"  {k}: {v}")
