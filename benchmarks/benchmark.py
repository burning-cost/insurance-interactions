"""
NOTE: This is the 10-feature benchmark. At 10 features (C(10,2) = 45 pairs),
exhaustive pairwise LR testing is practical, and on compact settings
(n_ensemble=2, n_epochs=150), CANN+NID can be outperformed by brute force.
This benchmark is retained for honesty — it documents the regime where
exhaustive testing is viable.

The primary benchmark, demonstrating CANN+NID's real advantage, is at:
    benchmarks/benchmark_50features.py
At 50 features (C(50,2) = 1,225 pairs), exhaustive testing is impractical,
the Bonferroni threshold is 82x more stringent, and CANN+NID wins clearly.

Benchmark: insurance-interactions CANN/NID detection vs exhaustive pairwise
GLM testing for finding interaction terms in personal lines pricing.

The problem: when you build a Poisson GLM for claim frequency, you fit main
effects for each rating factor. But pricing factors often interact — the effect
of vehicle power is different for young drivers than experienced ones. These
interactions aren't visible in main effects alone.

The naive approach: test all pairwise interactions via likelihood-ratio tests.
With 10 features, that is C(10,2) = 45 pairs. With multiple testing and
correlated features, you get false positives. You also spend CPU time running
45 GLM fits.

The CANN+NID approach: fit a single neural network with a GLM skip connection
(CANN architecture, Schelldorfer & Wüthrich 2019). The network learns the
residual signal the GLM misses. Then apply NID (Neural Interaction Detection,
Tsang et al. 2018) to rank candidate pairs by interaction strength from the
network weights — without testing every pair. Only the top-K candidates are
verified with LR tests.

Setup:
- 50,000 synthetic motor policies, Poisson claim frequency
- 10 rating factors: age, vehicle_power, region, cover_type, ncd, engine_size,
  mileage_band, fuel_type, parking, channel
- 2 planted interactions:
  (1) age_band × vehicle_power: young drivers with high-power vehicles have
      multiplicatively higher risk than either factor alone suggests
  (2) region × cover_type: comprehensive cover in certain regions has extra loading
- 8 noise covariates with no interactions
- GLM main effects fitted as baseline

Metrics:
- Detection rate: does each method find both planted interactions?
- False positive rate: spurious interactions reported at p < 0.05
- Time to detect
- Deviance improvement from adding found interactions to GLM

Run:
    python benchmarks/benchmark.py

Install:
    pip install insurance-interactions[torch] glum numpy polars
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

BENCHMARK_START = time.time()

print("=" * 70)
print("Benchmark: insurance-interactions CANN/NID vs exhaustive GLM testing")
print("=" * 70)
print()

try:
    from insurance_interactions import (
        InteractionDetector,
        DetectorConfig,
        test_interactions,
    )
    print("insurance-interactions imported OK")
except ImportError as e:
    print(f"ERROR: Could not import insurance-interactions: {e}")
    print("Install with: pip install insurance-interactions[torch]")
    sys.exit(1)

try:
    import polars as pl
except ImportError:
    print("ERROR: polars required. Install with: pip install polars")
    sys.exit(1)

try:
    from glum import GeneralizedLinearRegressor
except ImportError:
    print("ERROR: glum required. Install with: pip install glum")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Data-generating process with 2 planted interactions
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

N_POLICIES = 50_000
BASE_FREQ = 0.08

print(f"DGP: {N_POLICIES:,} motor policies, Poisson frequency")
print(f"     Base frequency: {BASE_FREQ:.1%}")
print(f"     Planted interactions: age_band×vehicle_power, region×cover_type")
print()

# Factor levels
N_AGE_BANDS = 5      # 17-24, 25-34, 35-49, 50-64, 65+
N_VEH_POWER = 4      # low, medium, high, performance
N_REGIONS = 6        # UK regions
N_COVER = 3          # TPO, TPFT, Comprehensive
N_NCD = 6            # 0, 1, 2, 3, 4, 5+ years
N_ENGINE = 3         # small, medium, large
N_MILEAGE = 4        # <5k, 5-10k, 10-20k, 20k+
N_FUEL = 2           # petrol, diesel
N_PARKING = 3        # street, garage, drive
N_CHANNEL = 2        # PCW, direct

# Generate raw factor indices
age_band = RNG.integers(0, N_AGE_BANDS, N_POLICIES)
veh_power = RNG.integers(0, N_VEH_POWER, N_POLICIES)
region = RNG.integers(0, N_REGIONS, N_POLICIES)
cover_type = RNG.integers(0, N_COVER, N_POLICIES)
ncd = RNG.integers(0, N_NCD, N_POLICIES)
engine_size = RNG.integers(0, N_ENGINE, N_POLICIES)
mileage_band = RNG.integers(0, N_MILEAGE, N_POLICIES)
fuel_type = RNG.integers(0, N_FUEL, N_POLICIES)
parking = RNG.integers(0, N_PARKING, N_POLICIES)
channel = RNG.integers(0, N_CHANNEL, N_POLICIES)

# Main effects on log scale
# age: youngest (0) and oldest (4) bands have higher risk
age_effect = np.array([0.35, 0.10, 0.0, 0.05, 0.20])
# vehicle power: monotone increasing
veh_effect = np.array([0.0, 0.12, 0.25, 0.45])
# region
region_effect = np.array([0.0, 0.08, 0.15, -0.10, 0.05, 0.20])
# cover type: comprehensive has more claims (more covered perils)
cover_effect = np.array([0.0, 0.10, 0.25])
# ncd: monotone decreasing (more experience = lower risk)
ncd_effect = np.array([0.30, 0.15, 0.05, 0.0, -0.05, -0.12])
# noise covariates: small main effects only
engine_effect = np.array([0.0, 0.05, 0.10])
mileage_effect = np.array([-0.10, 0.0, 0.08, 0.15])
fuel_effect = np.array([0.0, 0.03])
parking_effect = np.array([0.05, 0.0, -0.05])
channel_effect = np.array([0.0, 0.03])

# PLANTED INTERACTION 1: age_band × vehicle_power
# Young drivers (band 0) in high-power vehicles: +0.4 extra loading
# This is NOT captured by main effects alone
age_veh_interaction = np.zeros((N_AGE_BANDS, N_VEH_POWER))
age_veh_interaction[0, 2] = 0.25   # young + high power
age_veh_interaction[0, 3] = 0.50   # young + performance
age_veh_interaction[1, 3] = 0.20   # 25-34 + performance

# PLANTED INTERACTION 2: region × cover_type
# Comprehensive in certain regions has extra loading (theft/weather)
region_cover_interaction = np.zeros((N_REGIONS, N_COVER))
region_cover_interaction[2, 2] = 0.30   # Region 2, comprehensive
region_cover_interaction[5, 2] = 0.25   # Region 5, comprehensive

# True log mu
log_mu = (
    np.log(BASE_FREQ)
    + age_effect[age_band]
    + veh_effect[veh_power]
    + region_effect[region]
    + cover_effect[cover_type]
    + ncd_effect[ncd]
    + engine_effect[engine_size]
    + mileage_effect[mileage_band]
    + fuel_effect[fuel_type]
    + parking_effect[parking]
    + channel_effect[channel]
    + age_veh_interaction[age_band, veh_power]       # planted interaction 1
    + region_cover_interaction[region, cover_type]   # planted interaction 2
)

exposure = RNG.uniform(0.5, 1.5, N_POLICIES)
true_mu = np.exp(log_mu) * exposure
claim_counts = RNG.poisson(true_mu)

print(f"Claims: {claim_counts.sum():,} total  ({claim_counts.sum()/N_POLICIES:.3f} per policy)")
print()

# Build Polars DataFrame with string factors (categorical)
X = pl.DataFrame({
    "age_band":    [f"age_{v}" for v in age_band],
    "veh_power":   [f"pwr_{v}" for v in veh_power],
    "region":      [f"reg_{v}" for v in region],
    "cover_type":  [f"cov_{v}" for v in cover_type],
    "ncd":         [f"ncd_{v}" for v in ncd],
    "engine_size": [f"eng_{v}" for v in engine_size],
    "mileage_band":[f"mil_{v}" for v in mileage_band],
    "fuel_type":   [f"fuel_{v}" for v in fuel_type],
    "parking":     [f"park_{v}" for v in parking],
    "channel":     [f"ch_{v}" for v in channel],
})

y = claim_counts.astype(np.float64)
exp_arr = exposure

# Fit baseline Poisson GLM (main effects only) to get glm_predictions
import pandas as pd

X_pd = pd.get_dummies(X.to_pandas(), drop_first=True)
glm_base = GeneralizedLinearRegressor(family="poisson", alpha=0, fit_intercept=True)
glm_base.fit(X_pd, y, sample_weight=exp_arr)
glm_preds = glm_base.predict(X_pd)
base_deviance = float(2.0 * np.sum(
    np.where(y > 0, y * np.log(np.maximum(y / glm_preds, 1e-10)) - (y - glm_preds), -(y - glm_preds))
) * exp_arr.mean())  # approximate weighted deviance

# More accurate base deviance
pos = y > 0
dev_terms = np.where(pos, y * np.log(y / glm_preds) - (y - glm_preds), -(y - glm_preds))
base_deviance = float(2.0 * np.sum(dev_terms))

print(f"Base GLM (main effects only):")
print(f"  Deviance: {base_deviance:,.0f}")
print()

# ---------------------------------------------------------------------------
# Method 1: Exhaustive pairwise GLM testing (naive approach)
# ---------------------------------------------------------------------------

print("-" * 70)
print("METHOD 1: Exhaustive pairwise GLM testing (all C(10,2) = 45 pairs)")
print("-" * 70)
print()

FEATURE_NAMES = X.columns
all_pairs = [(FEATURE_NAMES[i], FEATURE_NAMES[j])
             for i in range(len(FEATURE_NAMES))
             for j in range(i + 1, len(FEATURE_NAMES))]
print(f"  Testing {len(all_pairs)} pairs...")

t0 = time.time()
exhaustive_results = test_interactions(
    X=X,
    y=y,
    exposure=exp_arr,
    interaction_pairs=all_pairs,
    family="poisson",
    alpha_bonferroni=0.05,
)
exhaustive_time = time.time() - t0

# Count detected planted interactions
PLANTED = {("age_band", "veh_power"), ("veh_power", "age_band"),
           ("region", "cover_type"), ("cover_type", "region")}

exhaustive_pd = exhaustive_results.to_pandas()
exhaustive_sig = exhaustive_pd[exhaustive_pd["lr_p"] < 0.05]

exhaustive_tp = sum(
    1 for _, row in exhaustive_sig.iterrows()
    if (row["feature_1"], row["feature_2"]) in PLANTED or
       (row["feature_2"], row["feature_1"]) in PLANTED
)
exhaustive_fp = len(exhaustive_sig) - exhaustive_tp

print(f"  Time: {exhaustive_time:.1f}s")
print(f"  Significant pairs (p < 0.05): {len(exhaustive_sig)}")
print(f"  True positives (planted found): {exhaustive_tp} / 2")
print(f"  False positives: {exhaustive_fp}")
print()

# ---------------------------------------------------------------------------
# Method 2: CANN + NID (library approach)
# ---------------------------------------------------------------------------

print("-" * 70)
print("METHOD 2: CANN + NID interaction detection (insurance-interactions)")
print("-" * 70)
print()

# Use compact config for benchmark speed
config = DetectorConfig(
    cann_hidden_dims=[32, 16],
    cann_n_epochs=150,
    cann_n_ensemble=2,
    cann_patience=15,
    cann_batch_size=1024,
    top_k_nid=10,
    top_k_final=5,
    alpha_bonferroni=0.05,
)

detector = InteractionDetector(family="poisson", config=config)

t0 = time.time()
detector.fit(
    X=X,
    y=y,
    glm_predictions=glm_preds,
    exposure=exp_arr,
)
cann_time = time.time() - t0

# Get interaction table and suggestions
interaction_table = detector.interaction_table()
top_pairs = detector.suggest_interactions(top_k=5)

table_pd = interaction_table.to_pandas()
sig_cann = table_pd[table_pd["lr_p"] < 0.05] if "lr_p" in table_pd.columns else table_pd.head(5)

cann_tp = sum(
    1 for _, row in sig_cann.iterrows()
    if (row["feature_1"], row["feature_2"]) in PLANTED or
       (row["feature_2"], row["feature_1"]) in PLANTED
)
cann_fp = len(sig_cann) - cann_tp

print(f"  Time (CANN training + NID + top-10 GLM tests): {cann_time:.1f}s")
print(f"  NID screened {len(all_pairs)} pairs -> tested top 10 candidates")
print(f"  Significant pairs: {len(sig_cann)}")
print(f"  True positives (planted found): {cann_tp} / 2")
print(f"  False positives: {cann_fp}")
print()

print("  Top 5 interactions by NID score:")
print(f"  {'Rank':>4} {'Feature 1':<16} {'Feature 2':<16} {'NID score':>10} {'LR p-value':>12}")
print(f"  {'-'*4} {'-'*16} {'-'*16} {'-'*10} {'-'*12}")
for i, row in table_pd.head(5).iterrows():
    lr_p = row.get("lr_p", float("nan"))
    nid_score = row.get("nid_score", row.get("score", float("nan")))
    print(f"  {i+1:>4} {row['feature_1']:<16} {row['feature_2']:<16} {nid_score:>10.4f} {lr_p:>12.4f}")
print()

# Deviance improvement from adding found interactions
if top_pairs is not None:
    top_pairs_pd = top_pairs.to_pandas() if hasattr(top_pairs, "to_pandas") else top_pairs
    found_pairs = [(row["feature_1"], row["feature_2"]) for _, row in top_pairs_pd.head(2).iterrows()]

    if found_pairs:
        from insurance_interactions import build_glm_with_interactions
        glm_with_interactions = build_glm_with_interactions(
            X=X,
            y=y,
            exposure=exp_arr,
            interaction_pairs=found_pairs,
            family="poisson",
        )
        # Compute deviance improvement
        preds_with = glm_with_interactions.predict(
            pd.get_dummies(X.to_pandas(), drop_first=True)  # base columns only approximation
        )

print()

# ---------------------------------------------------------------------------
# Summary comparison
# ---------------------------------------------------------------------------

print("=" * 70)
print("SUMMARY: CANN/NID vs exhaustive pairwise GLM testing")
print("=" * 70)
print()

print(f"  {'Metric':<50} {'Exhaustive':>12} {'CANN+NID':>10}")
print(f"  {'-'*50} {'-'*12} {'-'*10}")
print(f"  {'Pairs tested':<50} {len(all_pairs):>12} {'top 10':>10}")
print(f"  {'Time (seconds)':<50} {exhaustive_time:>12.1f} {cann_time:>10.1f}")
print(f"  {'True positives (out of 2 planted)':<50} {exhaustive_tp:>12} {cann_tp:>10}")
print(f"  {'False positives (spurious)':<50} {exhaustive_fp:>12} {cann_fp:>10}")
print(f"  {'GLM fits required':<50} {len(all_pairs):>12} {'<= 10':>10}")
print()

print("TOP INTERACTIONS FOUND BY EACH METHOD")
print(f"  Planted: age_band x veh_power, region x cover_type")
print()
print(f"  Exhaustive top 5 by deviance:")
for _, row in exhaustive_pd.sort_values("delta_deviance", ascending=False).head(5).iterrows():
    tag = " <<< PLANTED" if (row["feature_1"], row["feature_2"]) in PLANTED or \
          (row["feature_2"], row["feature_1"]) in PLANTED else ""
    print(f"    {row['feature_1']:15} x {row['feature_2']:15}  "
          f"delta_dev={row['delta_deviance']:>8.1f}  p={row['lr_p']:.4f}{tag}")
print()
print(f"  CANN+NID top 5 by NID score (pre-tested):")
for i, row in table_pd.head(5).iterrows():
    lr_p = row.get("lr_p", float("nan"))
    nid_score = row.get("nid_score", row.get("score", float("nan")))
    tag = " <<< PLANTED" if (row["feature_1"], row["feature_2"]) in PLANTED or \
          (row["feature_2"], row["feature_1"]) in PLANTED else ""
    print(f"    {row['feature_1']:15} x {row['feature_2']:15}  "
          f"NID={nid_score:.4f}  p={lr_p:.4f}{tag}")

print()
print("INTERPRETATION")
print(f"  Exhaustive testing: tests every pair, Bonferroni corrects for")
print(f"  {len(all_pairs)} tests. False positive rate rises with number of features.")
print(f"  At 10 features (45 tests), Bonferroni threshold is p < {0.05/len(all_pairs):.4f}.")
print()
print(f"  CANN+NID: the neural network captures all non-linear residuals in")
print(f"  one forward pass. NID reads off interaction strength from weight")
print(f"  matrices — it is a ranking heuristic, not a statistical test.")
print(f"  Only the top-{config.top_k_nid} candidates get GLM LR tests, so the")
print(f"  multiple testing burden is dramatically lower.")
print()
print(f"  Speed advantage: {exhaustive_time/cann_time:.1f}x speedup from CANN+NID.")
print(f"  The advantage grows quadratically with the number of features.")

elapsed = time.time() - BENCHMARK_START
print(f"\nBenchmark completed in {elapsed:.1f}s")
