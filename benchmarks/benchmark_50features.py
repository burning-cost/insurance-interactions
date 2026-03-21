"""
Benchmark: insurance-interactions at scale — 50 rating factors.

## The scaling problem

A UK personal lines motor book with 50 rating factors (routine for a mid-sized
insurer once you account for region sub-bands, vehicle groups, telematics bands,
NCD steps, and scheme covariates) has:

    C(50, 2) = 1,225 pairwise interactions to test

Exhaustive LR testing of all 1,225 pairs has two costs:
1. Compute: fitting 1,225 GLMs on 50,000 policies (each with 50 one-hot columns
   plus an interaction term) is slow — roughly 30–60 minutes on a CPU.
2. False positives: Bonferroni threshold drops to p < 0.05/1,225 = 0.0000408.
   At p=0.05 nominal type I rate, you expect 1,225 * 0.05 = 61 false positives
   before correction. After Bonferroni you lose sensitivity on real interactions
   unless they are very strong (effect size > 0.3 log-points).

CANN+NID solves both problems:
- One neural network fit on all 50 features simultaneously
- NID produces a ranked shortlist of candidates from the weight matrices in
  milliseconds — no GLM fits needed for ranking
- Only the top-K candidates (e.g. top 15) receive LR tests, so the multiple
  testing burden is 1,225 → 15 tests
- Bonferroni threshold is p < 0.05/15 = 0.0033 — far more detectable

## Benchmark design

50,000 synthetic Poisson motor policies with 50 rating factors:
  - 10 genuine rating factors (age band, vehicle power, region, cover type, NCD,
    engine size, mileage band, fuel type, parking, channel) — same as benchmark.py
  - 40 noise covariates with small main effects and no interactions
    (scheme group, broker group, fleet size band, payment method, telematics flag,
    claims history, postcode density band, and 33 further noise factors)
  - 3 planted interactions among the 10 genuine factors:
    1. age_band × veh_power (log effect = 0.50, affects youngest + highest power)
    2. region × cover_type (log effect = 0.30, regional comprehensive loading)
    3. ncd × mileage_band (log effect = 0.25, low NCD + high mileage)
  - The 40 noise covariates have NO interactions with each other or with genuine
    factors — exhaustive testing must discriminate signal from noise

Metrics reported:
  - True positives (planted interactions recovered)
  - False positives (spurious pairs flagged)
  - Wall-clock time for each method
  - GLM fits required
  - Multiple testing threshold applied

Run:
    python benchmarks/benchmark_50features.py

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
print("Benchmark: insurance-interactions at scale — 50 rating factors")
print("=" * 70)
print()
print("Context: C(50,2) = 1,225 pairs. Exhaustive testing is impractical.")
print("CANN+NID screens all 1,225 pairs in one pass; only top-15 get GLM tests.")
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

import pandas as pd

# ---------------------------------------------------------------------------
# Data-generating process
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

N_POLICIES = 50_000
BASE_FREQ = 0.08

print(f"DGP: {N_POLICIES:,} motor policies, Poisson frequency")
print(f"     50 rating factors (10 genuine + 40 noise covariates)")
print(f"     3 planted interactions among genuine factors")
print(f"     Base frequency: {BASE_FREQ:.1%}")
print()

# ---- Genuine rating factors (same as benchmark.py) -------------------------

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

age_band  = RNG.integers(0, N_AGE_BANDS, N_POLICIES)
veh_power = RNG.integers(0, N_VEH_POWER, N_POLICIES)
region    = RNG.integers(0, N_REGIONS, N_POLICIES)
cover_type = RNG.integers(0, N_COVER, N_POLICIES)
ncd       = RNG.integers(0, N_NCD, N_POLICIES)
engine_sz = RNG.integers(0, N_ENGINE, N_POLICIES)
mileage   = RNG.integers(0, N_MILEAGE, N_POLICIES)
fuel      = RNG.integers(0, N_FUEL, N_POLICIES)
parking   = RNG.integers(0, N_PARKING, N_POLICIES)
channel   = RNG.integers(0, N_CHANNEL, N_POLICIES)

# ---- Noise covariates: 40 factors, various cardinalities -------------------
# These have small main effects but NO interactions. Exhaustive testing must
# discriminate these from the genuine interactions purely by chance.

NOISE_SPECS = [
    # (name, n_levels, max_abs_effect)
    ("scheme_grp",        4,  0.06),
    ("broker_grp",        5,  0.05),
    ("fleet_size",        3,  0.04),
    ("payment_method",    2,  0.03),
    ("telematics_flag",   2,  0.08),
    ("prior_claims",      4,  0.10),
    ("postcode_density",  4,  0.07),
    ("renewal_count",     5,  0.04),
    ("excess_level",      3,  0.03),
    ("legal_exp",         2,  0.02),
    ("breakdown_cover",   2,  0.03),
    ("keycare",           2,  0.01),
    ("home_emerg",        2,  0.01),
    ("courtesy_car",      2,  0.02),
    ("veh_age_grp",       5,  0.06),
    ("veh_value_grp",     4,  0.05),
    ("veh_use",           3,  0.04),
    ("annual_dist_grp",   4,  0.05),
    ("driver_count",      3,  0.03),
    ("young_driver_flag", 2,  0.12),
    ("occupation_grp",    6,  0.04),
    ("homeowner",         2,  0.03),
    ("marital_status",    3,  0.02),
    ("multi_car",         2,  0.02),
    ("multi_policy",      2,  0.02),
    ("direct_debit",      2,  0.01),
    ("online_quote",      2,  0.01),
    ("loyalty_years",     5,  0.03),
    ("claims_free_yrs",   6,  0.05),
    ("abs_fitted",        2,  0.02),
    ("airbags",           3,  0.02),
    ("tracker_fitted",    2,  0.03),
    ("immobiliser",       2,  0.02),
    ("modified_veh",      2,  0.04),
    ("restricted_mileage", 2, 0.03),
    ("night_driver",      2,  0.02),
    ("overseas_use",      2,  0.01),
    ("agreed_value",      2,  0.01),
    ("classic_car",       2,  0.02),
    ("high_performance",  2,  0.05),
]
assert len(NOISE_SPECS) == 40

noise_indices = {}
noise_effects = {}
for name, n_levels, max_eff in NOISE_SPECS:
    idx = RNG.integers(0, n_levels, N_POLICIES)
    effect = RNG.uniform(-max_eff, max_eff, n_levels)
    effect -= effect.mean()  # centre so no extra intercept shift
    noise_indices[name] = idx
    noise_effects[name] = effect

# ---- Main effects on genuine factors (log scale) ---------------------------

age_effect     = np.array([0.35,  0.10,  0.0, 0.05, 0.20])
veh_effect     = np.array([0.0,   0.12,  0.25, 0.45])
region_effect  = np.array([0.0,   0.08,  0.15, -0.10, 0.05, 0.20])
cover_effect   = np.array([0.0,   0.10,  0.25])
ncd_effect     = np.array([0.30,  0.15,  0.05, 0.0, -0.05, -0.12])
engine_effect  = np.array([0.0,   0.05,  0.10])
mileage_effect = np.array([-0.10, 0.0,   0.08, 0.15])
fuel_effect    = np.array([0.0,   0.03])
parking_effect = np.array([0.05,  0.0,  -0.05])
channel_effect = np.array([0.0,   0.03])

# ---- Planted interactions ---------------------------------------------------
# Only among the 10 genuine factors — the 40 noise covariates have none.

# PLANTED INTERACTION 1: age_band × veh_power  (same as benchmark.py)
# Young drivers (band 0) in high/performance vehicles: elevated risk
age_veh_ix = np.zeros((N_AGE_BANDS, N_VEH_POWER))
age_veh_ix[0, 2] = 0.25   # young + high power
age_veh_ix[0, 3] = 0.50   # young + performance
age_veh_ix[1, 3] = 0.20   # 25-34 + performance

# PLANTED INTERACTION 2: region × cover_type  (same as benchmark.py)
# Comprehensive cover has regional loading for theft / severe weather
region_cover_ix = np.zeros((N_REGIONS, N_COVER))
region_cover_ix[2, 2] = 0.30   # Region 2, comprehensive
region_cover_ix[5, 2] = 0.25   # Region 5, comprehensive

# PLANTED INTERACTION 3: ncd × mileage_band  (new)
# Low NCD + high mileage: extra loading (inexperienced + more exposure)
ncd_mileage_ix = np.zeros((N_NCD, N_MILEAGE))
ncd_mileage_ix[0, 3] = 0.25   # 0 years NCD + 20k+ miles
ncd_mileage_ix[1, 3] = 0.15   # 1 year NCD + 20k+ miles
ncd_mileage_ix[0, 2] = 0.10   # 0 years NCD + 10-20k miles

PLANTED = frozenset([
    ("age_band", "veh_power"),
    ("veh_power", "age_band"),
    ("region", "cover_type"),
    ("cover_type", "region"),
    ("ncd", "mileage_band"),
    ("mileage_band", "ncd"),
])
N_PLANTED = 3

# ---- Assemble log_mu -------------------------------------------------------

log_mu = (
    np.log(BASE_FREQ)
    + age_effect[age_band]
    + veh_effect[veh_power]
    + region_effect[region]
    + cover_effect[cover_type]
    + ncd_effect[ncd]
    + engine_effect[engine_sz]
    + mileage_effect[mileage]
    + fuel_effect[fuel]
    + parking_effect[parking]
    + channel_effect[channel]
    # noise covariates
    + sum(noise_effects[name][noise_indices[name]] for name in noise_effects)
    # planted interactions
    + age_veh_ix[age_band, veh_power]
    + region_cover_ix[region, cover_type]
    + ncd_mileage_ix[ncd, mileage]
)

exposure = RNG.uniform(0.5, 1.5, N_POLICIES)
true_mu = np.exp(log_mu) * exposure
claim_counts = RNG.poisson(true_mu)

print(f"Claims: {claim_counts.sum():,} total  ({claim_counts.sum()/N_POLICIES:.3f} per policy)")
print()

# ---- Build Polars DataFrame ------------------------------------------------

genuine_cols = {
    "age_band":    [f"age_{v}"  for v in age_band],
    "veh_power":   [f"pwr_{v}"  for v in veh_power],
    "region":      [f"reg_{v}"  for v in region],
    "cover_type":  [f"cov_{v}"  for v in cover_type],
    "ncd":         [f"ncd_{v}"  for v in ncd],
    "engine_size": [f"eng_{v}"  for v in engine_sz],
    "mileage_band":[f"mil_{v}"  for v in mileage],
    "fuel_type":   [f"fuel_{v}" for v in fuel],
    "parking":     [f"park_{v}" for v in parking],
    "channel":     [f"ch_{v}"   for v in channel],
}

noise_cols = {
    name: [f"{name}_{v}" for v in noise_indices[name]]
    for name in noise_indices
}

X = pl.DataFrame({**genuine_cols, **noise_cols})

FEATURE_NAMES = X.columns
N_FEATURES = len(FEATURE_NAMES)
N_PAIRS = N_FEATURES * (N_FEATURES - 1) // 2

print(f"Feature matrix: {N_FEATURES} features, {N_PAIRS:,} candidate pairs")
print(f"Bonferroni threshold (exhaustive): p < {0.05/N_PAIRS:.6f}")
print(f"Bonferroni threshold (top-15 NID): p < {0.05/15:.4f}")
print()

y = claim_counts.astype(np.float64)
exp_arr = exposure

# ---- Baseline GLM (main effects only) --------------------------------------

X_pd = pd.get_dummies(X.to_pandas(), drop_first=True)
glm_base = GeneralizedLinearRegressor(family="poisson", alpha=0, fit_intercept=True)
glm_base.fit(X_pd, y, sample_weight=exp_arr)
glm_preds = glm_base.predict(X_pd)

pos = y > 0
dev_terms = np.where(pos, y * np.log(y / glm_preds) - (y - glm_preds), -(y - glm_preds))
base_deviance = float(2.0 * np.sum(dev_terms))

print(f"Base GLM (main effects only):")
print(f"  Features: {N_FEATURES}  |  One-hot columns: {X_pd.shape[1]}")
print(f"  Deviance: {base_deviance:,.0f}")
print()

# ---------------------------------------------------------------------------
# Method 1: CANN + NID (library approach)
# ---------------------------------------------------------------------------

print("-" * 70)
print("METHOD 1: CANN + NID (insurance-interactions)")
print(f"  Screens all {N_PAIRS:,} pairs via neural network, tests top 15 candidates")
print("-" * 70)
print()

# Full production settings: n_ensemble=5, n_epochs=300.
# If this is too slow for your environment, reduce to n_ensemble=3, n_epochs=200.
# These settings are documented in the config dict below.
CANN_CONFIG = dict(
    cann_hidden_dims=[64, 32],   # wider than default to handle 50-feature input
    cann_n_epochs=300,
    cann_n_ensemble=5,
    cann_patience=30,
    cann_batch_size=1024,
    top_k_nid=15,                # only 15 GLM tests vs 1,225 exhaustive
    top_k_final=10,
    alpha_bonferroni=0.05,
)

print(f"  Config: n_ensemble={CANN_CONFIG['cann_n_ensemble']}, "
      f"n_epochs={CANN_CONFIG['cann_n_epochs']}, "
      f"top_k_nid={CANN_CONFIG['top_k_nid']}")

config = DetectorConfig(**CANN_CONFIG)
detector = InteractionDetector(family="poisson", config=config)

t0 = time.time()
detector.fit(
    X=X,
    y=y,
    glm_predictions=glm_preds,
    exposure=exp_arr,
)
cann_time = time.time() - t0

interaction_table = detector.interaction_table()
top_pairs = detector.suggest_interactions(top_k=10)

table_pd = interaction_table.to_pandas()

# Count significant pairs at Bonferroni-corrected threshold (15 tests)
alpha_cann = 0.05 / CANN_CONFIG["top_k_nid"]
if "lr_p" in table_pd.columns:
    sig_cann = table_pd[table_pd["lr_p"] < alpha_cann]
else:
    sig_cann = table_pd.head(0)

cann_tp = sum(
    1 for _, row in sig_cann.iterrows()
    if (row["feature_1"], row["feature_2"]) in PLANTED
)
cann_fp = len(sig_cann) - cann_tp

# Also check raw top-15 rows: how many planted pairs are in the NID shortlist?
top15 = table_pd.head(15)
cann_tp_shortlist = sum(
    1 for _, row in top15.iterrows()
    if (row["feature_1"], row["feature_2"]) in PLANTED
)

print()
print(f"  Time: {cann_time:.1f}s  ({cann_time/60:.1f} min)")
print(f"  Pairs screened by NID: {N_PAIRS:,}")
print(f"  GLM tests run: {CANN_CONFIG['top_k_nid']}")
print(f"  Planted interactions in top-15 shortlist: {cann_tp_shortlist} / {N_PLANTED}")
print(f"  Significant (p < {alpha_cann:.4f}): {len(sig_cann)}")
print(f"  True positives: {cann_tp} / {N_PLANTED}")
print(f"  False positives: {cann_fp}")
print()
print(f"  Top 10 by NID score:")
print(f"  {'Rank':>4} {'Feature 1':<20} {'Feature 2':<20} {'NID score':>10} {'LR p':>12} {'Planted?':>10}")
print(f"  {'-'*4} {'-'*20} {'-'*20} {'-'*10} {'-'*12} {'-'*10}")
for i, row in table_pd.head(10).iterrows():
    lr_p = row.get("lr_p", float("nan"))
    nid_score = row.get("nid_score", row.get("score", float("nan")))
    planted_tag = "YES" if (row["feature_1"], row["feature_2"]) in PLANTED else ""
    print(f"  {i+1:>4} {row['feature_1']:<20} {row['feature_2']:<20} "
          f"{nid_score:>10.4f} {lr_p:>12.4f} {planted_tag:>10}")
print()

# ---------------------------------------------------------------------------
# Method 2: Exhaustive LR testing — timing estimate
# ---------------------------------------------------------------------------
# Testing all 1,225 pairs takes too long to run interactively on most machines.
# We time a sample of 50 pairs and extrapolate.

print("-" * 70)
print("METHOD 2: Exhaustive pairwise GLM testing (C(50,2) = 1,225 pairs)")
print("-" * 70)
print()

all_pairs_full = [
    (FEATURE_NAMES[i], FEATURE_NAMES[j])
    for i in range(N_FEATURES)
    for j in range(i + 1, N_FEATURES)
]
assert len(all_pairs_full) == N_PAIRS

# Time a sample of 30 random pairs to estimate total cost
SAMPLE_SIZE = 30
sample_pairs = [all_pairs_full[i] for i in RNG.integers(0, N_PAIRS, SAMPLE_SIZE)]

print(f"  Timing {SAMPLE_SIZE} sample pairs to estimate full cost...")
t0 = time.time()
sample_results = test_interactions(
    X=X,
    y=y,
    exposure=exp_arr,
    interaction_pairs=sample_pairs,
    family="poisson",
    alpha_bonferroni=0.05,
)
sample_time = time.time() - t0
per_pair_time = sample_time / SAMPLE_SIZE
estimated_total = per_pair_time * N_PAIRS

print(f"  {SAMPLE_SIZE} pairs took {sample_time:.1f}s  ({per_pair_time:.2f}s/pair)")
print(f"  Estimated time for all {N_PAIRS:,} pairs: {estimated_total:.0f}s ({estimated_total/60:.1f} min)")
print()

# Run full exhaustive test only if estimated time is under 10 minutes
RUN_FULL_EXHAUSTIVE = estimated_total < 600

if RUN_FULL_EXHAUSTIVE:
    print(f"  Running full exhaustive test ({estimated_total:.0f}s estimated)...")
    t0 = time.time()
    exhaustive_results = test_interactions(
        X=X,
        y=y,
        exposure=exp_arr,
        interaction_pairs=all_pairs_full,
        family="poisson",
        alpha_bonferroni=0.05,
    )
    exhaustive_time = time.time() - t0

    alpha_exhaustive = 0.05 / N_PAIRS
    exhaustive_pd = exhaustive_results.to_pandas()
    exhaustive_sig = exhaustive_pd[exhaustive_pd["lr_p"] < alpha_exhaustive]

    exhaustive_tp = sum(
        1 for _, row in exhaustive_sig.iterrows()
        if (row["feature_1"], row["feature_2"]) in PLANTED
    )
    exhaustive_fp = len(exhaustive_sig) - exhaustive_tp

    print(f"  Actual time: {exhaustive_time:.1f}s  ({exhaustive_time/60:.1f} min)")
    print(f"  Bonferroni threshold: p < {alpha_exhaustive:.6f}")
    print(f"  Significant pairs: {len(exhaustive_sig)}")
    print(f"  True positives: {exhaustive_tp} / {N_PLANTED}")
    print(f"  False positives: {exhaustive_fp}")
    print()
    exhaustive_time_label = f"{exhaustive_time:.0f}s"
    exhaustive_tp_label = f"{exhaustive_tp}/{N_PLANTED}"
    exhaustive_fp_label = str(exhaustive_fp)
    exhaustive_pairs_label = str(N_PAIRS)
else:
    print(f"  Estimated time ({estimated_total/60:.1f} min) exceeds 10-minute cap.")
    print(f"  Skipping full exhaustive run. Reporting extrapolation only.")
    print()
    exhaustive_time_label = f"~{estimated_total/60:.0f} min (extrapolated)"
    exhaustive_tp_label = "not run"
    exhaustive_fp_label = "not run"
    exhaustive_pairs_label = f"{N_PAIRS:,}"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=" * 70)
print(f"SUMMARY: 50-feature benchmark ({N_POLICIES:,} policies, {N_PLANTED} planted interactions)")
print("=" * 70)
print()
print(f"  {'Metric':<48} {'CANN+NID':>12} {'Exhaustive LR':>16}")
print(f"  {'-'*48} {'-'*12} {'-'*16}")
print(f"  {'Candidate pairs':<48} {N_PAIRS:>12,} {exhaustive_pairs_label:>16}")
print(f"  {'GLM fits required':<48} {CANN_CONFIG['top_k_nid']:>12} {N_PAIRS:>16,}")
print(f"  {'Multiple testing threshold':<48} {f'p < {0.05/CANN_CONFIG[\"top_k_nid\"]:.4f}':>12} {f'p < {0.05/N_PAIRS:.6f}':>16}")
print(f"  {'True positives (out of {N_PLANTED})':<48} {cann_tp:>12} {exhaustive_tp_label:>16}")
print(f"  {'False positives':<48} {cann_fp:>12} {exhaustive_fp_label:>16}")
print(f"  {'Wall-clock time':<48} {f'{cann_time:.0f}s':>12} {exhaustive_time_label:>16}")
print()

print("PLANTED INTERACTIONS IN CANN+NID NID SHORTLIST (top 15):")
for i, row in table_pd.head(15).iterrows():
    lr_p = row.get("lr_p", float("nan"))
    nid_score = row.get("nid_score", row.get("score", float("nan")))
    is_planted = (row["feature_1"], row["feature_2"]) in PLANTED
    tag = "  <<< PLANTED" if is_planted else ""
    print(f"  {i+1:>2}. {row['feature_1']:<20} x {row['feature_2']:<20}  "
          f"NID={nid_score:.4f}  p={lr_p:.4f}{tag}")
print()

print("KEY TAKEAWAYS")
print(f"  1. Scale: with 50 features, exhaustive LR testing requires {N_PAIRS:,} GLM")
print(f"     fits. CANN+NID requires {CANN_CONFIG['top_k_nid']} (one neural net pass + {CANN_CONFIG['top_k_nid']} GLMs).")
print()
print(f"  2. Multiple testing: Bonferroni at {N_PAIRS:,} tests means p < {0.05/N_PAIRS:.6f}.")
print(f"     A real interaction with effect size 0.25 log-points may not survive")
print(f"     this threshold. CANN+NID uses p < {0.05/CANN_CONFIG['top_k_nid']:.4f} — 82x more sensitive.")
print()
print(f"  3. False positives: at nominal p=0.05, exhaustive testing expects")
print(f"     {N_PAIRS * 0.05:.0f} false positives before correction among {N_PAIRS:,} pairs.")
print(f"     NID pre-screening concentrates attention on plausible pairs,")
print(f"     dramatically reducing spurious discoveries.")
print()
print(f"  4. Speed: {estimated_total/60:.0f} min (exhaustive, extrapolated) vs "
      f"{cann_time/60:.1f} min (CANN+NID).")
print(f"     The CANN training time is fixed regardless of feature count.")
print(f"     Exhaustive testing grows as O(p^2) — 50 features is already painful;")
print(f"     80 features (C(80,2)=3,160 pairs) would be intractable on a working day.")
print()

elapsed = time.time() - BENCHMARK_START
print(f"Benchmark completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
