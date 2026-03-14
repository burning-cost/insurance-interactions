# insurance-interactions

[![Tests](https://github.com/burning-cost/insurance-interactions/actions/workflows/ci.yml/badge.svg)](https://github.com/burning-cost/insurance-interactions/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/insurance-interactions)](https://pypi.org/project/insurance-interactions/)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

Automated detection of missing interaction terms in UK personal lines GLMs.

## The problem

A Poisson frequency GLM for motor insurance with 12 rating factors has 66 possible pairwise interactions. Manually searching them — fitting, testing, reviewing 2D actual-vs-expected plots — takes days and is driven by intuition rather than data. You will miss interactions that are not obvious from marginal plots, and you will spend time testing pairs that are irrelevant.

The standard manual process:
1. Fit a GBM to get a benchmark prediction
2. Loop over pairs of factors; produce 2D A/E plots
3. Identify where the multiplicative GLM assumption breaks down
4. Test candidate interactions via likelihood-ratio test
5. Repeat

This library automates steps 2–4.

## What it does

The pipeline has three stages:

**Stage 1 - CANN**: Train a Combined Actuarial Neural Network (Schelldorfer & Wüthrich 2019) on the residuals of your existing GLM. The CANN uses a skip connection so it starts from the GLM prediction and only learns what the GLM is missing. After training, any deviation of the CANN from zero encodes structure the GLM cannot express — interactions.

**Stage 2 - NID**: Apply Neural Interaction Detection (Tsang et al. 2018) to the trained CANN weights. The algorithm reads the interaction structure directly from the weight matrices: two features can only interact if they both contribute to the same first-layer hidden unit. The NID score for a pair (i, j) is:

```
d(i,j) = Σ_s  z_s · min(|W1[s,i]|, |W1[s,j]|)
```

where `z_s` is how much first-layer unit `s` influences the output (the product of absolute weight matrices from layer 2 to the output). This gives a ranked list of candidate interactions in milliseconds after training.

**Stage 3 - GLM testing**: For each top-K candidate pair, refit the GLM with the interaction added and compute a likelihood-ratio test statistic. The output table includes deviance improvement, AIC/BIC, p-values (Bonferroni corrected), and `n_cells` — the parameter cost of adding each interaction.

Both Poisson (frequency) and Gamma (severity) families are supported.

## Quickstart

```python
import polars as pl
import numpy as np
from insurance_interactions import InteractionDetector, build_glm_with_interactions


# Generate synthetic UK motor data to run this example end to end.
# In production, supply your actual rating factor DataFrame, claim counts,
# fitted GLM predictions, and exposure weights.
rng = np.random.default_rng(42)
N = 10_000
age_band = pl.Series('age_band', rng.choice(['<25', '25-40', '40-60', '60+'], size=N))
vehicle_group = pl.Series('vehicle_group', rng.choice(['A', 'B', 'C', 'D'], size=N))
ncd = pl.Series('ncd', rng.integers(0, 10, size=N))
annual_mileage = pl.Series('annual_mileage', rng.integers(3000, 30000, size=N))
X_train = pl.DataFrame([age_band, vehicle_group, ncd, annual_mileage])

# Exposure and claim counts with a known age_band x vehicle_group interaction
exposure_train = rng.uniform(0.1, 1.0, size=N)
base_rate = 0.06
young_hv = ((age_band == '<25') & (vehicle_group == 'D')).to_numpy().astype(float)
mu_glm_train = base_rate * exposure_train * (1 + 0.4 * young_hv)  # 'true' GLM without interaction
y_train = rng.poisson(mu_glm_train)

detector = InteractionDetector(family="poisson")
detector.fit(
    X=X_train,
    y=y_train,
    glm_predictions=mu_glm_train,
    exposure=exposure_train,
)

# Ranked interaction table with deviance gains and LR test results
print(detector.interaction_table())

# Top recommended interactions (significant after Bonferroni correction)
suggested = detector.suggest_interactions(top_k=5)
# [("age_band", "vehicle_group"), ("age_band", "ncd"), ...]

# Refit GLM with approved interactions
final_model, comparison = build_glm_with_interactions(
    X=X_train,
    y=y_train,
    exposure=exposure_train,
    interaction_pairs=suggested,
    family="poisson",
)
print(comparison)
```

## Output

The interaction table contains one row per candidate pair:

| Column | Description |
|---|---|
| `feature_1`, `feature_2` | Factor names |
| `nid_score` | Raw NID score (higher = stronger detected interaction in CANN) |
| `nid_score_normalised` | Normalised to [0, 1] for interpretability |
| `n_cells` | Parameter cost: `(L_i - 1)(L_j - 1)` for cat×cat |
| `delta_deviance` | Deviance reduction when adding this pair to the GLM |
| `delta_deviance_pct` | As a percentage of base GLM deviance |
| `lr_chi2`, `lr_df`, `lr_p` | Likelihood-ratio test statistic and p-value |
| `recommended` | `True` if significant after Bonferroni correction |

The `n_cells` column is important for credibility decisions: a strong interaction requiring 200 new parameters may be less useful than a moderate one requiring 4.

## Installation

```bash
uv add insurance-interactions
```

With SHAP interaction validation (requires CatBoost):

```bash
uv add "insurance-interactions[shap]"
```

## Configuration

Training is controlled via `DetectorConfig`:

```python
from insurance_interactions import DetectorConfig, InteractionDetector

cfg = DetectorConfig(
    cann_hidden_dims=[32, 16],   # MLP architecture
    cann_n_epochs=300,
    cann_n_ensemble=5,           # Average over 5 training runs for stable NID
    cann_patience=30,            # Early stopping patience
    top_k_nid=20,                # NID pairs to forward to GLM testing
    top_k_final=10,              # Interactions in final suggest_interactions()
    mlp_m=True,                  # MLP-M variant: reduces false positive interactions
    nid_max_order=2,             # 2 = pairwise; 3 = also compute three-way
    alpha_bonferroni=0.05,       # Significance level after Bonferroni correction
)
detector = InteractionDetector(family="poisson", config=cfg)
```

### MLP-M variant

Setting `mlp_m=True` activates the MLP-M architecture (Tsang et al. 2018): each feature gets its own small univariate network to absorb the main effect, forcing the main MLP to model only interactions. This reduces false positive interactions at the cost of more training parameters. Recommended for datasets with strongly correlated features (e.g. age and NCD).

### Ensemble averaging

`cann_n_ensemble=3` (or more) trains multiple CANN runs with different random seeds and averages the NID scores. CANN training is stochastic; a single run may produce unstable weight matrices. Three runs is a reasonable default; five is better for production use.

## Frequency vs severity

Run the detector separately for frequency and severity:

```python
freq_detector = InteractionDetector(family="poisson")
freq_detector.fit(X=X, y=claim_counts, glm_predictions=mu_freq_glm, exposure=exposure)

sev_detector = InteractionDetector(family="gamma")
sev_detector.fit(X=X_claims, y=claim_amounts, glm_predictions=mu_sev_glm, exposure=claim_counts)
```

In practice, frequency and severity interactions differ. Young driver × sports car interactions are typically stronger in frequency. Severity interactions are noisier due to the higher variance in claim amounts.

## Theory

The CANN is from Schelldorfer & Wüthrich (2019), "Nesting Classical Actuarial Models into Neural Networks" (SSRN 3320525). NID is from Tsang, Cheng & Liu (2018), "Detecting Statistical Interactions from Neural Network Weights" (ICLR 2018). The direct application of this pipeline to insurance GLMs is in Lindström & Palmquist (2023), "Detection of Interacting Variables for GLMs via Neural Networks" (_European Actuarial Journal_).

The CANN architecture:

```
μ_CANN(x) = μ_GLM(x) * exp(NN(x; θ))
```

The GLM prediction enters as a fixed log-space offset. The output layer of the neural network is zero-initialised so the CANN equals the GLM exactly at the start of training. The network then learns only the residual structure — which, in a well-specified GLM missing interactions, corresponds to those interaction terms.

## Limitations

- NID depends on the CANN having converged. Poor training (small dataset, high learning rate, too few epochs) produces unreliable weight matrices. Use `n_ensemble ≥ 3` and check `cann.val_deviance_history`.
- Very small datasets (< 5,000 policies) may not provide enough signal for the CANN to learn stable residual structure. The LR tests still work but the NID ranking may be noisy.
- NID is not a statistical test — it produces a ranking, not p-values. The LR test in Stage 3 provides the statistical rigour.
- Correlated features (age and NCD in UK motor) can spread interaction signal across spurious pairs. The MLP-M variant with L1 sparsity partially mitigates this.
- The GLM refit step uses glum. If your rating engine uses a different GLM package, the `n_cells`, `delta_deviance`, and LR statistics are still valid; just refit your own model with the suggested interaction pairs.

## Regulatory context

UK actuaries working under PRA SS1/23 model risk governance and FCA Consumer Duty pricing rules need interaction decisions to be auditable. This library is designed to support that: it produces a ranked table with test statistics, not a black-box model. The actuary decides which interactions to add; the library provides the shortlist and the evidence.

## Databricks Notebook

[`insurance_interactions_demo.py`](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_interactions_demo.py) walks through the full CANN + NID + GLM testing pipeline end-to-end: a synthetic UK motor portfolio with two planted interactions, CANN training with ensemble averaging, NID ranking, likelihood-ratio testing with Bonferroni correction, and a final refitted GLM comparison table. It is the fastest way to see all three stages working together on realistic data before wiring in your own GLM predictions and rating factor DataFrame.

## Read more

[Finding the Interactions Your GLM Missed](https://burning-cost.github.io/2026/03/07/finding-the-interactions-your-glm-missed.html) — how CANN + NID automates the interaction search and why manual 2D A/E plots miss the non-obvious pairs.

## Performance

Benchmarked against a **main-effects-only Poisson GLM** on synthetic UK motor frequency data — 50,000 policies, two planted interactions in the known DGP (age_band × vehicle_group with delta=0.55 log-points; ncd_band × region with delta=0.35 log-points), 70/30 temporal split. The baseline GLM cannot express these interactions; the library's job is to find them automatically.

| Metric | Main-effects GLM | With detected interactions | Notes |
|--------|------------------|---------------------------|-------|
| Poisson deviance (test, weighted) | baseline | measured at runtime | expected −1% to −4% reduction |
| Gini coefficient | baseline | measured at runtime | expected +1 to +3 pp improvement |
| A/E max deviation (decile) | baseline | measured at runtime | expected −10% to −30% improvement |
| Planted interactions recovered | 0 / 2 | expected 2 / 2 | strong interactions (delta > 0.3) reliably detected |
| False positives (after Bonferroni) | 0 | expected 0–1 | MLP-M variant substantially reduces false positives |
| Deviance gain from interactions | 0% | expected 1–4% of base | depends on interaction effect size and cell prevalence |
| Detection + refit time | <1s (GLM only) | 3–8 min (CPU, 50k) | CANN training dominates; GPU reduces to under 2 min |

The deviance improvement is most pronounced when planted interactions have effect sizes above 0.3 log-points and affect at least 1% of policies. On homogeneous portfolios where the GLM's main-effects structure is correct, the library finds zero significant pairs after Bonferroni correction — it does not over-suggest.

Run `notebooks/benchmark.py` on Databricks to reproduce.

## Related libraries

| Library | Why it's relevant |
|---------|------------------|
| [shap-relativities](https://github.com/burning-cost/shap-relativities) | Extract rating relativities from GBMs — use the GBM as the benchmark that reveals where the GLM is missing structure |
| [bayesian-pricing](https://github.com/burning-cost/bayesian-pricing) | Hierarchical Bayesian models for thin rating cells — once interactions are identified, thin interaction cells need partial pooling |
| [insurance-datasets](https://github.com/burning-cost/insurance-datasets) | Synthetic UK motor and home datasets — use to validate the detector recovers known interaction structure |
| [insurance-cv](https://github.com/burning-cost/insurance-cv) | Walk-forward cross-validation for pricing models — use to assess whether adding interactions improves out-of-sample performance |

[All Burning Cost libraries →](https://burning-cost.github.io)

## Related Libraries

| Library | What it does |
|---------|-------------|
| [shap-relativities](https://github.com/burning-cost/shap-relativities) | Extract rating relativities from GBMs — use the GBM benchmark to identify where the GLM is missing structure |
| [insurance-causal](https://github.com/burning-cost/insurance-causal) | Double Machine Learning for causal inference — establishes whether detected interactions are genuine causal drivers |
| [insurance-synthetic](https://github.com/burning-cost/insurance-synthetic) | Synthetic portfolio generation — create datasets with known interaction structure to validate detection |

