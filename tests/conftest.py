"""Shared test fixtures for insurance-interactions tests.

All synthetic data is generated here so individual test files stay focused
on assertions rather than data setup. The synthetic datasets are designed to
have known, detectable interactions so we can verify the ranking properties.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def synthetic_poisson_data(rng: np.random.Generator) -> dict:
    """200 rows, 5 features. age_band × vehicle_group have a known interaction.

    Data generating process:
      log(mu) = intercept + beta_age[age] + beta_veh[veh] + interaction_term
    where interaction_term is non-zero only for (age_young × veh_sport).
    """
    n = 200

    age_cats = ["young", "mid", "old"]
    veh_cats = ["hatchback", "saloon", "sport"]
    region_cats = ["north", "south", "midlands"]

    age = rng.choice(age_cats, size=n)
    vehicle = rng.choice(veh_cats, size=n)
    region = rng.choice(region_cats, size=n)
    mileage = rng.uniform(5_000, 30_000, size=n)
    ncd = rng.integers(0, 9, size=n).astype(float)
    exposure = rng.uniform(0.5, 1.0, size=n)

    # Main effects
    age_effect = {"young": 0.8, "mid": 0.0, "old": -0.3}
    veh_effect = {"hatchback": 0.0, "saloon": -0.2, "sport": 0.5}

    log_mu = np.array([
        -1.5
        + age_effect[a]
        + veh_effect[v]
        - 0.02 * (m / 10_000)
        - 0.05 * n_
        for a, v, m, n_ in zip(age, vehicle, mileage, ncd)
    ])

    # Known interaction: young × sport adds +1.0 to log_mu
    interaction_mask = (age == "young") & (vehicle == "sport")
    log_mu[interaction_mask] += 1.0

    mu = np.exp(log_mu) * exposure
    y = rng.poisson(mu).astype(np.float32)

    X = pl.DataFrame({
        "age_band": pl.Series(age).cast(pl.String),
        "vehicle_group": pl.Series(vehicle).cast(pl.String),
        "region": pl.Series(region).cast(pl.String),
        "annual_mileage": mileage.astype(np.float32),
        "ncd": ncd.astype(np.float32),
    })

    # GLM predictions: main effects only (missing the interaction)
    glm_log_pred = np.array([
        -1.5 + age_effect[a] + veh_effect[v] - 0.02 * (m / 10_000) - 0.05 * n_
        for a, v, m, n_ in zip(age, vehicle, mileage, ncd)
    ]) + np.log(exposure)
    glm_predictions = np.exp(glm_log_pred).astype(np.float32)

    return {
        "X": X,
        "y": y,
        "exposure": exposure.astype(np.float32),
        "glm_predictions": glm_predictions,
        "known_interaction": ("age_band", "vehicle_group"),
    }


@pytest.fixture(scope="session")
def synthetic_gamma_data(rng: np.random.Generator) -> dict:
    """Gamma severity data with a known interaction for testing the Gamma family."""
    n = 150

    age_cats = ["young", "mid", "old"]
    veh_cats = ["small", "large"]
    age = rng.choice(age_cats, size=n)
    vehicle = rng.choice(veh_cats, size=n)
    mileage = rng.uniform(5_000, 20_000, size=n)

    age_effect = {"young": 0.3, "mid": 0.0, "old": -0.1}
    veh_effect = {"small": 0.0, "large": 0.4}

    log_mu = np.array([
        7.5 + age_effect[a] + veh_effect[v] + 0.01 * (m / 10_000)
        for a, v, m in zip(age, vehicle, mileage)
    ])
    # Interaction: young × large adds +0.6
    interaction_mask = (age == "young") & (vehicle == "large")
    log_mu[interaction_mask] += 0.6

    mu = np.exp(log_mu)
    # Gamma: shape=5, scale=mu/5
    y = rng.gamma(shape=5.0, scale=mu / 5.0).astype(np.float32)
    exposure = np.ones(n, dtype=np.float32)

    glm_log_pred = np.array([
        7.5 + age_effect[a] + veh_effect[v] + 0.01 * (m / 10_000)
        for a, v, m in zip(age, vehicle, mileage)
    ])
    glm_predictions = np.exp(glm_log_pred).astype(np.float32)

    X = pl.DataFrame({
        "age_band": pl.Series(age).cast(pl.String),
        "vehicle_group": pl.Series(vehicle).cast(pl.String),
        "annual_mileage": mileage.astype(np.float32),
    })

    return {
        "X": X,
        "y": y,
        "exposure": exposure,
        "glm_predictions": glm_predictions,
        "known_interaction": ("age_band", "vehicle_group"),
    }
