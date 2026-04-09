"""
data_generator.py
Generates a 100k-row synthetic loan-approval dataset with intentional bias:
  - Rural applicants have lower base approval rates
  - Low-income applicants face steeper penalties
  - Subtle gender bias embedded in the approval logic
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42


def generate_dataset(n_samples: int = 100_000) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)

    # ── Feature generation ──────────────────────────────────────────────────
    age = rng.integers(18, 75, size=n_samples).astype(float)

    # Income: log-normal, roughly $5k–$250k; rural skews lower
    region = rng.choice(["urban", "rural"], size=n_samples, p=[0.6, 0.4])
    income_base = np.where(region == "urban",
                           rng.lognormal(mean=11.0, sigma=0.7, size=n_samples),
                           rng.lognormal(mean=10.3, sigma=0.7, size=n_samples))
    income = np.clip(income_base, 5_000, 300_000)

    # Education: 0=none, 1=high school, 2=some college, 3=bachelor's, 4=graduate
    edu_probs = [0.08, 0.28, 0.22, 0.28, 0.14]
    education = rng.choice([0, 1, 2, 3, 4], size=n_samples, p=edu_probs)

    gender = rng.choice(["male", "female"], size=n_samples, p=[0.50, 0.50])

    # ── Biased approval logic ────────────────────────────────────────────────
    # Base log-odds from legitimate features
    log_odds = (
        -2.5
        + 0.000018 * income
        + 0.35 * education
        + 0.012 * (age - 18)
        - 0.00009 * (age - 18) ** 2   # age penalty after ~middle age
    )

    # Intentional bias penalties (these are the unfair signals)
    rural_penalty = np.where(region == "rural", -0.80, 0.0)
    gender_penalty = np.where(gender == "female", -0.25, 0.0)
    low_income_penalty = np.where(income < 30_000, -0.60, 0.0)

    log_odds_biased = log_odds + rural_penalty + gender_penalty + low_income_penalty

    prob_approval = 1.0 / (1.0 + np.exp(-log_odds_biased))
    approved = rng.binomial(1, prob_approval).astype(int)

    df = pd.DataFrame({
        "age": age,
        "income": income.round(0),
        "education": education,
        "gender": gender,
        "region": region,
        "approved": approved,
    })

    return df


def encode_features(df: pd.DataFrame):
    """Return (X, y, feature_names) with numeric encoding suitable for sklearn."""
    enc = df.copy()
    enc["gender_enc"] = (enc["gender"] == "female").astype(float)
    enc["region_enc"] = (enc["region"] == "rural").astype(float)

    feature_cols = ["income", "education", "age", "gender_enc", "region_enc"]
    X = enc[feature_cols].values
    y = enc["approved"].values
    return X, y, feature_cols
