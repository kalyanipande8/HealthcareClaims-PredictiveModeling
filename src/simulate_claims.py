"""Simulate healthcare claims data where a 30-day readmission outcome
is driven by a known subset of true risk factors plus irrelevant noise
features -- so a predictive model's feature-importance recovery can be
validated against real ground truth."""
from __future__ import annotations

import numpy as np
import pandas as pd

N_PATIENTS = 3000
SEED = 331

TRUE_RISK_FEATURES = ["prior_admissions", "num_chronic_conditions", "age",
                      "length_of_stay", "er_visits_past_year"]
NOISE_FEATURES = [f"noise_{i}" for i in range(10)]


def simulate(n: int = N_PATIENTS, seed: int = SEED):
    rng = np.random.default_rng(seed)

    prior_admissions = rng.poisson(1.2, n)
    num_chronic = rng.poisson(2.0, n)
    age = rng.integers(18, 95, n)
    length_of_stay = rng.poisson(4, n)
    er_visits = rng.poisson(1.0, n)

    df = pd.DataFrame({
        "patient_id": [f"PT-{i:05d}" for i in range(n)],
        "prior_admissions": prior_admissions, "num_chronic_conditions": num_chronic,
        "age": age, "length_of_stay": length_of_stay, "er_visits_past_year": er_visits,
    })

    for nf in NOISE_FEATURES:
        df[nf] = rng.normal(0, 1, n)

    risk_score = (0.5 * prior_admissions + 0.4 * num_chronic + 0.02 * (age - 50)
                  + 0.15 * length_of_stay + 0.3 * er_visits)
    risk_score = (risk_score - risk_score.mean()) / risk_score.std()
    prob = 1 / (1 + np.exp(-(risk_score - 0.5)))
    df["readmitted_30d"] = rng.binomial(1, prob)

    return df


if __name__ == "__main__":
    df = simulate()
    print(df.shape, df["readmitted_30d"].mean())
