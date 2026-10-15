import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulate_claims import simulate, TRUE_RISK_FEATURES
from src.predictive_model import train_and_evaluate, validate_feature_recovery


def test_cohort_has_3000_patients():
    df = simulate()
    assert len(df) == 3000


def test_model_beats_random_auc():
    df = simulate()
    result = train_and_evaluate(df)
    assert result["auc"] > 0.6


def test_feature_recovery_mostly_finds_true_risk_factors():
    df = simulate()
    result = train_and_evaluate(df)
    recovery = validate_feature_recovery(result["importances"])
    assert recovery["recovery_pct"] >= 60.0


def test_noise_features_rank_below_true_features_on_average():
    df = simulate()
    result = train_and_evaluate(df)
    imp = result["importances"]
    true_mean = imp[TRUE_RISK_FEATURES].mean()
    noise_cols = [c for c in imp.index if c.startswith("noise_")]
    noise_mean = imp[noise_cols].mean()
    assert true_mean > noise_mean


def test_readmission_rate_is_realistic():
    df = simulate()
    rate = df["readmitted_30d"].mean()
    assert 0.05 < rate < 0.60
