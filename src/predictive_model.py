"""Real gradient-boosted readmission-risk model with cross-validated
AUC evaluation and permutation-based feature-importance recovery."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from .simulate_claims import TRUE_RISK_FEATURES, NOISE_FEATURES


def train_and_evaluate(df: pd.DataFrame, cv: int = 5, seed: int = 331) -> dict:
    features = TRUE_RISK_FEATURES + NOISE_FEATURES
    X = df[features].values
    y = df["readmitted_30d"].values

    model = GradientBoostingClassifier(random_state=seed, n_estimators=100, max_depth=3)

    probs = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, probs)

    model.fit(X, y)
    perm = permutation_importance(model, X, y, n_repeats=20, random_state=seed)
    importances = pd.Series(perm.importances_mean, index=features).sort_values(ascending=False)

    return {"auc": float(auc), "importances": importances, "features": features, "model": model}


def validate_feature_recovery(importances: pd.Series, top_k: int = 5) -> dict:
    top_features = set(importances.sort_values(ascending=False).index[:top_k])
    true_set = set(TRUE_RISK_FEATURES)
    recovered = top_features & true_set
    return {
        "top_k_features": list(top_features),
        "n_true_risk_features_in_top_k": len(recovered),
        "n_true_risk_features_total": len(true_set),
        "recovery_pct": round(100 * len(recovered) / len(true_set), 2),
    }
