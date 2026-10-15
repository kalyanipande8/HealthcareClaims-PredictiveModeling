import json
import os

from src.simulate_claims import simulate
from src.predictive_model import train_and_evaluate, validate_feature_recovery


def main():
    df = simulate()
    result = train_and_evaluate(df)
    recovery = validate_feature_recovery(result["importances"])

    summary = {
        "n_patients": len(df),
        "readmission_rate_pct": round(100 * df["readmitted_30d"].mean(), 2),
        "cv_auc": round(result["auc"], 4),
        "feature_recovery": recovery,
        "top_importances": result["importances"].round(5).to_dict(),
    }

    os.makedirs("results", exist_ok=True)
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
