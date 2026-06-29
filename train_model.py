"""

Training pipeline, step 2:

    training_features.csv -> Isolation Forest -> models/isolation_forest.pkl
                                               -> models/feature_baseline.json


Run:
    python train_model.py

Inputs:
    data/training_features.csv  (produced by generate_training_data.py)

Outputs:
    models/isolation_forest.pkl
    models/feature_baseline.json
"""

import json

import numpy as np
import pandas as pd

import config
import isolation_forest_model as model_lib
from feature_extraction import FEATURE_COLUMNS


def build_baseline(df: pd.DataFrame, model) -> dict:
    feature_stats = {
        col: {"mean": float(df[col].mean()), "std": float(df[col].std())}
        for col in FEATURE_COLUMNS
    }

    raw_scores = model.decision_function(df[FEATURE_COLUMNS])
    score_range = {
        "p1": float(np.percentile(raw_scores, 1)),
        "p99": float(np.percentile(raw_scores, 99)),
    }

    return {"feature_stats": feature_stats, "score_range": score_range}


def main():
    print(f"Loading {config.TRAINING_FEATURES_CSV} ...")
    df = pd.read_csv(config.TRAINING_FEATURES_CSV)
    print(f"  -> {len(df)} normal feature windows")

    if len(df) < 50:
        print("Not enough training data. Run generate_training_data.py first.")
        return

    print("Training Isolation Forest...")
    model = model_lib.train(df)
    model_lib.save(model)
    print(f"  -> Model saved to {config.MODEL_PATH}")

    print("Computing threat-engine baseline (feature stats + score range)...")
    baseline = build_baseline(df, model)
    with open(config.BASELINE_PATH, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"  -> Baseline saved to {config.BASELINE_PATH}")


if __name__ == "__main__":
    main()
