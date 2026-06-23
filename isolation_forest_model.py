"""
isolation_forest_model.py
============================
Thin wrapper around `sklearn.ensemble.IsolationForest` so that every other
file (train_model.py, evaluate_model.py, predict.py, dashboard.py) talks
to the model through one consistent, three-method interface instead of
calling `joblib.load` / `model.predict` / `model.decision_function`
directly in five different places.

Inputs
------
- `train()`: a pandas DataFrame of feature vectors (columns =
  `feature_extraction.FEATURE_COLUMNS`), normal data only.
- `score()`: a single flat feature vector (list[float], in
  `FEATURE_COLUMNS` order).

Outputs
-------
- `train()` returns the fitted model.
- `score()` returns `(label, raw_score)` where `label` is
  `"Normal"` or `"Anomaly"` and `raw_score` is sklearn's
  `decision_function` value (lower = more anomalous).
"""

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

import config
from feature_extraction import FEATURE_COLUMNS


def train(training_features_df: pd.DataFrame) -> IsolationForest:
    model = IsolationForest(
        contamination=config.ISOLATION_FOREST_CONTAMINATION,
        random_state=config.RANDOM_SEED,
    )
    model.fit(training_features_df[FEATURE_COLUMNS])
    return model


def save(model: IsolationForest, path=config.MODEL_PATH) -> None:
    joblib.dump(model, path)


def load(path=config.MODEL_PATH) -> IsolationForest:
    return joblib.load(path)


def score(model: IsolationForest, feature_vector: list[float]):
    """Score a single feature vector. Returns (label, raw_score)."""
    feats_df = pd.DataFrame([feature_vector], columns=FEATURE_COLUMNS)
    prediction = model.predict(feats_df)[0]
    raw_score = float(model.decision_function(feats_df)[0])
    label = "Anomaly" if prediction == -1 else "Normal"
    return label, raw_score
