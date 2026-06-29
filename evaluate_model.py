"""


    test_features.csv (labeled) -> Isolation Forest -> Accuracy / Precision /
                                                          Recall / F1 / Confusion Matrix

Run:
    python evaluate_model.py

Inputs:
    data/test_features.csv     (produced by generate_test_data.py)
    models/isolation_forest.pkl

Output:
    Printed Accuracy / Precision / Recall / F1 / Confusion Matrix.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import config
import isolation_forest_model as model_lib
from feature_extraction import FEATURE_COLUMNS


def main():
    print(f"Loading {config.TEST_FEATURES_CSV} ...")
    df = pd.read_csv(config.TEST_FEATURES_CSV)
    print(f"  -> {len(df)} labeled windows "
          f"({int((df['label'] == 0).sum())} normal, {int((df['label'] == 1).sum())} anomaly)")

    print(f"Loading model from {config.MODEL_PATH} ...")
    model = model_lib.load()

    raw_predictions = model.predict(df[FEATURE_COLUMNS])
    # sklearn's IsolationForest: -1 = anomaly, 1 = normal. Map to our 0/1 label convention.
    y_pred = (raw_predictions == -1).astype(int)
    y_true = df["label"].astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print("\n--- Evaluation Results ---")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print("\nConfusion Matrix (rows = true, cols = predicted; order = [Normal, Anomaly]):")
    print(cm)


if __name__ == "__main__":
    main()
