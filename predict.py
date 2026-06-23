"""
predict.py
============
Offline prediction CLI. Reads a raw radar track CSV (timestamp + every
channel in `config.RADAR_CHANNELS`), slides a window across it, and for
each full window: computes features (via `feature_extraction.py`, the one
canonical implementation), scores it with the Isolation Forest, and runs
the result through `threat_engine.py` to get an explainable 0-100 score.

This replaces V1's `predict.py`, which used its own bolted-on
`explain_window()` heuristic instead of the project's real threat engine,
and `test_model.py`, which fed the model the wrong number of features and
would crash on import.

Run:
    python predict.py                          # uses data/test_tracks.csv
    python predict.py path/to/your_track.csv

Inputs:
    A CSV with columns: timestamp, range, bearing, power, frequency, relative_velocity
    (optionally a `true_label` column, ignored for scoring but used to
    print a quick accuracy summary if present)

Outputs:
    data/prediction_results.csv -- one row per window: window_start_index,
    window_end_index, label, threat_score, threat_level, contributing_factors
"""

import sys

import pandas as pd

import config
import isolation_forest_model as model_lib
import threat_engine
from feature_extraction import compute_window_features, feature_vector_to_dict


def predict_track(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    rows = df.to_dict("records")

    model = model_lib.load()
    baseline = threat_engine.load_baseline()

    results = []
    for start in range(0, len(rows) - config.WINDOW_SIZE + 1, config.WINDOW_STEP):
        window = rows[start:start + config.WINDOW_SIZE]
        feats = compute_window_features(window)

        label, raw_score = model_lib.score(model, feats)
        feature_dict = feature_vector_to_dict(feats)
        assessment = threat_engine.compute_threat(feature_dict, raw_score, baseline)

        results.append({
            "window_start": start,
            "window_end": start + config.WINDOW_SIZE - 1,
            "label": label,
            "raw_if_score": round(raw_score, 4),
            "threat_score": assessment.score,
            "threat_level": assessment.level,
            "contributing_factors": "; ".join(assessment.contributing_factors),
        })

    return pd.DataFrame(results)


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else config.TEST_TRACKS_CSV
    print(f"Predicting on {csv_path} ...")

    results_df = predict_track(csv_path)
    results_df.to_csv(config.PREDICTION_RESULTS_CSV, index=False)

    n_anomaly = int((results_df["label"] == "Anomaly").sum())
    print(f"\n{len(results_df)} windows scored. {n_anomaly} flagged as Anomaly.")
    print(f"Results written to {config.PREDICTION_RESULTS_CSV}\n")

    flagged = results_df[results_df["label"] == "Anomaly"]
    if len(flagged) > 0:
        print("Flagged windows:")
        for _, row in flagged.iterrows():
            print(
                f"  rows {row['window_start']}-{row['window_end']} | "
                f"threat={row['threat_score']:.0f} ({row['threat_level']}) | "
                f"{row['contributing_factors']}"
            )
    else:
        print("No windows flagged as anomalous.")


if __name__ == "__main__":
    main()
