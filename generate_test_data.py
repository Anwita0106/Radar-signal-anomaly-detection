"""

Testing pipeline, step 1:

    Aircraft -> Inject Anomaly -> Radar Data -> Feature Extraction -> test_features.csv


Run:
    python generate_test_data.py

Output:
    data/test_features.csv  -- one row per window: FEATURE_COLUMNS + label
                               (0 = Normal, 1 = Anomaly)
    data/test_tracks.csv    -- ONE representative raw track (timestamp +
                               every radar channel) with a single injected
                               anomaly segment in the middle, for use as a
                               demo input to predict.py
"""

import numpy as np
import pandas as pd

import config
from behaviors import NORMAL_BEHAVIORS, ANOMALOUS_BEHAVIORS
from feature_extraction import FEATURE_COLUMNS, compute_window_features
from simulate_track import simulate_track

RNG = np.random.default_rng(config.RANDOM_SEED + 1)

NUM_PASSES = 200
NORMAL_LEAD_TICKS = (40, 80)
NORMAL_TAIL_TICKS = (40, 80)
ANOMALY_FRACTION = 0.5


def random_test_schedule():
    """Build a schedule: normal lead-in, optionally an anomalous segment,
    normal tail-out. Returns (schedule, contains_anomaly)."""
    lead = int(RNG.integers(*NORMAL_LEAD_TICKS))
    tail = int(RNG.integers(*NORMAL_TAIL_TICKS))
    lead_behavior = RNG.choice(list(NORMAL_BEHAVIORS.keys()))
    tail_behavior = RNG.choice(list(NORMAL_BEHAVIORS.keys()))

    schedule = [(lead_behavior, lead)]
    contains_anomaly = RNG.random() < ANOMALY_FRACTION

    if contains_anomaly:
        anomaly_name = RNG.choice(list(ANOMALOUS_BEHAVIORS.keys()))
        anomaly_len = int(RNG.integers(*config.ANOMALY_SEGMENT_LENGTH_TICKS))
        schedule.append((anomaly_name, anomaly_len))

    schedule.append((tail_behavior, tail))
    return schedule, contains_anomaly


def rows_flags_to_window_rows(rows, flags):
    feature_rows = []
    n = len(rows)
    for start in range(0, n - config.WINDOW_SIZE + 1, config.WINDOW_STEP):
        window = rows[start:start + config.WINDOW_SIZE]
        window_flags = flags[start:start + config.WINDOW_SIZE]
        feats = compute_window_features(window)
        anomaly_ratio = sum(window_flags) / config.WINDOW_SIZE
        label = 1 if anomaly_ratio >= config.WINDOW_ANOMALY_LABEL_RATIO else 0
        feature_rows.append(feats + [label])
    return feature_rows


def generate_labeled_features():
    print(f"Simulating {NUM_PASSES} test passes (anomaly fraction = {ANOMALY_FRACTION})...")
    all_feature_rows = []
    n_anomalous_passes = 0

    for i in range(NUM_PASSES):
        schedule, contains_anomaly = random_test_schedule()
        rows, flags = simulate_track(schedule, rng=RNG)

        if len(rows) < config.WINDOW_SIZE:
            continue

        if contains_anomaly:
            n_anomalous_passes += 1

        all_feature_rows.extend(rows_flags_to_window_rows(rows, flags))

        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{NUM_PASSES} passes simulated")

    columns = FEATURE_COLUMNS + ["label"]
    df = pd.DataFrame(all_feature_rows, columns=columns)
    df.to_csv(config.TEST_FEATURES_CSV, index=False)

    n_anom_windows = int(df["label"].sum())
    print(f"\nWrote {len(df)} windows to {config.TEST_FEATURES_CSV}")
    print(f"  Passes with an injected anomaly: {n_anomalous_passes}/{NUM_PASSES}")
    print(f"  Normal windows: {len(df) - n_anom_windows}, Anomaly windows: {n_anom_windows}")


def generate_demo_track():
    print("\nGenerating data/test_tracks.csv (single demo track with one injected anomaly)...")
    schedule = [
        ("straight_flight", 60),
        ("zig_zag", 20),
        ("gentle_turn", 60),
    ]
    rows, flags = simulate_track(schedule, rng=RNG)

    df = pd.DataFrame(rows)
    df["true_label"] = [1 if f else 0 for f in flags]
    df.to_csv(config.TEST_TRACKS_CSV, index=False)
    print(f"  -> {len(df)} rows written to {config.TEST_TRACKS_CSV} "
          f"({sum(flags)} ground-truth anomalous samples, behavior='zig_zag')")


if __name__ == "__main__":
    generate_labeled_features()
    generate_demo_track()
    print("\nTest data generation complete.")
