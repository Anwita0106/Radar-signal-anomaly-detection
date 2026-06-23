"""
generate_training_data.py
============================
Training pipeline, step 1:

    Normal Aircraft -> Radar Data -> Feature Extraction -> training_features.csv

Simulates many aircraft passes through radar coverage, where EVERY pass is
built entirely out of `behaviors.NORMAL_BEHAVIORS` (straight_flight,
gentle_turn, climb, descent) -- no anomalous behavior is ever scheduled
here. This is what guarantees the Isolation Forest is trained on a
genuinely normal-only distribution, rather than V1's `behavior_dataset.csv`,
which was built from a live log that had anomalies randomly injected into
it 3% of the time with no way to exclude them.

Run:
    python generate_training_data.py

Output:
    data/training_features.csv  -- one row per behavioral window, columns
    = feature_extraction.FEATURE_COLUMNS (no label column -- this dataset
    is unsupervised by construction, since every row is already known to
    be normal).
"""

import numpy as np
import pandas as pd

import config
from behaviors import NORMAL_BEHAVIORS
from data_logger import DataLogger
from feature_extraction import FEATURE_COLUMNS, compute_window_features
from simulate_track import simulate_track

RNG = np.random.default_rng(config.RANDOM_SEED)

NUM_PASSES = 400
MIN_TICKS_PER_SEGMENT = 60
MAX_TICKS_PER_SEGMENT = 140


def random_normal_schedule() -> list[tuple[str, int]]:
    """Build a normal-only behavior schedule: 1-2 segments drawn from
    NORMAL_BEHAVIORS, each lasting a randomized number of ticks."""
    behavior_names = list(NORMAL_BEHAVIORS.keys())
    num_segments = RNG.integers(1, 3)  # 1 or 2 segments per pass
    schedule = []
    for _ in range(num_segments):
        name = RNG.choice(behavior_names)
        ticks = int(RNG.integers(MIN_TICKS_PER_SEGMENT, MAX_TICKS_PER_SEGMENT))
        schedule.append((name, ticks))
    return schedule


def rows_to_window_features(rows: list[dict]) -> list[list[float]]:
    feature_rows = []
    n = len(rows)
    for start in range(0, n - config.WINDOW_SIZE + 1, config.WINDOW_STEP):
        window = rows[start:start + config.WINDOW_SIZE]
        feature_rows.append(compute_window_features(window))
    return feature_rows


def main():
    print(f"Simulating {NUM_PASSES} normal-only aircraft passes...")
    all_feature_rows = []

    for i in range(NUM_PASSES):
        schedule = random_normal_schedule()
        rows, _flags = simulate_track(schedule, rng=RNG)

        if len(rows) < config.WINDOW_SIZE:
            continue

        all_feature_rows.extend(rows_to_window_features(rows))

        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{NUM_PASSES} passes simulated")

    df = pd.DataFrame(all_feature_rows, columns=FEATURE_COLUMNS)
    df.to_csv(config.TRAINING_FEATURES_CSV, index=False)

    print(f"\nWrote {len(df)} normal feature windows to {config.TRAINING_FEATURES_CSV}")


if __name__ == "__main__":
    main()
