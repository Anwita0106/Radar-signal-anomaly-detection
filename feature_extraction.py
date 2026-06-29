"""

The single, canonical feature-engineering implementation for the whole
project.


Inputs
------
- A "window": a list of `WINDOW_SIZE` raw measurement dicts (each with the
  keys in `config.RADAR_CHANNELS`), in time order.

Outputs
-------
- `compute_window_features(window)` -> a flat list of floats, in the exact
  order of `FEATURE_COLUMNS`: for each radar channel, the mean, standard
  deviation, variance, maximum sample-to-sample jump, and linear trend
  slope across the window.
"""

import numpy as np

import config

FEATURE_COLUMNS = [
    f"{channel}_{stat}"
    for channel in config.RADAR_CHANNELS
    for stat in config.WINDOW_STATS
]


def compute_window_features(window: list[dict]) -> list[float]:
    """
    Compute the flat behavioral feature vector for one window of raw
    radar measurements.

    `window` must be a list of dicts, each containing every key in
    `config.RADAR_CHANNELS`, in time order.
    """
    features: list[float] = []

    for channel in config.RADAR_CHANNELS:
        values = np.array([row[channel] for row in window], dtype=float)

        mean = float(np.mean(values))
        std = float(np.std(values))
        var = float(np.var(values))

        if len(values) > 1:
            max_jump = float(np.max(np.abs(np.diff(values))))
            slope = float(np.polyfit(np.arange(len(values)), values, 1)[0])
        else:
            max_jump = 0.0
            slope = 0.0

        features.extend([mean, std, var, max_jump, slope])

    return features


def feature_vector_to_dict(feature_vector: list[float]) -> dict:
    """Convenience: zip a flat feature vector back up with its column names."""
    return dict(zip(FEATURE_COLUMNS, feature_vector))
