"""
threat_engine.py
==================
Turns three independent signals into ONE explainable threat assessment:

  1. The Isolation Forest's raw anomaly score (the ML component).
  2. How abnormal the *signal* channels (power, frequency) look, relative
     to the training baseline (the signal component).
  3. How abnormal the *flight behavior* channels (range, bearing,
     relative_velocity) look, relative to the training baseline (the
     behavior component).

These are combined into a single 0-100 `threat_score` and a four-level
`threat_level` (LOW / MEDIUM / HIGH / CRITICAL), plus a short list of
human-readable `contributing_factors` -- e.g. "frequency_max_jump is
4.1 sigma from baseline" -- so the dashboard (and a presenter) can always
explain *why* a given score was assigned, not just what the number is.


Inputs
------
- A feature vector (dict, via `feature_extraction.feature_vector_to_dict`).
- The Isolation Forest's raw `decision_function` score for that vector.
- A baseline (mean/std per feature, plus the training score range),
  produced once by `train_model.py` and loaded via `load_baseline()`.

Outputs
-------
- A `ThreatAssessment` dataclass: `score` (0-100), `level` (str),
  `contributing_factors` (list[str]).
"""

import json
from dataclasses import dataclass, field

import numpy as np

import config

SIGNAL_CHANNELS = ["power", "frequency"]
BEHAVIOR_CHANNELS = ["range", "bearing", "relative_velocity"]


@dataclass
class ThreatAssessment:
    score: float
    level: str
    contributing_factors: list[str] = field(default_factory=list)
    ml_component: float = 0.0
    signal_component: float = 0.0
    behavior_component: float = 0.0


def load_baseline(path=config.BASELINE_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def _zscore(value: float, mean: float, std: float) -> float:
    if std < 1e-9:
        std = 1e-9
    return (value - mean) / std


def _channel_abnormality(feature_dict: dict, baseline: dict, channels: list[str]):
    """Average |z-score| across every stat of the given channels, scaled
    to 0-100. Also returns the list of individual flagged factors."""
    z_values = []
    factors = []
    stats = baseline["feature_stats"]

    for col, z_stats in stats.items():
        channel = next((ch for ch in config.RADAR_CHANNELS if col.startswith(ch + "_")), None)
        if channel is None or channel not in channels:
            continue
        value = feature_dict.get(col)
        if value is None:
            continue
        z = _zscore(value, z_stats["mean"], z_stats["std"])
        z_values.append(abs(z))
        if abs(z) >= config.Z_SCORE_FLAG_THRESHOLD:
            factors.append(f"{col} is {z:.1f} sigma from the normal baseline")

    if not z_values:
        return 0.0, factors

    avg_abs_z = float(np.mean(z_values))
    component = min(100.0, (avg_abs_z / config.Z_SCORE_SATURATION) * 100.0)
    return component, factors


def _ml_component(raw_score: float, baseline: dict) -> float:
    """Map the IF decision_function score onto 0-100 using the training
    score distribution: scores at/below the 1st percentile (most
    anomalous within the training spread) saturate at 100; scores at/above
    the 99th percentile (clearly normal) map to 0."""
    p1 = baseline["score_range"]["p1"]
    p99 = baseline["score_range"]["p99"]
    if p99 <= p1:
        return 0.0
    pct = (p99 - raw_score) / (p99 - p1)
    return float(min(100.0, max(0.0, pct * 100.0)))


def level_for_score(score: float) -> str:
    thresholds = config.THREAT_LEVEL_THRESHOLDS
    if score >= thresholds["CRITICAL"]:
        return "CRITICAL"
    if score >= thresholds["HIGH"]:
        return "HIGH"
    if score >= thresholds["MEDIUM"]:
        return "MEDIUM"
    return "LOW"


def compute_threat(feature_dict: dict, raw_if_score: float, baseline: dict) -> ThreatAssessment:
    ml_component = _ml_component(raw_if_score, baseline)
    signal_component, signal_factors = _channel_abnormality(feature_dict, baseline, SIGNAL_CHANNELS)
    behavior_component, behavior_factors = _channel_abnormality(feature_dict, baseline, BEHAVIOR_CHANNELS)

    score = (
        config.THREAT_WEIGHT_ML * ml_component
        + config.THREAT_WEIGHT_SIGNAL * signal_component
        + config.THREAT_WEIGHT_BEHAVIOR * behavior_component
    )
    score = round(min(100.0, max(0.0, score)), 1)

    factors = signal_factors + behavior_factors
    if ml_component >= 60:
        factors.insert(0, f"Isolation Forest flags this window as anomalous (component {ml_component:.0f}/100)")

    return ThreatAssessment(
        score=score,
        level=level_for_score(score),
        contributing_factors=factors,
        ml_component=ml_component,
        signal_component=signal_component,
        behavior_component=behavior_component,
    )
