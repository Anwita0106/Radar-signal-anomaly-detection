"""
config.py
=========
Single source of truth for every constant used across the V2 system.

Why this file exists
---------------------
In V1, the same numbers (radar coverage rectangle, window size, noise
levels) were hardcoded independently in `dashboard.py` and `radar_logger.py`,
and were expected to "match by convention." If one was edited and the other
wasn't, the two would silently drift apart.

In V2, every other module imports its constants from here. There is exactly
one coverage rectangle, one window size, one set of noise parameters, one
set of threat thresholds. Change a number once, and it's correct everywhere.
"""

from pathlib import Path

# ----------------------------------------------------------------------
# Project paths
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

TRAINING_FEATURES_CSV = DATA_DIR / "training_features.csv"
TEST_FEATURES_CSV = DATA_DIR / "test_features.csv"
TEST_TRACKS_CSV = DATA_DIR / "test_tracks.csv"
LIVE_LOG_CSV = DATA_DIR / "live_log.csv"
PREDICTION_RESULTS_CSV = DATA_DIR / "prediction_results.csv"

MODEL_PATH = MODELS_DIR / "isolation_forest.pkl"
BASELINE_PATH = MODELS_DIR / "feature_baseline.json"

# ----------------------------------------------------------------------
# Radar coverage area (screen / simulation coordinate space, in "units")
# Used by the dashboard to draw the radar field AND by the simulators to
# know when an aircraft has entered / left coverage. Defined ONCE.
# ----------------------------------------------------------------------
COVERAGE_LEFT = 0
COVERAGE_TOP = 0
COVERAGE_WIDTH = 1000
COVERAGE_HEIGHT = 800
COVERAGE_RIGHT = COVERAGE_LEFT + COVERAGE_WIDTH
COVERAGE_BOTTOM = COVERAGE_TOP + COVERAGE_HEIGHT

# ----------------------------------------------------------------------
# Radar station network (ground-based, altitude = 0)
# ----------------------------------------------------------------------
RADAR_STATIONS = {
    "R1": (75, 190),
    "R2": (650, 35),
    "R3": (935, 545),
    "R4": (300, 760),
}
REFERENCE_RADAR = "R1"  # radar used for the single-track CSV / live dashboard log

# ----------------------------------------------------------------------
# Radar physics constants (abstracted, not real-world SI units -- this is
# a simulation, and the formulas are explicitly modeled on real radar
# principles: inverse-fourth-power signal falloff and Doppler frequency
# shift, rather than the arbitrary 1/distance formulas used in V1.)
# ----------------------------------------------------------------------
REFERENCE_RANGE = 200.0          # range (units) at which REFERENCE_POWER_DB is observed
REFERENCE_POWER_DB = 40.0        # signal power (dB) at REFERENCE_RANGE
CARRIER_FREQUENCY_MHZ = 1000.0   # nominal radar carrier frequency
DOPPLER_COEFF = 0.08             # MHz of frequency shift per unit/tick of relative velocity

# ----------------------------------------------------------------------
# Measurement noise (Gaussian standard deviation, per channel)
# ----------------------------------------------------------------------
NOISE_STD = {
    "range": 1.5,
    "bearing": 0.7,        # degrees
    "power": 0.6,          # dB
    "frequency": 0.3,      # MHz
    "relative_velocity": 0.4,
}

# ----------------------------------------------------------------------
# Aircraft kinematics defaults
# ----------------------------------------------------------------------
BASE_SPEED = 6.0          # units / tick, horizontal ground speed
BASE_ALTITUDE = 500.0     # starting altitude (units)
CLIMB_RATE = 1.2          # altitude units / tick during CLIMB
DESCENT_RATE = 1.2        # altitude units / tick during DESCENT
GENTLE_TURN_RATE = 0.6    # degrees / tick

# ----------------------------------------------------------------------
# Anomalous behavior parameters
# ----------------------------------------------------------------------
SUDDEN_HEADING_CHANGE_DEG = (70, 150)      # min/max instantaneous heading jump
ZIG_ZAG_PERIOD_TICKS = 6                   # ticks between heading flips
ZIG_ZAG_AMPLITUDE_DEG = 45
EXTREME_ACCEL_MULTIPLIER = (3.0, 5.0)      # speed multiplier range
LOITER_TURN_RATE_DEG = 18.0                # degrees/tick -> tight circling
LOITER_SPEED_FACTOR = 0.4                  # loitering aircraft slows down
ALTITUDE_SPIKE_RATE_MULTIPLIER = (6.0, 10.0)  # multiple of CLIMB_RATE

ANOMALY_SEGMENT_LENGTH_TICKS = (15, 25)    # how long an injected anomaly lasts

# ----------------------------------------------------------------------
# Feature extraction / windowing
# ----------------------------------------------------------------------
WINDOW_SIZE = 20      # samples per behavioral feature window
WINDOW_STEP = 2        # stride used when generating training/test windows

RADAR_CHANNELS = ["range", "bearing", "power", "frequency", "relative_velocity"]
WINDOW_STATS = ["mean", "std", "var", "max_jump", "trend_slope"]

# A window is labeled anomalous (for the TEST set only -- training data is
# pure-normal by construction and therefore never labeled) if at least this
# fraction of its raw samples fall inside an injected anomaly segment.
WINDOW_ANOMALY_LABEL_RATIO = 0.15

# ----------------------------------------------------------------------
# Isolation Forest
# ----------------------------------------------------------------------
ISOLATION_FOREST_CONTAMINATION = 0.05
RANDOM_SEED = 42

# ----------------------------------------------------------------------
# Threat engine
# ----------------------------------------------------------------------
# Final threat score (0-100) = weighted blend of three components:
THREAT_WEIGHT_ML = 0.5
THREAT_WEIGHT_SIGNAL = 0.25
THREAT_WEIGHT_BEHAVIOR = 0.25

# A z-score (vs. the training baseline) at/above this magnitude is treated
# as "contributing" for the purposes of building the human-readable
# explanation list.
Z_SCORE_FLAG_THRESHOLD = 2.0
# z-score magnitude that maps to a 100/100 component score (clipped above this)
Z_SCORE_SATURATION = 6.0

THREAT_LEVEL_THRESHOLDS = {
    "LOW": 0,
    "MEDIUM": 35,
    "HIGH": 60,
    "CRITICAL": 82,
}

# ----------------------------------------------------------------------
# Simulation tick rate (used by the live dashboard's QTimer, milliseconds)
# ----------------------------------------------------------------------
TICK_INTERVAL_MS = 30
