"""
dashboard.py
==============
The live PyQt6 dashboard -- the final stage of the pipeline:

    Aircraft Simulator -> Radar Simulator -> Data Logger -> Feature Extraction
        -> Isolation Forest -> Threat Engine -> Dashboard (this file)

Every tick (config.TICK_INTERVAL_MS): the aircraft advances one step, the
reference radar takes a measurement, the measurement is buffered by
`DataLogger`, and once a full window is available, `feature_extraction`,
`isolation_forest_model`, and `threat_engine` are run in sequence (the
exact same functions used offline by `predict.py`) to produce a threat
score. The UI panels below are pure rendering of that result -- there is
no parallel/duplicate scoring logic living inside this file.

Why this file exists (vs. V1's `dashboard.py`)
------------------------------------------------
V1's dashboard hardcoded "RADAR STATUS: ONLINE" regardless of anything,
never set `warning_message` in the anomaly branch (so the alert banner's
text was always blank), and read decisions from `radar_logger.py`'s
internal `(label, score)` tuple with no explanation of *why*. This V2
dashboard surfaces the actual threat score, level, and the contributing
factors list straight from `threat_engine.py`, keeps a real event log and
alert history, and turns the aircraft red using the same `threat_level`
the panels display -- one source of truth, not several.

Run:
    python dashboard.py     (or python main.py)

Requires:
    models/isolation_forest.pkl and models/feature_baseline.json to
    already exist (run generate_training_data.py then train_model.py
    first).
"""

import sys
from collections import deque
from datetime import datetime
import time

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import QPointF

import config
import isolation_forest_model as model_lib
import threat_engine
from aircraft_simulator import Aircraft
from behaviors import NORMAL_BEHAVIORS, ANOMALOUS_BEHAVIORS
from data_logger import DataLogger
from feature_extraction import compute_window_features, feature_vector_to_dict
from radar_simulator import build_radar_network
from voice_alert import speak_alert

# ----------------------------------------------------------------------
# Color palette -- dark "ops room" theme
# ----------------------------------------------------------------------
BG_COLOR = QColor(8, 14, 12)
PANEL_BG = QColor(14, 22, 19)
GRID_COLOR = QColor(22, 60, 40)
TEXT_DIM = QColor(110, 150, 130)
TEXT_BRIGHT = QColor(170, 230, 190)
ACCENT_GREEN = QColor(60, 220, 130)
ACCENT_AMBER = QColor(235, 175, 60)
ACCENT_RED = QColor(235, 70, 70)
ACCENT_CYAN = QColor(80, 200, 220)

LEVEL_COLORS = {
    "LOW": ACCENT_GREEN,
    "MEDIUM": ACCENT_AMBER,
    "HIGH": QColor(240, 130, 40),
    "CRITICAL": ACCENT_RED,
}

PASS_NORMAL_TICKS = (60, 110)
PASS_ANOMALY_FRACTION = 2


def random_pass_schedule(rng: np.random.Generator):
    lead = int(rng.integers(*PASS_NORMAL_TICKS))
    tail = int(rng.integers(*PASS_NORMAL_TICKS))
    lead_behavior = rng.choice(list(NORMAL_BEHAVIORS.keys()))
    tail_behavior = rng.choice(list(NORMAL_BEHAVIORS.keys()))

    schedule = [(str(lead_behavior), lead)]
    if rng.random() < PASS_ANOMALY_FRACTION:
        anomaly_name = str(rng.choice(list(ANOMALOUS_BEHAVIORS.keys())))
        anomaly_len = int(rng.integers(*config.ANOMALY_SEGMENT_LENGTH_TICKS))
        schedule.append((anomaly_name, anomaly_len))
    schedule.append((str(tail_behavior), tail))
    return schedule


class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RADAR SIGNAL ANOMALY DETECTION SYSTEM ")
        self.resize(1300, 950)

        self.rng = np.random.default_rng()
        self.aircraft = Aircraft(rng=self.rng)
        self.radar_network = build_radar_network()
        self.reference_radar = self.radar_network[config.REFERENCE_RADAR]

        self.logger = DataLogger(csv_path=config.LIVE_LOG_CSV)

        try:
            self.model = model_lib.load()
            self.baseline = threat_engine.load_baseline()
            self.model_ready = True
        except FileNotFoundError:
            self.model = None
            self.baseline = None
            self.model_ready = False

        self._schedule = []
        self._schedule_index = 0
        self._segment_ticks_remaining = 0
        self._start_new_pass()

        self.last_measurement = None
        self.last_assessment = threat_engine.ThreatAssessment(score=0.0, level="LOW")
        self.threat_history = deque(maxlen=120)
        self.event_log = deque(maxlen=12)
        self.alert_history = deque(maxlen=8)
        self._last_level = "LOW"
        self.last_voice_time = 0
        

        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(config.TICK_INTERVAL_MS)

        self.layout = self._compute_layout()

    # ------------------------------------------------------------------
    def _start_new_pass(self):
        self.aircraft.reset()
        self.reference_radar.reset()
        self.logger.reset_window()
        self._schedule = random_pass_schedule(self.rng)
        self._schedule_index = 0
        self._apply_next_segment()

    def _apply_next_segment(self):
        name, ticks = self._schedule[self._schedule_index]
        self.aircraft.set_behavior(name)
        self._segment_ticks_remaining = ticks

    def _log_event(self, text: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.event_log.appendleft(f"[{timestamp}] {text}")

    # ------------------------------------------------------------------
    def tick(self):
        self.aircraft.step()
        self._segment_ticks_remaining -= 1

        if self._segment_ticks_remaining <= 0:
            self._schedule_index += 1
            if self._schedule_index < len(self._schedule):
                self._apply_next_segment()
            elif self.aircraft.behavior_name != "straight_flight":
                # Schedule exhausted: fly straight until the aircraft exits
                # coverage, instead of repeating the last behavior forever
                # (which could otherwise let a turning/looping behavior
                # accumulate heading change far beyond anything seen in
                # training and trigger repeated false alerts).
                self.aircraft.set_behavior("straight_flight")

        if self.aircraft.has_exited():
            self._start_new_pass()
            self.update()
            return

        if self.aircraft.is_inside_coverage():
            measurement = self.reference_radar.measure(
                self.aircraft.state.x, self.aircraft.state.y, self.aircraft.state.z, self.rng
            )
            self.last_measurement = measurement
            row = {"timestamp": round(datetime.now().timestamp(), 3), **measurement}
            self.logger.add(row)

            if self.model_ready and self.logger.window_ready():
                feats = compute_window_features(self.logger.window)
                label, raw_score = model_lib.score(self.model, feats)
                feature_dict = feature_vector_to_dict(feats)
                assessment = threat_engine.compute_threat(feature_dict, raw_score, self.baseline)
                self.last_assessment = assessment
                self.threat_history.append(assessment.score)
                if assessment.level in ("HIGH", "CRITICAL"):

                    current_time = time.time()
                    print(
                        "LEVEL=", assessment.level,
                        "LAST_VOICE=", current_time - self.last_voice_time
                    )

                    if current_time - self.last_voice_time > 5:

                        print("VOICE ALERT TRIGGERED")

                        speak_alert(
                            f"Warning. {assessment.level} threat detected."
                        )

                        self.last_voice_time = current_time

                # Only log level changes once
                if assessment.level != self._last_level:

                    self._log_event(
                        f"Threat level -> {assessment.level} (score {assessment.score:.0f}), "
                        f"behavior={self.aircraft.behavior_name}"
                    )

                    if assessment.level in ("HIGH", "CRITICAL"):
                        self.alert_history.appendleft({
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "level": assessment.level,
                            "score": assessment.score,
                            "behavior": self.aircraft.behavior_name,
                        })

                    self._last_level = assessment.level

        self.update()

    # ------------------------------------------------------------------
    # Geometry helpers (coverage-space -> screen-space)
    #
    # The layout is computed fresh every paintEvent from the window's
    # CURRENT size, rather than fixed pixel constants -- so the dashboard
    # fills whatever screen it's maximized on instead of sitting in a
    # fixed 1300x950 box in the corner. Every _draw_* method below reads
    # positions from `self.layout` (set at the top of paintEvent) instead
    # of hardcoded numbers.
    # ------------------------------------------------------------------
    MARGIN = 30
    TOP = 90
    RIGHT_PANEL_W = 480
    GAP = 18
    EVENT_LOG_H = 160

    def _compute_layout(self) -> dict:
        win_w = max(self.width(), 900)
        win_h = max(self.height(), 700)

        panel_x = win_w - self.RIGHT_PANEL_W - self.MARGIN
        field_left = self.MARGIN
        field_top = self.TOP
        field_w = max(300, panel_x - self.MARGIN - self.GAP)
        field_h = max(300, win_h - self.TOP - self.EVENT_LOG_H - self.GAP - self.MARGIN)

        event_log_y = field_top + field_h + self.GAP
        event_log_h = max(0, win_h - event_log_y - self.MARGIN)

        status_y = self.TOP
        status_h = 95
        channel_y = status_y + status_h + 10
        channel_h = 130
        threat_y = channel_y + channel_h + 10
        threat_h = 140
        alert_y = threat_y + threat_h + 10
        alert_h = 130
        history_y = alert_y + alert_h + 10
        history_h = max(60, win_h - history_y - self.MARGIN)

        return {
            "field_left": field_left, "field_top": field_top,
            "field_w": field_w, "field_h": field_h,
            "panel_x": panel_x, "panel_w": self.RIGHT_PANEL_W,
            "status_y": status_y, "status_h": status_h,
            "channel_y": channel_y, "channel_h": channel_h,
            "threat_y": threat_y, "threat_h": threat_h,
            "alert_y": alert_y, "alert_h": alert_h,
            "history_y": history_y, "history_h": history_h,
            "event_log_x": field_left, "event_log_y": event_log_y,
            "event_log_w": field_w, "event_log_h": event_log_h,
        }

    def to_screen(self, x, y):
        L = self.layout
        sx = L["field_left"] + (x / config.COVERAGE_WIDTH) * L["field_w"]
        sy = L["field_top"] + (y / config.COVERAGE_HEIGHT) * L["field_h"]
        return sx, sy

    # ------------------------------------------------------------------
    def paintEvent(self, event):
        self.layout = self._compute_layout()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), BG_COLOR)

        self._draw_title(painter)
        self._draw_radar_field(painter)
        self._draw_aircraft(painter)
        self._draw_status_panel(painter)
        self._draw_channel_panel(painter)
        self._draw_threat_panel(painter)
        self._draw_alert_panel(painter)
        self._draw_event_log(painter)
        self._draw_alert_history(painter)

        painter.end()

    # ------------------------------------------------------------------
    def _draw_title(self, p: QPainter):
        p.setPen(QPen(ACCENT_GREEN))
        p.setFont(QFont("Consolas", 16, QFont.Weight.Bold))
        p.drawText(30, 35, "RADAR SIGNAL ANOMALY DETECTION SYSTEM")
        p.setFont(QFont("Consolas", 9))
        p.setPen(QPen(TEXT_DIM))
        p.drawText(30, 55, " 4-Station Doppler Radar Network")
        if not self.model_ready:
            p.setPen(QPen(ACCENT_RED))
            p.drawText(
                30, 72,
                "NO MODEL LOADED — run generate_training_data.py then train_model.py"
            )

    def _draw_radar_field(self, p: QPainter):
        L = self.layout
        rect_x, rect_y = L["field_left"], L["field_top"]
        rect_w, rect_h = L["field_w"], L["field_h"]

        p.setPen(QPen(GRID_COLOR, 1))
        p.setBrush(QBrush(PANEL_BG))
        p.drawRect(rect_x, rect_y, rect_w, rect_h)

        # range rings
        cx, cy = rect_x + rect_w / 2, rect_y + rect_h / 2
        for frac in (0.25, 0.5, 0.75, 1.0):
            r = (min(rect_w, rect_h) / 2) * frac
            p.setPen(QPen(GRID_COLOR, 1, Qt.PenStyle.DashLine))
            p.drawEllipse(QPointF(cx, cy), r, r)

        # radar stations
        p.setFont(QFont("Consolas", 8))
        for name, station in self.radar_network.items():
            sx, sy = self.to_screen(station.x, station.y)
            color = ACCENT_CYAN if name == config.REFERENCE_RADAR else TEXT_DIM
            p.setPen(QPen(color))
            p.setBrush(QBrush(color))
            p.drawEllipse(QPointF(sx, sy), 5, 5)
            p.drawText(int(sx + 8), int(sy - 6), name)

        # trail
        if len(self.aircraft.trail) > 1:
            pen = QPen(ACCENT_GREEN if not self.aircraft.is_anomalous else ACCENT_RED, 1)
            p.setPen(pen)
            pts = [QPointF(*self.to_screen(x, y)) for x, y in self.aircraft.trail]
            for i in range(len(pts) - 1):
                p.drawLine(pts[i], pts[i + 1])

    def _draw_aircraft(self, p: QPainter):
        level = self.last_assessment.level
        is_alert = level in ("HIGH", "CRITICAL")
        color = LEVEL_COLORS.get(level, ACCENT_GREEN) if is_alert else ACCENT_GREEN

        sx, sy = self.to_screen(self.aircraft.state.x, self.aircraft.state.y)
        L = self.layout
        if not (L["field_left"] <= sx <= L["field_left"] + L["field_w"]):
            return

        import math
        heading_rad = math.radians(self.aircraft.state.heading)
        size = 11
        tip = QPointF(sx + size * math.cos(heading_rad), sy + size * math.sin(heading_rad))
        left = QPointF(
            sx + size * math.cos(heading_rad + 2.6), sy + size * math.sin(heading_rad + 2.6)
        )
        right = QPointF(
            sx + size * math.cos(heading_rad - 2.6), sy + size * math.sin(heading_rad - 2.6)
        )

        p.setBrush(QBrush(color))
        p.setPen(QPen(QColor(255, 255, 255), 1.2))
        p.drawPolygon(QPolygonF([tip, left, right]))

        if is_alert:
            p.setPen(QPen(color, 1, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(sx, sy), 16, 16)

    def _panel_box(self, p: QPainter, x, y, w, h, title):
        p.setPen(QPen(GRID_COLOR))
        p.setBrush(QBrush(PANEL_BG))
        p.drawRect(x, y, w, h)
        p.setPen(QPen(ACCENT_CYAN))
        p.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        p.drawText(x + 8, y + 16, title)

    def _draw_status_panel(self, p: QPainter):
        L = self.layout
        x, y, w, h = L["panel_x"], L["status_y"], L["panel_w"], L["status_h"]
        self._panel_box(p, x, y, w, h, "TRACK STATUS")
        p.setFont(QFont("Consolas", 9))

        in_cov = self.aircraft.is_inside_coverage()
        p.setPen(QPen(ACCENT_GREEN if in_cov else TEXT_DIM))
        status_text = "TRACKING" if in_cov else "ACQUIRING"
        p.drawText(x + 12, y + 34, f"Target: {status_text}")

        p.setPen(QPen(TEXT_DIM))
        p.drawText(
            x + 150, y + 34,
            f"Alt {self.aircraft.state.z:.0f}   "
            f"Hdg {self.aircraft.state.heading % 360:.0f} deg   "
            f"Spd {self.aircraft.state.speed:.1f} u/t"
        )

        p.setPen(QPen(TEXT_BRIGHT))
        p.drawText(x + 12, y + 56, f"Behavior (ground truth): {self.aircraft.behavior_name}")

        gt_color = ACCENT_RED if self.aircraft.is_anomalous else ACCENT_GREEN
        p.setPen(QPen(gt_color))
        gt_label = "ANOMALOUS PATTERN" if self.aircraft.is_anomalous else "NORMAL PATTERN"
        p.drawText(x + 12, y + 76, f"Ground truth class: {gt_label}")

    def _draw_channel_panel(self, p: QPainter):
        L = self.layout
        x, y, w, h = L["panel_x"], L["channel_y"], L["panel_w"], L["channel_h"]
        self._panel_box(p, x, y, w, h, f"RADAR CHANNELS  ({config.REFERENCE_RADAR})")
        p.setFont(QFont("Consolas", 9))
        p.setPen(QPen(TEXT_BRIGHT))

        if self.last_measurement is None:
            p.drawText(x + 12, y + 40, "No target in coverage")
            return

        labels = [
            ("Range", "range", "u"),
            ("Bearing", "bearing", "deg"),
            ("Power", "power", "dB"),
            ("Frequency", "frequency", "MHz"),
            ("Rel. Velocity", "relative_velocity", "u/tick"),
        ]
        for i, (label, key, unit) in enumerate(labels):
            val = self.last_measurement[key]
            p.drawText(x + 12, y + 38 + i * 18, f"{label:14s} {val:9.2f} {unit}")

    def _draw_threat_panel(self, p: QPainter):
        L = self.layout
        x, y, w, h = L["panel_x"], L["threat_y"], L["panel_w"], L["threat_h"]
        self._panel_box(p, x, y, w, h, "THREAT ENGINE")

        a = self.last_assessment
        color = LEVEL_COLORS.get(a.level, ACCENT_GREEN)

        p.setFont(QFont("Consolas", 22, QFont.Weight.Bold))
        p.setPen(QPen(color))
        p.drawText(x + 12, y + 50, f"{a.score:.0f}")
        p.setFont(QFont("Consolas", 10))
        p.drawText(x + 70, y + 50, "/ 100")

        p.setFont(QFont("Consolas", 13, QFont.Weight.Bold))
        p.drawText(x + 12, y + 75, a.level)

        # component bars
        components = [
            ("ML", a.ml_component),
            ("Signal", a.signal_component),
            ("Behavior", a.behavior_component),
        ]
        bar_x = x + 180
        for i, (label, value) in enumerate(components):
            by = y + 30 + i * 20
            p.setFont(QFont("Consolas", 8))
            p.setPen(QPen(TEXT_DIM))
            p.drawText(bar_x, by + 7, label)
            bar_w = 200
            p.setPen(QPen(GRID_COLOR))
            p.drawRect(bar_x + 65, by, bar_w, 10)
            p.setBrush(QBrush(color))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRect(bar_x + 65, by, int(bar_w * value / 100), 10)

        # sparkline of threat score history
        spark_y = y + 100
        spark_h = 30
        spark_x = x + 12
        spark_w = w - 24
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(GRID_COLOR))
        p.drawRect(spark_x, spark_y, spark_w, spark_h)
        if len(self.threat_history) > 1:
            pts = []
            history = list(self.threat_history)
            step = spark_w / max(1, len(history) - 1)
            for i, val in enumerate(history):
                px = spark_x + i * step
                py = spark_y + spark_h - (val / 100) * spark_h
                pts.append(QPointF(px, py))
            p.setPen(QPen(ACCENT_CYAN, 1))
            for i in range(len(pts) - 1):
                p.drawLine(pts[i], pts[i + 1])

    def _draw_alert_panel(self, p: QPainter):
        L = self.layout
        x, y, w, h = L["panel_x"], L["alert_y"], L["panel_w"], L["alert_h"]
        title = "ALERT — ACTIVE" if self.last_assessment.level in ("HIGH", "CRITICAL") else "ALERT"
        self._panel_box(p, x, y, w, h, title)

        a = self.last_assessment
        p.setFont(QFont("Consolas", 9))

        if a.level not in ("HIGH", "CRITICAL"):
            p.setPen(QPen(TEXT_DIM))
            p.drawText(x + 12, y + 40, "No active alert.")
            return

        color = LEVEL_COLORS.get(a.level)
        p.setPen(QPen(color))
        p.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
        p.drawText(x + 12, y + 38, f"⚠ {a.level} THREAT DETECTED")

        p.setFont(QFont("Consolas", 8))
        p.setPen(QPen(TEXT_BRIGHT))
        for i, factor in enumerate(a.contributing_factors[:4]):
            p.drawText(x + 12, y + 58 + i * 16, f"- {factor}"[:75])

    def _draw_event_log(self, p: QPainter):
        L = self.layout
        x, y, w, h = L["event_log_x"], L["event_log_y"], L["event_log_w"], L["event_log_h"]
        if h < 40:
            return
        self._panel_box(p, x, y, w, h, "EVENT LOG")
        p.setFont(QFont("Consolas", 8))
        p.setPen(QPen(TEXT_DIM))
        for i, line in enumerate(list(self.event_log)[: (h - 30) // 14]):
            p.drawText(x + 12, y + 32 + i * 14, line[: max(20, w // 7)])

    def _draw_alert_history(self, p: QPainter):
        L = self.layout
        x, y, w, h = L["panel_x"], L["history_y"], L["panel_w"], L["history_h"]
        if h < 40:
            return
        self._panel_box(p, x, y, w, h, "ALERT HISTORY")
        p.setFont(QFont("Consolas", 8))
        if not self.alert_history:
            p.setPen(QPen(TEXT_DIM))
            p.drawText(x + 12, y + 36, "No alerts yet this session.")
            return
        for i, alert in enumerate(list(self.alert_history)[: (h - 30) // 16]):
            color = LEVEL_COLORS.get(alert["level"], TEXT_DIM)
            p.setPen(QPen(color))
            line = (
                f"[{alert['time']}] {alert['level']:8s} "
                f"score={alert['score']:.0f}  behavior={alert['behavior']}"
            )
            p.drawText(x + 12, y + 34 + i * 16, line[:80])


def run_dashboard():
    app = QApplication(sys.argv)
    window = Dashboard()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_dashboard()
