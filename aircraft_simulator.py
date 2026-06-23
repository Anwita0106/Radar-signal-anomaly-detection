"""
aircraft_simulator.py
======================
The kinematics engine. `Aircraft` owns position/heading/speed/altitude and
knows how to integrate them forward one tick at a time. It does NOT decide
*what* the aircraft should be doing -- that decision belongs entirely to
`behaviors.py`. This split is the key architectural fix over V1, where a
single `Aircraft.update()` method's `if/elif` chain mixed "how to move"
and "which pattern is normal vs anomalous" into one unreadable block.

Inputs
------
- A `behavior_name` (a key into `behaviors.NORMAL_BEHAVIORS` or
  `behaviors.ANOMALOUS_BEHAVIORS`), set via `Aircraft.set_behavior()`.

Outputs
-------
- `Aircraft.state` (an `AircraftState`): x, y, z, heading, speed, vspeed --
  read by `radar_simulator.py` every tick to produce a measurement, and by
  `dashboard.py` to draw the aircraft on screen.
"""

import math

import numpy as np

import config
from behaviors import AircraftState, ALL_BEHAVIORS, is_anomalous


class Aircraft:
    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.trail: list[tuple[float, float]] = []
        self._behavior_name = "straight_flight"
        self._behavior_fn = ALL_BEHAVIORS[self._behavior_name]
        self._ticks_in_behavior = 0
        self.state = self._initial_state()

    # ------------------------------------------------------------------
    def _initial_state(self) -> AircraftState:
        return AircraftState(
            x=float(config.COVERAGE_LEFT - 50),
            y=float(config.COVERAGE_TOP + config.COVERAGE_HEIGHT / 2),
            z=config.BASE_ALTITUDE,
            heading=0.0,            # flying in +x direction
            speed=config.BASE_SPEED,
            vspeed=0.0,
        )

    def reset(self, entry_y: float | None = None) -> None:
        """Re-enter from the left edge of coverage on a fresh, randomized lane."""
        self.trail.clear()
        self.state = self._initial_state()
        if entry_y is not None:
            self.state.y = entry_y
        else:
            margin = 80
            self.state.y = self.rng.uniform(
                config.COVERAGE_TOP + margin,
                config.COVERAGE_BOTTOM - margin,
            )
        self.set_behavior("straight_flight")

    # ------------------------------------------------------------------
    def set_behavior(self, behavior_name: str) -> None:
        """Switch the active behavior. Resets the per-behavior tick counter."""
        if behavior_name not in ALL_BEHAVIORS:
            raise ValueError(f"Unknown behavior: {behavior_name}")
        self._behavior_name = behavior_name
        self._behavior_fn = ALL_BEHAVIORS[behavior_name]
        self._ticks_in_behavior = 0

    @property
    def behavior_name(self) -> str:
        return self._behavior_name

    @property
    def is_anomalous(self) -> bool:
        return is_anomalous(self._behavior_name)

    # ------------------------------------------------------------------
    def step(self) -> AircraftState:
        """Advance the simulation by one tick. Returns the new state."""
        # 1. Let the active behavior adjust heading / speed / vspeed.
        self._behavior_fn(self.state, self._ticks_in_behavior, self.rng)
        self._ticks_in_behavior += 1

        # 2. Integrate position from the (possibly just-updated) kinematics.
        heading_rad = math.radians(self.state.heading)
        self.state.x += self.state.speed * math.cos(heading_rad)
        self.state.y += self.state.speed * math.sin(heading_rad)
        self.state.z += self.state.vspeed

        self.trail.append((self.state.x, self.state.y))
        if len(self.trail) > 300:
            self.trail.pop(0)

        return self.state

    # ------------------------------------------------------------------
    def is_inside_coverage(self) -> bool:
        return (
            config.COVERAGE_LEFT <= self.state.x <= config.COVERAGE_RIGHT
            and config.COVERAGE_TOP <= self.state.y <= config.COVERAGE_BOTTOM
        )

    def has_exited_right(self) -> bool:
        return self.state.x > config.COVERAGE_RIGHT + 50

    def has_exited(self) -> bool:
        """True once the aircraft has drifted out of coverage in ANY
        direction by more than a small margin -- not just past the right
        edge. A pass should end here so a track never silently flies
        outside the field indefinitely (e.g. after a turn-heavy normal
        behavior or a sudden_heading_change anomaly redirects it)."""
        margin = 60
        return (
            self.state.x > config.COVERAGE_RIGHT + margin
            or self.state.x < config.COVERAGE_LEFT - margin
            or self.state.y < config.COVERAGE_TOP - margin
            or self.state.y > config.COVERAGE_BOTTOM + margin
        )
