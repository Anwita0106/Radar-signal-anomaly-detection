"""
radar_simulator.py
====================
Models a single ground-based radar station and the five measurement
channels it produces for a given aircraft state:

    range               - 3D distance from radar to aircraft (includes altitude)
    bearing             - horizontal direction of arrival (DOA), degrees
    power               - received signal power (dB), via the inverse-fourth-power
                          radar range equation
    frequency           - carrier frequency (MHz) plus a Doppler shift
                          proportional to relative velocity
    relative_velocity   - closing/opening rate (true range-rate), units/tick

This replaces V1's `radar_simulator.py`, which used arbitrary
`10000/(distance+1)` style formulas with no physical motivation and no
velocity/Doppler concept at all. Here, `frequency` and `relative_velocity`
are deliberately coupled the way a real Doppler radar's are -- an
aircraft that suddenly accelerates or dives doesn't just change one
random number, it produces a *physically consistent* shift across
multiple channels, which is exactly the kind of signature the anomaly
detector is meant to pick up on.

Inputs
------
- A `RadarStation` (fixed ground position).
- A `behaviors.AircraftState` (x, y, z) for the current tick.

Outputs
-------
- A dict with keys `RADAR_CHANNELS` (see config.py) -- the raw, optionally
  noisy measurement for that tick.
"""

import math
from dataclasses import dataclass, field

import numpy as np

import config


@dataclass
class RadarStation:
    name: str
    x: float
    y: float
    z: float = 0.0  # ground-based
    _last_true_range: float | None = field(default=None, repr=False)

    def reset(self) -> None:
        """Call this when a new aircraft pass begins, so the first
        relative-velocity reading isn't computed against a stale range
        from a previous (unrelated) track."""
        self._last_true_range = None

    def true_range(self, ax: float, ay: float, az: float) -> float:
        return math.sqrt((ax - self.x) ** 2 + (ay - self.y) ** 2 + (az - self.z) ** 2)

    def measure(
        self,
        ax: float,
        ay: float,
        az: float,
        rng: np.random.Generator,
        noise: bool = True,
    ) -> dict:
        true_rng = self.true_range(ax, ay, az)

        bearing = math.degrees(math.atan2(ay - self.y, ax - self.x))

        # Inverse-fourth-power radar range equation, expressed directly in dB:
        # power_db = ref_power_db - 40*log10(range / ref_range)
        safe_range = max(true_rng, 1e-3)
        power_db = config.REFERENCE_POWER_DB - 40 * math.log10(
            safe_range / config.REFERENCE_RANGE
        )

        # Relative velocity = true range-rate (negative = closing/approaching).
        if self._last_true_range is None:
            relative_velocity = 0.0
        else:
            relative_velocity = true_rng - self._last_true_range
        self._last_true_range = true_rng

        # Doppler-style frequency shift proportional to relative velocity.
        frequency = config.CARRIER_FREQUENCY_MHZ - (
            config.DOPPLER_COEFF * relative_velocity
        )

        if noise:
            true_rng += rng.normal(0, config.NOISE_STD["range"])
            bearing += rng.normal(0, config.NOISE_STD["bearing"])
            power_db += rng.normal(0, config.NOISE_STD["power"])
            frequency += rng.normal(0, config.NOISE_STD["frequency"])
            relative_velocity += rng.normal(0, config.NOISE_STD["relative_velocity"])

        return {
            "range": round(true_rng, 3),
            "bearing": round(bearing, 3),
            "power": round(power_db, 3),
            "frequency": round(frequency, 3),
            "relative_velocity": round(relative_velocity, 3),
        }


def build_radar_network() -> dict[str, RadarStation]:
    """Build the fixed 4-station radar network from config.RADAR_STATIONS."""
    return {
        name: RadarStation(name=name, x=float(pos[0]), y=float(pos[1]))
        for name, pos in config.RADAR_STATIONS.items()
    }
