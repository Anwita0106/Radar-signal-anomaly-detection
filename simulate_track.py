"""
simulate_track.py
====================
One shared function that simulates a single aircraft pass through radar
coverage and returns its raw measurement rows. Both
`generate_training_data.py` (normal-only passes) and
`generate_test_data.py` (normal passes with an anomalous segment injected)
call this SAME function -- they only differ in which `behavior_schedule`
they hand it. This is what guarantees training and testing are built on
identical simulation mechanics, unlike V1, where the live dashboard's
motion model and `dataset_generator.py`'s motion model were two separate
(if similar) implementations.

Inputs
------
- `behavior_schedule`: a list of `(behavior_name, num_ticks)` tuples
  describing what the aircraft should do and for how long, in order.
- `radar_name`: which station in the network to take measurements from
  (defaults to `config.REFERENCE_RADAR`).
- `rng`: a `numpy.random.Generator` (pass a seeded one for reproducibility).

Outputs
-------
- `rows`: a list of raw measurement dicts (`timestamp` + every channel in
  `config.RADAR_CHANNELS`).
- `anomaly_flags`: a parallel list of bools, True for any row produced
  while an anomalous behavior was active (used only by the TEST generator
  to build ground-truth labels -- the training generator ignores this,
  since its schedule never contains an anomalous behavior in the first
  place).
"""

import numpy as np

import config
from aircraft_simulator import Aircraft
from behaviors import is_anomalous
from radar_simulator import build_radar_network


def simulate_track(
    behavior_schedule: list[tuple[str, int]],
    radar_name: str = config.REFERENCE_RADAR,
    rng: np.random.Generator | None = None,
    noise: bool = True,
):
    rng = rng if rng is not None else np.random.default_rng()

    aircraft = Aircraft(rng=rng)
    aircraft.reset()

    network = build_radar_network()
    radar = network[radar_name]
    radar.reset()

    rows = []
    anomaly_flags = []
    toa = 0.0

    for behavior_name, num_ticks in behavior_schedule:
        aircraft.set_behavior(behavior_name)
        flag = is_anomalous(behavior_name)

        for _ in range(num_ticks):
            aircraft.step()

            if not aircraft.is_inside_coverage():
                if aircraft.has_exited():
                    return rows, anomaly_flags
                continue

            measurement = radar.measure(
                aircraft.state.x, aircraft.state.y, aircraft.state.z, rng, noise=noise
            )
            toa += 0.03
            row = {"timestamp": round(toa, 3), **measurement}
            rows.append(row)
            anomaly_flags.append(flag)

    return rows, anomaly_flags
