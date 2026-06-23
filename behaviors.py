"""
behaviors.py
============
The single, explicit catalog of every flight behavior the simulator can
produce -- cleanly split into NORMAL and ANOMALOUS.

Why this file exists
---------------------
In V1, `aircraft_simulator.Aircraft` mixed normal and anomalous motion
patterns together inside one unlabeled `mode` switch (0-5), with comments
like "# DOA Drift Anomaly" sitting next to normal "Climbing"/"Descending"
modes. There was no programmatic way to ask "is this behavior supposed to
be normal or anomalous?" -- you had to read the source and guess.

In V2, that question is answered by data, not by reading code:

    NORMAL_BEHAVIORS = {"straight_flight": ..., "gentle_turn": ..., ...}
    ANOMALOUS_BEHAVIORS = {"sudden_heading_change": ..., ...}

The training pipeline only ever samples from NORMAL_BEHAVIORS. The testing
pipeline samples a normal behavior, then injects one entry from
ANOMALOUS_BEHAVIORS over a limited segment. The live dashboard does the
same thing, using the exact same functions -- so "training," "testing,"
and "the live demo" all agree, by construction, on what counts as normal
and what counts as anomalous.

Behavior function contract
---------------------------
Every behavior function has the signature:

    behavior(state: AircraftState, t: int, rng: np.random.Generator) -> None

It mutates `state.heading`, `state.speed`, and/or `state.vspeed` in place
for the current tick. `t` is the number of ticks since this behavior
became active (starts at 0), which lets behaviors like ZIG_ZAG or
SUDDEN_HEADING_CHANGE do something different at specific points in their
lifetime. Actually moving the aircraft (applying heading/speed/vspeed to
x/y/z) is the job of `aircraft_simulator.Aircraft.step()`, not of the
behavior functions -- behaviors only decide *how the aircraft should be
flying right now*, not how to integrate position.
"""

from dataclasses import dataclass

import numpy as np

import config


@dataclass
class AircraftState:
    """Plain kinematic state. No behavior logic lives here."""
    x: float
    y: float
    z: float           # altitude
    heading: float      # degrees, 0 = +x direction
    speed: float        # horizontal units / tick
    vspeed: float       # altitude units / tick


# ----------------------------------------------------------------------
# NORMAL behaviors
# ----------------------------------------------------------------------

def straight_flight(state: AircraftState, t: int, rng: np.random.Generator) -> None:
    """Constant heading, constant speed, level altitude."""
    state.vspeed = 0.0


def gentle_turn(state: AircraftState, t: int, rng: np.random.Generator) -> None:
    """Slow, constant-rate heading change -- a normal course correction."""
    state.heading += config.GENTLE_TURN_RATE
    state.vspeed = 0.0


def climb(state: AircraftState, t: int, rng: np.random.Generator) -> None:
    """Steady, modest rate of altitude gain."""
    state.vspeed = config.CLIMB_RATE


def descent(state: AircraftState, t: int, rng: np.random.Generator) -> None:
    """Steady, modest rate of altitude loss."""
    state.vspeed = -config.DESCENT_RATE


NORMAL_BEHAVIORS = {
    "straight_flight":straight_flight,
    "gentle_turn": gentle_turn,
    "climb": climb,
    "descent": descent,

}


# ----------------------------------------------------------------------
# ANOMALOUS behaviors
# ----------------------------------------------------------------------

def sudden_heading_change(state: AircraftState, t: int, rng: np.random.Generator) -> None:
    """
    An abrupt, large heading jump (simulating an evasive maneuver or a
    non-cooperative target reacting to something). The jump happens once,
    on the first tick this behavior is active, then the aircraft continues
    straight on the new heading.
    """
    if t == 0:
        lo, hi = config.SUDDEN_HEADING_CHANGE_DEG
        #jump = rng.uniform(lo, hi) * rng.choice([-1, 1])\
        jump = 180
        state.heading += jump
    state.vspeed = 0.0


def zig_zag(state: AircraftState, t: int, rng: np.random.Generator) -> None:
    """
    Rapidly oscillating heading -- the aircraft repeatedly snaps between
    two headings every `ZIG_ZAG_PERIOD_TICKS` ticks. Distinct from
    `gentle_turn`: this is high-frequency, high-amplitude, and has no net
    directional intent.
    """
    period = config.ZIG_ZAG_PERIOD_TICKS
    amplitude = config.ZIG_ZAG_AMPLITUDE_DEG
    phase = (t // period) % 2
    state.heading = state.heading if t > 0 else state.heading
    # Snap to +amplitude or -amplitude relative to the heading captured at t=0
    if t == 0:
        state._zigzag_base_heading = state.heading  # type: ignore[attr-defined]
    base = getattr(state, "_zigzag_base_heading", state.heading)
    state.heading = base + (amplitude if phase == 0 else -amplitude)
    state.vspeed = 0.0


def evasive_maneuver(state: AircraftState, t: int, rng: np.random.Generator) -> None:
    """
    Aircraft repeatedly performs sudden tactical turns.
    Simulates evasive movement.
    """

    if t % 10 == 0:
        state.heading += rng.choice([-90, 90])

    state.vspeed = 0.0

def extreme_acceleration(state: AircraftState, t: int, rng: np.random.Generator) -> None:
    """
    A sudden, large jump in ground speed -- far beyond what the aircraft's
    normal cruise speed would allow. Heading and altitude are unaffected;
    only `speed` (and therefore `range`/`relative_velocity` as seen by
    radar) spikes.
    """
    if t == 0:
        lo, hi = config.EXTREME_ACCEL_MULTIPLIER
        #multiplier = rng.uniform(lo, hi)
        multiplier = 10
        state._base_speed_before_accel = state.speed  # type: ignore[attr-defined]
        state.speed = state.speed * multiplier
    state.vspeed = 0.0


def suspicious_loitering(state: AircraftState, t: int, rng: np.random.Generator) -> None:
    """
    Tight, continuous circling at reduced speed with near-zero net
    displacement -- behavior consistent with reconnaissance / loitering
    rather than transiting through the area.
    """
    if t == 0:
        state._base_speed_before_loiter = state.speed  # type: ignore[attr-defined]
        state.speed = state.speed * config.LOITER_SPEED_FACTOR
    state.heading += config.LOITER_TURN_RATE_DEG
    state.vspeed = 0.0


def altitude_spike(state: AircraftState, t: int, rng: np.random.Generator) -> None:
    """
    A sharp, fast altitude change -- climbing or diving at a rate far
    beyond normal `climb`/`descent` behavior. Heading and speed are
    unaffected.
    """
    if t == 0:
        lo, hi = config.ALTITUDE_SPIKE_RATE_MULTIPLIER
        multiplier = rng.uniform(lo, hi) * rng.choice([-1, 1])
        state._spike_vspeed = config.CLIMB_RATE * multiplier  # type: ignore[attr-defined]
    state.vspeed = getattr(state, "_spike_vspeed", config.CLIMB_RATE)


ANOMALOUS_BEHAVIORS = {
    "sudden_heading_change": sudden_heading_change,
    "zig_zag": zig_zag,
    "extreme_acceleration": extreme_acceleration,
    "suspicious_loitering": suspicious_loitering,
    "altitude_spike": altitude_spike,
    "evasive_maneuver": evasive_maneuver,
}


ALL_BEHAVIORS = {**NORMAL_BEHAVIORS, **ANOMALOUS_BEHAVIORS}


def is_anomalous(behavior_name: str) -> bool:
    return behavior_name in ANOMALOUS_BEHAVIORS
