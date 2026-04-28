"""
Workload pattern generators used by the Locust load test.

Patterns:
  periodic  – sinusoidal oscillation (simulates day/night traffic cycles)
  spike     – sudden burst then decay (flash sale / viral event)
  ramp      – linearly increasing load (organic growth)
  steady    – constant baseline load
"""
import math
import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class PatternConfig:
    name: str
    base_users: int = 10
    peak_users: int = 50
    period_seconds: float = 300.0    # for periodic
    spike_duration_seconds: float = 60.0
    ramp_duration_seconds: float = 300.0


def periodic_rate(cfg: PatternConfig, elapsed: float) -> int:
    """Sinusoidal user count between base and peak."""
    amplitude = (cfg.peak_users - cfg.base_users) / 2
    midpoint = cfg.base_users + amplitude
    rate = midpoint + amplitude * math.sin(2 * math.pi * elapsed / cfg.period_seconds)
    return max(1, int(rate))


def spike_rate(cfg: PatternConfig, elapsed: float) -> int:
    """
    Spike: rise quickly to peak, hold briefly, then decay back to base.
    Spike repeats every (3 × spike_duration) seconds.
    """
    cycle = cfg.spike_duration_seconds * 3
    t = elapsed % cycle
    if t < cfg.spike_duration_seconds * 0.2:
        # Fast rise
        frac = t / (cfg.spike_duration_seconds * 0.2)
        return int(cfg.base_users + frac * (cfg.peak_users - cfg.base_users))
    elif t < cfg.spike_duration_seconds:
        # Hold peak
        return cfg.peak_users
    else:
        # Exponential decay
        decay_t = t - cfg.spike_duration_seconds
        frac = math.exp(-decay_t / (cfg.spike_duration_seconds * 0.5))
        return max(cfg.base_users, int(cfg.base_users + frac * (cfg.peak_users - cfg.base_users)))


def ramp_rate(cfg: PatternConfig, elapsed: float) -> int:
    """Linear ramp from base to peak over ramp_duration, then reset."""
    t = elapsed % (cfg.ramp_duration_seconds * 1.2)
    if t <= cfg.ramp_duration_seconds:
        frac = t / cfg.ramp_duration_seconds
        return int(cfg.base_users + frac * (cfg.peak_users - cfg.base_users))
    else:
        return cfg.base_users


def steady_rate(cfg: PatternConfig, _elapsed: float) -> int:
    return cfg.base_users


PATTERN_FN: dict[str, Callable] = {
    "periodic": periodic_rate,
    "spike": spike_rate,
    "ramp": ramp_rate,
    "steady": steady_rate,
}


class PatternScheduler:
    """
    Cycles through workload patterns on a fixed schedule.
    Used by the Locust master to adjust spawn rates dynamically.
    """

    SCHEDULE = [
        ("steady",   120),   # 2 min baseline
        ("ramp",     300),   # 5 min ramp
        ("steady",   120),
        ("periodic", 300),   # 5 min periodic
        ("steady",   120),
        ("spike",    180),   # 3 min spikes
        ("steady",   120),
    ]

    def __init__(self, cfg: PatternConfig = None):
        self.cfg = cfg or PatternConfig(name="default")
        self._start = time.time()
        self._phase_index = 0
        self._phase_start = time.time()

    def current_pattern(self) -> str:
        name, duration = self.SCHEDULE[self._phase_index]
        if time.time() - self._phase_start >= duration:
            self._phase_index = (self._phase_index + 1) % len(self.SCHEDULE)
            self._phase_start = time.time()
            name = self.SCHEDULE[self._phase_index][0]
        return name

    def current_user_count(self) -> int:
        pattern = self.current_pattern()
        elapsed = time.time() - self._phase_start
        fn = PATTERN_FN[pattern]
        return fn(self.cfg, elapsed)
