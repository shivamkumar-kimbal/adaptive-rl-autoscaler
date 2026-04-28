"""
Workload pattern classifier.

Analyses recent request-rate time series and classifies into:
  periodic  – dominant oscillation detected via FFT
  spike     – sudden large increase relative to recent baseline
  ramp      – steady upward trend (linear regression slope > 0)
  steady    – none of the above

Pattern labels are attached to logged trajectories for post-hoc analysis
and can optionally be used to select specialised RL sub-policies.
"""
import logging
from collections import deque
from typing import List

import numpy as np

from autoscaler.config import settings

logger = logging.getLogger(__name__)


class WorkloadPatternClassifier:
    """
    Lightweight time-series classifier for request-rate patterns.
    Uses FFT for periodicity and linear regression for trend detection.
    """

    def __init__(
        self,
        window: int = settings.PATTERN_WINDOW,
        spike_threshold: float = settings.PATTERN_SPIKE_THRESHOLD,
        periodic_freq_threshold: float = settings.PATTERN_PERIODIC_FREQ_THRESHOLD,
        ramp_slope_threshold: float = 0.5,
    ):
        self.window = window
        self.spike_threshold = spike_threshold
        self.periodic_freq_threshold = periodic_freq_threshold
        self.ramp_slope_threshold = ramp_slope_threshold
        self._history: deque = deque(maxlen=window)
        self._current_pattern: str = "steady"

    def update(self, request_rate: float) -> str:
        """
        Add the latest request rate observation and return the current pattern.
        """
        self._history.append(float(request_rate))
        if len(self._history) >= max(10, self.window // 2):
            self._current_pattern = self._classify(list(self._history))
        return self._current_pattern

    def current_pattern(self) -> str:
        return self._current_pattern

    # ── Classification logic ──────────────────────────────────────────────────

    def _classify(self, series: List[float]) -> str:
        arr = np.array(series, dtype=np.float64)

        # 1. Spike detection: last value >> recent mean by spike_threshold × std
        if len(arr) >= 5:
            baseline = arr[:-1]
            std = np.std(baseline) + 1e-9
            mean = np.mean(baseline)
            if arr[-1] > mean + self.spike_threshold * std:
                return "spike"

        # 2. Periodic detection via FFT
        if self._is_periodic(arr):
            return "periodic"

        # 3. Ramp detection via linear regression slope
        slope = self._linear_slope(arr)
        if slope > self.ramp_slope_threshold:
            return "ramp"

        return "steady"

    def _is_periodic(self, arr: np.ndarray) -> bool:
        """True if FFT reveals a dominant non-DC frequency above threshold."""
        if len(arr) < 16:
            return False
        # Detrend before FFT to avoid DC dominance
        detrended = arr - np.linspace(arr[0], arr[-1], len(arr))
        fft_vals = np.abs(np.fft.rfft(detrended))
        freqs = np.fft.rfftfreq(len(arr))
        # Ignore DC component (index 0)
        if len(fft_vals) <= 1:
            return False
        dominant_freq = freqs[1:][np.argmax(fft_vals[1:])]
        total_power = np.sum(fft_vals[1:]) + 1e-9
        dominant_power_ratio = np.max(fft_vals[1:]) / total_power
        return (
            dominant_freq > self.periodic_freq_threshold
            and dominant_power_ratio > 0.3
        )

    @staticmethod
    def _linear_slope(arr: np.ndarray) -> float:
        """Normalised slope of a linear regression fit (per-step change / mean)."""
        x = np.arange(len(arr), dtype=np.float64)
        if np.std(x) == 0:
            return 0.0
        slope = np.polyfit(x, arr, 1)[0]
        mean = np.mean(np.abs(arr)) + 1e-9
        return slope / mean
