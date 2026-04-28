"""
Sliding-window reward monitor for AWARE continuous retraining.

Triggers retraining when:
  - Running average drops below RETRAIN_AVG_THRESHOLD, OR
  - Running variance exceeds RETRAIN_VAR_THRESHOLD
"""
import logging
from collections import deque
from typing import Optional

import numpy as np

from autoscaler.config import settings

logger = logging.getLogger(__name__)


class RetrainingMonitor:
    """
    Monitors recent rewards and decides when to re-trigger offline training.
    Implements the AWARE paper's performance monitoring component.
    """

    def __init__(
        self,
        window: int = settings.REWARD_WINDOW,
        avg_threshold: float = settings.RETRAIN_AVG_THRESHOLD,
        var_threshold: float = settings.RETRAIN_VAR_THRESHOLD,
    ):
        self.window = window
        self.avg_threshold = avg_threshold
        self.var_threshold = var_threshold
        self._buffer: deque = deque(maxlen=window)
        self._triggered: bool = False

    def record(self, reward: float) -> None:
        self._buffer.append(float(reward))

    def should_retrain(self) -> bool:
        """
        Returns True if the reward window indicates performance degradation.
        Only triggers once per degradation event (reset() must be called to re-arm).
        """
        if self._triggered:
            return False

        if len(self._buffer) < self.window:
            return False  # not enough data yet

        arr = np.array(list(self._buffer))
        avg = float(np.mean(arr))
        var = float(np.var(arr))

        if avg < self.avg_threshold:
            logger.warning(
                "Retraining trigger: avg reward %.3f < threshold %.3f",
                avg, self.avg_threshold,
            )
            self._triggered = True
            return True

        if var > self.var_threshold:
            logger.warning(
                "Retraining trigger: reward variance %.3f > threshold %.3f",
                var, self.var_threshold,
            )
            self._triggered = True
            return True

        return False

    def reset(self) -> None:
        """Re-arm the monitor after a retrain has been scheduled."""
        self._buffer.clear()
        self._triggered = False

    def get_stats(self) -> dict:
        if not self._buffer:
            return {"avg": 0.0, "var": 0.0, "n": 0}
        arr = np.array(list(self._buffer))
        return {
            "avg": float(np.mean(arr)),
            "var": float(np.var(arr)),
            "n": len(arr),
        }
