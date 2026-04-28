"""
AWARE-inspired lifecycle state machine.

States:
  INITIALIZED       → HPA controls system; collecting bootstrap trajectories
  OFFLINE_TRAINING  → Enough HPA data collected; training RL offline
  ONLINE_TRAINING   → RL controls system and trains incrementally
  SERVING           → RL trained and deployed; monitoring for retraining triggers

Transitions:
  INITIALIZED       → OFFLINE_TRAINING : buffer_size >= MIN_BOOTSTRAP_STEPS
  OFFLINE_TRAINING  → ONLINE_TRAINING  : offline training done AND rl_avg >= threshold * hpa_avg
  ONLINE_TRAINING   → SERVING          : rl_avg >= hpa_avg (sustained over REWARD_WINDOW)
  SERVING           → OFFLINE_TRAINING : retraining_monitor.should_retrain()
"""
import logging
from enum import Enum
from typing import Callable, Optional

from autoscaler.config import settings

logger = logging.getLogger(__name__)


class LifecycleState(str, Enum):
    INITIALIZED = "INITIALIZED"
    OFFLINE_TRAINING = "OFFLINE_TRAINING"
    ONLINE_TRAINING = "ONLINE_TRAINING"
    SERVING = "SERVING"


class LifecycleManager:
    """
    Manages autoscaler lifecycle transitions following the AWARE paper design.
    """

    def __init__(self):
        self.state: LifecycleState = LifecycleState.INITIALIZED
        self._offline_training_done: bool = False
        self._on_transition_callbacks: list = []

    # ── Transition logic ──────────────────────────────────────────────────────

    def check_and_transition(
        self,
        hpa_buffer_size: int,
        hpa_avg_reward: float,
        rl_avg_reward: float,
        retraining_monitor=None,
    ) -> Optional[LifecycleState]:
        """
        Evaluate all transition conditions and apply the first valid one.
        Returns the new state if a transition occurred, else None.
        """
        prev = self.state

        if self.state == LifecycleState.INITIALIZED:
            if hpa_buffer_size >= settings.MIN_BOOTSTRAP_STEPS:
                self._transition(LifecycleState.OFFLINE_TRAINING)

        elif self.state == LifecycleState.OFFLINE_TRAINING:
            if self._offline_training_done:
                # Only go online if RL is competitive with HPA
                hpa_ref = max(hpa_avg_reward, 1e-9)
                if rl_avg_reward >= settings.RL_BETTER_THRESHOLD * hpa_ref:
                    self._transition(LifecycleState.ONLINE_TRAINING)
                else:
                    logger.info(
                        "RL avg reward %.3f < %.1f%% of HPA avg reward %.3f — "
                        "staying offline.",
                        rl_avg_reward,
                        settings.RL_BETTER_THRESHOLD * 100,
                        hpa_avg_reward,
                    )

        elif self.state == LifecycleState.ONLINE_TRAINING:
            hpa_ref = max(hpa_avg_reward, 1e-9)
            if rl_avg_reward >= hpa_ref:
                self._transition(LifecycleState.SERVING)

        elif self.state == LifecycleState.SERVING:
            if retraining_monitor is not None and retraining_monitor.should_retrain():
                logger.warning(
                    "Performance degradation detected — triggering retraining."
                )
                retraining_monitor.reset()
                self._offline_training_done = False
                self._transition(LifecycleState.OFFLINE_TRAINING)

        return self.state if self.state != prev else None

    def mark_offline_training_done(self) -> None:
        self._offline_training_done = True

    def _transition(self, new_state: LifecycleState) -> None:
        old = self.state
        self.state = new_state
        logger.info("Lifecycle transition: %s → %s", old.value, new_state.value)
        for cb in self._on_transition_callbacks:
            try:
                cb(old, new_state)
            except Exception as exc:
                logger.warning("Transition callback error: %s", exc)

    def on_transition(self, callback: Callable) -> None:
        """Register a callback(old_state, new_state) for state transitions."""
        self._on_transition_callbacks.append(callback)

    def is_hpa_active(self) -> bool:
        return self.state in (
            LifecycleState.INITIALIZED,
            LifecycleState.OFFLINE_TRAINING,
        )

    def is_rl_active(self) -> bool:
        return self.state in (
            LifecycleState.ONLINE_TRAINING,
            LifecycleState.SERVING,
        )
