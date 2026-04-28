"""
HPA-mimicking baseline agent.

Implements the same scale-up / scale-down logic as Kubernetes HPA
(CPU-threshold based) and collects (s, a, r, s') trajectories for
offline RL pre-training (AWARE bootstrapping phase).
"""
import logging
import pickle
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from autoscaler.config import settings

logger = logging.getLogger(__name__)

# Trajectory tuple type
Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


class HPABaseline:
    """
    CPU-threshold autoscaler that mirrors Kubernetes HPA behaviour.

    Collects experience tuples (obs, action, reward, next_obs, done)
    during bootstrapping for later offline PPO training.
    """

    def __init__(
        self,
        cpu_scale_up_threshold: float = settings.HPA_CPU_SCALE_UP_THRESHOLD,
        cpu_hysteresis: float = settings.HPA_CPU_HYSTERESIS,
        max_buffer_size: int = 50_000,
    ):
        self.cpu_up = cpu_scale_up_threshold
        self.cpu_down = cpu_scale_up_threshold - cpu_hysteresis
        self._replay_buffer: deque = deque(maxlen=max_buffer_size)
        self._last_obs: Optional[np.ndarray] = None
        self._last_action: Optional[int] = None
        self._reward_history: deque = deque(maxlen=settings.REWARD_WINDOW)

    # ── Policy ────────────────────────────────────────────────────────────────

    def decide(self, obs: np.ndarray) -> int:
        """
        Decide action from normalised observation vector.

        obs = [cpu_pct, mem_pct, req_rate_norm, p95_lat_norm, replicas_norm]
        Returns action index: 0=scale-down, 1=maintain, 2=scale-up
        """
        cpu_pct = float(obs[0])

        if cpu_pct >= self.cpu_up:
            action = 2  # scale up
        elif cpu_pct < self.cpu_down:
            action = 0  # scale down
        else:
            action = 1  # maintain

        return action

    # ── Trajectory collection ─────────────────────────────────────────────────

    def record_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool = False,
    ) -> None:
        """Store a single (s, a, r, s', done) transition."""
        self._replay_buffer.append((obs.copy(), action, reward, next_obs.copy(), done))
        self._reward_history.append(reward)

    def step_and_record(
        self,
        obs: np.ndarray,
        reward: Optional[float],
        next_obs: Optional[np.ndarray],
    ) -> int:
        """
        Convenience: record previous transition (if any), decide next action.
        Returns the chosen action.
        """
        if self._last_obs is not None and reward is not None and next_obs is not None:
            self.record_transition(self._last_obs, self._last_action, reward, next_obs)

        action = self.decide(obs)
        self._last_obs = obs.copy()
        self._last_action = action
        return action

    # ── Buffer access ─────────────────────────────────────────────────────────

    def get_replay_buffer(self) -> List[Transition]:
        return list(self._replay_buffer)

    def buffer_size(self) -> int:
        return len(self._replay_buffer)

    def get_avg_reward(self) -> float:
        if not self._reward_history:
            return 0.0
        return float(np.mean(list(self._reward_history)))

    def save_buffer(self, path: str = settings.REPLAY_BUFFER_PATH) -> None:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(list(self._replay_buffer), f)
        logger.info("Replay buffer saved to %s (%d transitions)", path, len(self._replay_buffer))

    def load_buffer(self, path: str = settings.REPLAY_BUFFER_PATH) -> None:
        with open(path, "rb") as f:
            data: List[Transition] = pickle.load(f)
        self._replay_buffer = deque(data, maxlen=self._replay_buffer.maxlen)
        logger.info("Replay buffer loaded from %s (%d transitions)", path, len(data))
