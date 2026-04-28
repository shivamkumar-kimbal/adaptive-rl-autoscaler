"""
Gymnasium environment wrapping the Kubernetes autoscaling problem.

Observation space (5-D, all normalised to [0, 1]):
  [cpu_pct, mem_pct, req_rate_norm, p95_latency_norm, replicas_norm]

Action space:
  Discrete(3)  →  0 = scale down, 1 = maintain, 2 = scale up

Reward:
  α · slo_score  +  (1-α) · utilisation_score  −  oscillation_penalty
"""
import logging
import time
from collections import deque
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from autoscaler.config import settings
from autoscaler.controller.scaler import KubernetesScaler
from autoscaler.environment.metrics import PrometheusClient

logger = logging.getLogger(__name__)

# Action index → replica delta mapping
ACTION_DELTA = {0: -1, 1: 0, 2: 1}

# Upper bound used to normalise request rate (req/s)
REQ_RATE_NORM_CEILING = float(
    __import__("os").getenv("REQ_RATE_NORM_CEILING", "500")
)


class KubernetesAutoscalerEnv(gym.Env):
    """
    Gymnasium environment for the AWARE-inspired Kubernetes autoscaler.

    In real operation the step() method actually invokes the Kubernetes API
    and waits for CONTROL_INTERVAL_SECONDS to collect the next observation.
    For offline / replay training a RecordedEnv subclass is used instead.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scaler: Optional[KubernetesScaler] = None,
        prom_client: Optional[PrometheusClient] = None,
        dry_run: bool = False,
    ):
        super().__init__()

        self.scaler = scaler or KubernetesScaler()
        self.prom = prom_client or PrometheusClient()
        self.dry_run = dry_run  # skip real k8s calls during replay training

        self.observation_space = spaces.Box(
            low=np.zeros(5, dtype=np.float32),
            high=np.ones(5, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        self._prev_action: Optional[int] = None
        self._current_replicas: int = settings.MIN_REPLICAS
        self._step_count: int = 0

        # Keep a short history of raw observations for pattern detector
        self._obs_history: deque = deque(maxlen=settings.PATTERN_WINDOW)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_raw_obs(self) -> Dict[str, float]:
        return {
            "cpu_pct": self.prom.get_cpu_pct(),
            "mem_pct": self.prom.get_memory_pct(),
            "req_rate": self.prom.get_request_rate(),
            "p95_latency_ms": self.prom.get_p95_latency_ms(),
            "replicas": float(self.scaler.get_replicas()),
        }

    def _normalise(self, raw: Dict[str, float]) -> np.ndarray:
        return np.array(
            [
                np.clip(raw["cpu_pct"], 0.0, 1.0),
                np.clip(raw["mem_pct"], 0.0, 1.0),
                np.clip(raw["req_rate"] / REQ_RATE_NORM_CEILING, 0.0, 1.0),
                np.clip(
                    raw["p95_latency_ms"] / (settings.SLO_LATENCY_MS * 2), 0.0, 1.0
                ),
                np.clip(
                    (raw["replicas"] - settings.MIN_REPLICAS)
                    / (settings.MAX_REPLICAS - settings.MIN_REPLICAS),
                    0.0,
                    1.0,
                ),
            ],
            dtype=np.float32,
        )

    def _compute_reward(
        self,
        p95_ms: float,
        replicas: float,
        action: int,
    ) -> float:
        # SLO score: 1 if within SLO, degrades linearly beyond
        if p95_ms <= settings.SLO_LATENCY_MS:
            slo_score = 1.0
        else:
            slo_score = max(
                0.0,
                1.0 - (p95_ms - settings.SLO_LATENCY_MS) / settings.SLO_LATENCY_MS,
            )

        # Utilisation score: fewer replicas = higher score (efficiency)
        util_score = 1.0 - (replicas - settings.MIN_REPLICAS) / (
            settings.MAX_REPLICAS - settings.MIN_REPLICAS + 1e-9
        )
        util_score = max(0.0, util_score)

        # Oscillation penalty: discourage flip-flopping
        oscillation_penalty = (
            0.1 if (self._prev_action is not None and action != self._prev_action and action != 1)
            else 0.0
        )

        reward = (
            settings.ALPHA * slo_score
            + (1.0 - settings.ALPHA) * util_score
            - oscillation_penalty
        )
        return float(reward)

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._prev_action = None
        self._step_count = 0

        raw = self._get_raw_obs()
        self._obs_history.clear()
        self._obs_history.append(raw)
        obs = self._normalise(raw)
        return obs, {"raw": raw}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        delta = ACTION_DELTA[int(action)]

        if not self.dry_run:
            new_replicas = self.scaler.apply_delta(delta)
            time.sleep(settings.CONTROL_INTERVAL_SECONDS)
        else:
            new_replicas = max(
                settings.MIN_REPLICAS,
                min(settings.MAX_REPLICAS, self._current_replicas + delta),
            )
        self._current_replicas = new_replicas

        raw = self._get_raw_obs()
        self._obs_history.append(raw)
        obs = self._normalise(raw)

        reward = self._compute_reward(
            p95_ms=raw["p95_latency_ms"],
            replicas=raw["replicas"],
            action=action,
        )

        self._prev_action = action
        self._step_count += 1

        info = {
            "raw": raw,
            "reward_breakdown": {
                "p95_latency_ms": raw["p95_latency_ms"],
                "replicas": raw["replicas"],
                "action_delta": delta,
            },
        }
        # Never terminates on its own (continuous control task)
        return obs, reward, False, False, info

    def render(self):
        pass

    def get_obs_history(self):
        return list(self._obs_history)
