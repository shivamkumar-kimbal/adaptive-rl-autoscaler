"""
Main autoscaler control loop (Orchestrator).

Runs every CONTROL_INTERVAL_SECONDS and delegates to HPA or RL
depending on the current lifecycle state.

Lifecycle flow (AWARE-inspired):
  INITIALIZED       → HPA controls; collect bootstrap data
  OFFLINE_TRAINING  → HPA still controls; RL trains in background thread
  ONLINE_TRAINING   → RL controls; PPO updates incrementally
  SERVING           → RL controls; monitor for retraining triggers
"""
import logging
import os
import threading
import time
from typing import Optional

import numpy as np

from autoscaler.agents.hpa_baseline import HPABaseline
from autoscaler.agents.ppo_agent import PPOAgent
from autoscaler.config import settings
from autoscaler.controller.scaler import KubernetesScaler
from autoscaler.environment.k8s_env import KubernetesAutoscalerEnv
from autoscaler.environment.metrics import PrometheusClient
from autoscaler.lifecycle.retraining_monitor import RetrainingMonitor
from autoscaler.lifecycle.state_machine import LifecycleManager, LifecycleState
from autoscaler.workload_detector.pattern_classifier import WorkloadPatternClassifier

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Central controller that ties together the HPA baseline, PPO agent,
    lifecycle manager, and Kubernetes scaler.
    """

    def __init__(self):
        self.prom = PrometheusClient()
        self.scaler = KubernetesScaler()
        self.env = KubernetesAutoscalerEnv(scaler=self.scaler, prom_client=self.prom)

        self.hpa = HPABaseline()
        self.rl_agent = PPOAgent(env=self.env)
        self.lifecycle = LifecycleManager()
        self.retrain_monitor = RetrainingMonitor()
        self.pattern_classifier = WorkloadPatternClassifier()

        # Lazy-import metrics logger to avoid circular deps
        from evaluation.metrics_logger import MetricsLogger
        self.metrics_logger = MetricsLogger()

        self._prev_obs: Optional[np.ndarray] = None
        self._prev_reward: Optional[float] = None
        self._running = False
        self._offline_train_thread: Optional[threading.Thread] = None

        # Register lifecycle transition logging
        self.lifecycle.on_transition(self._on_lifecycle_transition)

    # ── Lifecycle hooks ───────────────────────────────────────────────────────

    def _on_lifecycle_transition(self, old: LifecycleState, new: LifecycleState):
        logger.info("=== LIFECYCLE: %s → %s ===", old.value, new.value)
        self.metrics_logger.log_event("lifecycle_transition", {"from": old.value, "to": new.value})

        if new == LifecycleState.OFFLINE_TRAINING:
            self._start_offline_training()

    def _start_offline_training(self):
        """Launch offline training in a background thread so HPA keeps running."""
        def _train():
            logger.info("Offline training thread started.")
            buffer = self.hpa.get_replay_buffer()
            self.rl_agent.train_offline(buffer)
            self.lifecycle.mark_offline_training_done()
            logger.info("Offline training thread done.")

        self._offline_train_thread = threading.Thread(target=_train, daemon=True)
        self._offline_train_thread.start()

    # ── Core control tick ─────────────────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        """Query Prometheus and return a normalised observation vector."""
        obs, _ = self.env.reset() if self._prev_obs is None else (self._prev_obs, {})
        # Always refresh from Prometheus
        raw = self.env._get_raw_obs()
        obs = self.env._normalise(raw)
        return obs, raw

    def _compute_reward(self, raw: dict, action: int) -> float:
        return self.env._compute_reward(
            p95_ms=raw["p95_latency_ms"],
            replicas=raw["replicas"],
            action=action,
        )

    def tick(self):
        """One control loop iteration."""
        obs, raw = self._get_observation()
        pattern = self.pattern_classifier.update(raw["req_rate"])
        state = self.lifecycle.state

        # ── Choose action ─────────────────────────────────────────────────────
        if self.lifecycle.is_hpa_active():
            action = self.hpa.decide(obs)
            controller = "HPA"
        else:
            action = self.rl_agent.predict(obs)
            controller = "RL"

        # ── Apply scaling ─────────────────────────────────────────────────────
        delta = {0: -1, 1: 0, 2: 1}[action]
        new_replicas = self.scaler.apply_delta(delta)

        # ── Compute reward ────────────────────────────────────────────────────
        reward = self._compute_reward(raw, action)

        # ── Record trajectories ───────────────────────────────────────────────
        if self._prev_obs is not None:
            self.hpa.record_transition(self._prev_obs, action, reward, obs)

        if self.lifecycle.is_rl_active():
            self.rl_agent.record_reward(reward)
            self.retrain_monitor.record(reward)

        # ── Log metrics ───────────────────────────────────────────────────────
        self.metrics_logger.log_step(
            state=state.value,
            controller=controller,
            action=action,
            reward=reward,
            pattern=pattern,
            **raw,
        )

        # ── Lifecycle transitions ─────────────────────────────────────────────
        self.lifecycle.check_and_transition(
            hpa_buffer_size=self.hpa.buffer_size(),
            hpa_avg_reward=self.hpa.get_avg_reward(),
            rl_avg_reward=self.rl_agent.get_avg_reward(),
            retraining_monitor=self.retrain_monitor,
        )

        self._prev_obs = obs
        self._prev_reward = reward

        logger.info(
            "[%s/%s] action=%d replicas=%d p95=%.0fms reward=%.3f pattern=%s",
            state.value, controller, action, new_replicas,
            raw["p95_latency_ms"], reward, pattern,
        )

    # ── Run loop ──────────────────────────────────────────────────────────────

    def run(self):
        """Start the blocking control loop."""
        self._running = True
        logger.info(
            "Orchestrator started. Interval=%ds, Namespace=%s, Deployment=%s",
            settings.CONTROL_INTERVAL_SECONDS,
            settings.K8S_NAMESPACE,
            settings.K8S_DEPLOYMENT_NAME,
        )
        while self._running:
            try:
                self.tick()
            except KeyboardInterrupt:
                logger.info("Orchestrator interrupted by user.")
                break
            except Exception as exc:
                logger.error("Tick error (continuing): %s", exc, exc_info=True)

            time.sleep(settings.CONTROL_INTERVAL_SECONDS)

        # Persist everything on exit
        self.hpa.save_buffer()
        self.rl_agent.save()
        logger.info("Orchestrator stopped. Models saved.")

    def stop(self):
        self._running = False
