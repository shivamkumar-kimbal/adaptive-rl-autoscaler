"""
PPO-based RL agent wrapping stable-baselines3.

Supports:
  - Offline pre-training from HPA replay buffer (behavioural cloning → PPO fine-tune)
  - Online training against the live Kubernetes environment
  - Model persistence (save / load)
  - Reward tracking for lifecycle transitions
"""
import logging
import os
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from autoscaler.config import settings

logger = logging.getLogger(__name__)

Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


class _ReplayBufferEnv:
    """
    Minimal fake environment backed by a replay buffer.
    Used so stable-baselines3 can call model.learn() offline.
    """

    def __init__(self, transitions: List[Transition], obs_shape=(5,), n_actions=3):
        self.transitions = transitions
        self._idx = 0
        self._current_obs = transitions[0][0] if transitions else np.zeros(obs_shape)

        try:
            import gymnasium as gym
            from gymnasium import spaces
        except ImportError:
            import gym
            from gym import spaces

        self.observation_space = spaces.Box(
            low=np.zeros(obs_shape, dtype=np.float32),
            high=np.ones(obs_shape, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(n_actions)
        self.reward_range = (-np.inf, np.inf)
        self.spec = None
        self.metadata = {}
        self.np_random = np.random.default_rng()

    def reset(self, **kwargs):
        self._idx = 0
        self._current_obs = self.transitions[0][0].copy()
        return self._current_obs, {}

    def step(self, action):
        if self._idx >= len(self.transitions):
            self._idx = 0
        obs, _, reward, next_obs, done = self.transitions[self._idx]
        self._idx += 1
        self._current_obs = next_obs.copy()
        truncated = self._idx >= len(self.transitions)
        return next_obs, reward, done, truncated, {}

    def render(self):
        pass


class PPOAgent:
    """PPO autoscaler agent with offline + online training support."""

    def __init__(
        self,
        env=None,
        model_path: Optional[str] = None,
    ):
        self._env = env
        self._model = None
        self._reward_history: deque = deque(maxlen=settings.REWARD_WINDOW)

        if model_path and os.path.exists(model_path + ".zip"):
            self.load(model_path)

    # ── Model construction ────────────────────────────────────────────────────

    def _build_model(self, env):
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env

        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=settings.PPO_LEARNING_RATE,
            n_steps=settings.PPO_N_STEPS,
            batch_size=settings.PPO_BATCH_SIZE,
            n_epochs=settings.PPO_N_EPOCHS,
            gamma=settings.PPO_GAMMA,
            gae_lambda=settings.PPO_GAE_LAMBDA,
            clip_range=settings.PPO_CLIP_RANGE,
            verbose=1,
            tensorboard_log=os.path.join(settings.LOG_DIR, "tensorboard"),
        )
        return model

    # ── Offline training ──────────────────────────────────────────────────────

    def train_offline(
        self,
        replay_buffer: List[Transition],
        total_timesteps: int = settings.OFFLINE_TRAIN_TIMESTEPS,
        bc_epochs: int = settings.BC_N_EPOCHS,
    ) -> None:
        """
        Pre-train on HPA-collected trajectories.

        Phase 1: Behavioural cloning (supervised, fast convergence).
        Phase 2: PPO fine-tuning on the replay buffer (improves on HPA).
        """
        if not replay_buffer:
            logger.warning("Offline training called with empty replay buffer — skipping.")
            return

        logger.info(
            "Offline training: BC (%d epochs) + PPO (%d steps) on %d transitions",
            bc_epochs, total_timesteps, len(replay_buffer),
        )

        # ── Phase 1: Behavioural cloning ──────────────────────────────────────
        self._behavioural_clone(replay_buffer, n_epochs=bc_epochs)

        # ── Phase 2: PPO fine-tune on replay buffer env ───────────────────────
        replay_env = _ReplayBufferEnv(replay_buffer)
        if self._model is None:
            self._model = self._build_model(replay_env)
        else:
            self._model.set_env(replay_env)

        self._model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
        logger.info("Offline training complete.")

    def _behavioural_clone(
        self, transitions: List[Transition], n_epochs: int = 10
    ) -> None:
        """
        Supervised behavioural cloning: minimise cross-entropy(π(s), a_hpa).
        Builds a lightweight MLP policy from scratch using torch directly
        so we don't need the `imitation` library as a hard dependency.
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim

        obs_arr = np.array([t[0] for t in transitions], dtype=np.float32)
        act_arr = np.array([t[1] for t in transitions], dtype=np.int64)

        obs_t = torch.tensor(obs_arr)
        act_t = torch.tensor(act_arr)

        # Simple 2-layer MLP
        policy_net = nn.Sequential(
            nn.Linear(5, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 3),
        )
        optimiser = optim.Adam(policy_net.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        dataset = torch.utils.data.TensorDataset(obs_t, act_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for batch_obs, batch_act in loader:
                logits = policy_net(batch_obs)
                loss = loss_fn(logits, batch_act)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item()
            logger.debug("BC epoch %d/%d — loss: %.4f", epoch + 1, n_epochs, epoch_loss)

        # Warm-start the SB3 PPO policy with the BC weights
        if self._model is not None:
            try:
                sb3_policy = self._model.policy
                with torch.no_grad():
                    # Map BC weights → SB3 MlpPolicy mlp_extractor + action_net
                    state = sb3_policy.state_dict()
                    bc_state = policy_net.state_dict()
                    # Layer mapping (SB3 MlpPolicy internal names)
                    key_map = {
                        "0.weight": "mlp_extractor.policy_net.0.weight",
                        "0.bias": "mlp_extractor.policy_net.0.bias",
                        "2.weight": "mlp_extractor.policy_net.2.weight",
                        "2.bias": "mlp_extractor.policy_net.2.bias",
                        "4.weight": "action_net.weight",
                        "4.bias": "action_net.bias",
                    }
                    for bc_key, sb3_key in key_map.items():
                        if sb3_key in state and bc_key in bc_state:
                            if state[sb3_key].shape == bc_state[bc_key].shape:
                                state[sb3_key] = bc_state[bc_key]
                    sb3_policy.load_state_dict(state)
                logger.info("BC weights transferred to SB3 PPO policy.")
            except Exception as exc:
                logger.warning("BC weight transfer failed (non-fatal): %s", exc)

    # ── Online training ───────────────────────────────────────────────────────

    def train_online(
        self,
        env=None,
        total_timesteps: int = settings.ONLINE_TRAIN_TIMESTEPS,
    ) -> None:
        """Standard PPO online training against the live environment."""
        live_env = env or self._env
        if live_env is None:
            raise ValueError("No environment provided for online training.")

        if self._model is None:
            self._model = self._build_model(live_env)
        else:
            self._model.set_env(live_env)

        logger.info("Online training: %d timesteps", total_timesteps)
        self._model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
        logger.info("Online training complete.")

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, obs: np.ndarray) -> int:
        if self._model is None:
            logger.warning("Model not trained yet — defaulting to maintain (action=1).")
            return 1
        action, _ = self._model.predict(obs, deterministic=True)
        return int(action)

    def record_reward(self, reward: float) -> None:
        self._reward_history.append(reward)

    def get_avg_reward(self) -> float:
        if not self._reward_history:
            return 0.0
        return float(np.mean(list(self._reward_history)))

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = None) -> None:
        path = path or os.path.join(settings.MODEL_DIR, "ppo_autoscaler")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if self._model:
            self._model.save(path)
            logger.info("Model saved to %s", path)

    def load(self, path: str = None) -> None:
        from stable_baselines3 import PPO
        path = path or os.path.join(settings.MODEL_DIR, "ppo_autoscaler")
        self._model = PPO.load(path)
        logger.info("Model loaded from %s", path)
