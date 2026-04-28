"""
Central configuration for the AWARE-inspired RL autoscaler.
All values are overridable via environment variables.
"""
import os


# ── SLO ────────────────────────────────────────────────────────────────────────
SLO_LATENCY_MS: float = float(os.getenv("SLO_LATENCY_MS", "500"))
# Weight for SLO score vs utilisation score in reward (0-1)
ALPHA: float = float(os.getenv("ALPHA", "0.7"))

# ── Scaling bounds ──────────────────────────────────────────────────────────────
MIN_REPLICAS: int = int(os.getenv("MIN_REPLICAS", "1"))
MAX_REPLICAS: int = int(os.getenv("MAX_REPLICAS", "10"))
# Cooldown between scale actions (seconds)
SCALE_COOLDOWN_SECONDS: int = int(os.getenv("SCALE_COOLDOWN_SECONDS", "30"))

# ── Control loop ────────────────────────────────────────────────────────────────
CONTROL_INTERVAL_SECONDS: int = int(os.getenv("CONTROL_INTERVAL_SECONDS", "30"))

# ── Kubernetes ──────────────────────────────────────────────────────────────────
K8S_NAMESPACE: str = os.getenv("K8S_NAMESPACE", "rl-autoscaler")
K8S_DEPLOYMENT_NAME: str = os.getenv("K8S_DEPLOYMENT_NAME", "workload-app")
# Set to "incluster" when running inside a pod; "local" uses ~/.kube/config
K8S_CONFIG_MODE: str = os.getenv("K8S_CONFIG_MODE", "local")

# ── Prometheus ──────────────────────────────────────────────────────────────────
PROMETHEUS_URL: str = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
# PromQL selectors – adjust to match your deployment labels
PROMETHEUS_JOB_WORKLOAD: str = os.getenv("PROMETHEUS_JOB_WORKLOAD", "workload-app")
PROMETHEUS_CONTAINER_NAME: str = os.getenv("PROMETHEUS_CONTAINER_NAME", "workload-app")

# ── HPA baseline ───────────────────────────────────────────────────────────────
HPA_CPU_SCALE_UP_THRESHOLD: float = float(os.getenv("HPA_CPU_SCALE_UP_THRESHOLD", "0.7"))
# Scale down when CPU drops below (threshold - hysteresis)
HPA_CPU_HYSTERESIS: float = float(os.getenv("HPA_CPU_HYSTERESIS", "0.2"))

# ── AWARE bootstrapping ────────────────────────────────────────────────────────
# Minimum HPA-controlled steps to collect before offline training starts
MIN_BOOTSTRAP_STEPS: int = int(os.getenv("MIN_BOOTSTRAP_STEPS", "1000"))
# RL avg_reward must be >= this fraction of HPA avg_reward to go ONLINE
RL_BETTER_THRESHOLD: float = float(os.getenv("RL_BETTER_THRESHOLD", "0.95"))

# ── Offline training ───────────────────────────────────────────────────────────
OFFLINE_TRAIN_TIMESTEPS: int = int(os.getenv("OFFLINE_TRAIN_TIMESTEPS", "10000"))
BC_N_EPOCHS: int = int(os.getenv("BC_N_EPOCHS", "10"))

# ── Online training ────────────────────────────────────────────────────────────
ONLINE_TRAIN_TIMESTEPS: int = int(os.getenv("ONLINE_TRAIN_TIMESTEPS", "50000"))
# After this many online steps, check if SERVING transition applies
ONLINE_EVAL_INTERVAL: int = int(os.getenv("ONLINE_EVAL_INTERVAL", "200"))

# ── Retraining monitor ─────────────────────────────────────────────────────────
REWARD_WINDOW: int = int(os.getenv("REWARD_WINDOW", "100"))
RETRAIN_AVG_THRESHOLD: float = float(os.getenv("RETRAIN_AVG_THRESHOLD", "0.5"))
RETRAIN_VAR_THRESHOLD: float = float(os.getenv("RETRAIN_VAR_THRESHOLD", "0.1"))

# ── PPO hyperparameters ────────────────────────────────────────────────────────
PPO_LEARNING_RATE: float = float(os.getenv("PPO_LEARNING_RATE", "3e-4"))
PPO_N_STEPS: int = int(os.getenv("PPO_N_STEPS", "2048"))
PPO_BATCH_SIZE: int = int(os.getenv("PPO_BATCH_SIZE", "64"))
PPO_N_EPOCHS: int = int(os.getenv("PPO_N_EPOCHS", "10"))
PPO_GAMMA: float = float(os.getenv("PPO_GAMMA", "0.99"))
PPO_GAE_LAMBDA: float = float(os.getenv("PPO_GAE_LAMBDA", "0.95"))
PPO_CLIP_RANGE: float = float(os.getenv("PPO_CLIP_RANGE", "0.2"))

# ── Model persistence ──────────────────────────────────────────────────────────
MODEL_DIR: str = os.getenv("MODEL_DIR", "/tmp/rl_autoscaler/models")
REPLAY_BUFFER_PATH: str = os.getenv("REPLAY_BUFFER_PATH", "/tmp/rl_autoscaler/replay_buffer.pkl")

# ── Evaluation / logging ───────────────────────────────────────────────────────
LOG_DIR: str = os.getenv("LOG_DIR", "/tmp/rl_autoscaler/logs")
METRICS_CSV_PATH: str = os.getenv("METRICS_CSV_PATH", "/tmp/rl_autoscaler/logs/metrics.csv")

# ── Workload pattern detector ──────────────────────────────────────────────────
PATTERN_WINDOW: int = int(os.getenv("PATTERN_WINDOW", "50"))
PATTERN_SPIKE_THRESHOLD: float = float(os.getenv("PATTERN_SPIKE_THRESHOLD", "2.0"))
PATTERN_PERIODIC_FREQ_THRESHOLD: float = float(
    os.getenv("PATTERN_PERIODIC_FREQ_THRESHOLD", "0.05")
)
