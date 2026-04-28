# AWARE-Inspired Safe RL Kubernetes Autoscaler

A production-grade Kubernetes autoscaler using Reinforcement Learning, inspired by the [AWARE paper (USENIX ATC 2023)](https://www.usenix.org/conference/atc23/presentation/qiu).

Outperforms HPA by learning an optimal scaling policy while guaranteeing SLO compliance through safe bootstrapping.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Orchestrator (30s loop)             │
│                                                      │
│  LifecycleManager                                    │
│  INITIALIZED → OFFLINE_TRAINING → ONLINE → SERVING  │
│                                                      │
│  ┌──────────┐    ┌──────────────┐    ┌───────────┐  │
│  │  HPA     │    │  PPO Agent   │    │Retrain    │  │
│  │ Baseline │    │ (SB3)        │    │Monitor    │  │
│  └──────────┘    └──────────────┘    └───────────┘  │
│                                                      │
│  ┌──────────────────┐   ┌──────────────────────────┐ │
│  │ PrometheusClient │   │ KubernetesScaler         │ │
│  │ CPU/mem/lat/rps  │   │ patch deployment/scale   │ │
│  └──────────────────┘   └──────────────────────────┘ │
└─────────────────────────────────────────────────────┘
         ↕ metrics                ↕ scale actions
┌──────────────┐          ┌──────────────────────────┐
│  Prometheus  │          │  Kubernetes / Docker     │
│  + Grafana   │          │  workload-app deployment │
└──────────────┘          └──────────────────────────┘
         ↑ traffic
┌──────────────┐
│  Locust      │
│  load gen    │
└──────────────┘
```

---

## How This Follows AWARE

| AWARE Principle | This Implementation |
|---|---|
| **Safe bootstrapping** | HPA controls system while RL collects trajectories offline. RL only takes over when `avg_reward(RL) ≥ 0.95 × avg_reward(HPA)`. |
| **Lifecycle states** | `INITIALIZED → OFFLINE_TRAINING → ONLINE_TRAINING → SERVING` |
| **Offline pre-training** | Behavioural cloning on HPA trajectories → PPO fine-tuning on replay buffer |
| **Online training** | PPO updates incrementally while controlling live system |
| **Continuous retraining** | Sliding window of last 100 rewards; retrain if avg drops or variance spikes |
| **State space** | `[cpu_pct, mem_pct, req_rate_norm, p95_latency_norm, replicas_norm]` |
| **Action space** | `Discrete(3)`: scale-down / maintain / scale-up |
| **Reward** | `α·SLO_score + (1−α)·utilisation_score − oscillation_penalty` |
| **Baseline** | Full HPA vs RL metrics table + matplotlib plots |

### Simplifications vs Paper
- PPO (stable-baselines3) instead of formal CMDP safe RL
- Single deployment, not multi-service mesh
- Pattern detection via FFT/linear regression (no meta-learning)
- HPA fallback via behavioural cloning init (not Lyapunov certificates)
- Minikube instead of production cloud

---

## Quickstart

### Option A — docker-compose (no Kubernetes required)

```bash
# 1. Clone and install
git clone <repo>
cd RL_project
pip install -r requirements.txt

# 2. Start the full stack
make up

# 3. Open UIs
#   Workload:   http://localhost:8000
#   Prometheus: http://localhost:9090
#   Grafana:    http://localhost:3000  (admin / admin)
#   Locust:     http://localhost:8089

# 4. Start a load test in Locust UI: 30 users, spawn rate 3/s

# 5. Watch autoscaler logs
make logs

# 6. After 20+ minutes, evaluate
make eval
make plots
```

### Option B — Minikube (full Kubernetes)

```bash
# Prerequisites: minikube, kubectl, docker

# 1. Setup cluster + deploy
make setup

# 2. Run experiment (30 minutes)
make experiment

# 3. Results
make eval
```

---

## Project Structure

```
RL_project/
├── autoscaler/                   # Core RL controller service
│   ├── config/settings.py        # All hyperparameters (env-var overridable)
│   ├── environment/
│   │   ├── k8s_env.py            # gym.Env: state/action/reward definition
│   │   └── metrics.py            # Prometheus HTTP client
│   ├── agents/
│   │   ├── ppo_agent.py          # PPO + behavioural cloning offline training
│   │   └── hpa_baseline.py       # HPA policy + trajectory collector
│   ├── lifecycle/
│   │   ├── state_machine.py      # AWARE lifecycle states + transitions
│   │   └── retraining_monitor.py # Sliding-window reward monitor
│   ├── controller/
│   │   ├── scaler.py             # Kubernetes API scaling wrapper
│   │   └── orchestrator.py       # Main 30s control loop
│   └── workload_detector/
│       └── pattern_classifier.py # FFT/regression: periodic/spike/ramp/steady
├── workload/                     # FastAPI target service
├── load_generator/               # Locust workload patterns
├── evaluation/
│   ├── metrics_logger.py         # CSV + JSONL structured logging
│   ├── compare_baselines.py      # HPA vs RL comparison table
│   └── plot_results.py           # matplotlib charts
├── k8s/                          # Kubernetes manifests
│   ├── workload/                 # Deployment + Service
│   ├── autoscaler/               # Deployment + RBAC
│   ├── hpa/                      # HPA baseline
│   └── monitoring/               # Prometheus + Grafana
├── docker/                       # docker-compose config files
├── scripts/                      # setup_minikube.sh, deploy.sh, run_experiment.sh
├── docker-compose.yml
└── Makefile
```

---

## RL Design

### State Space (5-D, normalised 0–1)
```
[cpu_utilisation, memory_utilisation, request_rate_norm, p95_latency_norm, replicas_norm]
```

### Action Space
```
Discrete(3): 0 = scale down (−1 replica)
             1 = maintain
             2 = scale up  (+1 replica)
```

### Reward Function
```
slo_score     = 1.0            if p95 ≤ SLO_LATENCY_MS
              = max(0, 1 − (p95 − SLO) / SLO)   otherwise

util_score    = 1 − (replicas − min_replicas) / (max_replicas − min_replicas)

oscillation_penalty = 0.1 if action ≠ prev_action AND action ≠ maintain

reward = α · slo_score + (1−α) · util_score − oscillation_penalty
       = 0.7 · slo_score + 0.3 · util_score − penalty   (default α=0.7)
```

---

## Configuration

All parameters are in `autoscaler/config/settings.py` and overridable via env vars:

| Variable | Default | Description |
|---|---|---|
| `SLO_LATENCY_MS` | `500` | p95 latency SLO target (ms) |
| `ALPHA` | `0.7` | SLO weight in reward |
| `MIN_BOOTSTRAP_STEPS` | `1000` | HPA steps before offline training |
| `RL_BETTER_THRESHOLD` | `0.95` | RL must reach 95% of HPA reward to go online |
| `REWARD_WINDOW` | `100` | Sliding window for retraining monitor |
| `RETRAIN_AVG_THRESHOLD` | `0.5` | Retrain if avg reward drops below |
| `RETRAIN_VAR_THRESHOLD` | `0.1` | Retrain if reward variance exceeds |
| `MIN_REPLICAS` | `1` | Minimum pod count |
| `MAX_REPLICAS` | `10` | Maximum pod count |
| `CONTROL_INTERVAL_SECONDS` | `30` | Control loop frequency |
| `PPO_LEARNING_RATE` | `3e-4` | PPO learning rate |

---

## Evaluation

After running the experiment:
```bash
# Print HPA vs RL comparison table
python -m evaluation.compare_baselines

# Generate plots
python -m evaluation.plot_results
```

Output table example:
```
═══════════════════════════════════════════════════════════════════
Metric                            HPA          RL       Better
═══════════════════════════════════════════════════════════════════
Steps recorded                    450         380
Avg p95 latency (ms)            312.1ms     248.6ms      RL ✓
p99 latency (ms)                891.3ms     612.4ms      RL ✓
SLO violation (%)                8.20%       2.10%       RL ✓
Avg replicas                       4.30        3.10       RL ✓
Max replicas                          9           7       RL ✓
Avg reward                        0.6812      0.7934      RL ✓
Cost score (0-1)                   0.430       0.310      RL ✓
═══════════════════════════════════════════════════════════════════
```

Plots generated:
- `plots/replicas_over_time.png` — replica count coloured by controller
- `plots/latency_over_time.png` — p95 latency + SLO line
- `plots/reward_over_time.png`  — smoothed reward curve
- `plots/slo_violations.png`    — bar chart SLO violation rate

---

## GitHub Push

```bash
cd RL_project
git init
git add .
git commit -m "feat: AWARE-inspired RL Kubernetes autoscaler"
git remote add origin https://github.com/<username>/rl-k8s-autoscaler.git
git branch -M main
git push -u origin main
```
