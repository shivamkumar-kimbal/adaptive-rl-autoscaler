.PHONY: setup install build up down logs eval plots clean train help

NAMESPACE ?= rl-autoscaler
METRICS_CSV ?= /tmp/rl_autoscaler/logs/metrics.csv

help:
	@echo "AWARE-inspired RL Autoscaler"
	@echo ""
	@echo "  make install       Install Python dependencies"
	@echo "  make build         Build Docker images"
	@echo "  make up            Start full stack (docker-compose)"
	@echo "  make down          Stop docker-compose stack"
	@echo "  make logs          Tail autoscaler logs"
	@echo "  make eval          Run HPA vs RL evaluation"
	@echo "  make plots         Generate result plots"
	@echo "  make train         Run training script offline"
	@echo "  make setup         Setup Minikube cluster (full k8s)"
	@echo "  make deploy        Rebuild + redeploy to Minikube"
	@echo "  make experiment    Run full timed experiment"
	@echo "  make clean         Remove generated files"

install:
	pip install -r requirements.txt

build:
	docker build -t workload-app:latest ./workload
	docker build -t rl-autoscaler:latest ./autoscaler

up:
	docker-compose up --build -d
	@echo ""
	@echo "Stack running:"
	@echo "  Workload:   http://localhost:8000"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Grafana:    http://localhost:3000  (admin/admin)"
	@echo "  Locust:     http://localhost:8089"
	@echo ""
	@echo "Tail logs: make logs"

down:
	docker-compose down

logs:
	docker-compose logs -f autoscaler

eval:
	python -m evaluation.compare_baselines --csv $(METRICS_CSV)

plots:
	python -m evaluation.plot_results --csv $(METRICS_CSV)

train:
	@echo "Offline training from saved replay buffer..."
	PYTHONPATH=. python -c "\
import pickle, os; \
from autoscaler.agents.ppo_agent import PPOAgent; \
from autoscaler.config import settings; \
buf_path = settings.REPLAY_BUFFER_PATH; \
assert os.path.exists(buf_path), f'Buffer not found: {buf_path}'; \
with open(buf_path, 'rb') as f: buf = pickle.load(f); \
agent = PPOAgent(); \
agent.train_offline(buf); \
agent.save(); \
print('Training complete.')"

setup:
	bash scripts/setup_minikube.sh

deploy:
	bash scripts/deploy.sh

experiment:
	bash scripts/run_experiment.sh

clean:
	rm -rf /tmp/rl_autoscaler
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
