#!/usr/bin/env bash
# Rebuild images and roll out updates to an existing Minikube cluster.

set -euo pipefail

NAMESPACE="rl-autoscaler"

echo "==> Rebuilding images in Minikube Docker context..."
eval "$(minikube docker-env)"

docker build -t workload-app:latest ./workload
docker build -t rl-autoscaler:latest ./autoscaler

echo "==> Rolling out workload..."
kubectl rollout restart deployment/workload-app -n "$NAMESPACE"

echo "==> Rolling out RL autoscaler..."
kubectl rollout restart deployment/rl-autoscaler -n "$NAMESPACE"

echo "==> Waiting for rollout..."
kubectl rollout status deployment/workload-app -n "$NAMESPACE" --timeout=120s
kubectl rollout status deployment/rl-autoscaler -n "$NAMESPACE" --timeout=120s

echo "==> Deploy complete."
