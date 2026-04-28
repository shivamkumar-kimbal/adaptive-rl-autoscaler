#!/usr/bin/env bash
# Setup Minikube cluster + deploy full AWARE RL autoscaler stack.
# Prerequisites: minikube, kubectl, docker installed.

set -euo pipefail

NAMESPACE="rl-autoscaler"

echo "==> Starting Minikube..."
minikube start \
  --cpus=4 \
  --memory=6144 \
  --driver=docker \
  --addons=metrics-server \
  --addons=ingress

echo "==> Enabling kube-state-metrics..."
kubectl apply -f https://github.com/kubernetes/kube-state-metrics/releases/download/v2.10.0/kube-state-metrics-deployment.yaml 2>/dev/null || true

echo "==> Pointing Docker to Minikube registry..."
eval "$(minikube docker-env)"

echo "==> Building workload image..."
docker build -t workload-app:latest ./workload

echo "==> Building autoscaler image..."
docker build -t rl-autoscaler:latest ./autoscaler

echo "==> Creating namespace and applying manifests..."
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/autoscaler/rbac.yaml
kubectl apply -f k8s/workload/deployment.yaml
kubectl apply -f k8s/workload/service.yaml
kubectl apply -f k8s/monitoring/prometheus-config.yaml
kubectl apply -f k8s/monitoring/prometheus-deployment.yaml
kubectl apply -f k8s/monitoring/grafana-deployment.yaml
kubectl apply -f k8s/autoscaler/deployment.yaml
kubectl apply -f k8s/autoscaler/service.yaml

# Apply Grafana dashboard ConfigMap
kubectl create configmap grafana-dashboard-json \
  --from-file=rl-autoscaler.json=k8s/monitoring/grafana-dashboard.json \
  -n "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

echo ""
echo "==> Stack deployed! Services:"
echo "    Workload:   $(minikube service workload-app-external -n $NAMESPACE --url 2>/dev/null)"
echo "    Prometheus: $(minikube service prometheus-service -n $NAMESPACE --url 2>/dev/null)"
echo "    Grafana:    $(minikube service grafana-service -n $NAMESPACE --url 2>/dev/null)  (admin/admin)"
echo ""
echo "==> Wait for pods to be ready:"
echo "    kubectl get pods -n $NAMESPACE -w"
