#!/usr/bin/env bash
# Run a full bootstrap → train → evaluate experiment.
# Can be run locally (against a running docker-compose stack)
# or pointed at a Minikube deployment via env vars.

set -euo pipefail

NAMESPACE="${K8S_NAMESPACE:-rl-autoscaler}"
METRICS_CSV="${METRICS_CSV_PATH:-/tmp/rl_autoscaler/logs/metrics.csv}"
DURATION_MINUTES="${EXPERIMENT_DURATION_MINUTES:-30}"

echo "========================================"
echo " AWARE RL Autoscaler Experiment Runner"
echo "========================================"
echo " Duration:    ${DURATION_MINUTES} minutes"
echo " Namespace:   ${NAMESPACE}"
echo " Metrics CSV: ${METRICS_CSV}"
echo "========================================"

# 1. Start load generator (Locust headless)
echo ""
echo "==> Starting load generator (${DURATION_MINUTES}m)..."
WORKLOAD_URL="${WORKLOAD_URL:-http://localhost:8000}"

locust -f load_generator/locustfile.py \
  --host="$WORKLOAD_URL" \
  --headless \
  -u 30 -r 3 \
  --run-time "${DURATION_MINUTES}m" \
  --csv /tmp/rl_autoscaler/logs/locust \
  &
LOCUST_PID=$!

echo "  Locust PID: $LOCUST_PID"
echo "  Autoscaler should be running separately (docker-compose or k8s)."
echo ""
echo "==> Experiment running... waiting ${DURATION_MINUTES} minutes."
sleep $(( DURATION_MINUTES * 60 ))

# 2. Kill load generator
echo "==> Stopping load generator..."
kill "$LOCUST_PID" 2>/dev/null || true

# 3. Run evaluation
echo ""
echo "==> Running baseline comparison..."
python -m evaluation.compare_baselines --csv "$METRICS_CSV"

echo ""
echo "==> Generating plots..."
python -m evaluation.plot_results --csv "$METRICS_CSV"

echo ""
echo "==> Experiment complete. Results in:"
echo "    CSV:   $METRICS_CSV"
echo "    Plots: $(dirname $METRICS_CSV)/../plots/"
