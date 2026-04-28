"""
HPA vs RL autoscaler comparison.

Reads the metrics CSV produced by the autoscaler controller and
computes side-by-side statistics for each controller phase.

Usage:
  python -m evaluation.compare_baselines
  python -m evaluation.compare_baselines --csv /path/to/metrics.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

from autoscaler.config import settings


def load_metrics(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        print(f"ERROR: metrics file not found: {csv_path}")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


def compute_stats(df: pd.DataFrame, slo_ms: float = settings.SLO_LATENCY_MS) -> dict:
    """Compute comparison metrics for a slice of the dataframe."""
    if df.empty:
        return {}

    p95_arr = df["p95_latency_ms"].values
    replicas_arr = df["replicas"].values
    rewards_arr = df["reward"].values

    slo_violations = np.mean(p95_arr > slo_ms) * 100  # percentage

    return {
        "n_steps": len(df),
        "avg_p95_latency_ms": round(float(np.mean(p95_arr)), 1),
        "p99_latency_ms": round(float(np.percentile(p95_arr, 99)), 1),
        "slo_violation_pct": round(float(slo_violations), 2),
        "avg_replicas": round(float(np.mean(replicas_arr)), 2),
        "max_replicas": int(np.max(replicas_arr)),
        "avg_reward": round(float(np.mean(rewards_arr)), 4),
        "cost_score": round(float(np.mean(replicas_arr)) / settings.MAX_REPLICAS, 3),
    }


def print_comparison_table(hpa_stats: dict, rl_stats: dict):
    """Print a markdown-style comparison table."""
    metrics = [
        ("Steps recorded",       "n_steps",              "",       False),
        ("Avg p95 latency (ms)", "avg_p95_latency_ms",   "ms",     True),
        ("p99 latency (ms)",     "p99_latency_ms",        "ms",     True),
        ("SLO violation (%)",    "slo_violation_pct",    "%",      True),
        ("Avg replicas",         "avg_replicas",          "",       True),
        ("Max replicas",         "max_replicas",          "",       True),
        ("Avg reward",           "avg_reward",            "",       False),
        ("Cost score (0-1)",     "cost_score",            "",       True),
    ]

    print("\n" + "=" * 65)
    print(f"{'Metric':<30} {'HPA':>12} {'RL':>12} {'Better':>8}")
    print("=" * 65)
    for label, key, unit, lower_is_better in metrics:
        h = hpa_stats.get(key, "N/A")
        r = rl_stats.get(key, "N/A")
        if isinstance(h, (int, float)) and isinstance(r, (int, float)):
            if lower_is_better:
                winner = "RL ✓" if r < h else ("HPA ✓" if h < r else "tie")
            else:
                winner = "RL ✓" if r > h else ("HPA ✓" if h > r else "tie")
        else:
            winner = ""
        print(f"{label:<30} {str(h)+unit:>12} {str(r)+unit:>12} {winner:>8}")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare HPA vs RL autoscaler metrics")
    parser.add_argument("--csv", default=settings.METRICS_CSV_PATH,
                        help="Path to metrics CSV file")
    parser.add_argument("--slo", type=float, default=settings.SLO_LATENCY_MS,
                        help="SLO latency threshold in ms")
    args = parser.parse_args()

    df = load_metrics(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")

    hpa_df = df[df["controller"] == "HPA"]
    rl_df = df[df["controller"] == "RL"]

    print(f"HPA steps: {len(hpa_df)}  |  RL steps: {len(rl_df)}")

    if hpa_df.empty:
        print("WARNING: No HPA data found — run in INITIALIZED/OFFLINE_TRAINING state first.")
    if rl_df.empty:
        print("WARNING: No RL data found — system has not reached ONLINE_TRAINING yet.")

    hpa_stats = compute_stats(hpa_df, slo_ms=args.slo)
    rl_stats = compute_stats(rl_df, slo_ms=args.slo)

    print_comparison_table(hpa_stats, rl_stats)

    # Trigger plot generation
    try:
        from evaluation.plot_results import plot_all
        plot_all(df)
    except Exception as exc:
        print(f"Plotting skipped: {exc}")


if __name__ == "__main__":
    main()
