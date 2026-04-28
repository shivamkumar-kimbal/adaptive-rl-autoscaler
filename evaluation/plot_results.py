"""
Visualisation for HPA vs RL autoscaler comparison.

Generates:
  plots/replicas_over_time.png   — replica count timeline coloured by controller
  plots/latency_over_time.png    — p95 latency + SLO threshold line
  plots/reward_over_time.png     — cumulative reward curve
  plots/slo_violations.png       — SLO violation rate per phase

Usage:
  python -m evaluation.plot_results
  python -m evaluation.plot_results --csv /path/to/metrics.csv
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from autoscaler.config import settings

PLOT_DIR = os.path.join(os.path.dirname(settings.METRICS_CSV_PATH), "..", "plots")
CONTROLLER_COLORS = {"HPA": "#4c72b0", "RL": "#dd8452"}


def _ensure_plot_dir():
    os.makedirs(PLOT_DIR, exist_ok=True)


def plot_replicas_over_time(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(14, 4))
    for ctrl, color in CONTROLLER_COLORS.items():
        subset = df[df["controller"] == ctrl]
        ax.scatter(subset["timestamp"], subset["replicas"],
                   c=color, s=8, label=ctrl, alpha=0.7)
    ax.set_xlabel("Time")
    ax.set_ylabel("Replicas")
    ax.set_title("Replica Count Over Time — HPA vs RL")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "replicas_over_time.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_latency_over_time(df: pd.DataFrame, slo_ms: float = settings.SLO_LATENCY_MS):
    fig, ax = plt.subplots(figsize=(14, 4))
    for ctrl, color in CONTROLLER_COLORS.items():
        subset = df[df["controller"] == ctrl]
        ax.scatter(subset["timestamp"], subset["p95_latency_ms"],
                   c=color, s=8, label=ctrl, alpha=0.7)
    ax.axhline(slo_ms, color="red", linestyle="--", linewidth=1.5,
               label=f"SLO ({slo_ms:.0f}ms)")
    ax.set_xlabel("Time")
    ax.set_ylabel("p95 Latency (ms)")
    ax.set_title("p95 Request Latency — HPA vs RL")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "latency_over_time.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_reward_over_time(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(14, 4))
    window = 20
    for ctrl, color in CONTROLLER_COLORS.items():
        subset = df[df["controller"] == ctrl].sort_values("timestamp")
        if len(subset) < window:
            continue
        smoothed = subset["reward"].rolling(window=window, min_periods=1).mean()
        ax.plot(subset["timestamp"], smoothed, color=color, label=f"{ctrl} (rolling mean)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Reward")
    ax.set_title("Smoothed Reward Over Time — HPA vs RL")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "reward_over_time.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_slo_violations(df: pd.DataFrame, slo_ms: float = settings.SLO_LATENCY_MS):
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = []
    violation_rates = []
    colors = []
    for ctrl, color in CONTROLLER_COLORS.items():
        subset = df[df["controller"] == ctrl]
        if subset.empty:
            continue
        rate = (subset["p95_latency_ms"] > slo_ms).mean() * 100
        categories.append(ctrl)
        violation_rates.append(rate)
        colors.append(color)

    bars = ax.bar(categories, violation_rates, color=colors, width=0.4, edgecolor="black")
    for bar, val in zip(bars, violation_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=12)
    ax.set_ylabel("SLO Violation Rate (%)")
    ax.set_title(f"SLO Violation Rate (threshold: {slo_ms:.0f}ms)")
    ax.set_ylim(0, max(violation_rates or [10]) * 1.3 + 2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "slo_violations.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_all(df: pd.DataFrame):
    _ensure_plot_dir()
    plot_replicas_over_time(df)
    plot_latency_over_time(df)
    plot_reward_over_time(df)
    plot_slo_violations(df)
    print(f"\nAll plots saved to: {os.path.abspath(PLOT_DIR)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=settings.METRICS_CSV_PATH)
    parser.add_argument("--slo", type=float, default=settings.SLO_LATENCY_MS)
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"ERROR: {args.csv} not found.")
        return

    df = pd.read_csv(args.csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    plot_all(df)


if __name__ == "__main__":
    main()
