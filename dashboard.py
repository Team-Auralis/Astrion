"""
Live training dashboard for ORION Pokemon AI.

Shows real-time charts for all concurrent training runs.

Usage:
    python dashboard.py                          # Auto-discover all runs
    python dashboard.py --dirs a b c             # Specific checkpoint dirs
    python dashboard.py --refresh 5              # Refresh every 5 seconds
    python dashboard.py --save dashboard.png     # Save a snapshot instead of live
"""

import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict

import matplotlib

# Headless backend when only writing a snapshot, so the dashboard can run
# unattended (no display / background service); interactive Tk otherwise.
matplotlib.use("Agg" if "--save" in sys.argv else "TkAgg")
import matplotlib.pyplot as plt
import numpy as np

# Pretty colors per run
COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]


def find_runs(base_dir="checkpoints/pokemon"):
    """Find all training runs with a training_log.json."""
    runs = {}
    if not os.path.isdir(base_dir):
        return runs
    for name in sorted(os.listdir(base_dir)):
        log_path = os.path.join(base_dir, name, "training_log.json")
        if os.path.exists(log_path):
            runs[name] = log_path
    return runs


def load_history(log_path, history):
    """Append latest log entry to history (deduplicated by rollout)."""
    try:
        with open(log_path) as f:
            entry = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return history

    rollout = entry.get("rollout", 0)
    if rollout not in history["rollout"]:
        for key in [
            "rollout",
            "global_step",
            "avg_reward",
            "avg_badges",
            "max_badges",
            "best_badges",
            "policy_loss",
            "value_loss",
            "entropy",
            "lr",
            "episodes",
        ]:
            history[key].append(entry.get(key, 0))
        # Sanitize inf
        if history["best_badges"][-1] is None:
            history["best_badges"][-1] = 0
    return history


def new_history():
    return defaultdict(list)


def draw(fig, all_histories, title_suffix=""):
    """Redraw the dashboard with current data."""
    fig.clear()

    n_runs = len(all_histories)
    if n_runs == 0:
        fig.text(
            0.5,
            0.5,
            "Waiting for training logs...",
            ha="center",
            va="center",
            fontsize=16,
            color="gray",
        )
        fig.canvas.draw_idle()
        return

    # 2x3 grid: reward, badges, entropy, losses, LR, steps
    axes = fig.subplots(2, 3)
    fig.suptitle(
        f"ORION Pokemon Training Dashboard {title_suffix}",
        fontsize=14,
        fontweight="bold",
    )

    labels = list(all_histories.keys())

    # 1. Average Reward
    ax = axes[0][0]
    for i, (name, h) in enumerate(all_histories.items()):
        if h["avg_reward"]:
            ax.plot(
                h["rollout"],
                h["avg_reward"],
                color=COLORS[i % len(COLORS)],
                label=name,
                linewidth=2,
            )
    ax.set_title("Average Reward", fontsize=10, fontweight="bold")
    ax.set_xlabel("Rollout")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    # 2. Badges (max + best)
    ax = axes[0][1]
    for i, (name, h) in enumerate(all_histories.items()):
        if h["max_badges"]:
            ax.plot(
                h["rollout"],
                h["max_badges"],
                color=COLORS[i % len(COLORS)],
                label=f"{name} max",
                linewidth=2,
                linestyle="-",
            )
            ax.plot(
                h["rollout"],
                h["best_badges"],
                color=COLORS[i % len(COLORS)],
                label=f"{name} best",
                linewidth=1,
                linestyle="--",
            )
    ax.set_title("Badges Earned", fontsize=10, fontweight="bold")
    ax.set_xlabel("Rollout")
    ax.set_ylabel("Badges")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    # 3. Entropy (policy health)
    ax = axes[0][2]
    for i, (name, h) in enumerate(all_histories.items()):
        if h["entropy"]:
            ax.plot(
                h["rollout"],
                h["entropy"],
                color=COLORS[i % len(COLORS)],
                label=name,
                linewidth=2,
            )
    ax.axhline(y=0.5, color="red", linestyle=":", alpha=0.5, label="collapse threshold")
    ax.set_title("Policy Entropy (higher = healthier)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Rollout")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    # 4. Losses
    ax = axes[1][0]
    for i, (name, h) in enumerate(all_histories.items()):
        if h["policy_loss"]:
            ax.plot(
                h["rollout"],
                h["policy_loss"],
                color=COLORS[i % len(COLORS)],
                label=f"{name} P",
                linewidth=1.5,
            )
            # Use log scale for value loss (can be huge)
            ax.plot(
                h["rollout"],
                h["value_loss"],
                color=COLORS[i % len(COLORS)],
                label=f"{name} V",
                linewidth=1.5,
                linestyle="--",
            )
    ax.set_title("Losses (P=solid, V=dashed)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Rollout")
    ax.set_yscale("symlog")
    ax.legend(fontsize=6, loc="best")
    ax.grid(True, alpha=0.3)

    # 5. Learning Rate
    ax = axes[1][1]
    for i, (name, h) in enumerate(all_histories.items()):
        if h["lr"]:
            ax.plot(
                h["rollout"],
                h["lr"],
                color=COLORS[i % len(COLORS)],
                label=name,
                linewidth=2,
            )
    ax.set_title("Learning Rate", fontsize=10, fontweight="bold")
    ax.set_xlabel("Rollout")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    # 6. Progress table (text)
    ax = axes[1][2]
    ax.axis("off")
    table_data = [["Run", "Rollout", "Steps", "Episodes", "Badges"]]
    for name, h in all_histories.items():
        if h["rollout"]:
            table_data.append(
                [
                    name[:12],
                    f"{h['rollout'][-1]}",
                    f"{h['global_step'][-1]:,}",
                    f"{h['episodes'][-1]}",
                    f"{h['max_badges'][-1]}",
                ]
            )
    table = ax.table(
        cellText=table_data[1:], colLabels=table_data[0], loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax.set_title("Status", fontsize=10, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description="Live Pokemon training dashboard")
    parser.add_argument(
        "--dirs", nargs="*", default=None, help="Specific checkpoint dirs to monitor"
    )
    parser.add_argument(
        "--base", default="checkpoints/pokemon", help="Base directory to search"
    )
    parser.add_argument(
        "--refresh", type=float, default=3.0, help="Refresh interval in seconds"
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Save a single snapshot to file instead of live window",
    )
    args = parser.parse_args()

    # Discover runs
    if args.dirs:
        runs = {}
        for d in args.dirs:
            log = os.path.join(d, "training_log.json")
            if os.path.exists(log):
                runs[os.path.basename(d)] = log
    else:
        runs = find_runs(args.base)

    if not runs:
        print(f"No training logs found in {args.base}")
        print("Waiting for training to produce first log (every 10 rollouts)...")

    histories = {name: new_history() for name in runs}

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    plt.subplots_adjust(hspace=0.35, wspace=0.3)
    manager = plt.get_current_fig_manager()
    try:
        manager.window.wm_geometry("+50+50")
    except Exception:
        pass

    if args.save:
        # Single snapshot mode
        for name, path in runs.items():
            histories[name] = load_history(path, histories[name])
        draw(fig, histories, "(snapshot)")
        fig.savefig(args.save, dpi=120, bbox_inches="tight")
        print(f"Saved: {args.save}")
        return

    # Live mode
    print("=" * 60)
    print("  ORION Pokemon Training Dashboard")
    print("=" * 60)
    print(f"  Monitoring: {', '.join(runs.keys()) or 'waiting...'}")
    print(f"  Refresh:    {args.refresh}s")
    print(f"  Close the window or press Ctrl+C to stop")
    print("=" * 60)

    try:
        while True:
            # Re-discover runs (training may have started since)
            if not args.dirs:
                new_runs = find_runs(args.base)
                for name, path in new_runs.items():
                    if name not in histories:
                        histories[name] = new_history()
                        print(f"  [NEW] Discovered run: {name}")
                runs = new_runs

            # Load latest data
            for name, path in runs.items():
                histories[name] = load_history(path, histories[name])

            # Remove runs that no longer exist
            histories = {k: v for k, v in histories.items() if k in runs}

            draw(fig, histories, f"({time.strftime('%H:%M:%S')})")
            plt.pause(args.refresh)

    except (KeyboardInterrupt, Exception) as e:
        print(f"\nDashboard stopped: {e}")
        # Save final snapshot
        snapshot = "checkpoints/pokemon/dashboard_final.png"
        os.makedirs(os.path.dirname(snapshot), exist_ok=True)
        draw(fig, histories, "(final)")
        fig.savefig(snapshot, dpi=120, bbox_inches="tight")
        print(f"Final snapshot saved: {snapshot}")


if __name__ == "__main__":
    main()
