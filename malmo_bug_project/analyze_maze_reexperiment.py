"""
analyze_maze_reexperiment.py

Aggregate maze re-experiment logs and generate meeting/paper plots.

Run:
    python3 analyze_maze_reexperiment.py --root ./logs_maze_reexperiment --out ./analysis_maze_reexperiment
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ALGO_ORDER = ["RANDOM", "RELINE", "BEAGT", "DQN_RND", "CAE_FINAL", "RND_CAE_FINAL"]


def read_all_runs(root):
    rows = []
    root = Path(root)
    for csv_path in root.glob("*/maze_progress.csv"):
        run_dir = csv_path.parent
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            df["run_dir"] = str(run_dir)
            if "algorithm" not in df.columns:
                df["algorithm"] = run_dir.name.split("_MAZE21_")[0]
            rows.append(df)
        except Exception as e:
            print(f"[WARN] Failed to read {csv_path}: {e}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    return tuple(np.quantile(means, [alpha / 2, 1 - alpha / 2]))


def plot_curve(df, out_dir):
    plt.figure(figsize=(10, 6))
    for algo in ALGO_ORDER:
        sub = df[df["algorithm"] == algo].copy()
        if sub.empty:
            continue
        sub["global_step_bin"] = (sub["global_step"] // 1000) * 1000
        g = sub.groupby("global_step_bin")["coverage_ratio"].agg(["mean", "std", "count"]).reset_index()
        x = g["global_step_bin"].values
        y = g["mean"].values * 100.0
        std = g["std"].fillna(0).values * 100.0
        n = g["count"].replace(0, np.nan).values
        se = std / np.sqrt(n)
        plt.plot(x, y, label=algo)
        plt.fill_between(x, y - 1.96 * se, y + 1.96 * se, alpha=0.15)
    plt.title("Maze coverage learning curve")
    plt.xlabel("Training step")
    plt.ylabel("Coverage ratio (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "maze_coverage_curve.png", dpi=200)
    plt.close()


def summarize_final(df, out_dir):
    finals = []
    for (algo, run_dir), sub in df.groupby(["algorithm", "run_dir"]):
        last = sub.sort_values("global_step").iloc[-1]
        finals.append({
            "algorithm": algo,
            "run_dir": run_dir,
            "final_coverage_ratio": float(last["coverage_ratio"]),
            "final_visited_count": float(last["visited_count"]),
            "final_reachable_count": float(last.get("reachable_count", np.nan)),
        })
    final_df = pd.DataFrame(finals)
    final_df.to_csv(Path(out_dir) / "maze_final_runs.csv", index=False)

    algos = [a for a in ALGO_ORDER if a in final_df["algorithm"].unique()]
    data = [final_df[final_df["algorithm"] == a]["final_coverage_ratio"].values * 100 for a in algos]

    plt.figure(figsize=(9, 6))
    plt.boxplot(data, tick_labels=algos, showmeans=True)
    plt.title("Final maze coverage by algorithm")
    plt.ylabel("Final coverage ratio (%)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "maze_final_coverage_boxplot.png", dpi=200)
    plt.close()

    rows = []
    for algo in algos:
        vals = final_df[final_df["algorithm"] == algo]["final_coverage_ratio"].values * 100
        ci_lo, ci_hi = bootstrap_ci(vals)
        rows.append({
            "algorithm": algo,
            "runs": len(vals),
            "mean_final_coverage_pct": float(np.mean(vals)),
            "std_final_coverage_pct": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "ci95_low": float(ci_lo),
            "ci95_high": float(ci_hi),
        })
    pd.DataFrame(rows).to_csv(Path(out_dir) / "maze_final_coverage_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./logs_maze_reexperiment")
    parser.add_argument("--out", default="./analysis_maze_reexperiment")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = read_all_runs(args.root)
    if df.empty:
        print("No maze_progress.csv files found.")
        return
    df.to_csv(out_dir / "maze_all_progress.csv", index=False)
    plot_curve(df, out_dir)
    summarize_final(df, out_dir)
    print("=" * 70)
    print("Maze analysis completed.")
    print(f"Input : {args.root}")
    print(f"Output: {args.out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
