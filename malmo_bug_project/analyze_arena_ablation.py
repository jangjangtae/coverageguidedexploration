"""Aggregate Full/Spatial/NoStag Arena runs for the manuscript ablation.

Example:
    python3 analyze_arena_ablation.py \
      --roots logs_arena_issue10_final logs_arena_ablation \
      --seeds 1,2,3,4,5,6,7,8,9,10 \
      --out analysis_arena_ablation
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


ALGORITHMS = ["RND_CAE_FINAL", "RND_CAE_SPATIAL", "RND_CAE_NO_STAG"]
FULL_ALGORITHM = "RND_CAE_FINAL"
COMPARISON_ALGORITHMS = ["RND_CAE_SPATIAL", "RND_CAE_NO_STAG"]
METRICS = [
    "final_unique_bug_count",
    "bug_auc_count_scale",
    "time_to_1_or_horizon",
    "time_to_3_or_horizon",
    "time_to_5_or_horizon",
    "cae_stagnation_trigger_count",
]


def parse_seeds(raw: str) -> Optional[set[int]]:
    if not raw.strip():
        return None
    return {int(x.strip()) for x in raw.split(",") if x.strip()}


def read_json(path: Path) -> Dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def unique_bug_steps(events_path: Path, expected_unique_count: Optional[int] = None) -> List[int]:
    """Read first-detection steps without parsing the very large evidence JSON.

    Arena event rows can exceed tens of thousands because repeated detections
    retain full evidence payloads. The first eight CSV fields never contain
    commas, so splitting only that prefix is both safe and substantially less
    memory intensive than materializing each evidence field with DictReader.
    """
    if not events_path.exists():
        return []
    rows = []
    with events_path.open(encoding="utf-8") as f:
        next(f, None)
        for line in f:
            try:
                fields = line.split(",", 8)
                is_first = int(float(fields[6] or 0))
                if is_first:
                    rows.append(int(float(fields[2] or 0)))
                    if expected_unique_count and len(rows) >= expected_unique_count:
                        break
            except (IndexError, TypeError, ValueError):
                continue
    return sorted(rows)


def last_stagnation_count(progress_path: Path) -> Optional[int]:
    if not progress_path.exists():
        return None
    last = 0
    with progress_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "cae_stagnation_trigger_count" not in (reader.fieldnames or []):
            return None
        for row in reader:
            try:
                last = max(last, int(float(row.get("cae_stagnation_trigger_count", "0") or 0)))
            except (TypeError, ValueError):
                continue
    return last


def run_metrics(summary_path: Path) -> Optional[Dict]:
    summary = read_json(summary_path)
    algorithm = str(summary.get("algorithm", ""))
    if algorithm not in ALGORITHMS:
        return None

    seed = int(summary.get("seed"))
    horizon = int(summary.get("steps") or summary.get("global_step") or 100_000)
    target_count = int(summary.get("target_bug_count") or 10)
    bug_auc = float(summary.get("bug_auc") or 0.0)
    final_bug_count = int(summary.get("final_unique_bug_count") or 0)
    events_path = summary_path.parent / "bug_events.csv"
    steps = unique_bug_steps(events_path, expected_unique_count=final_bug_count)
    detected_bug_ids = {str(x) for x in summary.get("unique_bugs", [])}
    stagnation_count = summary.get("cae_stagnation_trigger_count")
    if stagnation_count is None:
        stagnation_count = last_stagnation_count(summary_path.parent / "arena_progress.csv")

    row = {
        "algorithm": algorithm,
        "seed": seed,
        "run_dir": str(summary_path.parent),
        "horizon": horizon,
        "final_unique_bug_count": float(summary.get("final_unique_bug_count") or 0.0),
        "bug_auc_normalized": bug_auc,
        "bug_auc_count_scale": bug_auc * target_count,
        "cae_stagnation_trigger_count": float(stagnation_count) if stagnation_count is not None else np.nan,
        "target_bugs": ";".join(str(x) for x in summary.get("target_bugs", [])),
        "detected_bugs": ";".join(sorted(detected_bug_ids)),
        "mtime": summary_path.stat().st_mtime,
    }
    for k in (1, 3, 5):
        reached = len(steps) >= k
        row[f"reached_{k}_bugs"] = int(reached)
        row[f"time_to_{k}"] = int(steps[k - 1]) if reached else ""
        row[f"time_to_{k}_or_horizon"] = int(steps[k - 1]) if reached else horizon
    return row


def collect_runs(roots: Sequence[Path], seeds: Optional[set[int]]) -> List[Dict]:
    by_key: Dict[Tuple[str, int], List[Dict]] = defaultdict(list)
    for root in roots:
        if not root.exists():
            print(f"[WARN] Missing root: {root}")
            continue
        for path in root.rglob("run_summary.json"):
            try:
                row = run_metrics(path)
            except Exception as exc:
                print(f"[WARN] Failed to read {path}: {exc}")
                continue
            if row is None or (seeds is not None and row["seed"] not in seeds):
                continue
            by_key[(row["algorithm"], row["seed"])].append(row)

    selected = []
    for key, candidates in sorted(by_key.items()):
        candidates.sort(key=lambda x: x["mtime"])
        if len(candidates) > 1:
            print(f"[WARN] Duplicate completed runs for {key}; using newest: {candidates[-1]['run_dir']}")
        row = dict(candidates[-1])
        row.pop("mtime", None)
        selected.append(row)
    return selected


def write_csv(path: Path, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.quantile(samples, [0.025, 0.975]))


def paired_sign_flip_p(diff: np.ndarray, n_perm: int, seed: int) -> float:
    if len(diff) == 0:
        return np.nan
    observed = abs(float(np.mean(diff)))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diff)))
    permuted = np.abs((signs * diff).mean(axis=1))
    return float((np.sum(permuted >= observed) + 1) / (n_perm + 1))


def summarize(rows: Sequence[Dict], n_boot: int, n_perm: int, random_seed: int):
    summary_rows = []
    by_algorithm: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        by_algorithm[row["algorithm"]].append(row)

    for algorithm in ALGORITHMS:
        subset = by_algorithm.get(algorithm, [])
        if not subset:
            continue
        for metric in METRICS:
            values = np.asarray([float(x[metric]) for x in subset], dtype=float)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                continue
            lo, hi = bootstrap_mean_ci(values, n_boot=n_boot, seed=random_seed)
            summary_rows.append({
                "algorithm": algorithm,
                "metric": metric,
                "n": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "bootstrap_ci95_low": lo,
                "bootstrap_ci95_high": hi,
            })
        for k in (1, 3, 5):
            reached = np.asarray([float(x[f"reached_{k}_bugs"]) for x in subset], dtype=float)
            summary_rows.append({
                "algorithm": algorithm,
                "metric": f"reach_rate_{k}_bugs",
                "n": len(reached),
                "mean": float(np.mean(reached)),
                "std": float(np.std(reached, ddof=1)) if len(reached) > 1 else 0.0,
                "bootstrap_ci95_low": "",
                "bootstrap_ci95_high": "",
            })

    paired_rows = []
    full_by_seed = {x["seed"]: x for x in by_algorithm.get(FULL_ALGORITHM, [])}
    for algorithm in COMPARISON_ALGORITHMS:
        comparison_by_seed = {x["seed"]: x for x in by_algorithm.get(algorithm, [])}
        common_seeds = sorted(set(full_by_seed) & set(comparison_by_seed))
        mismatched = [
            s for s in common_seeds
            if int(full_by_seed[s]["horizon"]) != int(comparison_by_seed[s]["horizon"])
        ]
        if mismatched:
            print(
                f"[WARN] Excluding horizon-mismatched {algorithm}/Full pairs "
                f"for seeds: {','.join(str(x) for x in mismatched)}"
            )
        paired_seeds = [s for s in common_seeds if s not in mismatched]
        for metric in METRICS:
            metric_pairs = [
                (
                    float(comparison_by_seed[s][metric]),
                    float(full_by_seed[s][metric]),
                    s,
                )
                for s in paired_seeds
                if np.isfinite(float(comparison_by_seed[s][metric]))
                and np.isfinite(float(full_by_seed[s][metric]))
            ]
            diff = np.asarray([a - b for a, b, _ in metric_pairs], dtype=float)
            metric_seeds = [s for _, _, s in metric_pairs]
            lo, hi = bootstrap_mean_ci(diff, n_boot=n_boot, seed=random_seed)
            paired_rows.append({
                "comparison_minus_full": algorithm,
                "metric": metric,
                "paired_n": len(diff),
                "paired_seeds": ";".join(str(x) for x in metric_seeds),
                "mean_difference": float(np.mean(diff)) if len(diff) else np.nan,
                "bootstrap_ci95_low": lo,
                "bootstrap_ci95_high": hi,
                "paired_sign_flip_p": paired_sign_flip_p(diff, n_perm=n_perm, seed=random_seed),
            })
    return summary_rows, paired_rows


def summarize_fault_rates(rows: Sequence[Dict]) -> List[Dict]:
    output = []
    by_algorithm: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        by_algorithm[row["algorithm"]].append(row)
    all_targets = sorted({
        bug_id
        for row in rows
        for bug_id in str(row.get("target_bugs", "")).split(";")
        if bug_id
    })
    for algorithm in ALGORITHMS:
        subset = by_algorithm.get(algorithm, [])
        if not subset:
            continue
        detected_sets = [set(str(row.get("detected_bugs", "")).split(";")) - {""} for row in subset]
        for bug_id in all_targets:
            count = sum(bug_id in detected for detected in detected_sets)
            output.append({
                "algorithm": algorithm,
                "bug_id": bug_id,
                "detected_runs": count,
                "n": len(subset),
                "discovery_rate": count / len(subset),
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", default=["logs_arena_issue10_final", "logs_arena_ablation"])
    parser.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--out", default="analysis_arena_ablation")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260810)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = collect_runs([Path(x) for x in args.roots], parse_seeds(args.seeds))
    if not rows:
        raise SystemExit("No completed Full/Spatial/NoStag runs were found.")

    summary_rows, paired_rows = summarize(
        rows,
        n_boot=args.bootstrap,
        n_perm=args.permutations,
        random_seed=args.random_seed,
    )
    write_csv(out / "ablation_run_metrics.csv", rows)
    write_csv(out / "ablation_summary.csv", summary_rows)
    write_csv(out / "ablation_paired_differences.csv", paired_rows)
    write_csv(out / "ablation_per_fault_rates.csv", summarize_fault_rates(rows))
    print(f"Runs: {len(rows)}")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
