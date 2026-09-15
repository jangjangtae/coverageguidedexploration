"""
train_random_arena_issue10_with_coverage.py

RANDOM baseline 전용 Arena ISSUE10 재실험 스크립트.
목적:
  - RANDOM도 다른 알고리즘과 동일하게 coverage / visited-cell 로그를 남김
  - TensorBoard tag:
      custom/visited_cells
      main/visited_cells
      custom/exploration_rate
      main/exploration_rate
    를 반드시 기록
  - CSV:
      arena_progress.csv에 visited_cells, visited_count, coverage_count,
      coverage_rate, coverage_ratio 등을 기록

기본 실행:
    python3 agents/train_random_arena_issue10_with_coverage_v2.py

단일 run 병렬 실행 예:
    python3 agents/train_random_arena_issue10_with_coverage_v2.py --runs 1 --seed-start 1 --run-offset 0 --port 10000 --log-root ./logs_arena_issue10_final/RANDOM
    python3 agents/train_random_arena_issue10_with_coverage_v2.py --runs 1 --seed-start 2 --run-offset 1 --port 10001 --log-root ./logs_arena_issue10_final/RANDOM

기본 설정:
    algo      = RANDOM
    runs      = 10
    steps     = 100000
    port      = 10000
    log_root  = ./logs_arena_issue10_final/RANDOM
    cooldown  = 100 sec
    arena     = 20 x 20 = 400 cells

병렬 실행 시:
    - 모든 프로세스의 --log-root를 같은 폴더로 지정하면 한 폴더 아래에 저장됨
    - 각 run 폴더 이름에 port/seed/run_idx가 포함되어 충돌을 피함

중요:
  기존 logs_arena_issue10_final/RANDOM 폴더에 coverage 없는 old RANDOM 로그가 있으면
  분석 코드가 old run을 먼저 집을 수 있다.
  논문용 그래프를 깔끔하게 만들려면 기존 RANDOM 폴더를 root 밖으로 백업한 뒤 실행하는 것을 권장한다.

예:
    mv logs_arena_issue10_final/RANDOM logs_arena_issue10_final_RANDOM_old_no_coverage
    python3 agents/train_random_arena_issue10_with_coverage.py
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


# ============================================================
# Target bugs
# ============================================================

TARGET_BUGS = [
    "BUG_HEADING_DYNAMICS_ANOMALY",
    "BUG_TRANSITION_TELEPORT",
    "BUG_MOVEMENT_LOCK_ZONE",
    "BUG_COLLISION_IMPULSE_GLITCH",
    "BUG_CONTEXTUAL_SEQUENCE_FAILURE",
    "BUG_CONTEXTUAL_INTERACTION_OMISSION",
    "BUG_WORLD_STATE_MUTATION",
    "BUG_VIEW_DEPENDENT_MOVEMENT_LOCK",
    "BUG_SLOT_SELECTION_DESYNC",
    "BUG_BREAK_EVENT_CORRUPTION",
]


# ============================================================
# TensorBoard writer
# ============================================================

class SafeSummaryWriter:
    def __init__(self, log_dir: Path):
        self.writer = None
        self.enabled = False
        self.error = None

        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(log_dir))
            self.enabled = True
        except Exception as e1:
            try:
                from tensorboardX import SummaryWriter
                self.writer = SummaryWriter(logdir=str(log_dir))
                self.enabled = True
            except Exception as e2:
                self.error = f"torch/tensorboard writer unavailable: {repr(e1)} | {repr(e2)}"

    def add_scalar(self, tag: str, value: Any, step: int) -> None:
        if not self.enabled or self.writer is None:
            return
        try:
            if value is None:
                return
            val = float(value)
            if not np.isfinite(val):
                return
            self.writer.add_scalar(tag, val, int(step))
        except Exception:
            pass

    def flush(self) -> None:
        if self.enabled and self.writer is not None:
            try:
                self.writer.flush()
            except Exception:
                pass

    def close(self) -> None:
        if self.enabled and self.writer is not None:
            try:
                self.writer.close()
            except Exception:
                pass


# ============================================================
# JSON helpers
# ============================================================

def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(list(obj))
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=json_default)


def read_json_if_exists(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def copy_or_create_bug_snapshot(run_dir: Path, bug_json_path: str = "") -> None:
    candidates = []
    if bug_json_path:
        candidates.append(Path(bug_json_path))
    candidates.extend([
        Path("envs/bug_definitions_issue10_consistent.json"),
        Path("envs/bug_definitions_issue10.json"),
        Path("bug_definitions_issue10_consistent.json"),
    ])

    for src in candidates:
        if src.exists():
            try:
                shutil.copyfile(src, run_dir / "bug_definitions_snapshot.json")
                return
            except Exception:
                pass

    save_json(
        run_dir / "bug_definitions_snapshot.json",
        {
            "bugs": [
                {
                    "id": bug,
                    "legacy_id": "",
                    "description": "",
                    "reward": 100,
                }
                for bug in TARGET_BUGS
            ]
        },
    )


# ============================================================
# Env compatibility
# ============================================================

def import_env_class(module_name: str, class_name: str):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def set_possible_port(env: Any, port: int) -> None:
    for attr in ["port", "server_port", "_port", "malmo_port"]:
        try:
            if hasattr(env, attr):
                setattr(env, attr, int(port))
        except Exception:
            pass


def create_env(args: argparse.Namespace, run_dir: Path, seed: int):
    EnvCls = import_env_class(args.env_module, args.env_class)

    constructor_attempts = [
        {"port": args.port, "seed": seed},
        {"port": args.port},
        {"server_port": args.port, "seed": seed},
        {"server_port": args.port},
        {},
    ]

    last_exc = None

    for kwargs in constructor_attempts:
        try:
            print(f"    [Env] Trying {args.env_module}.{args.env_class} kwargs={kwargs}")
            env = EnvCls(**kwargs)
            set_possible_port(env, args.port)

            # seed
            try:
                env.reset(seed=seed)
            except Exception:
                try:
                    env.seed(seed)
                except Exception:
                    pass

            try:
                env.action_space.seed(seed)
            except Exception:
                pass

            print(f"    [Env] Created. requested_port={args.port}, env.port={getattr(env, 'port', None)}")
            return env
        except TypeError as e:
            # constructor signature mismatch
            last_exc = e
            continue
        except Exception as e:
            last_exc = e
            # 실제 Malmo connection 실패일 수 있으므로 다음 kwargs로 넘어가기보다는 break하지 않고 시도
            continue

    raise RuntimeError(f"Failed to create env. Last error: {repr(last_exc)}")


def reset_env(env: Any, seed: Optional[int] = None):
    try:
        if seed is not None:
            out = env.reset(seed=seed)
        else:
            out = env.reset()
    except TypeError:
        out = env.reset()

    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
    else:
        obs, info = out, {}

    if info is None:
        info = {}

    return obs, info


def step_env(env: Any, action: Any):
    out = env.step(action)

    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
    elif isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, info = out
        terminated = bool(done)
        truncated = False
    else:
        raise RuntimeError(f"Unsupported env.step output: type={type(out)}, value={out}")

    if info is None:
        info = {}

    return obs, float(reward), bool(done), bool(terminated), bool(truncated), info


def close_env(env: Any) -> None:
    try:
        env.close()
    except Exception:
        pass


def sample_random_action(env: Any):
    return env.action_space.sample()


# ============================================================
# Recursive extraction helpers
# ============================================================

def iter_nested(obj: Any, prefix: str = "", depth: int = 0, max_depth: int = 5):
    if depth > max_depth:
        return

    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            yield key, v
            yield from iter_nested(v, key, depth + 1, max_depth)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]"
            yield key, v
            yield from iter_nested(v, key, depth + 1, max_depth)


def as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return float(int(x))
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


def find_first_numeric(data_sources: Sequence[Any], candidate_names: Sequence[str]) -> Optional[float]:
    cand = [c.lower() for c in candidate_names]

    for src in data_sources:
        if src is None:
            continue

        # direct dict first
        if isinstance(src, dict):
            lower_map = {str(k).lower(): v for k, v in src.items()}
            for name in cand:
                if name in lower_map:
                    v = as_float(lower_map[name])
                    if v is not None:
                        return v

        # nested
        for key, val in iter_nested(src):
            lk = key.lower().replace("/", "_")
            base = lk.split(".")[-1]
            for name in cand:
                n = name.replace("/", "_").lower()
                if base == n or lk.endswith("." + n) or lk.endswith("_" + n) or lk == n:
                    v = as_float(val)
                    if v is not None:
                        return v

    return None


def get_env_attr_numeric(env: Any, candidate_names: Sequence[str]) -> Optional[float]:
    objects = []
    cur = env
    for _ in range(5):
        if cur is None:
            break
        objects.append(cur)
        cur = getattr(cur, "env", None)

    for obj in objects:
        for name in candidate_names:
            for attr in {name, "_" + name, name.replace("/", "_"), name.replace("/", "_").lower()}:
                try:
                    if hasattr(obj, attr):
                        val = getattr(obj, attr)
                        v = as_float(val)
                        if v is not None:
                            return v
                        if isinstance(val, (set, list, tuple, dict)):
                            return float(len(val))
                except Exception:
                    pass

    return None


def get_env_visited_set_size(env: Any) -> Optional[float]:
    attr_names = [
        "visited_cells",
        "_visited_cells",
        "visited",
        "_visited",
        "visited_tiles",
        "_visited_tiles",
        "visited_positions",
        "_visited_positions",
        "coverage_cells",
        "_coverage_cells",
        "seen_cells",
        "_seen_cells",
    ]

    cur = env
    for _ in range(5):
        if cur is None:
            break

        for attr in attr_names:
            try:
                if hasattr(cur, attr):
                    val = getattr(cur, attr)
                    if isinstance(val, (set, list, tuple, dict)):
                        return float(len(val))
            except Exception:
                pass

        cur = getattr(cur, "env", None)

    return None


def extract_position(obs: Any, info: Dict[str, Any], env: Any = None) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    sources = [info]

    # obs가 dict면 위치가 들어있을 수 있음
    if isinstance(obs, dict):
        sources.append(obs)

    x_candidates = ["x", "xpos", "x_pos", "xposition", "x_position", "agent_x", "pos_x", "XPos"]
    y_candidates = ["y", "ypos", "y_pos", "yposition", "y_position", "agent_y", "pos_y", "YPos"]
    z_candidates = ["z", "zpos", "z_pos", "zposition", "z_position", "agent_z", "pos_z", "ZPos"]
    yaw_candidates = ["yaw", "Yaw", "rotation", "heading", "agent_yaw"]

    x = find_first_numeric(sources, x_candidates)
    y = find_first_numeric(sources, y_candidates)
    z = find_first_numeric(sources, z_candidates)
    yaw = find_first_numeric(sources, yaw_candidates)

    # env attrs fallback
    if env is not None:
        if x is None:
            x = get_env_attr_numeric(env, x_candidates)
        if y is None:
            y = get_env_attr_numeric(env, y_candidates)
        if z is None:
            z = get_env_attr_numeric(env, z_candidates)
        if yaw is None:
            yaw = get_env_attr_numeric(env, yaw_candidates)

    return x, y, z, yaw


def cell_from_position(x: Optional[float], z: Optional[float]) -> Optional[Tuple[int, int]]:
    if x is None or z is None:
        return None
    try:
        if not np.isfinite(x) or not np.isfinite(z):
            return None
        # Minecraft 좌표는 .5 중심값이 많으므로 floor 기반 cell로 변환
        return int(math.floor(float(x))), int(math.floor(float(z)))
    except Exception:
        return None


def extract_coverage(obs: Any, info: Dict[str, Any], env: Any, local_seen_cells: Set[Tuple[int, int]]) -> Dict[str, Any]:
    sources = [info]
    if isinstance(obs, dict):
        sources.append(obs)

    visited_candidates = [
        "visited_cells",
        "main/visited_cells",
        "custom/visited_cells",
        "visited_count",
        "coverage_count",
        "unique_cells_visited",
        "unique_tiles_visited",
        "episode_unique_cells_visited",
        "episode_unique_tiles_visited",
        "cell_count",
        "tile_count",
    ]

    coverage_candidates = [
        "coverage_rate",
        "coverage_ratio",
        "exploration_rate",
        "main/exploration_rate",
        "custom/exploration_rate",
        "main/coverage_rate",
        "custom/coverage_rate",
    ]

    visited = find_first_numeric(sources, visited_candidates)
    coverage_rate = find_first_numeric(sources, coverage_candidates)

    if visited is None:
        visited = get_env_attr_numeric(env, visited_candidates)

    if visited is None:
        visited = get_env_visited_set_size(env)

    if coverage_rate is None:
        coverage_rate = get_env_attr_numeric(env, coverage_candidates)

    # local position fallback
    x, y, z, yaw = extract_position(obs, info, env)
    cell = cell_from_position(x, z)
    if cell is not None:
        local_seen_cells.add(cell)

    if visited is None and len(local_seen_cells) > 0:
        visited = float(len(local_seen_cells))

    if coverage_rate is None and visited is not None:
        coverage_rate = float(visited) / 400.0 * 100.0

    # 0~1 ratio scale이면 percent로 변환
    if coverage_rate is not None and coverage_rate <= 1.5:
        coverage_rate = coverage_rate * 100.0

    coverage_ratio = None
    if coverage_rate is not None:
        coverage_ratio = coverage_rate / 100.0

    return {
        "x": x,
        "y": y,
        "z": z,
        "yaw": yaw,
        "cell": cell,
        "visited_cells": visited,
        "visited_count": visited,
        "coverage_count": visited,
        "coverage_rate": coverage_rate,
        "coverage_ratio": coverage_ratio,
        "coverage_source": (
            "env_or_info" if visited is not None and cell is None else
            "local_position_fallback" if cell is not None else
            "unknown"
        ),
    }


def extract_bug_ids(info: Dict[str, Any]) -> Set[str]:
    found: Set[str] = set()

    if not isinstance(info, dict):
        return found

    for key, val in iter_nested(info):
        lk = key.lower()

        # target list / definitions는 실제 탐지가 아니므로 제외
        if "target" in lk or "definition" in lk:
            continue

        # bug 관련 key에 있는 값만 탐색
        if "bug" not in lk and "fault" not in lk:
            continue

        if isinstance(val, str):
            for bug in TARGET_BUGS:
                if bug == val or bug in val:
                    found.add(bug)

        elif isinstance(val, (list, tuple, set)):
            for item in val:
                if isinstance(item, str):
                    for bug in TARGET_BUGS:
                        if bug == item or bug in item:
                            found.add(bug)

        elif isinstance(val, dict):
            # {"BUG_...": true} 형태
            for k2, v2 in val.items():
                if isinstance(k2, str) and k2 in TARGET_BUGS and bool(v2):
                    found.add(k2)

    return found


# ============================================================
# Logging
# ============================================================

PROGRESS_FIELDS = [
    "algorithm",
    "run_idx",
    "seed",
    "global_step",
    "episode_idx",
    "episode_step",
    "action",
    "reward",
    "extrinsic_reward",
    "intrinsic_reward",
    "total_reward",
    "done",
    "terminated",
    "truncated",
    "x",
    "y",
    "z",
    "yaw",
    "visited_cells",
    "visited_count",
    "coverage_count",
    "coverage_rate",
    "coverage_ratio",
    "coverage_source",
    "unique_bug_count",
    "unique_bug_fraction",
    "new_bugs",
    "unique_bugs",
]

BUG_EVENT_FIELDS = [
    "algorithm",
    "run_idx",
    "seed",
    "global_step",
    "episode_idx",
    "episode_step",
    "bug_id",
    "is_first_detection",
    "unique_bug_count",
]


class CSVLogger:
    def __init__(self, path: Path, fieldnames: List[str]):
        self.path = path
        self.fieldnames = fieldnames
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        self.writer.writeheader()
        self.rows = 0

    def write(self, row: Dict[str, Any]):
        safe = {}
        for k in self.fieldnames:
            v = row.get(k, "")
            if isinstance(v, (list, tuple, set)):
                v = ";".join(map(str, sorted(v)))
            elif isinstance(v, dict):
                v = json.dumps(v, ensure_ascii=False, default=json_default)
            safe[k] = v
        self.writer.writerow(safe)
        self.rows += 1

    def flush(self):
        try:
            self.file.flush()
        except Exception:
            pass

    def close(self):
        try:
            self.file.flush()
            self.file.close()
        except Exception:
            pass


# ============================================================
# Main experiment
# ============================================================

def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--port", type=int, default=10000)
    p.add_argument("--log-root", type=str, default="./logs_arena_issue10_final/RANDOM")
    p.add_argument("--cooldown", type=int, default=100)

    p.add_argument("--seed-start", type=int, default=1)
    p.add_argument("--seeds", type=str, default="")
    p.add_argument("--run-offset", type=int, default=0,
                   help="병렬 실행 시 전체 run 번호를 맞추기 위한 offset. 실제 run_idx = run-offset + local_idx")

    p.add_argument("--env-module", type=str, default="envs.malmo_env_issue10_distributed_ep90")
    p.add_argument("--env-class", type=str, default="MalmoEnv")
    p.add_argument("--bug-json-path", type=str, default="")

    p.add_argument("--episode-seconds", type=int, default=90)
    p.add_argument("--target-bug-count", type=int, default=10)

    p.add_argument("--progress-every", type=int, default=100)
    p.add_argument("--tb-every", type=int, default=1000)
    p.add_argument("--retry", type=int, default=5)
    p.add_argument("--retry-sleep", type=int, default=10)

    return p


def parse_seeds(args: argparse.Namespace) -> List[int]:
    if args.seeds.strip():
        return [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    return list(range(args.seed_start, args.seed_start + args.runs))


def make_run_dir(args: argparse.Namespace, run_idx: int, seed: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # port를 폴더명에 포함해서 병렬 실행 시 폴더 충돌과 사후 추적 문제를 줄인다.
    name = f"RANDOM_ISSUE10_COVERAGE_{timestamp}_port_{int(args.port):05d}_seed_{seed:03d}_run_{run_idx:02d}"
    run_dir = Path(args.log_root) / name

    # 매우 드문 timestamp 충돌 방지
    if run_dir.exists():
        suffix = 1
        while True:
            alt = Path(args.log_root) / f"{name}_{suffix:02d}"
            if not alt.exists():
                run_dir = alt
                break
            suffix += 1

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_static_configs(args: argparse.Namespace, run_dir: Path, run_idx: int, seed: int, writer_error: Optional[str] = None) -> None:
    training_config = {
        "algo": "RANDOM",
        "algorithm": "RANDOM",
        "run_idx": run_idx,
        "seed": seed,
        "runs": args.runs,
        "run_offset": args.run_offset,
        "steps": args.steps,
        "port": args.port,
        "log_root": args.log_root,
        "log_dir": str(run_dir),
        "env_module": args.env_module,
        "env_class": args.env_class,
        "episode_seconds": args.episode_seconds,
        "target_bug_count": args.target_bug_count,
        "progress_every": args.progress_every,
        "tb_every": args.tb_every,
        "coverage_logging": True,
        "coverage_tags": [
            "custom/visited_cells",
            "main/visited_cells",
            "custom/exploration_rate",
            "main/exploration_rate",
        ],
        "writer_error": writer_error,
        "created_at": datetime.now().isoformat(),
    }

    algorithm_config = {
        "algorithm": "RANDOM",
        "seed": seed,
        "steps_per_run": args.steps,
        "episode_seconds": args.episode_seconds,
        "target_bugs": TARGET_BUGS,
        "coverage_logging": True,
        "coverage_note": "RANDOM baseline with explicit visited-cell and coverage-rate logging.",
    }

    manifest = {
        "created_at": datetime.now().isoformat(),
        "algo_arg": "RANDOM",
        "algorithms": ["RANDOM"],
        "runs": args.runs,
        "steps": args.steps,
        "seeds": parse_seeds(args),
        "run_offset": args.run_offset,
        "target_bugs": TARGET_BUGS,
        "args": vars(args),
    }

    save_json(run_dir / "training_config.json", training_config)
    save_json(run_dir / "algorithm_config.json", algorithm_config)
    save_json(run_dir / "final_experiment_manifest.json", manifest)
    copy_or_create_bug_snapshot(run_dir, args.bug_json_path)


def create_env_with_retry(args: argparse.Namespace, run_dir: Path, seed: int):
    last = None

    for attempt in range(1, args.retry + 1):
        try:
            print(f"    [Env] Connecting to Malmo on port {args.port} (Attempt {attempt}/{args.retry})...")
            env = create_env(args, run_dir, seed)
            return env
        except Exception as e:
            last = e
            print(f"    [Env] Connection failed: {repr(e)}")
            traceback.print_exc()
            if attempt < args.retry:
                time.sleep(args.retry_sleep)

    raise RuntimeError(f"Could not create environment after {args.retry} attempts: {repr(last)}")


def cooldown(seconds: int) -> None:
    if seconds <= 0:
        return
    print(f">>> 🛑 Cooling down for {seconds} seconds to avoid Malmo port reuse issues...")
    remain = seconds
    while remain > 0:
        step = min(10, remain)
        print(f"    ... {remain} seconds remaining")
        time.sleep(step)
        remain -= step


def run_single(args: argparse.Namespace, run_idx: int, seed: int) -> bool:
    random.seed(seed)
    np.random.seed(seed)

    run_dir = make_run_dir(args, run_idx, seed)
    tb_dir = run_dir / "tensorboard"
    writer = SafeSummaryWriter(tb_dir)

    write_static_configs(args, run_dir, run_idx, seed, writer_error=writer.error)

    progress_logger = CSVLogger(run_dir / "arena_progress.csv", PROGRESS_FIELDS)
    bug_logger = CSVLogger(run_dir / "bug_events.csv", BUG_EVENT_FIELDS)

    print("=" * 88)
    print(f"🚀 Starting RANDOM coverage run {run_idx}/{args.runs} | seed={seed} | port={args.port}")
    print(f"📁 log_dir={run_dir}")
    print("=" * 88)

    env = None
    obs = None
    info = {}

    detected_bugs: Set[str] = set()
    local_seen_cells: Set[Tuple[int, int]] = set()

    global_step = 0
    episode_idx = 0
    episode_step = 0
    event_count = 0
    progress_rows = 0
    first_bug_step = None

    max_visited_cells = 0.0
    max_coverage_rate = 0.0

    try:
        env = create_env_with_retry(args, run_dir, seed)
        obs, info = reset_env(env, seed=seed)

        # reset 직후 coverage도 업데이트
        cov = extract_coverage(obs, info, env, local_seen_cells)

        for step in range(1, args.steps + 1):
            global_step = step
            episode_step += 1

            action = sample_random_action(env)
            obs, reward, done, terminated, truncated, info = step_env(env, action)

            cov = extract_coverage(obs, info, env, local_seen_cells)

            visited_cells = cov.get("visited_cells")
            coverage_rate = cov.get("coverage_rate")
            coverage_ratio = cov.get("coverage_ratio")

            if visited_cells is not None:
                max_visited_cells = max(max_visited_cells, float(visited_cells))
            if coverage_rate is not None:
                max_coverage_rate = max(max_coverage_rate, float(coverage_rate))

            # bug extraction
            new_bugs = extract_bug_ids(info)
            first_detection_bugs = []

            for bug in sorted(new_bugs):
                is_first = bug not in detected_bugs
                if is_first:
                    detected_bugs.add(bug)
                    first_detection_bugs.append(bug)
                    if first_bug_step is None:
                        first_bug_step = global_step

                event_count += 1
                bug_logger.write({
                    "algorithm": "RANDOM",
                    "run_idx": run_idx,
                    "seed": seed,
                    "global_step": global_step,
                    "episode_idx": episode_idx,
                    "episode_step": episode_step,
                    "bug_id": bug,
                    "is_first_detection": int(is_first),
                    "unique_bug_count": len(detected_bugs),
                })

            unique_bug_count = len(detected_bugs)
            unique_bug_fraction = unique_bug_count / max(len(TARGET_BUGS), 1)

            # info에 이미 unique_bug_count가 있으면 그것도 반영
            info_unique = find_first_numeric([info], ["unique_bug_count", "final_unique_bug_count", "current_unique_bug_count"])
            if info_unique is not None and info_unique > unique_bug_count:
                # 구체 bug id가 안 보이더라도 count는 보존
                unique_bug_count = int(info_unique)
                unique_bug_fraction = unique_bug_count / max(len(TARGET_BUGS), 1)

            should_log_progress = (
                global_step == 1
                or global_step % args.progress_every == 0
                or done
                or bool(first_detection_bugs)
                or global_step == args.steps
            )

            if should_log_progress:
                progress_logger.write({
                    "algorithm": "RANDOM",
                    "run_idx": run_idx,
                    "seed": seed,
                    "global_step": global_step,
                    "episode_idx": episode_idx,
                    "episode_step": episode_step,
                    "action": action,
                    "reward": reward,
                    "extrinsic_reward": reward,
                    "intrinsic_reward": 0.0,
                    "total_reward": reward,
                    "done": int(done),
                    "terminated": int(terminated),
                    "truncated": int(truncated),
                    "x": cov.get("x"),
                    "y": cov.get("y"),
                    "z": cov.get("z"),
                    "yaw": cov.get("yaw"),
                    "visited_cells": visited_cells,
                    "visited_count": visited_cells,
                    "coverage_count": visited_cells,
                    "coverage_rate": coverage_rate,
                    "coverage_ratio": coverage_ratio,
                    "coverage_source": cov.get("coverage_source"),
                    "unique_bug_count": unique_bug_count,
                    "unique_bug_fraction": unique_bug_fraction,
                    "new_bugs": first_detection_bugs,
                    "unique_bugs": detected_bugs,
                })
                progress_rows += 1

            should_log_tb = (
                global_step == 1
                or global_step % args.tb_every == 0
                or done
                or bool(first_detection_bugs)
                or global_step == args.steps
            )

            if should_log_tb:
                if visited_cells is not None:
                    writer.add_scalar("custom/visited_cells", visited_cells, global_step)
                    writer.add_scalar("main/visited_cells", visited_cells, global_step)
                    writer.add_scalar("custom/coverage_count", visited_cells, global_step)
                    writer.add_scalar("main/coverage_count", visited_cells, global_step)

                if coverage_rate is not None:
                    writer.add_scalar("custom/exploration_rate", coverage_rate, global_step)
                    writer.add_scalar("main/exploration_rate", coverage_rate, global_step)
                    writer.add_scalar("custom/coverage_rate", coverage_rate, global_step)
                    writer.add_scalar("main/coverage_rate", coverage_rate, global_step)

                writer.add_scalar("main/unique_bug_count", unique_bug_count, global_step)
                writer.add_scalar("custom/unique_bug_count", unique_bug_count, global_step)
                writer.add_scalar("main/reward", reward, global_step)

            if global_step % max(1000, args.tb_every) == 0:
                print(
                    f"    step={global_step:6d} | ep={episode_idx:3d} | "
                    f"visited={visited_cells} | coverage={coverage_rate} | "
                    f"bugs={unique_bug_count}/{len(TARGET_BUGS)}"
                )
                progress_logger.flush()
                bug_logger.flush()
                writer.flush()

            if done:
                episode_idx += 1
                episode_step = 0
                obs, info = reset_env(env)

        # summary
        final_unique_bug_count = len(detected_bugs)
        # count fallback
        if unique_bug_count > final_unique_bug_count:
            final_unique_bug_count = int(unique_bug_count)

        final_unique_bug_fraction = final_unique_bug_count / max(len(TARGET_BUGS), 1)

        summary = {
            "algorithm": "RANDOM",
            "run_idx": run_idx,
            "seed": seed,
            "steps": args.steps,
            "port": args.port,
            "log_dir": str(run_dir),
            "env_module": args.env_module,
            "finished_at": datetime.now().isoformat(),
            "global_step": global_step,
            "episode_idx": episode_idx,
            "final_unique_bug_count": final_unique_bug_count,
            "final_unique_bug_fraction": final_unique_bug_fraction,
            "unique_bugs": sorted(detected_bugs),
            "target_bugs": TARGET_BUGS,
            "target_bug_count": len(TARGET_BUGS),
            "event_count": event_count,
            "progress_rows": progress_rows,
            "first_bug_step": first_bug_step,
            "max_visited_cells": max_visited_cells,
            "max_coverage_rate": max_coverage_rate,
            "final_visited_cells": cov.get("visited_cells") if "cov" in locals() else None,
            "final_coverage_rate": cov.get("coverage_rate") if "cov" in locals() else None,
            "coverage_logging": True,
            "coverage_tags": [
                "custom/visited_cells",
                "main/visited_cells",
                "custom/exploration_rate",
                "main/exploration_rate",
            ],
            "tensorboard_writer_enabled": writer.enabled,
            "tensorboard_writer_error": writer.error,
        }

        # bug_auc는 정확한 per-bug first step이 없을 수 있으므로 기존 분석 코드에서 다시 계산 가능.
        # 여기서는 최소한 summary 형식 유지.
        summary["bug_auc"] = None

        save_json(run_dir / "run_summary.json", summary)
        save_json(run_dir / "arena_logging_summary.json", summary)

        print(
            f"✅ Finished RANDOM run {run_idx} | "
            f"bugs={final_unique_bug_count}/{len(TARGET_BUGS)} | "
            f"max_visited={max_visited_cells:.1f} | max_coverage={max_coverage_rate:.2f}% | "
            f"progress_rows={progress_rows}"
        )

        return True

    except KeyboardInterrupt:
        print(">>> Interrupted by user.")
        raise

    except Exception as e:
        print(f"❌ Run failed: {repr(e)}")
        traceback.print_exc()
        save_json(
            run_dir / "error.json",
            {
                "algorithm": "RANDOM",
                "run_idx": run_idx,
                "seed": seed,
                "error": repr(e),
                "traceback": traceback.format_exc(),
                "global_step": global_step,
            },
        )
        return False

    finally:
        progress_logger.close()
        bug_logger.close()
        writer.close()
        if env is not None:
            close_env(env)


def main():
    parser = build_parser()
    args = parser.parse_args()

    seeds = parse_seeds(args)
    args.runs = len(seeds)

    print("=" * 88)
    print("RANDOM Arena ISSUE10 coverage-fixed experiment")
    print(f"Runs       : {args.runs}")
    print(f"Steps/run  : {args.steps}")
    print(f"Port       : {args.port}")
    print(f"Log root   : {args.log_root}")
    print(f"Seeds      : {seeds}")
    print(f"Run offset : {args.run_offset}")
    print(f"Shared root: {Path(args.log_root).resolve()}")
    print(f"Env        : {args.env_module}.{args.env_class}")
    print("=" * 88)

    root = Path(args.log_root)
    root.mkdir(parents=True, exist_ok=True)

    ok_count = 0
    for local_idx, seed in enumerate(seeds, start=1):
        # 병렬 실행 시 전체 run 번호를 유지하기 위해 offset을 더한다.
        run_idx = int(args.run_offset) + local_idx
        ok = run_single(args, run_idx, seed)
        ok_count += int(ok)

        if local_idx < len(seeds):
            cooldown(args.cooldown)

    print("=" * 88)
    print(f"Done. Success runs: {ok_count}/{len(seeds)}")
    print(f"Log root: {Path(args.log_root).resolve()}")
    print("=" * 88)


if __name__ == "__main__":
    main()
