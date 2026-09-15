"""
train_arena_issue10_final.py

Final unified trainer for ISSUE10 distributed Arena experiments.

Design goal
-----------
Run the same algorithm set used in the Maze experiment on the Arena benchmark:
    RANDOM, RELINE, BEAGT, DQN_RND, CAE_FINAL, RND_CAE_FINAL

Default final setting:
    runs              = 10
    total timesteps   = 100000 per run
    episode horizon   = handled by the Arena env, expected 90 sec
    cooldown          = 100 sec between runs
    seeds             = 1..10, shared by all algorithms

Place this file in:
    malmo_bug_project/agents/train_arena_issue10_final.py

Example:
    python3 agents/train_arena_issue10_final.py --algo DQN_RND --port 10003 \
      --log-root ./logs_arena_issue10_final/DQN_RND

Parallel execution:
    Run each algorithm in a different terminal with different Malmo ports.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import inspect
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

# Stable-Baselines3 v2 uses Gymnasium internally.
# Import Gymnasium first and adapt legacy Gym-style Malmo envs below if needed.
try:
    import gymnasium as gym
except Exception:  # pragma: no cover
    import gym

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:  # pragma: no cover
    torch = None
    nn = None
    optim = None

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed


# ============================================================
# Final experiment constants
# ============================================================

BASE_ALGO_ORDER = [
    "RANDOM",
    "RELINE",
    "BEAGT",
    "DQN_RND",
    "CAE_FINAL",
    "RND_CAE_FINAL",
]

ABLATION_ALGO_ORDER = [
    "RND_CAE_SPATIAL",
    "RND_CAE_NO_STAG",
]

ALGO_ORDER = BASE_ALGO_ORDER + ABLATION_ALGO_ORDER

DEFAULT_TOTAL_RUNS = 10
DEFAULT_STEPS_PER_RUN = 100_000
DEFAULT_LOG_FREQ = 1000
DEFAULT_COOLDOWN = 100
DEFAULT_SAVE_FREQ = 100_000
DEFAULT_TARGET_BUG_COUNT = 10

DEFAULT_ENV_MODULE_CANDIDATES = [
    "envs.malmo_env_issue10_distributed_ep90",
    "envs.malmo_env_issue10_distributed",
    "envs.malmo_env_issue10_consistent",
]
DEFAULT_ENV_CLASS = "MalmoEnv"

DEFAULT_BUG_MODULE_CANDIDATES = [
    "envs.bug_targets_issue10_distributed",
    "envs.bug_targets_issue10",
]


# ============================================================
# Small helpers
# ============================================================

def now_str() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def project_root() -> Path:
    # agents/train_*.py -> project root is parent of agents
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def json_dump(path: Union[str, Path], data: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, ensure_ascii=False, indent=2)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    try:
        return str(obj)
    except Exception:
        return None


def import_first_module(candidates: Sequence[str]) -> Optional[Any]:
    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    return None


def import_object(module_name: str, class_name: str) -> Any:
    mod = importlib.import_module(module_name)
    return getattr(mod, class_name)


def get_action_dim(env: gym.Env) -> int:
    space = env.action_space
    if hasattr(space, "n"):
        return int(space.n)
    raise ValueError(f"DQN/RANDOM baseline requires Discrete action space, got {space}")


def flatten_observation(obs: Any) -> np.ndarray:
    """Robustly flatten Box/Dict/Tuple observations for RND hashing/network input."""
    if isinstance(obs, dict):
        parts = []
        for key in sorted(obs.keys()):
            parts.append(flatten_observation(obs[key]))
        return np.concatenate(parts).astype(np.float32) if parts else np.zeros(1, dtype=np.float32)
    if isinstance(obs, (list, tuple)):
        parts = [flatten_observation(x) for x in obs]
        return np.concatenate(parts).astype(np.float32) if parts else np.zeros(1, dtype=np.float32)
    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.zeros(1, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    return arr.astype(np.float32)


def stable_hash(value: Any, digits: int = 12) -> str:
    raw = json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True).encode("utf-8", errors="ignore")
    return hashlib.md5(raw).hexdigest()[:digits]


def vector_hash(vec: np.ndarray, digits: int = 12) -> str:
    # Quantize to avoid excessive hash sensitivity.
    q = np.round(vec.astype(np.float32), 3)
    return hashlib.md5(q.tobytes()).hexdigest()[:digits]


def unpack_reset(result: Any) -> Any:
    # Gymnasium reset returns (obs, info); legacy Gym reset returns obs.
    if isinstance(result, tuple) and len(result) == 2:
        return result[0]
    return result


def make_reset_return(result: Any) -> Tuple[Any, Dict[str, Any]]:
    # Always return Gymnasium-style (obs, info).
    if isinstance(result, tuple) and len(result) == 2:
        obs, info = result
        return obs, dict(info or {})
    return result, {}


def unpack_step(result: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
    # legacy Gym: obs, reward, done, info
    # Gymnasium: obs, reward, terminated, truncated, info
    if isinstance(result, tuple) and len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, float(reward), bool(terminated or truncated), dict(info or {})
    if isinstance(result, tuple) and len(result) == 4:
        obs, reward, done, info = result
        return obs, float(reward), bool(done), dict(info or {})
    raise ValueError(f"Unsupported env.step result format: {type(result)} length={len(result) if isinstance(result, tuple) else 'NA'}")


def make_step_return(obs: Any, reward: float, done: bool, info: Dict[str, Any], original_len: int = 5):
    # SB3 v2 Monitor expects Gymnasium-style 5 values.
    terminated = bool(done)
    truncated = False
    return obs, float(reward), terminated, truncated, info


def convert_space(space: Any):
    """Convert common legacy gym.spaces to gymnasium.spaces."""
    try:
        if isinstance(space, gym.spaces.Space):
            return space
    except Exception:
        pass

    cls_name = space.__class__.__name__

    if cls_name == "Box":
        return gym.spaces.Box(
            low=np.asarray(space.low),
            high=np.asarray(space.high),
            shape=tuple(space.shape),
            dtype=space.dtype,
        )
    if cls_name == "Discrete":
        start = int(getattr(space, "start", 0))
        try:
            return gym.spaces.Discrete(int(space.n), start=start)
        except TypeError:
            return gym.spaces.Discrete(int(space.n))
    if cls_name == "MultiBinary":
        return gym.spaces.MultiBinary(space.n)
    if cls_name == "MultiDiscrete":
        return gym.spaces.MultiDiscrete(np.asarray(space.nvec, dtype=np.int64))
    if cls_name == "Tuple":
        return gym.spaces.Tuple(tuple(convert_space(s) for s in space.spaces))
    if cls_name == "Dict":
        return gym.spaces.Dict({k: convert_space(v) for k, v in space.spaces.items()})

    # Fallback: leave as-is. This is still better than failing before diagnostics.
    return space


class GymnasiumEnvAdapter(gym.Env):
    """Adapter for legacy Gym-style Malmo envs.

    Existing MalmoEnv classes in this project may not inherit from
    gymnasium.Env and may return old Gym-style step/reset outputs.
    SB3 v2's Monitor asserts Gymnasium Env inheritance, so this adapter
    provides the required interface without modifying the original env.
    """

    metadata = {"render_modes": []}

    def __init__(self, legacy_env: Any):
        super().__init__()
        self.legacy_env = legacy_env
        self.observation_space = convert_space(getattr(legacy_env, "observation_space"))
        self.action_space = convert_space(getattr(legacy_env, "action_space"))
        self.metadata = getattr(legacy_env, "metadata", self.metadata)

    @property
    def unwrapped(self):
        return getattr(self.legacy_env, "unwrapped", self.legacy_env)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            try:
                result = self.legacy_env.reset(seed=seed, options=options)
                return make_reset_return(result)
            except TypeError:
                try:
                    if hasattr(self.legacy_env, "seed"):
                        self.legacy_env.seed(seed)
                except Exception:
                    pass
        try:
            result = self.legacy_env.reset()
        except TypeError:
            result = self.legacy_env.reset(seed=seed)
        return make_reset_return(result)

    def step(self, action):
        result = self.legacy_env.step(action)
        if isinstance(result, tuple) and len(result) == 5:
            obs, reward, terminated, truncated, info = result
            return obs, float(reward), bool(terminated), bool(truncated), dict(info or {})
        if isinstance(result, tuple) and len(result) == 4:
            obs, reward, done, info = result
            return obs, float(reward), bool(done), False, dict(info or {})
        raise ValueError(f"Unsupported legacy env.step result: {result}")

    def render(self):
        if hasattr(self.legacy_env, "render"):
            return self.legacy_env.render()
        return None

    def close(self):
        if hasattr(self.legacy_env, "close"):
            return self.legacy_env.close()
        return None

    def __getattr__(self, name: str):
        # Forward project-specific attributes/methods such as agent_host.
        if name == "legacy_env":
            raise AttributeError(name)
        return getattr(self.legacy_env, name)


def ensure_gymnasium_env(env: Any) -> gym.Env:
    if isinstance(env, gym.Env):
        return env
    return GymnasiumEnvAdapter(env)


# ============================================================
# Bug metadata loading
# ============================================================

def load_bug_metadata(args) -> Tuple[List[str], Optional[str], Dict[str, Any]]:
    bug_module = import_first_module(DEFAULT_BUG_MODULE_CANDIDATES)
    bug_ids: List[str] = []
    bug_json_path = None
    raw: Dict[str, Any] = {}

    if bug_module is not None:
        for attr in ["ISSUE10_BUGS", "TARGET_BUGS", "BUG_IDS"]:
            if hasattr(bug_module, attr):
                val = getattr(bug_module, attr)
                if isinstance(val, dict):
                    bug_ids = list(val.keys())
                else:
                    bug_ids = [str(x) for x in list(val)]
                break
        if hasattr(bug_module, "BUG_JSON_PATH"):
            bug_json_path = str(getattr(bug_module, "BUG_JSON_PATH"))

    if args.bug_json_path:
        bug_json_path = args.bug_json_path

    if bug_json_path and Path(bug_json_path).exists():
        try:
            with open(bug_json_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            extracted = extract_bug_ids_from_json(raw)
            if extracted:
                bug_ids = extracted
        except Exception as e:
            print(f"[WARN] Failed to read bug json {bug_json_path}: {e}")

    if not bug_ids:
        bug_ids = [f"BUG_{i:02d}" for i in range(1, args.target_bug_count + 1)]

    return bug_ids, bug_json_path, raw


def extract_bug_ids_from_json(raw: Any) -> List[str]:
    ids: List[str] = []
    if isinstance(raw, dict):
        if "bugs" in raw and isinstance(raw["bugs"], list):
            for b in raw["bugs"]:
                if isinstance(b, dict):
                    bid = b.get("id") or b.get("bug_id") or b.get("name")
                    if bid:
                        ids.append(str(bid))
        elif all(isinstance(v, dict) for v in raw.values()):
            ids = [str(k) for k in raw.keys()]
        elif "ISSUE10_BUGS" in raw:
            val = raw["ISSUE10_BUGS"]
            if isinstance(val, list):
                ids = [str(x) for x in val]
    elif isinstance(raw, list):
        for b in raw:
            if isinstance(b, dict):
                bid = b.get("id") or b.get("bug_id") or b.get("name")
                if bid:
                    ids.append(str(bid))
            else:
                ids.append(str(b))
    return list(dict.fromkeys(ids))


# ============================================================
# Intrinsic reward wrappers
# ============================================================

@dataclass
class RNDConfig:
    scale: float = 1.0
    cap: float = 1.0
    feature_dim: int = 64
    hidden_dim: int = 128
    lr: float = 1e-4
    warmup_steps: int = 100
    normalize: bool = True
    update_every: int = 1
    device: str = "auto"


class RunningMeanStd:
    def __init__(self, eps: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps

    def update(self, x: float) -> None:
        x = float(x)
        batch_mean = x
        batch_var = 0.0
        batch_count = 1.0
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
        self.mean = new_mean
        self.var = max(m2 / total, 1e-8)
        self.count = total

    def norm(self, x: float) -> float:
        return float((x - self.mean) / (math.sqrt(self.var) + 1e-8))


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class RNDIntrinsicRewardWrapper(gym.Wrapper):
    """Online Random Network Distillation intrinsic reward wrapper.

    It adds info keys:
        rnd_reward, rnd_raw, rnd_norm, intrinsic_reward
    """
    def __init__(self, env: gym.Env, config: Optional[RNDConfig] = None):
        super().__init__(env)
        if torch is None:
            raise ImportError("PyTorch is required for RNDIntrinsicRewardWrapper")
        self.config = config or RNDConfig()
        self.device = self._resolve_device(self.config.device)
        self.predictor = None
        self.target = None
        self.optimizer = None
        self.rms = RunningMeanStd()
        self.global_step = 0
        self.input_dim = None

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _lazy_init(self, obs: Any) -> None:
        if self.predictor is not None:
            return
        vec = flatten_observation(obs)
        self.input_dim = int(vec.size)
        self.predictor = MLP(self.input_dim, self.config.hidden_dim, self.config.feature_dim).to(self.device)
        self.target = MLP(self.input_dim, self.config.hidden_dim, self.config.feature_dim).to(self.device)
        for p in self.target.parameters():
            p.requires_grad = False
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=self.config.lr)

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        obs = unpack_reset(result)
        self._lazy_init(obs)
        return make_reset_return(result)

    def step(self, action):
        result = self.env.step(action)
        obs, reward, done, info = unpack_step(result)
        self._lazy_init(obs)
        self.global_step += 1

        vec = flatten_observation(obs)
        x = torch.as_tensor(vec, dtype=torch.float32, device=self.device).view(1, -1)

        with torch.no_grad():
            target_feat = self.target(x)
        pred_feat = self.predictor(x)
        loss = torch.mean((pred_feat - target_feat) ** 2)
        raw = float(loss.detach().cpu().item())

        if self.global_step % max(1, self.config.update_every) == 0:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        if self.config.normalize:
            self.rms.update(raw)
            norm = max(0.0, self.rms.norm(raw))
        else:
            norm = raw

        rnd_reward = norm if self.global_step > self.config.warmup_steps else raw
        rnd_reward = float(np.clip(rnd_reward * self.config.scale, 0.0, self.config.cap))

        info = dict(info)
        info["rnd_raw"] = raw
        info["rnd_norm"] = norm
        info["rnd_reward"] = rnd_reward
        info["intrinsic_reward"] = float(info.get("intrinsic_reward", 0.0)) + rnd_reward
        info["extrinsic_reward"] = float(info.get("extrinsic_reward", reward))
        info["total_reward_before_wrapper"] = float(reward)

        return make_step_return(obs, float(reward) + rnd_reward, done, info)


@dataclass
class CAEConfig:
    spatial: float = 0.005
    state_action: float = 0.05
    object_action: float = 0.10
    sequence: float = 0.07
    opportunity: float = 0.15
    room_action: float = 0.03
    stagnation: float = 0.03
    stagnation_threshold: int = 4000
    cap: float = 1.0
    hash_obs_fallback: bool = True
    spatial_only: bool = False
    spatial_only_weight: float = 0.405
    enable_stagnation: bool = True


def cae_active_weights(config: CAEConfig) -> Dict[str, float]:
    """Return the CAE terms active for the selected experiment condition.

    The accepted Full implementation remains unchanged. The Spatial ablation
    retains only a direct Arena ``(XPos, ZPos)`` cell key and matches its
    initial reward budget to the sum of the six Full key weights.
    """
    if config.spatial_only:
        return {"spatial": float(config.spatial_only_weight)}
    return {
        "spatial": float(config.spatial),
        "state_action": float(config.state_action),
        "object_action": float(config.object_action),
        "opportunity": float(config.opportunity),
        "room_action": float(config.room_action),
        "sequence": float(config.sequence),
    }


class CAEIntrinsicRewardWrapper(gym.Wrapper):
    """Coverage-aware exploration reward wrapper.

    The wrapper tries to use semantic info keys from the Arena env when available.
    If not available, it falls back to quantized observation hashes and action.

    It adds info keys:
        cae_reward, cae_raw, intrinsic_reward
    """
    def __init__(self, env: gym.Env, config: Optional[CAEConfig] = None):
        super().__init__(env)
        self.config = config or CAEConfig()
        self.global_step = 0
        self.last_new_coverage_step = 0
        self.prev_action = None
        self.action_seq = deque(maxlen=3)

        self.visited_spatial = defaultdict(int)
        self.visited_state_action = defaultdict(int)
        self.visited_object_action = defaultdict(int)
        self.visited_sequence = defaultdict(int)
        self.visited_opportunity = defaultdict(int)
        self.visited_room_action = defaultdict(int)
        self.stagnation_trigger_count = 0

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.prev_action = None
        self.action_seq.clear()
        return make_reset_return(result)

    @staticmethod
    def novelty(count: int) -> float:
        return 1.0 / math.sqrt(float(count) + 1.0)

    def _key_context(self, obs: Any, info: Dict[str, Any], action: Any) -> Dict[str, Any]:
        pos = extract_spatial_ablation_position(info, obs) if self.config.spatial_only else extract_position(info)
        room = first_existing(info, ["room", "zone", "current_room", "current_zone", "region"])
        obj = first_existing(info, ["front_block", "nearby_block", "target_block", "object", "target_object", "look_block"])
        affordance = first_existing(info, ["affordance", "available_action", "opportunity", "interaction_opportunity"])

        if pos is None:
            obs_key = vector_hash(flatten_observation(obs), digits=10)
            pos_key = ("obs", obs_key)
        elif self.config.spatial_only:
            pos_key = (int(float(pos[0])), int(float(pos[2])))
        else:
            pos_key = tuple(int(round(float(x))) for x in pos[:3])

        if room is None:
            room = "unknown"
        if obj is None:
            obj = "none"
        if affordance is None:
            affordance = obj

        return {
            "spatial": pos_key,
            "state_action": (pos_key, int(action) if np.isscalar(action) else str(action)),
            "object_action": (str(obj), int(action) if np.isscalar(action) else str(action)),
            "opportunity": (str(affordance), int(action) if np.isscalar(action) else str(action)),
            "room_action": (str(room), int(action) if np.isscalar(action) else str(action)),
        }

    def step(self, action):
        result = self.env.step(action)
        obs, reward, done, info = unpack_step(result)
        info = dict(info)
        self.global_step += 1

        ctx = self._key_context(obs, info, action)
        self.action_seq.append(int(action) if np.isscalar(action) else str(action))
        seq_key = tuple(self.action_seq)

        raw = 0.0
        new_signal = False

        tables = {
            "spatial": self.visited_spatial,
            "state_action": self.visited_state_action,
            "object_action": self.visited_object_action,
            "opportunity": self.visited_opportunity,
            "room_action": self.visited_room_action,
            "sequence": self.visited_sequence,
        }
        keys = {**ctx, "sequence": seq_key}
        active_weights = cae_active_weights(self.config)

        for name, weight in active_weights.items():
            table = tables[name]
            key = keys[name]
            cnt = table[key]
            if cnt == 0:
                new_signal = True
            raw += float(weight) * self.novelty(cnt)
            table[key] += 1

        if new_signal:
            self.last_new_coverage_step = self.global_step

        stagnated = bool(
            self.config.enable_stagnation
            and (self.global_step - self.last_new_coverage_step) >= self.config.stagnation_threshold
        )
        if stagnated:
            raw += float(self.config.stagnation)
            self.last_new_coverage_step = self.global_step
            self.stagnation_trigger_count += 1

        cae_reward = float(np.clip(raw, 0.0, self.config.cap))

        info["cae_raw"] = raw
        info["cae_reward"] = cae_reward
        info["cae_stagnated"] = int(stagnated)
        info["cae_stagnation_trigger_count"] = int(self.stagnation_trigger_count)
        info["cae_new_key"] = int(new_signal)
        info["cae_mode"] = "spatial_only" if self.config.spatial_only else "full"
        info["intrinsic_reward"] = float(info.get("intrinsic_reward", 0.0)) + cae_reward
        info["extrinsic_reward"] = float(info.get("extrinsic_reward", reward))
        info["total_reward_before_wrapper"] = float(reward)

        self.prev_action = action
        return make_step_return(obs, float(reward) + cae_reward, done, info)


@dataclass
class HybridConfig:
    beta_rnd: float = 0.5
    beta_cae: float = 0.5
    cap: float = 1.0
    rnd: RNDConfig = field(default_factory=RNDConfig)
    cae: CAEConfig = field(default_factory=CAEConfig)


class RNDCAEHybridRewardWrapper(gym.Wrapper):
    """RND + CAE hybrid reward.

    To avoid double-adding rewards, this wrapper internally composes RND/CAE signals and
    adds the weighted hybrid reward once.
    """
    def __init__(self, env: gym.Env, config: Optional[HybridConfig] = None):
        super().__init__(env)
        self.config = config or HybridConfig()
        # Use internal helpers by wrapping a dummy chain is cumbersome, so implement RND parts here
        if torch is None:
            raise ImportError("PyTorch is required for RNDCAEHybridRewardWrapper")
        self.rnd_config = self.config.rnd
        self.cae_config = self.config.cae
        self.device = "cuda" if (self.rnd_config.device == "auto" and torch.cuda.is_available()) else ("cpu" if self.rnd_config.device == "auto" else self.rnd_config.device)
        self.predictor = None
        self.target = None
        self.optimizer = None
        self.rms = RunningMeanStd()
        self.global_step = 0

        self.cae_counts = {
            "spatial": defaultdict(int),
            "state_action": defaultdict(int),
            "object_action": defaultdict(int),
            "opportunity": defaultdict(int),
            "room_action": defaultdict(int),
            "sequence": defaultdict(int),
        }
        self.last_new_coverage_step = 0
        self.action_seq = deque(maxlen=3)
        self.stagnation_trigger_count = 0

    def _lazy_init(self, obs: Any) -> None:
        if self.predictor is not None:
            return
        dim = int(flatten_observation(obs).size)
        self.predictor = MLP(dim, self.rnd_config.hidden_dim, self.rnd_config.feature_dim).to(self.device)
        self.target = MLP(dim, self.rnd_config.hidden_dim, self.rnd_config.feature_dim).to(self.device)
        for p in self.target.parameters():
            p.requires_grad = False
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=self.rnd_config.lr)

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        obs = unpack_reset(result)
        self._lazy_init(obs)
        self.action_seq.clear()
        return make_reset_return(result)

    def _compute_rnd(self, obs: Any) -> Tuple[float, float, float]:
        self._lazy_init(obs)
        vec = flatten_observation(obs)
        x = torch.as_tensor(vec, dtype=torch.float32, device=self.device).view(1, -1)
        with torch.no_grad():
            target_feat = self.target(x)
        pred_feat = self.predictor(x)
        loss = torch.mean((pred_feat - target_feat) ** 2)
        raw = float(loss.detach().cpu().item())

        if self.global_step % max(1, self.rnd_config.update_every) == 0:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        if self.rnd_config.normalize:
            self.rms.update(raw)
            norm = max(0.0, self.rms.norm(raw))
        else:
            norm = raw
        reward = norm if self.global_step > self.rnd_config.warmup_steps else raw
        reward = float(np.clip(reward * self.rnd_config.scale, 0.0, self.rnd_config.cap))
        return reward, raw, norm

    def _compute_cae(self, obs: Any, info: Dict[str, Any], action: Any) -> Tuple[float, float, int, bool]:
        pos = extract_spatial_ablation_position(info, obs) if self.cae_config.spatial_only else extract_position(info)
        if pos is None:
            pos_key = ("obs", vector_hash(flatten_observation(obs), digits=10))
        elif self.cae_config.spatial_only:
            pos_key = (int(float(pos[0])), int(float(pos[2])))
        else:
            pos_key = tuple(int(round(float(x))) for x in pos[:3])
        room = first_existing(info, ["room", "zone", "current_room", "current_zone", "region"]) or "unknown"
        obj = first_existing(info, ["front_block", "nearby_block", "target_block", "object", "target_object", "look_block"]) or "none"
        opp = first_existing(info, ["affordance", "available_action", "opportunity", "interaction_opportunity"]) or obj
        act = int(action) if np.isscalar(action) else str(action)
        self.action_seq.append(act)

        keys = {
            "spatial": pos_key,
            "state_action": (pos_key, act),
            "object_action": (str(obj), act),
            "opportunity": (str(opp), act),
            "room_action": (str(room), act),
            "sequence": tuple(self.action_seq),
        }
        weights = cae_active_weights(self.cae_config)

        raw = 0.0
        new_signal = False
        for name, weight in weights.items():
            key = keys[name]
            cnt = self.cae_counts[name][key]
            if cnt == 0:
                new_signal = True
            raw += float(weight) * (1.0 / math.sqrt(float(cnt) + 1.0))
            self.cae_counts[name][key] += 1

        if new_signal:
            self.last_new_coverage_step = self.global_step
        stagnated = int(
            self.cae_config.enable_stagnation
            and (self.global_step - self.last_new_coverage_step) >= self.cae_config.stagnation_threshold
        )
        if stagnated:
            raw += self.cae_config.stagnation
            self.last_new_coverage_step = self.global_step
            self.stagnation_trigger_count += 1

        reward = float(np.clip(raw, 0.0, self.cae_config.cap))
        return reward, raw, stagnated, new_signal

    def step(self, action):
        result = self.env.step(action)
        obs, reward, done, info = unpack_step(result)
        info = dict(info)
        self.global_step += 1

        rnd_reward, rnd_raw, rnd_norm = self._compute_rnd(obs)
        cae_reward, cae_raw, cae_stagnated, cae_new_key = self._compute_cae(obs, info, action)
        hybrid = float(np.clip(self.config.beta_rnd * rnd_reward + self.config.beta_cae * cae_reward, 0.0, self.config.cap))

        info.update({
            "rnd_reward": rnd_reward,
            "rnd_raw": rnd_raw,
            "rnd_norm": rnd_norm,
            "cae_reward": cae_reward,
            "cae_raw": cae_raw,
            "cae_stagnated": cae_stagnated,
            "cae_stagnation_trigger_count": int(self.stagnation_trigger_count),
            "cae_new_key": int(cae_new_key),
            "cae_mode": "spatial_only" if self.cae_config.spatial_only else "full",
            "hybrid_reward": hybrid,
            "intrinsic_reward": hybrid,
            "extrinsic_reward": float(info.get("extrinsic_reward", reward)),
            "total_reward_before_wrapper": float(reward),
        })
        return make_step_return(obs, float(reward) + hybrid, done, info)


# ============================================================
# Logging wrapper
# ============================================================

def first_existing(info: Dict[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in info and info[k] is not None:
            return info[k]
    return None


def extract_position(info: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    candidates = [
        info.get("agent_pos"), info.get("position"), info.get("pos"),
        info.get("agent_position"), info.get("location"),
    ]
    for c in candidates:
        if c is None:
            continue
        if isinstance(c, dict):
            xs = [c.get("x"), c.get("y", 0.0), c.get("z")]
            if xs[0] is not None and xs[2] is not None:
                return float(xs[0]), float(xs[1]), float(xs[2])
        if isinstance(c, (list, tuple, np.ndarray)) and len(c) >= 2:
            if len(c) == 2:
                return float(c[0]), 0.0, float(c[1])
            return float(c[0]), float(c[1]), float(c[2])
    # common flat keys
    if "x" in info and "z" in info:
        return float(info["x"]), float(info.get("y", 0.0)), float(info["z"])
    if "agent_x" in info and "agent_z" in info:
        return float(info["agent_x"]), float(info.get("agent_y", 0.0)), float(info["agent_z"])
    return None


def extract_logged_position(info: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    """Extract coordinates for diagnostics without changing accepted CAE behavior.

    The accepted Arena wrapper falls back to an observation hash because
    ``extract_position`` does not consume Malmo's uppercase position keys.  New
    ablation logs should still retain the actual coordinates, so logging uses
    this broader extractor while reward computation remains backward-compatible.
    """
    pos = extract_position(info)
    if pos is not None:
        return pos
    if "XPos" in info and "ZPos" in info:
        return (
            float(info["XPos"]),
            float(info.get("YPos", 0.0)),
            float(info["ZPos"]),
        )
    return None


def extract_spatial_ablation_position(info: Dict[str, Any], obs: Any) -> Optional[Tuple[float, float, float]]:
    """Return direct Arena coordinates for the Spatial-only ablation.

    Full RND+CAE deliberately keeps the accepted observation-hash fallback.
    This ablation consumes Malmo's uppercase coordinate metadata; the first
    two observation entries (Arena ``x, z``) are a defensive fallback.
    """
    pos = extract_logged_position(info)
    if pos is not None:
        return pos
    vec = flatten_observation(obs)
    if vec.size >= 2:
        return float(vec[0]), 0.0, float(vec[1])
    return None


def flatten_bug_ids(value: Any) -> List[str]:
    ids: List[str] = []
    if value is None:
        return ids
    if isinstance(value, str):
        if value:
            ids.append(value)
    elif isinstance(value, dict):
        bid = value.get("bug_id") or value.get("id") or value.get("name")
        if bid:
            ids.append(str(bid))
        for key in ["bugs", "bug_ids", "detected_bugs", "new_bugs"]:
            if key in value:
                ids.extend(flatten_bug_ids(value[key]))
    elif isinstance(value, (list, tuple, set)):
        for x in value:
            ids.extend(flatten_bug_ids(x))
    return [x for x in ids if x]


def extract_detected_bugs(info: Dict[str, Any]) -> List[str]:
    keys = [
        "bug_id", "detected_bug", "detected_bugs", "new_bug", "new_bugs",
        "found_bug", "found_bugs", "bugs_found", "bug_events", "unique_bugs",
        "log/bug_id", "log/detected_bugs", "log/bugs_found",
    ]
    ids: List[str] = []
    for k in keys:
        if k in info:
            ids.extend(flatten_bug_ids(info[k]))

    # Boolean + id pattern
    if info.get("bug_found") or info.get("found_bug_flag") or info.get("is_bug"):
        for k in ["current_bug_id", "bug", "bug_name"]:
            if k in info:
                ids.extend(flatten_bug_ids(info[k]))

    return list(dict.fromkeys(str(x) for x in ids if x))


class ArenaLoggingWrapper(gym.Wrapper):
    """Creates paper-ready logs independent of the existing callback stack."""
    def __init__(
        self,
        env: gym.Env,
        log_dir: Union[str, Path],
        algorithm: str,
        target_bugs: Sequence[str],
        log_freq: int = DEFAULT_LOG_FREQ,
    ):
        super().__init__(env)
        self.log_dir = ensure_dir(log_dir)
        self.algorithm = algorithm
        self.target_bugs = list(target_bugs)
        self.target_bug_count = max(1, len(self.target_bugs))
        self.log_freq = int(log_freq)

        self.global_step = 0
        self.episode_idx = 0
        self.episode_step = 0
        self.unique_bugs: set[str] = set()
        self.last_reward = 0.0
        self.discovery_curve: List[Tuple[int, int]] = []
        self.progress_rows: List[Dict[str, Any]] = []
        self.event_count = 0
        self.cae_stagnation_trigger_count = 0

        self.progress_path = self.log_dir / "arena_progress.csv"
        self.events_path = self.log_dir / "bug_events.csv"
        self.episode_path = self.log_dir / "arena_episode_summary.csv"

        self._init_csvs()

    def _init_csvs(self) -> None:
        with open(self.progress_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.progress_fields())
            w.writeheader()
        with open(self.events_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.event_fields())
            w.writeheader()
        with open(self.episode_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.episode_fields())
            w.writeheader()

    @staticmethod
    def progress_fields() -> List[str]:
        return [
            "algorithm", "global_step", "episode_idx", "episode_step",
            "total_reward", "extrinsic_reward", "intrinsic_reward",
            "rnd_reward", "cae_reward", "hybrid_reward",
            "cae_mode", "cae_new_key", "cae_stagnated", "cae_stagnation_trigger_count",
            "unique_bug_count", "unique_bug_fraction", "unique_bugs",
            "visited_count", "coverage_count", "coverage_ratio",
            "agent_x", "agent_y", "agent_z", "action", "done",
        ]

    @staticmethod
    def event_fields() -> List[str]:
        return [
            "algorithm", "event_index", "global_step", "episode_idx", "episode_step",
            "bug_id", "is_first_detection", "unique_bug_count",
            "agent_x", "agent_y", "agent_z", "action", "evidence_json",
        ]

    @staticmethod
    def episode_fields() -> List[str]:
        return [
            "algorithm", "episode_idx", "global_step", "episode_step",
            "unique_bug_count", "episode_reward", "done_reason",
        ]

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.episode_idx += 1
        self.episode_step = 0
        self.episode_reward = 0.0
        return make_reset_return(result)

    def step(self, action):
        result = self.env.step(action)
        obs, reward, done, info = unpack_step(result)
        info = dict(info)
        self.global_step += 1
        self.episode_step += 1
        self.cae_stagnation_trigger_count = max(
            self.cae_stagnation_trigger_count,
            int(info.get("cae_stagnation_trigger_count", 0)),
        )
        self.episode_reward = getattr(self, "episode_reward", 0.0) + float(reward)
        self.last_reward = float(reward)

        detected = extract_detected_bugs(info)
        for bug_id in detected:
            self._log_bug_event(bug_id, info, action)

        self.discovery_curve.append((self.global_step, len(self.unique_bugs)))

        if self.global_step % self.log_freq == 0 or done or detected:
            self._log_progress(info, action, reward, done)

        if done:
            self._log_episode(info)

        # Add convenient info keys for existing callbacks as well.
        info["arena_unique_bug_count"] = len(self.unique_bugs)
        info["arena_unique_bugs"] = sorted(self.unique_bugs)
        info["arena_bug_fraction"] = len(self.unique_bugs) / self.target_bug_count

        return make_step_return(obs, reward, done, info)

    def _log_bug_event(self, bug_id: str, info: Dict[str, Any], action: Any) -> None:
        is_first = bug_id not in self.unique_bugs
        if is_first:
            self.unique_bugs.add(bug_id)
        self.event_count += 1
        pos = extract_logged_position(info) or (np.nan, np.nan, np.nan)
        row = {
            "algorithm": self.algorithm,
            "event_index": self.event_count,
            "global_step": self.global_step,
            "episode_idx": self.episode_idx,
            "episode_step": self.episode_step,
            "bug_id": bug_id,
            "is_first_detection": int(is_first),
            "unique_bug_count": len(self.unique_bugs),
            "agent_x": pos[0], "agent_y": pos[1], "agent_z": pos[2],
            "action": action,
            "evidence_json": json.dumps(to_jsonable(info), ensure_ascii=False),
        }
        with open(self.events_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.event_fields()).writerow(row)

    def _log_progress(self, info: Dict[str, Any], action: Any, reward: float, done: bool) -> None:
        pos = extract_logged_position(info) or (np.nan, np.nan, np.nan)
        visited = first_existing(info, [
            "visited_count", "unique_tiles_visited", "unique_cells_visited",
            "log/unique_tiles_visited", "episode_unique_tiles_visited",
        ])
        cov_count = first_existing(info, ["coverage_count", "unique_coverage_count", "unique_cells_modified"])
        cov_ratio = first_existing(info, ["coverage_ratio", "bug_coverage", "arena_coverage_ratio"])
        row = {
            "algorithm": self.algorithm,
            "global_step": self.global_step,
            "episode_idx": self.episode_idx,
            "episode_step": self.episode_step,
            "total_reward": float(reward),
            "extrinsic_reward": float(info.get("extrinsic_reward", reward - float(info.get("intrinsic_reward", 0.0)))),
            "intrinsic_reward": float(info.get("intrinsic_reward", 0.0)),
            "rnd_reward": float(info.get("rnd_reward", 0.0)),
            "cae_reward": float(info.get("cae_reward", 0.0)),
            "hybrid_reward": float(info.get("hybrid_reward", 0.0)),
            "cae_mode": str(info.get("cae_mode", "")),
            "cae_new_key": int(info.get("cae_new_key", 0)),
            "cae_stagnated": int(info.get("cae_stagnated", 0)),
            "cae_stagnation_trigger_count": int(info.get("cae_stagnation_trigger_count", 0)),
            "unique_bug_count": len(self.unique_bugs),
            "unique_bug_fraction": len(self.unique_bugs) / self.target_bug_count,
            "unique_bugs": ";".join(sorted(self.unique_bugs)),
            "visited_count": visited if visited is not None else "",
            "coverage_count": cov_count if cov_count is not None else "",
            "coverage_ratio": cov_ratio if cov_ratio is not None else "",
            "agent_x": pos[0], "agent_y": pos[1], "agent_z": pos[2],
            "action": action,
            "done": int(done),
        }
        self.progress_rows.append(row)
        with open(self.progress_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.progress_fields()).writerow(row)

    def _log_episode(self, info: Dict[str, Any]) -> None:
        reason = first_existing(info, ["done_reason", "termination_reason", "truncation_reason"]) or "done"
        row = {
            "algorithm": self.algorithm,
            "episode_idx": self.episode_idx,
            "global_step": self.global_step,
            "episode_step": self.episode_step,
            "unique_bug_count": len(self.unique_bugs),
            "episode_reward": float(getattr(self, "episode_reward", 0.0)),
            "done_reason": reason,
        }
        with open(self.episode_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.episode_fields()).writerow(row)

    def bug_auc(self, total_steps: Optional[int] = None) -> float:
        if not self.discovery_curve:
            return 0.0
        steps = np.array([s for s, _ in self.discovery_curve], dtype=float)
        counts = np.array([c for _, c in self.discovery_curve], dtype=float) / self.target_bug_count
        max_step = float(total_steps or np.nanmax(steps))
        if max_step <= 0:
            return 0.0
        return float(np.trapz(counts, steps) / max_step)

    def final_summary(self, total_steps: Optional[int] = None) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "global_step": self.global_step,
            "episode_idx": self.episode_idx,
            "final_unique_bug_count": len(self.unique_bugs),
            "final_unique_bug_fraction": len(self.unique_bugs) / self.target_bug_count,
            "unique_bugs": sorted(self.unique_bugs),
            "target_bugs": self.target_bugs,
            "target_bug_count": self.target_bug_count,
            "bug_auc": self.bug_auc(total_steps=total_steps),
            "event_count": self.event_count,
            "progress_rows": len(self.progress_rows),
            "cae_stagnation_trigger_count": int(self.cae_stagnation_trigger_count),
        }

    def close(self):
        try:
            json_dump(self.log_dir / "arena_logging_summary.json", self.final_summary())
        except Exception:
            pass
        return self.env.close()


# ============================================================
# Optional BEAGT / RELINE callback fallback
# ============================================================

class EpsilonStagnationBoostCallback(BaseCallback):
    """Simple reward-stagnation exploration boost fallback for BEAGT.

    This is only used if the project-specific RewardOnlyBEAGTCallback cannot be imported.
    """
    def __init__(self, start_after=10_000, check_every=500, window_size=20, bump_to=0.90, boost_steps=3000, verbose=0):
        super().__init__(verbose=verbose)
        self.start_after = start_after
        self.check_every = check_every
        self.window_size = window_size
        self.bump_to = bump_to
        self.boost_steps = boost_steps
        self.recent_rewards = deque(maxlen=window_size)
        self.boost_until = -1
        self.original_eps = None

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        if len(rewards):
            self.recent_rewards.append(float(np.mean(rewards)))

        if self.num_timesteps < self.start_after or self.num_timesteps % self.check_every != 0:
            return True

        if len(self.recent_rewards) >= self.window_size:
            std = float(np.std(self.recent_rewards))
            mean = float(np.mean(self.recent_rewards))
            # sparse reward stagnation heuristic
            if std < 1e-3 and mean <= 1e-3:
                self.boost_until = self.num_timesteps + self.boost_steps
                if self.verbose:
                    print(f"[BEAGT fallback] epsilon boost until step {self.boost_until}")

        if hasattr(self.model, "exploration_rate"):
            if self.original_eps is None:
                self.original_eps = float(self.model.exploration_rate)
            if self.num_timesteps <= self.boost_until:
                self.model.exploration_rate = max(float(self.model.exploration_rate), self.bump_to)

        return True


def maybe_make_beagt_callback(verbose: int = 1) -> BaseCallback:
    try:
        mod = importlib.import_module("envs.controllers")
        cls = getattr(mod, "RewardOnlyBEAGTCallback")
        return cls(start_after=10_000, check_every=500, verbose=verbose)
    except Exception:
        return EpsilonStagnationBoostCallback(verbose=verbose)


def maybe_make_reline_callback(verbose: int = 1) -> Optional[BaseCallback]:
    """Attach original RELINE only if the project exposes it.

    This function intentionally does NOT map RELINE to CAE dual-check.
    If no original RELINE callback is found, RELINE becomes an R_env-only DQN baseline.
    """
    candidates = [
        ("envs.controllers", "RELINECallback"),
        ("envs.controllers", "RelineCallback"),
        ("envs.controllers", "RELINEController"),
    ]
    for module, name in candidates:
        try:
            cls = getattr(importlib.import_module(module), name)
            try:
                return cls(start_after=10_000, check_every=500, verbose=verbose)
            except TypeError:
                return cls(verbose=verbose)
        except Exception:
            continue
    return None


# ============================================================
# Env creation
# ============================================================

def _force_env_port(env, port: int):
    """Best-effort port correction for legacy Malmo envs.

    Some project MalmoEnv implementations expose a ``port`` attribute and a
    ``client_pool`` that are created during ``__init__``.  If the constructor
    silently used its default port, this function rewrites both before the first
    mission is started.
    """
    try:
        setattr(env, "port", int(port))
    except Exception:
        pass

    try:
        import MalmoPython
        if hasattr(env, "client_pool"):
            env.client_pool = MalmoPython.ClientPool()
            env.client_pool.add(MalmoPython.ClientInfo("127.0.0.1", int(port)))
    except Exception:
        pass

    return env


def _instantiate_env_with_port(EnvCls, args, run_dir: Path, bug_json_path: Optional[str], seed: int):
    """Instantiate MalmoEnv while always trying to pass the requested port.

    The previous version relied only on ``inspect.signature``.  In some legacy
    project files the signature can be incomplete or hidden, causing ``port`` not
    to be passed and the environment to fall back to 10000.  This helper tries a
    small set of safe constructor patterns and then forcibly rewrites the port /
    client_pool as a final guard.
    """
    common_kwargs = {
        "port": int(args.port),
        "log_root": str(run_dir / "env_logs"),
        "seed": int(seed),
        "episode_seconds": float(args.episode_seconds),
        "mission_time_limit_ms": int(args.episode_seconds * 1000),
    }
    if bug_json_path:
        common_kwargs["bug_json_path"] = bug_json_path

    try:
        sig = inspect.signature(EnvCls)
        params = sig.parameters
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        filtered = {}
        for k, v in common_kwargs.items():
            if has_varkw or k in params:
                filtered[k] = v

        # Always prefer keyword construction with the requested port if possible.
        if "port" not in filtered:
            filtered["port"] = int(args.port)

        try:
            print(f"    [Env] constructor kwargs={filtered}")
            env = EnvCls(**filtered)
            return _force_env_port(env, args.port)
        except TypeError as e:
            print(f"    [Env] keyword construction failed: {e}")
    except Exception as e:
        print(f"    [Env] signature inspection failed: {e}")

    # Fallback 1: minimal keyword port only.
    try:
        print(f"    [Env] constructor kwargs={{'port': {int(args.port)}}}")
        env = EnvCls(port=int(args.port))
        return _force_env_port(env, args.port)
    except TypeError as e:
        print(f"    [Env] port-only keyword construction failed: {e}")

    # Fallback 2: positional port.
    try:
        print(f"    [Env] constructor positional port={int(args.port)}")
        env = EnvCls(int(args.port))
        return _force_env_port(env, args.port)
    except TypeError as e:
        print(f"    [Env] positional port construction failed: {e}")

    # Fallback 3: no-arg construction, then force port/client_pool.
    print("    [Env] constructor kwargs={} then force port/client_pool")
    env = EnvCls()
    return _force_env_port(env, args.port)


def create_base_env(args, run_dir: Path, bug_json_path: Optional[str], seed: int):
    env_module = args.env_module
    env_class = args.env_class

    if env_module == "auto":
        mod = import_first_module(DEFAULT_ENV_MODULE_CANDIDATES)
        if mod is None:
            raise ImportError(f"Could not import any env module from {DEFAULT_ENV_MODULE_CANDIDATES}")
        EnvCls = getattr(mod, env_class)
        env_module_name = mod.__name__
    else:
        EnvCls = import_object(env_module, env_class)
        env_module_name = env_module

    print(f"    [Env] module={env_module_name}, class={env_class}, requested_port={args.port}")
    env = _instantiate_env_with_port(EnvCls, args, run_dir, bug_json_path, seed)

    # Verify and print final port state.
    final_port = getattr(env, "port", None)
    print(f"    [Env] final env.port={final_port}, requested_port={args.port}")
    if final_port is not None and int(final_port) != int(args.port):
        print("    [WARN] env.port still differs from requested port after force correction.")

    # best-effort seeding
    for method_name in ["seed", "set_seed"]:
        if hasattr(env, method_name):
            try:
                getattr(env, method_name)(seed)
            except Exception:
                pass

    # Make env Monitor-compatible with Gymnasium SB3 builds.
    env = ensure_gymnasium_env(env)
    return env, env_module_name


def cae_config_for_algorithm(args, algorithm: str) -> CAEConfig:
    """Build a recorded, condition-specific CAE configuration.

    ``RND_CAE_FINAL`` keeps the accepted implementation exactly as configured.
    Spatial activates only the direct Arena ``(XPos, ZPos)`` cell key and
    matches the Full key-weight budget (0.405 by default). NoStag disables only the
    stagnation intervention.
    """
    data = dict(args.cae_config)
    if algorithm == "RND_CAE_SPATIAL":
        data["spatial_only"] = True
        data["enable_stagnation"] = True
    elif algorithm == "RND_CAE_NO_STAG":
        data["spatial_only"] = False
        data["enable_stagnation"] = False
    return CAEConfig(**data)


def create_env_with_retry(args, run_dir: Path, algorithm: str, target_bugs: List[str], bug_json_path: Optional[str], seed: int):
    last_err = None
    for i in range(args.retries):
        try:
            print(f"    [Env] Connecting to Malmo on port {args.port} (Attempt {i+1}/{args.retries})...")
            env, env_module_name = create_base_env(args, run_dir, bug_json_path, seed)

            # Reward wrappers before logging wrapper, so logger sees intrinsic info.
            if algorithm == "DQN_RND":
                env = RNDIntrinsicRewardWrapper(env, config=RNDConfig(**args.rnd_config))
            elif algorithm == "CAE_FINAL":
                env = CAEIntrinsicRewardWrapper(env, config=cae_config_for_algorithm(args, algorithm))
            elif algorithm in {"RND_CAE_FINAL", "RND_CAE_SPATIAL", "RND_CAE_NO_STAG"}:
                rnd_cfg = RNDConfig(**args.rnd_config)
                cae_cfg = cae_config_for_algorithm(args, algorithm)
                env = RNDCAEHybridRewardWrapper(env, config=HybridConfig(rnd=rnd_cfg, cae=cae_cfg, **args.hybrid_config))

            env = ArenaLoggingWrapper(env, log_dir=run_dir, algorithm=algorithm, target_bugs=target_bugs, log_freq=args.log_freq)
            env = Monitor(env, filename=str(run_dir / "monitor"))
            return env, env_module_name
        except Exception as e:
            last_err = e
            print(f"    [Env] Connection failed: {e}")
            traceback.print_exc()
            if i < args.retries - 1:
                time.sleep(args.retry_sleep)
    raise last_err


# ============================================================
# Config archiving
# ============================================================

def snapshot_experiment_files(run_dir: Path, bug_json_path: Optional[str], env_module_name: str, args) -> None:
    snap = ensure_dir(run_dir / "source_snapshot")
    root = project_root()

    # Copy bug JSON exactly.
    if bug_json_path and Path(bug_json_path).exists():
        shutil.copy2(bug_json_path, run_dir / "bug_definitions_snapshot.json")

    # Best-effort copy of important project files.
    candidates = [
        Path(__file__).resolve(),
        root / "envs" / "controllers.py",
        root / "envs" / "callbacks_issue10_consistent.py",
        root / "envs" / "callbacks_issue10_distributed.py",
        root / "envs" / "stats_callbacks_issue10_consistent.py",
        root / "envs" / "stats_callbacks_issue10_distributed.py",
        root / "envs" / "bug_targets_issue10.py",
        root / "envs" / "bug_targets_issue10_distributed.py",
        root / "envs" / "bug_detector.py",
        root / "envs" / "bug_detector_issue10.py",
    ]

    # module file
    try:
        mod = importlib.import_module(env_module_name)
        if hasattr(mod, "__file__"):
            candidates.append(Path(mod.__file__).resolve())
    except Exception:
        pass

    seen = set()
    for p in candidates:
        try:
            if p.exists() and p.is_file() and str(p) not in seen:
                seen.add(str(p))
                dst = snap / p.name
                shutil.copy2(p, dst)
        except Exception:
            continue

    json_dump(run_dir / "training_config.json", vars(args))


# ============================================================
# Model / callbacks
# ============================================================

def build_dqn_model(env: gym.Env, args, seed: int) -> DQN:
    return DQN(
        "MlpPolicy",
        env,
        verbose=args.sb3_verbose,
        seed=seed,
        buffer_size=args.buffer_size,
        learning_rate=args.learning_rate,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps,
        exploration_fraction=args.exploration_fraction,
        tensorboard_log=str(args.log_root),
    )


def build_callbacks(args, run_dir: Path, algorithm: str, target_bugs: List[str]) -> CallbackList:
    callbacks: List[BaseCallback] = []

    # Optional project-native callbacks. These are added if available but the wrapper logs independently.
    try:
        mod = importlib.import_module("envs.callbacks_issue10_consistent")
        if hasattr(mod, "UnifiedUniqueBugCallback"):
            callbacks.append(mod.UnifiedUniqueBugCallback(target_bugs=target_bugs, verbose=1))
        if hasattr(mod, "ExplorationCallback"):
            callbacks.append(mod.ExplorationCallback(map_size=(20, 20)))
        if hasattr(mod, "StepLoggingCallback"):
            callbacks.append(mod.StepLoggingCallback(freq=args.log_freq))
        if hasattr(mod, "FinalBugStatusLogger"):
            callbacks.append(mod.FinalBugStatusLogger(log_dir=str(run_dir), target_bugs=target_bugs))
        if hasattr(mod, "PeriodicBugStatusLogger"):
            callbacks.append(mod.PeriodicBugStatusLogger(log_dir=str(run_dir), target_bugs=target_bugs, save_every=5000))
    except Exception as e:
        print(f"    [Callbacks] Native callbacks skipped: {e}")

    try:
        mod = importlib.import_module("envs.stats_callbacks_issue10_consistent")
        if hasattr(mod, "StatisticalSummaryCallback"):
            callbacks.append(mod.StatisticalSummaryCallback(
                log_dir=str(run_dir),
                target_bugs=target_bugs,
                algorithm_name=algorithm,
                verbose=1,
            ))
    except Exception as e:
        print(f"    [Callbacks] Stats callback skipped: {e}")

    if algorithm == "BEAGT":
        callbacks.append(maybe_make_beagt_callback(verbose=1))
    elif algorithm == "RELINE":
        cb = maybe_make_reline_callback(verbose=1)
        if cb is not None:
            callbacks.append(cb)
        else:
            print("    [RELINE] Original RELINE callback not found. Running as R_env-only DQN baseline.")

    callbacks.append(CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(run_dir / "checkpoints"),
        name_prefix=f"{algorithm}_ISSUE10_FINAL",
    ))
    return CallbackList(callbacks)


# ============================================================
# Run execution
# ============================================================

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    set_random_seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def find_logging_wrapper(env: gym.Env) -> Optional[ArenaLoggingWrapper]:
    cur = env
    depth = 0
    while cur is not None and depth < 20:
        if isinstance(cur, ArenaLoggingWrapper):
            return cur
        cur = getattr(cur, "env", None)
        depth += 1
    return None


def run_random_policy(env: gym.Env, total_steps: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    obs = unpack_reset(env.reset())
    for _ in range(total_steps):
        action = int(env.action_space.sample())
        result = env.step(action)
        obs, reward, done, info = unpack_step(result)
        if done:
            obs = unpack_reset(env.reset())


def run_single_experiment(args, algorithm: str, run_idx: int, seed: int, target_bugs: List[str], bug_json_path: Optional[str]) -> bool:
    run_name = f"{algorithm}_ISSUE10_FINAL_{now_str()}_seed_{seed:03d}_run_{run_idx:02d}"
    run_dir = ensure_dir(Path(args.log_root) / algorithm / run_name if args.split_log_by_algo else Path(args.log_root) / run_name)

    print("\n" + "=" * 80)
    print(f"🚀 Starting {algorithm} run {run_idx}/{args.runs} | seed={seed} | port={args.port}")
    print(f"📁 log_dir={run_dir}")
    print("=" * 80)

    set_all_seeds(seed)

    env = None
    model = None
    env_module_name = "unknown"
    ok = False

    try:
        env, env_module_name = create_env_with_retry(args, run_dir, algorithm, target_bugs, bug_json_path, seed)
        snapshot_experiment_files(run_dir, bug_json_path, env_module_name, args)

        effective_cae_config = (
            asdict(cae_config_for_algorithm(args, algorithm))
            if algorithm in {"CAE_FINAL", "RND_CAE_FINAL", "RND_CAE_SPATIAL", "RND_CAE_NO_STAG"}
            else args.cae_config
        )
        algorithm_config = {
            "algorithm": algorithm,
            "seed": seed,
            "steps_per_run": args.steps,
            "episode_seconds": args.episode_seconds,
            "target_bugs": target_bugs,
            "rnd_config": args.rnd_config,
            "cae_config": effective_cae_config,
            "hybrid_config": args.hybrid_config,
        }
        json_dump(run_dir / "algorithm_config.json", algorithm_config)

        if algorithm == "RANDOM":
            run_random_policy(env, total_steps=args.steps, seed=seed)
        else:
            model = build_dqn_model(env, args, seed)
            callbacks = build_callbacks(args, run_dir, algorithm, target_bugs)
            model.learn(
                total_timesteps=args.steps,
                callback=callbacks,
                tb_log_name=f"{algorithm}_ISSUE10_FINAL",
                log_interval=999_999_999,
            )
            model.save(str(run_dir / f"{algorithm}_final_model"))

        logging_wrapper = find_logging_wrapper(env)
        wrapper_summary = logging_wrapper.final_summary(total_steps=args.steps) if logging_wrapper else {}
        final_summary = {
            "algorithm": algorithm,
            "run_idx": run_idx,
            "seed": seed,
            "steps": args.steps,
            "port": args.port,
            "log_dir": str(run_dir),
            "env_module": env_module_name,
            "finished_at": datetime.now().isoformat(),
            **wrapper_summary,
        }
        json_dump(run_dir / "run_summary.json", final_summary)
        ok = True
        print(f">>> ✅ Finished {algorithm} run {run_idx}. final_unique_bug_count={final_summary.get('final_unique_bug_count')}")

    except KeyboardInterrupt:
        print("\n>>> Interrupted by user.")
        raise
    except Exception as e:
        print(f"\n❌ Error in {algorithm} run {run_idx}: {e}")
        traceback.print_exc()
        json_dump(run_dir / "error_summary.json", {
            "algorithm": algorithm,
            "run_idx": run_idx,
            "seed": seed,
            "error": repr(e),
            "traceback": traceback.format_exc(),
        })
        ok = False
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        try:
            del model
        except Exception:
            pass
        try:
            del env
        except Exception:
            pass
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

        print(f">>> 🛑 Cooling down for {args.cooldown} seconds to avoid Malmo port reuse issues...")
        for remain in range(args.cooldown, 0, -10):
            print(f"    ... {remain} seconds remaining")
            time.sleep(min(10, remain))

    return ok


def parse_json_arg(s: Optional[str], default: Dict[str, Any]) -> Dict[str, Any]:
    if s is None or s == "":
        return dict(default)
    try:
        out = dict(default)
        out.update(json.loads(s))
        return out
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {e}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Final ISSUE10 Arena trainer")
    p.add_argument(
        "--algo",
        type=str,
        default="all",
        choices=["all", "ablations", "ablations_reverse", "all_with_ablations"] + ALGO_ORDER,
        help=(
            "'all' preserves the accepted six-method suite; 'ablations' runs Spatial then NoStag; "
            "'ablations_reverse' counterbalances that order."
        ),
    )
    p.add_argument("--port", type=int, default=10000)
    p.add_argument("--runs", type=int, default=DEFAULT_TOTAL_RUNS)
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS_PER_RUN)
    p.add_argument("--log-root", type=str, default="./logs_arena_issue10_final")
    p.add_argument("--split-log-by-algo", action="store_true", default=False,
                   help="If set, logs go to log_root/ALGO/run_name. Recommended when using --algo all.")
    p.add_argument("--cooldown", type=int, default=DEFAULT_COOLDOWN)
    p.add_argument("--log-freq", type=int, default=DEFAULT_LOG_FREQ)
    p.add_argument("--save-freq", type=int, default=DEFAULT_SAVE_FREQ)
    p.add_argument("--episode-seconds", type=int, default=90)
    p.add_argument("--target-bug-count", type=int, default=DEFAULT_TARGET_BUG_COUNT)
    p.add_argument("--seed-start", type=int, default=1)
    p.add_argument("--seeds", type=str, default="", help="Comma-separated seed list. Overrides seed-start/runs.")

    p.add_argument("--env-module", type=str, default="auto")
    p.add_argument("--env-class", type=str, default=DEFAULT_ENV_CLASS)
    p.add_argument("--bug-json-path", type=str, default="")
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--retry-sleep", type=int, default=10)

    # DQN params: keep close to previous experiments.
    p.add_argument("--buffer-size", type=int, default=50_000)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--learning-starts", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--train-freq", type=int, default=4)
    p.add_argument("--gradient-steps", type=int, default=1)
    p.add_argument("--target-update-interval", type=int, default=1000)
    p.add_argument("--exploration-initial-eps", type=float, default=0.99)
    p.add_argument("--exploration-final-eps", type=float, default=0.10)
    p.add_argument("--exploration-fraction", type=float, default=0.50)
    p.add_argument("--sb3-verbose", type=int, default=0)

    p.add_argument("--rnd-config-json", type=str, default="")
    p.add_argument("--cae-config-json", type=str, default="")
    p.add_argument("--hybrid-config-json", type=str, default="")
    return p


def finalize_args(args) -> Any:
    default_rnd = asdict(RNDConfig())
    default_cae = asdict(CAEConfig())
    default_hybrid = {"beta_rnd": 0.5, "beta_cae": 0.5, "cap": 1.0}
    args.rnd_config = parse_json_arg(args.rnd_config_json, default_rnd)
    args.cae_config = parse_json_arg(args.cae_config_json, default_cae)
    args.hybrid_config = parse_json_arg(args.hybrid_config_json, default_hybrid)
    args.log_root = str(Path(args.log_root))
    return args


def select_algorithms(algo_arg: str) -> List[str]:
    if algo_arg == "all":
        return list(BASE_ALGO_ORDER)
    if algo_arg == "ablations":
        return list(ABLATION_ALGO_ORDER)
    if algo_arg == "ablations_reverse":
        return list(reversed(ABLATION_ALGO_ORDER))
    if algo_arg == "all_with_ablations":
        return list(ALGO_ORDER)
    return [algo_arg]


def main() -> None:
    args = finalize_args(build_arg_parser().parse_args())

    if args.seeds.strip():
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
        args.runs = len(seeds)
    else:
        seeds = list(range(args.seed_start, args.seed_start + args.runs))

    target_bugs, bug_json_path, bug_raw = load_bug_metadata(args)
    ensure_dir(args.log_root)
    algos = select_algorithms(args.algo)
    json_dump(Path(args.log_root) / "final_experiment_manifest.json", {
        "created_at": datetime.now().isoformat(),
        "algo_arg": args.algo,
        "algorithms": algos,
        "runs": args.runs,
        "steps": args.steps,
        "seeds": seeds,
        "target_bugs": target_bugs,
        "bug_json_path": bug_json_path,
        "args": vars(args),
    })

    print("=" * 80)
    print("Final Arena ISSUE10 experiment")
    print(f"Algorithms : {algos}")
    print(f"Runs       : {args.runs}")
    print(f"Steps/run  : {args.steps}")
    print(f"Port       : {args.port}")
    print(f"Log root   : {args.log_root}")
    print(f"Seeds      : {seeds}")
    print(f"Bug count  : {len(target_bugs)}")
    print("=" * 80)

    all_ok = True
    for algorithm in algos:
        for idx, seed in enumerate(seeds, start=1):
            ok = run_single_experiment(args, algorithm, idx, seed, target_bugs, bug_json_path)
            all_ok = all_ok and ok
            if not ok and args.algo not in {"all", "ablations", "ablations_reverse", "all_with_ablations"}:
                print("[WARN] A run failed. Continuing to next run after cooldown.")

    print("\n" + "=" * 80)
    print(f"Final Arena experiment completed. all_ok={all_ok}")
    print("=" * 80)


if __name__ == "__main__":
    main()
