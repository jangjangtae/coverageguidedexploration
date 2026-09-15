"""
maze_reward_wrappers.py

Reward wrappers for 21x21 maze re-experiments.

Place at:
    envs/maze_reward_wrappers.py
"""

import math
from collections import defaultdict
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class _RNDNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64):
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


class _ScalarEMA:
    def __init__(self, alpha: float = 0.01):
        self.alpha = float(alpha)
        self.mean = 0.0
        self.var = 1.0
        self.initialized = False

    def update(self, x: float):
        x = float(x)
        if not self.initialized:
            self.mean = x
            self.var = 1.0
            self.initialized = True
            return
        old_mean = self.mean
        self.mean = (1.0 - self.alpha) * self.mean + self.alpha * x
        diff = x - old_mean
        self.var = (1.0 - self.alpha) * self.var + self.alpha * diff * diff

    @property
    def std(self):
        return math.sqrt(max(self.var, 1e-8))


class RNDRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        rnd_weight: float = 1.0,
        intrinsic_cap: float = 1.0,
        hidden_dim: int = 128,
        output_dim: int = 64,
        lr: float = 1e-4,
        norm_alpha: float = 0.01,
        device: Optional[str] = None,
    ):
        super().__init__(env)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.rnd_weight = float(rnd_weight)
        self.intrinsic_cap = float(intrinsic_cap)

        input_dim = int(np.prod(env.observation_space.shape))
        self.target = _RNDNet(input_dim, hidden_dim, output_dim).to(self.device)
        self.predictor = _RNDNet(input_dim, hidden_dim, output_dim).to(self.device)
        for p in self.target.parameters():
            p.requires_grad = False
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=float(lr))
        self.ema = _ScalarEMA(alpha=float(norm_alpha))

    def _obs_tensor(self, obs):
        arr = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(arr).to(self.device)

    def _rnd_reward(self, obs):
        x = self._obs_tensor(obs)
        with torch.no_grad():
            target_feat = self.target(x)
        pred_feat = self.predictor(x)
        loss = torch.mean((pred_feat - target_feat) ** 2)
        raw = float(loss.detach().cpu().item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.ema.update(raw)
        norm = raw / (self.ema.std + 1e-8)
        norm = float(np.clip(norm, 0.0, self.intrinsic_cap))
        reward = self.rnd_weight * norm
        return reward, {
            "rnd_raw": raw,
            "rnd_norm": norm,
            "rnd_reward": reward,
            "rnd_running_mean": float(self.ema.mean),
            "rnd_running_std": float(self.ema.std),
        }

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        rnd_reward, rnd_info = self._rnd_reward(obs)
        info = dict(info or {})
        info.update(rnd_info)
        info["extrinsic_reward"] = float(reward)
        info["intrinsic_reward"] = float(rnd_reward)
        info["total_reward"] = float(reward + rnd_reward)
        return obs, float(reward + rnd_reward), terminated, truncated, info


class CAERewardWrapper(gym.Wrapper):
    def __init__(self, env, cae_weight: float = 1.0, cap: float = 1.0, config: Optional[Dict[str, float]] = None):
        super().__init__(env)
        self.cae_weight = float(cae_weight)
        self.cap = float(cap)
        self.config = {
            "spatial": 0.08,
            "state_action": 0.04,
            "object_action": 0.06,
            "sequence": 0.04,
            "opportunity": 0.06,
            "stagnation": 0.03,
            "stagnation_threshold": 1000,
        }
        if config:
            self.config.update(config)
        self.counts = defaultdict(int)
        self.prev_action_name = "none"
        self.steps_since_new_coverage = 0
        self.special_objects = {"diamond_block", "glass", "glowstone", "stone"}

    def reset(self, **kwargs):
        self.counts.clear()
        self.prev_action_name = "none"
        self.steps_since_new_coverage = 0
        return self.env.reset(**kwargs)

    def _novelty(self, key):
        n = self.counts[key]
        is_new = (n == 0)
        self.counts[key] += 1
        return 1.0 / math.sqrt(float(n) + 1.0), is_new

    def _decode_blocks(self, obs):
        try:
            return self.env.unwrapped.decode_blocks(obs)
        except Exception:
            inv = {0: "air", 1: "stone", 2: "bedrock", 3: "gold_block", 4: "diamond_block", 5: "glass", 6: "glowstone"}
            arr = np.asarray(obs).astype(int).flatten()
            return [inv.get(int(x), "unknown") for x in arr]

    def _cae_reward(self, obs, action, info):
        action_name = str(info.get("action_name", str(action)))
        x = int(np.floor(float(info.get("XPos", 0.0))))
        z = int(np.floor(float(info.get("ZPos", 0.0))))
        cell = (x, z)
        blocks = self._decode_blocks(obs)
        nearby_specials = sorted({b for b in blocks if b in self.special_objects})

        new_any = False
        r_spatial, is_new = self._novelty(("spatial", cell))
        new_any = new_any or is_new
        r_state_action, is_new = self._novelty(("state_action", cell, action_name))
        new_any = new_any or is_new

        object_vals = []
        for b in nearby_specials:
            v, is_new = self._novelty(("object_action", b, action_name))
            object_vals.append(v)
            new_any = new_any or is_new
        r_object_action = float(np.mean(object_vals)) if object_vals else 0.0

        r_sequence, is_new = self._novelty(("sequence", self.prev_action_name, action_name))
        new_any = new_any or is_new

        opp_vals = []
        for b in nearby_specials:
            if b in {"diamond_block", "glass", "stone"}:
                v, is_new = self._novelty(("opportunity", b, action_name, self.prev_action_name))
                opp_vals.append(v)
                new_any = new_any or is_new
        r_opportunity = float(np.mean(opp_vals)) if opp_vals else 0.0

        if new_any:
            self.steps_since_new_coverage = 0
        else:
            self.steps_since_new_coverage += 1

        threshold = int(self.config.get("stagnation_threshold", 1000))
        r_stag = 1.0 if self.steps_since_new_coverage >= threshold else 0.0

        cae_raw = (
            self.config["spatial"] * r_spatial
            + self.config["state_action"] * r_state_action
            + self.config["object_action"] * r_object_action
            + self.config["sequence"] * r_sequence
            + self.config["opportunity"] * r_opportunity
            + self.config["stagnation"] * r_stag
        )
        cae_norm = float(np.clip(cae_raw, 0.0, self.cap))
        cae_reward = self.cae_weight * cae_norm
        self.prev_action_name = action_name

        return cae_reward, {
            "cae_reward": float(cae_reward),
            "cae_raw": float(cae_raw),
            "cae_norm": float(cae_norm),
            "cae_spatial": float(r_spatial),
            "cae_state_action": float(r_state_action),
            "cae_object_action": float(r_object_action),
            "cae_sequence": float(r_sequence),
            "cae_opportunity": float(r_opportunity),
            "cae_stagnation": float(r_stag),
            "cae_steps_since_new": int(self.steps_since_new_coverage),
            "cae_coverage_keys": int(len(self.counts)),
        }

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        cae_reward, cae_info = self._cae_reward(obs, action, info)
        info = dict(info or {})
        info.update(cae_info)
        info["extrinsic_reward"] = float(reward)
        info["intrinsic_reward"] = float(cae_reward)
        info["total_reward"] = float(reward + cae_reward)
        return obs, float(reward + cae_reward), terminated, truncated, info


class RNDCAEHybridWrapper(gym.Wrapper):
    def __init__(self, env, rnd_weight: float = 0.5, cae_weight: float = 0.5, intrinsic_cap: float = 1.0, cae_config=None, device=None):
        super().__init__(env)
        self.rnd_part = RNDRewardWrapper(env, rnd_weight=1.0, intrinsic_cap=1.0, device=device)
        self.cae_part = CAERewardWrapper(env, cae_weight=1.0, cap=1.0, config=cae_config)
        self.rnd_weight = float(rnd_weight)
        self.cae_weight = float(cae_weight)
        self.intrinsic_cap = float(intrinsic_cap)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.cae_part.counts.clear()
        self.cae_part.prev_action_name = "none"
        self.cae_part.steps_since_new_coverage = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        rnd_reward, rnd_info = self.rnd_part._rnd_reward(obs)
        cae_reward, cae_info = self.cae_part._cae_reward(obs, action, info)
        hybrid = self.rnd_weight * rnd_reward + self.cae_weight * cae_reward
        hybrid = float(np.clip(hybrid, 0.0, self.intrinsic_cap))

        info = dict(info or {})
        info.update(rnd_info)
        info.update(cae_info)
        info["rnd_cae_hybrid_reward"] = hybrid
        info["extrinsic_reward"] = float(reward)
        info["intrinsic_reward"] = float(hybrid)
        info["total_reward"] = float(reward + hybrid)
        return obs, float(reward + hybrid), terminated, truncated, info
