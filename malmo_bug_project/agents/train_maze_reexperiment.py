"""
train_maze_reexperiment.py

Sequentially run representative algorithms on the 21x21 Malmo maze.

Algorithms:
- RANDOM       : random action baseline
- RELINE       : standard DQN / no dual-check booster in the maze setting
- BEAGT        : reward-only exploration boosting baseline
- DQN_RND      : RND intrinsic reward baseline
- CAE_FINAL    : CAE coverage intrinsic reward + CAE dual-check boosting
- RND_CAE_FINAL: RND+CAE hybrid intrinsic reward + CAE dual-check boosting

Place at:
    agents/train_maze_reexperiment.py

Example:
    python3 agents/train_maze_reexperiment.py --algo all --runs 5 --steps 100000 --port 10006
"""

import argparse
import csv
import gc
import json
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from envs.simple_voxel_maze_env_v3 import SimpleVoxelMazeEnv
from envs.maze_reward_wrappers import CAERewardWrapper, RNDCAEHybridWrapper, RNDRewardWrapper


ALGO_LIST = ["RANDOM", "RELINE", "BEAGT", "DQN_RND", "CAE_FINAL", "RND_CAE_FINAL"]


class MazeMetricsCallback(BaseCallback):
    """TensorBoard + CSV + robust summary logging for maze experiments."""

    def __init__(self, log_dir, algorithm_name, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.algorithm_name = algorithm_name
        self.log_freq = int(log_freq)
        self.csv_path = os.path.join(log_dir, "maze_progress.csv")
        self.summary_path = os.path.join(log_dir, "callback_summary.json")
        self.fieldnames = [
            "global_step", "algorithm", "maze_seed", "episode_step",
            "visited_count", "reachable_count", "coverage_ratio", "coverage_percent",
            "success", "distance_to_goal", "extrinsic_reward", "intrinsic_reward", "total_reward",
            "rnd_reward", "rnd_norm", "cae_reward", "hybrid_reward", "truncation_reason",
        ]
        self.max_coverage_ratio = 0.0
        self.max_visited_count = 0
        self.success_count = 0
        self.completed_episode_count = 0
        self.last_coverage_ratio = 0.0
        self.last_visited_count = 0
        self.last_reachable_count = 1
        self.coverage_samples = []
        self.logged_rows = []
        self._prev_done = False

    def _on_training_start(self):
        os.makedirs(self.log_dir, exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()

    def _write_summary(self):
        if self.coverage_samples:
            steps = np.array([s for s, _ in self.coverage_samples], dtype=float)
            vals = np.array([v for _, v in self.coverage_samples], dtype=float)
            if len(vals) >= 2 and steps[-1] > steps[0]:
                auc = float(np.trapz(vals, steps) / max(steps[-1] - steps[0], 1.0))
            else:
                auc = float(vals[-1])
            mean_cov = float(np.mean(vals))
        else:
            auc = 0.0
            mean_cov = 0.0

        summary = {
            "algorithm": self.algorithm_name,
            "timesteps": int(self.num_timesteps),
            "last_coverage_ratio": float(self.last_coverage_ratio),
            "last_visited_count": int(self.last_visited_count),
            "last_reachable_count": int(self.last_reachable_count),
            "max_coverage_ratio": float(self.max_coverage_ratio),
            "max_visited_count": int(self.max_visited_count),
            "mean_logged_coverage_ratio": mean_cov,
            "auc_logged_coverage_ratio": auc,
            "success_count": int(self.success_count),
            "completed_episode_count": int(self.completed_episode_count),
            "csv_path": self.csv_path,
        }
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        rewards = self.locals.get("rewards", [0.0])
        dones = self.locals.get("dones", [False])
        info = infos[0] if infos else {}

        visited = float(info.get("visited_count", 0))
        reachable = float(info.get("reachable_count", 1))
        coverage = float(info.get("coverage_ratio", visited / max(reachable, 1)))
        success = bool(info.get("success", False))

        extrinsic = float(info.get("extrinsic_reward", rewards[0] if len(rewards) else 0.0))
        intrinsic = float(info.get("intrinsic_reward", 0.0))
        total = float(info.get("total_reward", rewards[0] if len(rewards) else 0.0))
        rnd_norm = float(info.get("rnd_norm", 0.0))
        rnd_reward = float(info.get("rnd_reward", rnd_norm))
        cae_reward = float(info.get("cae_reward", 0.0))
        hybrid_reward = float(info.get("rnd_cae_hybrid_reward", 0.0))

        self.last_coverage_ratio = coverage
        self.last_visited_count = int(visited)
        self.last_reachable_count = int(reachable)
        self.max_coverage_ratio = max(self.max_coverage_ratio, coverage)
        self.max_visited_count = max(self.max_visited_count, int(visited))

        if success:
            self.success_count += 1

        done_now = bool(dones[0]) if len(dones) else False
        if done_now and not self._prev_done:
            self.completed_episode_count += 1
        self._prev_done = done_now

        self.logger.record("maze/visited_count", visited)
        self.logger.record("maze/reachable_count", reachable)
        self.logger.record("maze/coverage_ratio", coverage)
        self.logger.record("maze/coverage_percent", 100.0 * coverage)
        self.logger.record("maze/max_coverage_ratio", self.max_coverage_ratio)
        self.logger.record("maze/success", float(success))
        self.logger.record("maze/distance_to_goal", float(info.get("distance_to_goal", -1) or -1))
        self.logger.record("reward/extrinsic", extrinsic)
        self.logger.record("reward/intrinsic", intrinsic)
        self.logger.record("reward/total", total)
        self.logger.record("rnd/reward", rnd_reward)
        self.logger.record("rnd/norm", rnd_norm)
        self.logger.record("cae/reward", cae_reward)
        self.logger.record("rnd_cae/hybrid_reward", hybrid_reward)

        if self.num_timesteps % self.log_freq == 0:
            self.coverage_samples.append((int(self.num_timesteps), coverage))
            row = {
                "global_step": int(self.num_timesteps),
                "algorithm": self.algorithm_name,
                "maze_seed": info.get("maze_seed", ""),
                "episode_step": info.get("episode_step", ""),
                "visited_count": int(visited),
                "reachable_count": int(reachable),
                "coverage_ratio": coverage,
                "coverage_percent": 100.0 * coverage,
                "success": int(success),
                "distance_to_goal": info.get("distance_to_goal", ""),
                "extrinsic_reward": extrinsic,
                "intrinsic_reward": intrinsic,
                "total_reward": total,
                "rnd_reward": rnd_reward,
                "rnd_norm": rnd_norm,
                "cae_reward": cae_reward,
                "hybrid_reward": hybrid_reward,
                "truncation_reason": info.get("truncation_reason", ""),
            }
            self.logged_rows.append(row)
            with open(self.csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)
            self._write_summary()
        return True

    def _on_training_end(self):
        self._write_summary()


class RewardOnlyBoostCallback(BaseCallback):
    """BEAGT-style reward-only stagnation boosting."""
    def __init__(self, window_size=500, check_every=500, start_after=5000,
                 reward_std_threshold=0.01, bump_to=0.90, boost_steps=3000,
                 cooldown_steps=5000, verbose=0):
        super().__init__(verbose)
        self.window_size = int(window_size)
        self.check_every = int(check_every)
        self.start_after = int(start_after)
        self.reward_std_threshold = float(reward_std_threshold)
        self.bump_to = float(bump_to)
        self.boost_steps = int(boost_steps)
        self.cooldown_steps = int(cooldown_steps)
        self.recent_rewards = deque(maxlen=self.window_size)
        self.boost_left = 0
        self.cooldown_left = 0

    def _on_step(self):
        rewards = self.locals.get("rewards", [0.0])
        self.recent_rewards.append(float(rewards[0]) if len(rewards) else 0.0)
        if self.boost_left > 0:
            self.model.exploration_rate = max(float(self.model.exploration_rate), self.bump_to)
            self.boost_left -= 1
        elif self.cooldown_left > 0:
            self.cooldown_left -= 1
        if (self.num_timesteps > self.start_after and self.num_timesteps % self.check_every == 0
                and self.cooldown_left == 0 and len(self.recent_rewards) >= self.window_size):
            std = float(np.std(self.recent_rewards))
            mean = float(np.mean(self.recent_rewards))
            if std < self.reward_std_threshold and mean < 1e-6:
                self.boost_left = self.boost_steps
                self.cooldown_left = self.cooldown_steps
                if self.verbose:
                    print(f"[RewardOnlyBoost] step={self.num_timesteps} mean={mean:.4f} std={std:.4f}")
        self.logger.record("boost/reward_only_boost_left", self.boost_left)
        return True


class CAEDualCheckBoostCallback(BaseCallback):
    """CAE-style reward + coverage dual-check boosting. Not used for RELINE."""
    def __init__(self, window_size=500, check_every=500, start_after=5000,
                 reward_std_threshold=0.01, coverage_gain_threshold=0.002,
                 bump_to=0.90, boost_steps=3000, cooldown_steps=5000, verbose=0):
        super().__init__(verbose)
        self.window_size = int(window_size)
        self.check_every = int(check_every)
        self.start_after = int(start_after)
        self.reward_std_threshold = float(reward_std_threshold)
        self.coverage_gain_threshold = float(coverage_gain_threshold)
        self.bump_to = float(bump_to)
        self.boost_steps = int(boost_steps)
        self.cooldown_steps = int(cooldown_steps)
        self.recent_rewards = deque(maxlen=self.window_size)
        self.recent_coverage = deque(maxlen=self.window_size)
        self.boost_left = 0
        self.cooldown_left = 0

    def _on_step(self):
        rewards = self.locals.get("rewards", [0.0])
        infos = self.locals.get("infos", [{}])
        info = infos[0] if infos else {}
        self.recent_rewards.append(float(rewards[0]) if len(rewards) else 0.0)
        self.recent_coverage.append(float(info.get("coverage_ratio", 0.0)))
        if self.boost_left > 0:
            self.model.exploration_rate = max(float(self.model.exploration_rate), self.bump_to)
            self.boost_left -= 1
        elif self.cooldown_left > 0:
            self.cooldown_left -= 1
        if (self.num_timesteps > self.start_after and self.num_timesteps % self.check_every == 0
                and self.cooldown_left == 0 and len(self.recent_rewards) >= self.window_size):
            reward_std = float(np.std(self.recent_rewards))
            cov_gain = float(self.recent_coverage[-1] - self.recent_coverage[0])
            if reward_std < self.reward_std_threshold and cov_gain < self.coverage_gain_threshold:
                self.boost_left = self.boost_steps
                self.cooldown_left = self.cooldown_steps
                if self.verbose:
                    print(f"[CAEDualCheckBoost] step={self.num_timesteps} reward_std={reward_std:.4f} cov_gain={cov_gain:.4f}")
        self.logger.record("boost/cae_dual_check_boost_left", self.boost_left)
        return True


def build_env(algo, port, seed, log_dir, max_episode_steps, mission_time_limit_ms):
    env = SimpleVoxelMazeEnv(
        port=port,
        map_seed=seed,
        maze_size=21,
        max_episode_steps=max_episode_steps,
        mission_time_limit_ms=mission_time_limit_ms,
        step_penalty=0.0,
        goal_reward=100.0,
    )

    cae_config = {
        "spatial": 0.08,
        "state_action": 0.04,
        "object_action": 0.06,
        "sequence": 0.04,
        "opportunity": 0.06,
        "stagnation": 0.03,
        "stagnation_threshold": 1000,
    }

    if algo == "DQN_RND":
        env = RNDRewardWrapper(env, rnd_weight=1.0, intrinsic_cap=1.0)
    elif algo == "CAE_FINAL":
        env = CAERewardWrapper(env, cae_weight=1.0, cap=1.0, config=cae_config)
    elif algo == "RND_CAE_FINAL":
        env = RNDCAEHybridWrapper(env, rnd_weight=0.5, cae_weight=0.5, intrinsic_cap=1.0, cae_config=cae_config)

    env = Monitor(env, filename=os.path.join(log_dir, "monitor"))
    return env


def summarize_progress_csv(csv_path):
    rows = []
    if not os.path.exists(csv_path):
        return {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        return {}
    cov = np.array([float(r.get("coverage_ratio", 0.0) or 0.0) for r in rows], dtype=float)
    steps = np.array([float(r.get("global_step", 0.0) or 0.0) for r in rows], dtype=float)
    visited = np.array([float(r.get("visited_count", 0.0) or 0.0) for r in rows], dtype=float)
    success = np.array([float(r.get("success", 0.0) or 0.0) for r in rows], dtype=float)
    if len(cov) >= 2 and steps[-1] > steps[0]:
        auc = float(np.trapz(cov, steps) / max(steps[-1] - steps[0], 1.0))
    else:
        auc = float(cov[-1])
    return {
        "last_coverage_ratio": float(cov[-1]),
        "last_visited_count": int(visited[-1]),
        "max_coverage_ratio": float(np.max(cov)),
        "max_visited_count": int(np.max(visited)),
        "mean_logged_coverage_ratio": float(np.mean(cov)),
        "auc_logged_coverage_ratio": auc,
        "logged_success_count": int(np.sum(success)),
        "logged_rows": len(rows),
    }


def run_random(algo, port, run_idx, steps, seed, log_root, max_episode_steps, mission_time_limit_ms):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{algo}_MAZE21_{timestamp}_run_{run_idx}"
    log_dir = os.path.join(log_root, run_name)
    os.makedirs(log_dir, exist_ok=True)

    env = SimpleVoxelMazeEnv(
        port=port,
        map_seed=seed,
        maze_size=21,
        max_episode_steps=max_episode_steps,
        mission_time_limit_ms=mission_time_limit_ms,
        step_penalty=0.0,
        goal_reward=100.0,
    )

    rng = np.random.default_rng(seed)
    obs, info = env.reset()

    csv_path = os.path.join(log_dir, "maze_progress.csv")
    fields = [
        "global_step", "algorithm", "maze_seed", "episode_step",
        "visited_count", "reachable_count", "coverage_ratio", "coverage_percent",
        "success", "distance_to_goal", "extrinsic_reward", "truncation_reason",
    ]
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    successes = 0
    completed_episode_count = 0
    max_coverage_ratio = 0.0
    max_visited_count = 0
    last_info = {}

    for step in range(1, steps + 1):
        action = int(rng.integers(env.action_space.n))
        obs, reward, terminated, truncated, info = env.step(action)
        last_info = info
        cov = float(info.get("coverage_ratio", 0.0))
        max_coverage_ratio = max(max_coverage_ratio, cov)
        max_visited_count = max(max_visited_count, int(info.get("visited_count", 0)))
        if info.get("success", False):
            successes += 1
        if step % 1000 == 0:
            row = {
                "global_step": step,
                "algorithm": algo,
                "maze_seed": info.get("maze_seed", ""),
                "episode_step": info.get("episode_step", ""),
                "visited_count": info.get("visited_count", 0),
                "reachable_count": info.get("reachable_count", 1),
                "coverage_ratio": cov,
                "coverage_percent": 100.0 * cov,
                "success": int(bool(info.get("success", False))),
                "distance_to_goal": info.get("distance_to_goal", ""),
                "extrinsic_reward": reward,
                "truncation_reason": info.get("truncation_reason", ""),
            }
            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)
        if terminated or truncated:
            completed_episode_count += 1
            obs, info = env.reset()

    summary = {
        "algorithm": algo,
        "run_idx": run_idx,
        "seed": seed,
        "steps": steps,
        "success_count": int(successes),
        "completed_episode_count": int(completed_episode_count),
        "last_coverage_ratio": float(last_info.get("coverage_ratio", 0.0)),
        "last_visited_count": int(last_info.get("visited_count", 0)),
        "last_reachable_count": int(last_info.get("reachable_count", 1)),
        "max_coverage_ratio": float(max_coverage_ratio),
        "max_visited_count": int(max_visited_count),
        "log_dir": log_dir,
    }
    summary.update(summarize_progress_csv(csv_path))
    with open(os.path.join(log_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    env.close()
    return summary


def run_dqn(algo, port, run_idx, steps, seed, log_root, max_episode_steps, mission_time_limit_ms):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{algo}_MAZE21_{timestamp}_run_{run_idx}"
    log_dir = os.path.join(log_root, run_name)
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 70)
    print(f"[RUN] {run_name}")
    print(f"Algo   : {algo}")
    print(f"Port   : {port}")
    print(f"Seed   : {seed}")
    print(f"Horizon: {max_episode_steps} steps, safety timeout={mission_time_limit_ms} ms")
    print(f"Logdir : {log_dir}")
    print("=" * 70)

    env = build_env(algo, port, seed, log_dir, max_episode_steps, mission_time_limit_ms)
    model = DQN(
        "MlpPolicy", env, verbose=0,
        buffer_size=50_000,
        learning_rate=1e-4,
        learning_starts=1000,
        batch_size=128,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_initial_eps=0.99,
        exploration_final_eps=0.10,
        exploration_fraction=0.5,
        tensorboard_log=log_dir,
        seed=seed,
    )

    metrics_cb = MazeMetricsCallback(log_dir, algo, log_freq=1000)
    callbacks = [
        metrics_cb,
        CheckpointCallback(save_freq=100000, save_path=os.path.join(log_dir, "checkpoints"), name_prefix=run_name),
    ]

    if algo == "BEAGT":
        callbacks.append(RewardOnlyBoostCallback(verbose=1))
    elif algo in {"CAE_FINAL", "RND_CAE_FINAL"}:
        callbacks.append(CAEDualCheckBoostCallback(verbose=1))
    # RELINE intentionally has no dual-check callback here.

    model.learn(total_timesteps=steps, callback=CallbackList(callbacks), tb_log_name=f"{algo}_MAZE21", log_interval=999_999_999)
    model.save(os.path.join(log_dir, f"{run_name}_final"))

    summary = {
        "algorithm": algo,
        "run_idx": run_idx,
        "seed": seed,
        "steps": steps,
        "log_dir": log_dir,
    }
    summary.update(metrics_cb._write_summary())
    summary.update(summarize_progress_csv(os.path.join(log_dir, "maze_progress.csv")))

    with open(os.path.join(log_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    env.close()
    del model
    del env
    gc.collect()
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="all", choices=["all"] + ALGO_LIST)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--port", type=int, default=10006)
    p.add_argument("--seed", type=int, default=20260515)
    p.add_argument("--log-root", default="./logs_maze_reexperiment")
    p.add_argument("--cooldown", type=int, default=100, help="Seconds to wait between runs. Default is 100 to avoid Malmo port reuse issues.")
    p.add_argument("--max-episode-steps", type=int, default=4500)
    p.add_argument("--mission-time-limit-ms", type=int, default=120000)
    return p.parse_args()


def main():
    args = parse_args()
    algos = ALGO_LIST if args.algo == "all" else [args.algo]
    os.makedirs(args.log_root, exist_ok=True)

    all_summaries = []
    for algo in algos:
        for run_idx in range(1, args.runs + 1):
            run_seed = int(args.seed + run_idx * 1000 + ALGO_LIST.index(algo) * 100000)
            try:
                if algo == "RANDOM":
                    summary = run_random(algo, args.port, run_idx, args.steps, run_seed, args.log_root,
                                         args.max_episode_steps, args.mission_time_limit_ms)
                else:
                    summary = run_dqn(algo, args.port, run_idx, args.steps, run_seed, args.log_root,
                                      args.max_episode_steps, args.mission_time_limit_ms)
                all_summaries.append(summary)
            except KeyboardInterrupt:
                print("Interrupted by user.")
                raise
            except Exception as e:
                print(f"[ERROR] {algo} run={run_idx}: {e}")

            print(f">>> Cooling down for {args.cooldown} seconds to avoid Malmo port reuse issues")
            for left in range(args.cooldown, 0, -10):
                print(f"    ... {left} seconds remaining")
                time.sleep(min(10, left))

    summary_path = os.path.join(args.log_root, "maze_reexperiment_all_summaries.json")
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("=" * 70)
    print("Maze re-experiment completed.")
    print(f"Summary: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
