import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
import json
import os


class BugDetectionCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.total_bugs = 0
        self.rollout_bugs = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])[0]
        if infos.get("bug_detected", False):
            count = len(infos.get("bug_ids", []))
            self.total_bugs += count
            self.rollout_bugs += count
            
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record("custom/bugs_found_total", self.total_bugs)
        self.logger.record("custom/bugs_found_rollout", self.rollout_bugs)
        self.rollout_bugs = 0 # 초기화

class UniqueBugCallback(BaseCallback):
    """
    발견한 버그의 '종류'가 몇 개인지 기록합니다 (최대 7개).
    - custom/unique_bugs_count: 현재까지 찾은 버그 종류 수
    """
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.found_bug_ids = set()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])[0]
        if infos.get("bug_detected", False):
            current_ids = infos.get("bug_ids", [])
            for bid in current_ids:
                if bid not in self.found_bug_ids:
                    self.found_bug_ids.add(bid)
                    if self.verbose > 0:
                        print(f"\n[🎉 NEW BUG TYPE] {bid} (Total Unique: {len(self.found_bug_ids)}/10)")
        
        self.logger.record("custom/unique_bugs_count", len(self.found_bug_ids))
        return True


class ExplorationCallback(BaseCallback):

    def __init__(self, map_size=(20, 20), verbose=0):
        super().__init__(verbose)
        self.map_w, self.map_h = map_size
        self.total_cells = self.map_w * self.map_h 
        self.visited_global = set() # 전체 학습 동안 방문한 곳

    def _on_step(self) -> bool:
        env_wrapper = self.training_env.envs[0]

        if hasattr(env_wrapper, "unwrapped"):
            base_env = env_wrapper.unwrapped
        else:
            base_env = env_wrapper
        
        if hasattr(base_env, "visited_cells"):
            current_visited = base_env.visited_cells
            self.visited_global.update(current_visited)
        
        visited_count = len(self.visited_global)
        coverage_rate = (visited_count / self.total_cells) * 100.0
        
        self.logger.record("custom/visited_cells", visited_count)
        self.logger.record("custom/exploration_rate", coverage_rate)
        
        return True

class EpsilonCallback(BaseCallback):

    def __init__(
        self,
        reward_window=200,
        check_every=500,
        std_threshold=0.05,
        min_eps=0.10,
        bump_to=0.50,
        eps_bump_duration=3000,
        cooldown_steps=5000,
        verbose=0,
    ):
        super().__init__(verbose)
        from collections import deque
        self.rew_buf = deque(maxlen=int(reward_window))
        self.check_every = check_every
        self.std_threshold = std_threshold
        self.min_eps = min_eps
        self.bump_to = bump_to
        self.eps_bump_duration = eps_bump_duration
        self.cooldown_steps = cooldown_steps
        
        self._bump_active_until = -1
        self._cooldown_until = -1
        self._last_checked_step = 0

    def _apply_eps(self, val: float):
        self.model.exploration_rate = float(val)

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", None)
        if rewards is not None:
            self.rew_buf.append(float(np.mean(rewards)))

        step = self.num_timesteps

        if step <= self._bump_active_until:
            self._apply_eps(self.bump_to)
            self.logger.record("eps/boost_active", 1)
            self.logger.record("eps/value", self.model.exploration_rate)
            self.model.current_actual_epsilon = self.current_eps
            return True

        if step - self._last_checked_step >= self.check_every and len(self.rew_buf) == self.rew_buf.maxlen:
            self._last_checked_step = step
            std = np.std(self.rew_buf)
            eps = self.model.exploration_rate

            if step >= self._cooldown_until and std < self.std_threshold and eps <= self.min_eps:
                self._bump_active_until = step + self.eps_bump_duration
                self._cooldown_until = self._bump_active_until + self.cooldown_steps
                if self.verbose > 0:
                    print(f"\n[EpsilonCallback] ⚠️ Stagnation detected! Boosting Epsilon to {self.bump_to}")

        self.logger.record("eps/boost_active", 0)
        self.logger.record("eps/value", self.model.exploration_rate)
        return True

class TrajectoryLogger(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.data = []
        self.last_x = 0
        self.last_z = 0

    def _on_training_start(self):
        self.data = []

    def _get_room_id(self, x, z):
        if x < 9 and z < 9: return 1   # Room 1 (Start)
        if x < 9 and z > 9: return 2   # Room 2 (Top-Left)
        if x > 9 and z > 9: return 4   # Room 4 (Top-Right)
        if x > 9 and z < 9: return 3   # Room 3 (End)
        return 0 # 복도(Crossway)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])[0]
        x = infos.get("XPos", 0)
        z = infos.get("ZPos", 0)

        current_eps = 0.0
        if hasattr(self.model, "exploration_rate"): # Random Agent
            current_eps = self.model.exploration_rate
        elif hasattr(self.model, "exploration_schedule"): # DQN
             current_eps = self.model.exploration_schedule(self.num_timesteps / self.locals['total_timesteps'])
        
        real_eps = infos.get("epsilon", current_eps) 
        is_boosting = 1 if real_eps > 0.5 else 0 # 0.5 이상이면 부스팅으로 간주


        room_id = self._get_room_id(x, z)
        

        self.data.append({
            "step": self.num_timesteps,
            "episode": len(self.model.ep_info_buffer), # 현재 에피소드 번호
            "x": round(x, 2),
            "z": round(z, 2),
            "room_id": room_id,
            "epsilon": round(real_eps, 3),
            "is_boosting": is_boosting
        })
        
        return True

    def _on_training_end(self):

        df = pd.DataFrame(self.data)
        save_path = os.path.join(self.log_dir, "trajectory_log.csv")
        df.to_csv(save_path, index=False)
        print(f"saved trajectory log to {save_path}")

class DetailedBugLogger(BaseCallback):

    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.bug_history = []
        self.bug_list = [
            "BUG_REVERSE_TURN", "BUG_TELEPORT_TRAP", "BUG_DEAD_ZONE", 
            "BUG_SUPER_JUMP", "BUG_SEQUENCE_ERROR", "BUG_INTERACT_FAIL", 
            "BUG_TRANSMUTATION", "BUG_SKY_FREEZE", "BUG_HOTBAR_ERROR", "BUG_BREAK_CRASH"
        ]

    def _on_step(self) -> bool:
        # 에피소드 종료 시점(dones=True)에만 기록
        dones = self.locals.get("dones", [False])[0]
        if dones:
            infos = self.locals.get("infos", [{}])[0]
            found_bugs = infos.get("detected_bugs", []) # malmo_env.py에서 넘겨줘야 함

            #if infos.get("bug_detected"): 
            #    print(f"DEBUG INFO: {infos}")
            
            row = {"episode": len(self.bug_history) + 1}
            total_found = 0
            for bug_id in self.bug_list:
                is_found = 1 if bug_id in found_bugs else 0
                row[bug_id] = is_found
                total_found += is_found
            
            row["total_found"] = total_found
            self.bug_history.append(row)
        return True

    def _on_training_end(self):
        if self.bug_history:
            df = pd.DataFrame(self.bug_history)
            save_path = os.path.join(self.log_dir, "bug_matrix.csv")
            df.to_csv(save_path, index=False)
            print(f"✅ Detailed bug log saved: {save_path}")

class FinalBugStatusLogger(BaseCallback):

    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.found_bugs = set()

        self.target_bugs = [
            "BUG_REVERSE_TURN", "BUG_TELEPORT_TRAP", "BUG_DEAD_ZONE", 
            "BUG_SUPER_JUMP", "BUG_SEQUENCE_ERROR", "BUG_INTERACT_FAIL", 
            "BUG_TRANSMUTATION", "BUG_SKY_FREEZE", "BUG_HOTBAR_ERROR", "BUG_BREAK_CRASH"
        ]

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])[0]
        
        current_step_bugs = infos.get("bug_ids", [])
        for bug in current_step_bugs:
            self.found_bugs.add(bug)
            
        env_accumulated = infos.get("detected_bugs", [])
        for bug in env_accumulated:
            self.found_bugs.add(bug)
            
        return True

    def _on_training_end(self) -> None:
        result_data = {
            "meta": {
                "total_steps": self.num_timesteps,
                "found_count": len(self.found_bugs),
                "total_target": len(self.target_bugs),
                "success_rate": f"{(len(self.found_bugs)/len(self.target_bugs))*100:.1f}%"
            },
            "details": {}
        }

        print("\n" + "="*40)
        print("📊 FINAL BUG REPORT")
        print("="*40)
        
        for bug_name in self.target_bugs:
            is_found = (bug_name in self.found_bugs)
            result_data["details"][bug_name] = is_found

            status_icon = "✅ FOUND" if is_found else "❌ MISSED"
            print(f"{bug_name:<25} : {status_icon}")

        # JSON 파일 저장
        save_path = os.path.join(self.log_dir, "final_bug_status.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=4)
            
        print("="*40)
        print(f"📄 Report saved to: {save_path}\n")
