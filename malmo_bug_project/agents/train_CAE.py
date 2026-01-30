import os
import time
import gc
import json
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from envs.malmo_env import MalmoEnv
from envs.callbacks import (
    BugDetectionCallback, 
    UniqueBugCallback, 
    ExplorationCallback,
    DetailedBugLogger, 
    TrajectoryLogger
)

# [설정]
MALMO_PORT = 10008       
TOTAL_RUNS = 10
STEPS_PER_RUN = 100_000
LOG_FREQ = 1000

class StepLoggingCallback(BaseCallback):
    def __init__(self, freq=1000, verbose=0):
        super().__init__(verbose)
        self.freq = freq
    def _on_step(self) -> bool:
        if self.num_timesteps % self.freq == 0:
            self.logger.dump(self.num_timesteps)
        return True

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
        print("📊 FINAL BUG REPORT (CAE)")
        print("="*40)
        for bug_name in self.target_bugs:
            is_found = (bug_name in self.found_bugs)
            result_data["details"][bug_name] = is_found
            status_icon = "✅ FOUND" if is_found else "❌ MISSED"
            print(f"{bug_name:<25} : {status_icon}")

        save_path = os.path.join(self.log_dir, "final_bug_status.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=4)
        print("="*40)
        print(f"📄 Report saved to: {save_path}\n")

class MomentumSpatialCallback(BaseCallback):
    def __init__(self, check_every=500, min_eps=0.10, bump_to=0.90, 
                 start_after=10000,       
                 displacement_threshold=5.0, 
                 reward_threshold=0.5,
                 base_boost_steps=3000,   
                 momentum_bonus=2000,     
                 max_accumulated_boost=15000, 
                 verbose=0):
        super().__init__(verbose)
        self.check_every = check_every
        self.min_eps = min_eps
        self.bump_to = bump_to
        self.start_after = start_after 
        self.displacement_threshold = displacement_threshold
        self.reward_threshold = reward_threshold # [NEW]
        
        self.base_boost_steps = base_boost_steps
        self.momentum_bonus = momentum_bonus
        self.max_accumulated_boost = max_accumulated_boost
        
        self.last_check = 0
        self.last_visited_count = 0
        self.last_pos = None 
        
        self.reward_history = deque(maxlen=check_every)
        
        # 상태 변수
        self.is_boosting = False
        self.boost_end_step = 0
        self.original_schedule = None
        
        self.local_visited = set()

    def _on_training_start(self) -> None:
        self.original_schedule = self.model.exploration_schedule
        def dynamic_schedule(current_progress):
            if self.num_timesteps < self.boost_end_step:
                self.is_boosting = True
                return self.bump_to
            
            self.is_boosting = False
            return self.original_schedule(current_progress)
        self.model.exploration_schedule = dynamic_schedule

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [0])
        self.reward_history.append(np.mean(rewards))

        if not self.locals.get('infos') or len(self.locals['infos']) == 0:
            return True
        
        infos = self.locals['infos'][0]
        x = infos.get('XPos', 0)
        z = infos.get('ZPos', 0)
        cell = (int(x), int(z))
        
        dones = self.locals.get("dones", [False])[0]
        if dones:
            if self.is_boosting and self.verbose > 0:
                 print(f"🔄 [Reset] Episode Finished. Stopping Boost.")
            self.is_boosting = False
            self.boost_end_step = 0
            self.local_visited.clear()
            self.last_visited_count = 0
            self.last_pos = None
            self.last_check = self.num_timesteps
            self.reward_history.clear()
            return True

        if cell not in self.local_visited:
            self.local_visited.add(cell)
            
        current_visited_count = len(self.local_visited)
        base_eps = self.original_schedule(self.model._current_progress_remaining)
      
        if self.is_boosting and (current_visited_count > self.last_visited_count):
            remaining = self.boost_end_step - self.num_timesteps
            if remaining < self.max_accumulated_boost:
                self.boost_end_step += self.momentum_bonus
                if self.verbose > 0:
                    print(f"🔥 [Momentum] New cell found! Boost EXTENDED by {self.momentum_bonus} steps! (Ends at {self.boost_end_step})")

        if not self.is_boosting and self.num_timesteps > self.start_after:
            if self.num_timesteps > self.last_check + self.check_every:
                delta = current_visited_count - self.last_visited_count
                current_pos = np.array([x, z])
                dist = 0.0
                if self.last_pos is not None:
                    dist = np.linalg.norm(current_pos - self.last_pos)
                
                avg_reward = np.mean(self.reward_history) if self.reward_history else 0.0
                is_reward_stuck = avg_reward < self.reward_threshold

                self.last_visited_count = current_visited_count
                self.last_pos = current_pos
                self.last_check = self.num_timesteps
                
                is_spatially_stuck = (delta <= 0 and dist < self.displacement_threshold)
                
                if is_spatially_stuck and is_reward_stuck and base_eps < self.min_eps:
                    self.is_boosting = True
                    self.boost_end_step = self.num_timesteps + self.base_boost_steps
                    
                    if self.verbose > 0:
                        print(f"🚀 [Stagnation] DUAL-CHECK TRIGGERED!")
                        print(f"   - Spatial: Moved {dist:.1f} blocks (Threshold: {self.displacement_threshold})")
                        print(f"   - Reward: Avg {avg_reward:.4f} (Threshold: {self.reward_threshold})")
                        print(f"   -> Boosting START for {self.base_boost_steps} steps.")

        if not self.is_boosting:
             self.last_visited_count = current_visited_count
             self.last_pos = np.array([x, z])
             
        self.model.CAE_is_boosting = 1 if self.is_boosting else 0
        return True

def create_env_with_retry(port, log_dir, retries=5):
    for i in range(retries):
        try:
            print(f"    [Env] Connecting to Malmo on port {port} (Attempt {i+1}/{retries})...")
            env = MalmoEnv(port=port, log_root="runs")
            env = Monitor(env, filename=os.path.join(log_dir, "monitor"))
            return env
        except Exception as e:
            print(f"    [Env] Connection failed: {e}")
            if i < retries - 1:
                time.sleep(10)
            else:
                raise e

def run_single_experiment(run_index):
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"CAE_{time_str}_run_{run_index}"
    
    log_dir = f"./logs/{run_name}"
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"🚀 Starting Run {run_index}/{TOTAL_RUNS}: {run_name} (CAE)")
    print(f"🎯 Target Port: {MALMO_PORT}")
    print(f"{'='*60}")

    try:
        env = create_env_with_retry(MALMO_PORT, log_dir)
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=0,  
        buffer_size=50_000,       
        learning_rate=0.0001,
        learning_starts=1000,     
        batch_size=128,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_initial_eps=0.99, 
        exploration_final_eps=0.1,    
        exploration_fraction=0.5,     
        tensorboard_log=log_dir
    )

    callbacks = [
        BugDetectionCallback(),
        UniqueBugCallback(verbose=1),
        ExplorationCallback(),
        StepLoggingCallback(freq=LOG_FREQ),
        
        TrajectoryLogger(log_dir=log_dir),
        DetailedBugLogger(log_dir=log_dir),
        FinalBugStatusLogger(log_dir=log_dir),

        MomentumSpatialCallback(
            check_every=500,        
            min_eps=0.20,           
            bump_to=0.90,           
            start_after=10000,      
            displacement_threshold=5.0, 
            reward_threshold=0.1,
            base_boost_steps=3000,  
            momentum_bonus=2000,    
            verbose=1
        ),

        CheckpointCallback(save_freq=100000, save_path=os.path.join(log_dir, "checkpoints"), name_prefix=run_name)
    ]
    
    print(f">>> Training Started for {STEPS_PER_RUN} steps...")
    try:
        model.learn(
            total_timesteps=STEPS_PER_RUN,
            callback=CallbackList(callbacks),
            tb_log_name="CAE", 
            log_interval=999_999_999 
        )
        
        save_path = os.path.join(log_dir, f"{run_name}_final")
        model.save(save_path)
        print(f">>> Run {run_index} Finished.")
        
    except KeyboardInterrupt:
        print("\n>>> Interrupted.")
        env.close()
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
    finally:
        env.close()
        del model
        del env
        gc.collect()
        
        wait_time = 100
        print(f">>> 🛑 Cooling down for {wait_time} seconds...")
        for i in range(wait_time, 0, -10):
            print(f"    ... {i} seconds remaining")
            time.sleep(10)
    
    return True

def main():
    print(f"=== Starting Batch Experiment (CAE): {TOTAL_RUNS} runs ===")
    for i in range(1, TOTAL_RUNS + 1):
        if not run_single_experiment(i):
            break
    print(f"\n=== All {TOTAL_RUNS} runs completed! ===")

if __name__ == "__main__":
    main()
