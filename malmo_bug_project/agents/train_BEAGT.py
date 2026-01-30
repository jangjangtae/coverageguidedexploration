import os
import time
import gc
import json
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
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

MALMO_PORT = 10004      
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
        print("📊 FINAL BUG REPORT (Spatial BEAGT)")
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

class SpatialEpsilonCallback(BaseCallback):

    def __init__(self, check_every=500,  
                 min_eps=0.10, 
                 bump_to=0.90,           
                 eps_bump_duration=5000, 
                 cooldown_steps=500,    
                 verbose=0):
        super().__init__(verbose)
        self.check_every = check_every
        self.min_eps = min_eps
        self.bump_to = bump_to
        self.eps_bump_duration = eps_bump_duration
        self.cooldown_steps = cooldown_steps
        
        self.last_check = 0
        self.bump_end_step = 0
        self.cooldown_end_step = 0
        self.last_visited_count = 0
        self.original_schedule = None
        self.visited_cells = set()

    def _on_training_start(self) -> None:
        self.original_schedule = self.model.exploration_schedule
        
        def dynamic_schedule(current_progress):
            if self.num_timesteps < self.bump_end_step:
                return self.bump_to
            return self.original_schedule(current_progress)
        
        self.model.exploration_schedule = dynamic_schedule

    def _on_step(self) -> bool:
        if not self.locals.get('infos') or len(self.locals['infos']) == 0:
            return True
        
        infos = self.locals['infos'][0]
        if 'XPos' not in infos or 'ZPos' not in infos:
            return True

        x = infos.get('XPos', 0)
        z = infos.get('ZPos', 0)
        
        cell = (int(x), int(z)) 
        
        if cell not in self.visited_cells:
            self.visited_cells.add(cell)
            
        current_visited_count = len(self.visited_cells)
        base_eps = self.original_schedule(self.model._current_progress_remaining)

        if self.num_timesteps > self.last_check + self.check_every:
            delta = current_visited_count - self.last_visited_count
            self.last_visited_count = current_visited_count
            self.last_check = self.num_timesteps
            
    
            if self.num_timesteps > self.cooldown_end_step:
            
                if delta <= 0 and base_eps < self.min_eps:
                    self.bump_end_step = self.num_timesteps + self.eps_bump_duration
                    self.cooldown_end_step = self.bump_end_step + self.cooldown_steps
                    
                    if self.verbose > 0:
                        print(f"🚀 [Spatial-BEAGT] STUCK! (No new cells in {self.check_every} steps). Boosting to {self.bump_to}!")

        self.model.current_actual_epsilon = self.model.exploration_rate
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
    run_name = f"BEAGT_REWARD_{time_str}_run_{run_index}"
    
    log_dir = f"./logs/{run_name}"
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"🚀 Starting Run {run_index}/{TOTAL_RUNS}: {run_name} (Spatial Mode)")
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
        
        SpatialEpsilonCallback(
            min_eps=0.20,          
            bump_to=0.90,          
            eps_bump_duration=5000, 
            cooldown_steps=500,    
            verbose=1
        ),

        CheckpointCallback(save_freq=100000, save_path=os.path.join(log_dir, "checkpoints"), name_prefix=run_name)
    ]
    
    print(f">>> Training Started for {STEPS_PER_RUN} steps...")
    try:
        model.learn(
            total_timesteps=STEPS_PER_RUN,
            callback=CallbackList(callbacks),
            tb_log_name="BEAGT_REWARD", 
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
    print(f"=== Starting Batch Experiment (Spatial BEAGT): {TOTAL_RUNS} runs ===")
    for i in range(1, TOTAL_RUNS + 1):
        if not run_single_experiment(i):
            break
    print(f"\n=== All {TOTAL_RUNS} runs completed! ===")

if __name__ == "__main__":
    main()
