import sys
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

try:
    from envs.simple_voxel_env import SimpleVoxelEnv
except ImportError:
    print("❌ Error: 'envs/simple_voxel_env.py' not found.")
    sys.exit(1)

# ==============================================================================
# [setting]
# ==============================================================================
CONFIG = {
    "run_name": "run_reward_based",
    "port": 10008,               # port
    "total_steps": 100000,       # step
    "start_after": 100000,        # 
    "check_every": 1000,         # 
    "reward_window": 1000,       # 
    "std_threshold": 0.001,      # 
    "boost_eps": 1.0,            # 
    "normal_eps": 0.05           # 
}


class RewardBasedBEAGTCallback(BaseCallback):
    def __init__(self, cfg, verbose=0):
        super().__init__(verbose)
        self.cfg = cfg
        self.reward_history = deque(maxlen=cfg["reward_window"])
        self.last_check = 0
        self.is_boosting = False
        self.original_schedule = None

    def _on_training_start(self) -> None:
        # 동적 Epsilon 스케줄링 설정
        self.original_schedule = self.model.exploration_schedule
        def dynamic_schedule(current_progress):
            return self.cfg["boost_eps"] if self.is_boosting else self.original_schedule(current_progress)
        self.model.exploration_schedule = dynamic_schedule

    def _on_step(self) -> bool:
        rewards = self.locals.get('rewards', [0])[0]
        self.reward_history.append(rewards)

        if rewards > 0:
            if self.is_boosting and self.verbose > 0:
                print("✅ [Baseline] Reward received! Boost OFF.")
            self.is_boosting = False
            self.reward_history.clear()
            self._update_model_status()
            return True

        if self.num_timesteps > self.cfg["start_after"]:
            if self.num_timesteps > self.last_check + self.cfg["check_every"]:
                self.last_check = self.num_timesteps
                self._check_stagnation()
        
        self._update_model_status()
        return True

    def _check_stagnation(self):
        if len(self.reward_history) < self.cfg["reward_window"]:
            return

        std_dev = np.std(self.reward_history)
        
        if std_dev < self.cfg["std_threshold"]:
            if not self.is_boosting:
                self.is_boosting = True
                if self.verbose > 0:
                    print(f"⚠️ [Baseline] Stagnation detected (std={std_dev:.4f}). Boost ON!")

    def _update_model_status(self):
        self.model.is_boosting = self.is_boosting

class DataLogger(BaseCallback):
    def __init__(self, run_name):
        super().__init__()
        self.data = []
        self.log_path = f"{run_name}.csv"
        
    def _on_step(self):
        # 모델에서 상태 플래그 가져오기
        is_boosting = 1 if getattr(self.model, 'is_boosting', False) else 0
        
        # 정보 추출
        infos = self.locals.get('infos', [{}])[0]
        visited_count = infos.get('visited_count', 0)
        current_eps = self.model.exploration_rate

        # 데이터 저장
        self.data.append({
            "step": self.num_timesteps,
            "visited_count": visited_count,
            "epsilon": current_eps,
            "is_boosting": is_boosting
        })

        if self.num_timesteps % 1000 == 0:
            self.save_log()
            status = "🔥FAILING" if is_boosting else "Normal"
            print(f"[Step {self.num_timesteps:6d}] Visited: {visited_count:3d} | Eps: {current_eps:.2f} | {status}")
        return True
        
    def save_log(self):
        pd.DataFrame(self.data).to_csv(self.log_path, index=False)

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_ID = f"{CONFIG['run_name']}_{timestamp}"
    
    print(f"🚀 [Baseline Experiment] {run_ID}")
    print(f"📌 Config: Port={CONFIG['port']}, Steps={CONFIG['total_steps']}")

    env = SimpleVoxelEnv(port=CONFIG["port"]) 
    
    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=1e-4, 
        buffer_size=10000, 
        exploration_fraction=0.1, 
        exploration_final_eps=CONFIG["normal_eps"]
    )

    baseline_callback = RewardBasedBEAGTCallback(cfg=CONFIG, verbose=1)
    logger = DataLogger(run_name=run_ID)

    try:
        model.learn(total_timesteps=CONFIG["total_steps"], callback=[logger, baseline_callback])
        print("✅ Training Finished.")
    except KeyboardInterrupt:
        print("\n🛑 Training Interrupted by User.")
    finally:
        logger.save_log()
        env.close()
        print("🏁 Cleanup Done.")

if __name__ == "__main__":
    main()
