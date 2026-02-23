import sys
import numpy as np
import pandas as pd
from datetime import datetime
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

CONFIG = {
    "run_name": "pure_rl_baseline",
    "port": 10010,          # 
    "total_steps": 100000,  # 
    "learning_rate": 1e-4,  
    "buffer_size": 10000,
    "exploration_frac": 0.1, # 
    "final_eps": 0.05        # 
}

class SimpleDataLogger(BaseCallback):
    def __init__(self, run_name):
        super().__init__()
        self.data = []
        self.log_path = f"{run_name}.csv"
        
    def _on_step(self):
        current_eps = self.model.exploration_rate

        infos = self.locals.get('infos', [{}])[0]
        visited_count = infos.get('visited_count', 0)

        self.data.append({
            "step": self.num_timesteps,
            "visited_count": visited_count,
            "epsilon": current_eps
        })

        if self.num_timesteps % 1000 == 0:
            self.save_log()
            print(f"[Step {self.num_timesteps:6d}] Visited: {visited_count:3d} | Eps: {current_eps:.3f}")
        return True
        
    def save_log(self):
        pd.DataFrame(self.data).to_csv(self.log_path, index=False)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_ID = f"{CONFIG['run_name']}_{timestamp}"
    
    print(f"🚀 [Pure RL Start] {run_ID}")
    print(f"📌 Steps: {CONFIG['total_steps']} | Final Eps: {CONFIG['final_eps']}")

    env = SimpleVoxelEnv(port=CONFIG["port"]) 
    
    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=CONFIG["learning_rate"], 
        buffer_size=CONFIG["buffer_size"], 
        exploration_fraction=CONFIG["exploration_frac"], 
        exploration_final_eps=CONFIG["final_eps"]
    )

    logger = SimpleDataLogger(run_name=run_ID)

    try:
        model.learn(total_timesteps=CONFIG["total_steps"], callback=logger)
        print("✅ Training Finished.")
    except KeyboardInterrupt:
        print("\n🛑 Training Interrupted by User.")
    finally:
        logger.save_log()
        env.close()
        print(f"🏁 Log saved to {run_ID}.csv")

if __name__ == "__main__":
    main()
