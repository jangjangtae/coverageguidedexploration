import sys
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

# [환경 로드]
try:
    from envs.simple_voxel_env import SimpleVoxelEnv
except ImportError:
    print("❌ Error: 'envs/simple_voxel_env.py' not found.")
    sys.exit(1)

# ==============================================================================
# [설정] 실험 파라미터 관리
# ==============================================================================
CONFIG = {
    "run_name": "run_reward_based",
    "port": 10008,               # Baseline 전용 포트
    "total_steps": 100000,       # 총 학습 스텝
    "start_after": 100000,        # 정체 감지 시작 시점
    "check_every": 1000,         # 감지 주기
    "reward_window": 1000,       # 보상 기록 윈도우 크기
    "std_threshold": 0.001,      # 정체 판단 기준 (표준편차)
    "boost_eps": 1.0,            # 부스팅 시 Epsilon 값
    "normal_eps": 0.05           # 최종 Epsilon 값
}

# ==============================================================================
# 1. [Baseline] Reward-based BEAGT Callback
# ==============================================================================
class RewardBasedBEAGTCallback(BaseCallback):
    """
    [Baseline] 보상 기반 탐험 제어
    - 보상 변화가 없으면(Std < Threshold) -> 부스팅 ON (랜덤 탐색)
    - 보상을 획득하면(Reward > 0) -> 부스팅 OFF
    """
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
        
        # [A] 성공 감지: 보상을 받으면 즉시 부스팅 해제
        if rewards > 0:
            if self.is_boosting and self.verbose > 0:
                print("✅ [Baseline] Reward received! Boost OFF.")
            self.is_boosting = False
            self.reward_history.clear()
            self._update_model_status()
            return True

        # [B] 정체 감지 (일정 주기마다 체크)
        if self.num_timesteps > self.cfg["start_after"]:
            if self.num_timesteps > self.last_check + self.cfg["check_every"]:
                self.last_check = self.num_timesteps
                self._check_stagnation()
        
        self._update_model_status()
        return True

    def _check_stagnation(self):
        """보상 기록의 표준편차를 확인하여 정체 여부 판단"""
        if len(self.reward_history) < self.cfg["reward_window"]:
            return

        std_dev = np.std(self.reward_history)
        
        # 표준편차가 임계값보다 낮으면 정체로 판단
        if std_dev < self.cfg["std_threshold"]:
            if not self.is_boosting:
                self.is_boosting = True
                if self.verbose > 0:
                    print(f"⚠️ [Baseline] Stagnation detected (std={std_dev:.4f}). Boost ON!")

    def _update_model_status(self):
        """모델 객체에 현재 상태 주입 (로거용)"""
        self.model.is_boosting = self.is_boosting

# ==============================================================================
# 2. [Logger] 데이터 수집기
# ==============================================================================
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

        # 콘솔 출력 (1000 스텝마다)
        if self.num_timesteps % 1000 == 0:
            self.save_log()
            status = "🔥FAILING" if is_boosting else "Normal"
            print(f"[Step {self.num_timesteps:6d}] Visited: {visited_count:3d} | Eps: {current_eps:.2f} | {status}")
        return True
        
    def save_log(self):
        pd.DataFrame(self.data).to_csv(self.log_path, index=False)

# ==============================================================================
# 3. Main Execution
# ==============================================================================
def main():
    # 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_ID = f"{CONFIG['run_name']}_{timestamp}"
    
    print(f"🚀 [Baseline Experiment] {run_ID}")
    print(f"📌 Config: Port={CONFIG['port']}, Steps={CONFIG['total_steps']}")

    # 환경 및 모델 초기화
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

    # 콜백 초기화
    baseline_callback = RewardBasedBEAGTCallback(cfg=CONFIG, verbose=1)
    logger = DataLogger(run_name=run_ID)

    # 학습 시작
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
