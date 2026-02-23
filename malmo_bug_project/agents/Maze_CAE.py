import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from gymnasium import spaces
import MalmoPython
import json
import random

# ==============================================================================
# 0. [환경] Procedural Safe Maze (기존 유지)
# ==============================================================================
class SimpleVoxelEnv(gym.Env):
    def __init__(self, port=10006, map_seed=None):
        super().__init__()
        self.grid_shape = (5, 3, 5) 
        self.obs_dim = 5 * 3 * 5
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.obs_dim,), dtype=np.float32)
        self.action_list = ["move 1", "turn 1", "turn -1", "move 0"] 
        self.action_space = spaces.Discrete(len(self.action_list))
        
        self.port = port
        self.agent_host = MalmoPython.AgentHost()
        self.client_pool = MalmoPython.ClientPool()
        self.client_pool.add(MalmoPython.ClientInfo("127.0.0.1", self.port))
        
        self.visited_cells = set()
        self.block_map = {"air": 0, "stone": 1, "bedrock": 2, "gold_block": 3, "diamond_block": 4, "glass": 5, "glowstone": 6}
        
        self.maze_size = 21 
        self.start_height = 2
        self.goal_height = 2
        
        self.map_seed = map_seed
        self.spawn_x = 0
        self.spawn_z = 0
        self.goal_x = 0
        self.goal_z = 0

    def _generate_maze(self):
        maze = np.ones((self.maze_size, self.maze_size), dtype=int)
        start_x, start_z = 1, 1
        maze[start_x, start_z] = 0
        stack = [(start_x, start_z)]
        while stack:
            x, z = stack[-1]
            neighbors = []
            for dx, dz in [(-2,0), (2,0), (0,-2), (0,2)]:
                nx, nz = x + dx, z + dz
                if 0 < nx < self.maze_size and 0 < nz < self.maze_size and maze[nx, nz] == 1:
                    neighbors.append((nx, nz))
            if neighbors:
                nx, nz = random.choice(neighbors)
                maze[(x + nx)//2, (z + nz)//2] = 0
                maze[nx, nz] = 0
                stack.append((nx, nz))
            else:
                stack.pop()
        goal_x, goal_z = self.maze_size - 2, self.maze_size - 2
        maze[goal_x, goal_z] = 0
        return maze, (start_x, start_z), (goal_x, goal_z)

    def _get_mission_xml(self):
        if self.map_seed is None: seed = int(time.time() * 1000) % 10000
        else: seed = self.map_seed
        random.seed(seed)
        
        maze, start, goal = self._generate_maze()
        offset = self.maze_size // 2
        
        draw_cmds = ""
        draw_cmds += f'<DrawCuboid x1="-{offset+5}" y1="1" z1="-{offset+5}" x2="{offset+5}" y2="10" z2="{offset+5}" type="air"/>'
        draw_cmds += f'<DrawCuboid x1="-{offset+5}" y1="1" z1="-{offset+5}" x2="{offset+5}" y2="1" z2="{offset+5}" type="bedrock"/>'
        
        for r in range(self.maze_size):
            for c in range(self.maze_size):
                x = r - offset
                z = c - offset
                if maze[r, c] == 1:
                    draw_cmds += f'<DrawCuboid x1="{x}" y1="2" z1="{z}" x2="{x}" y2="4" z2="{z}" type="stone"/>'
                else:
                    draw_cmds += f'<DrawBlock x="{x}" y="1" z="{z}" type="gold_block"/>'
                    if random.random() < 0.2:
                        draw_cmds += f'<DrawBlock x="{x}" y="4" z="{z}" type="glowstone"/>'

        sx, sz = start[0] - offset, start[1] - offset
        gx, gz = goal[0] - offset, goal[1] - offset
        self.spawn_x, self.spawn_z, self.goal_x, self.goal_z = sx, sz, gx, gz
        
        draw_cmds += f'<DrawCuboid x1="{sx-1}" y1="2" z1="{sz-1}" x2="{sx+1}" y2="4" z2="{sz+1}" type="air"/>' 
        draw_cmds += f'<DrawBlock x="{sx}" y="1" z="{sz}" type="stone"/>'
        draw_cmds += f'<DrawBlock x="{gx}" y="1" z="{gz}" type="diamond_block"/>'
        draw_cmds += f'<DrawBlock x="{gx}" y="2" z="{gz}" type="glowstone"/>'
        
        draw_cmds += f'<DrawCuboid x1="-{offset+1}" y1="2" z1="-{offset+1}" x2="{offset+1}" y2="4" z2="-{offset+1}" type="glass"/>' 
        draw_cmds += f'<DrawCuboid x1="-{offset+1}" y1="2" z1="{offset+1}" x2="{offset+1}" y2="4" z2="{offset+1}" type="glass"/>'
        draw_cmds += f'<DrawCuboid x1="-{offset+1}" y1="2" z1="-{offset+1}" x2="-{offset+1}" y2="4" z2="{offset+1}" type="glass"/>'
        draw_cmds += f'<DrawCuboid x1="{offset+1}" y1="2" z1="-{offset+1}" x2="{offset+1}" y2="4" z2="{offset+1}" type="glass"/>'

        return f'''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
        <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
          <About><Summary>Safe Procedural Maze {seed}</Summary></About>
          <ServerSection>
            <ServerHandlers>
              <FlatWorldGenerator generatorString="3;7,2;1;"/>
              <DrawingDecorator>{draw_cmds}</DrawingDecorator>
              <ServerQuitWhenAnyAgentFinishes/>
            </ServerHandlers>
          </ServerSection>
          <AgentSection mode="Survival">
            <Name>VoxelAgent</Name>
            <AgentStart><Placement x="{sx}.5" y="2" z="{sz}.5" yaw="0"/></AgentStart>
            <AgentHandlers>
              <ObservationFromGrid>
                <Grid name="surrounding_blocks">
                  <min x="-2" y="-1" z="-2"/>
                  <max x="2"  y="1"  z="2"/>
                </Grid>
              </ObservationFromGrid>
              <ObservationFromFullStats/>
              <ContinuousMovementCommands/>
              <AbsoluteMovementCommands/>
            </AgentHandlers>
          </AgentSection>
        </Mission>'''

    def step(self, action_idx):
        action = self.action_list[action_idx]
        try: self.agent_host.sendCommand(action)
        except RuntimeError: pass
        time.sleep(0.02)
        
        ws = self.agent_host.getWorldState()
        obs = self._get_observation(ws)
        reward = 0.0
        done = False
        info = {}
        x, y, z = 0, 2, 0 
        has_valid_data = False

        if ws.number_of_observations_since_last_state > 0:
            try:
                msg = ws.observations[-1].text
                data = json.loads(msg)
                if 'XPos' in data and 'YPos' in data:
                    x, y, z = data.get(u'XPos'), data.get(u'YPos'), data.get(u'ZPos')
                    has_valid_data = True
                info['XPos'], info['YPos'], info['ZPos'] = x, y, z
                cell = (int(x), int(z)) 
                if cell not in self.visited_cells: self.visited_cells.add(cell)
            except: pass

        if has_valid_data:
            dist_to_goal = abs(x - self.goal_x) + abs(z - self.goal_z)
            if dist_to_goal < 1.0:
                reward = 100.0
                done = True
                print(f"🎉 [Success] Maze Solved! Seed: {self.map_seed}")
                
        info["visited_count"] = len(self.visited_cells)
        if not ws.is_mission_running: done = True
        return obs, reward, done, False, info

    def _get_observation(self, ws):
        grid_vec = np.zeros(self.obs_dim, dtype=np.float32)
        if ws.number_of_observations_since_last_state > 0:
            try:
                msg = ws.observations[-1].text
                data = json.loads(msg)
                if "surrounding_blocks" in data:
                    grid = data["surrounding_blocks"]
                    grid_vec = np.array([self.block_map.get(b, 0) for b in grid], dtype=np.float32)
            except: pass
        return grid_vec

    def reset(self, seed=None, options=None):
        xml = self._get_mission_xml()
        try:
            my_mission = MalmoPython.MissionSpec(xml, True)
            my_mission.forceWorldReset() 
            self.agent_host.startMission(my_mission, self.client_pool, MalmoPython.MissionRecordSpec(), 0, "procedural_exp")
        except RuntimeError: time.sleep(2)

        print("Generating new maze...", end="")
        while not self.agent_host.getWorldState().has_mission_begun: time.sleep(0.1)
        print(" Go!")
        time.sleep(1.0)
        
        while True:
            ws = self.agent_host.getWorldState()
            if ws.number_of_observations_since_last_state > 0: break
            time.sleep(0.1)
            
        self.agent_host.sendCommand(f"tp {self.spawn_x}.5 2 {self.spawn_z}.5")
        self.visited_cells.clear()
        return self._get_observation(ws), {}
    
    def close(self): pass

# ==============================================================================
# 1. [핵심] Momentum Spatial BEAGT (관성 기반 부스팅)
# ==============================================================================
class MomentumSpatialCallback(BaseCallback):
    def __init__(self, check_every=500, min_eps=0.10, bump_to=0.90, 
                 start_after=10000,       # 초반 탐색 후 가동
                 displacement_threshold=5.0,
                 base_boost_steps=3000,   # 갇혔을 때 최초 부스팅 시간
                 momentum_bonus=2000,     # 길 찾으면 추가 연장 시간 (관성)
                 max_accumulated_boost=10000, # 너무 오래 켜지는 것 방지
                 verbose=0):
        super().__init__(verbose)
        self.check_every = check_every
        self.min_eps = min_eps
        self.bump_to = bump_to
        self.start_after = start_after 
        self.displacement_threshold = displacement_threshold
        
        self.base_boost_steps = base_boost_steps
        self.momentum_bonus = momentum_bonus
        self.max_accumulated_boost = max_accumulated_boost
        
        self.last_check = 0
        self.last_visited_count = 0
        self.last_pos = None 
        
        # 상태 변수
        self.is_boosting = False
        self.boost_end_step = 0
        self.original_schedule = None

    def _on_training_start(self) -> None:
        self.original_schedule = self.model.exploration_schedule
        def dynamic_schedule(current_progress):
            # 부스팅 시간이 아직 안 끝났으면 강제 0.9 (Epsilon)
            if self.num_timesteps < self.boost_end_step:
                self.is_boosting = True
                return self.bump_to
            
            self.is_boosting = False
            return self.original_schedule(current_progress)
        self.model.exploration_schedule = dynamic_schedule

    def _on_step(self) -> bool:
        infos = self.locals['infos'][0]
        current_visited_count = infos.get('visited_count', 0)
        x, z = infos.get('XPos', 0), infos.get('ZPos', 0)
        
        # Validator 로깅용 플래그 (1=Boost, 0=Normal)
        self.model.beagt_is_boosting = 1 if self.is_boosting else 0
        base_eps = self.original_schedule(self.model._current_progress_remaining)

        # [0] 리셋 감지 (에피소드 변경)
        if current_visited_count < self.last_visited_count:
            if self.is_boosting and self.verbose > 0:
                 print(f"🔄 [Reset] New Episode. Stopping Boost.")
            self.is_boosting = False
            self.boost_end_step = 0 # 부스팅 즉시 종료
            self.last_visited_count = current_visited_count
            self.last_check = self.num_timesteps
            self.last_pos = np.array([x, z])
            return True

        # [1] 핵심: 관성(Momentum) 로직
        # 부스팅 중에 '새로운 곳'을 발견했는가?
        if self.is_boosting and (current_visited_count > self.last_visited_count):
            # 현재 남은 시간 계산
            remaining = self.boost_end_step - self.num_timesteps
            
            # 너무 오래 켜져있지 않다면 시간 연장!
            if remaining < self.max_accumulated_boost:
                self.boost_end_step += self.momentum_bonus
                if self.verbose > 0:
                    print(f"🔥 [Momentum] New cell found! Boost EXTENDED by {self.momentum_bonus} steps! (Ends at {self.boost_end_step})")

        # [2] 정체 감지 (부스팅이 꺼져있을 때만 체크)
        if not self.is_boosting and self.num_timesteps > self.start_after:
            if self.num_timesteps > self.last_check + self.check_every:
                delta = current_visited_count - self.last_visited_count
                current_pos = np.array([x, z])
                dist = 0.0
                if self.last_pos is not None:
                    dist = np.linalg.norm(current_pos - self.last_pos)
                
                # 업데이트
                self.last_visited_count = current_visited_count
                self.last_pos = current_pos
                self.last_check = self.num_timesteps
                
                # 정체 판단: 발견 0개 + 움직임 적음 + 입실론 낮음
                if delta <= 0 and dist < self.displacement_threshold and base_eps < self.min_eps:
                    self.is_boosting = True
                    self.boost_end_step = self.num_timesteps + self.base_boost_steps
                    
                    if self.verbose > 0:
                        print(f"🚀 [Stagnation] Stuck detected. Boosting START for {self.base_boost_steps} steps.")

        # 평소 상태 업데이트 (부스팅 중이 아닐 때 놓친 방문 카운트 갱신용)
        if not self.is_boosting:
             self.last_visited_count = current_visited_count
             self.last_pos = np.array([x, z])

        return True

# ==============================================================================
# 2. [로거] Validator (데이터 저장용)
# ==============================================================================
class HypothesesValidator(BaseCallback):
    def __init__(self, run_name):
        super().__init__()
        self.data = []
        self.run_name = run_name
        self.log_path = f"{run_name}_log.csv"
        
    def _on_step(self):
        current_eps = self.model.exploration_rate
        # 1=Boost, 0=Normal
        status_flag = getattr(self.model, 'beagt_is_boosting', 0)
        
        infos = self.locals.get('infos', [{}])[0]
        visited_count = infos.get('visited_count', 0)

        self.data.append({
            "step": self.num_timesteps,
            "visited_count": visited_count,
            "epsilon": current_eps,
            "is_boosting": status_flag
        })

        if self.num_timesteps % 1000 == 0:
            self.save_log()
            status = "Normal"
            if status_flag == 1: status = "🔥BOOST"
            
            print(f"[Ours] Step: {self.num_timesteps:5d} | Eps: {current_eps:.2f} | Visited: {visited_count:3d} | {status}")
        return True
        
    def save_log(self):
        pd.DataFrame(self.data).to_csv(self.log_path, index=False)

# ==============================================================================
# 3. 메인 실행
# ==============================================================================
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"Maze_CAE_{timestamp}"
    
    print(f"🚀 [실험 시작] {run_name} (Momentum Mode Applied)")
    
    # 포트는 본인 환경에 맞게 수정하세요
    env = SimpleVoxelEnv(port=10006) 

    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=1e-4, 
        buffer_size=10000, 
        exploration_fraction=0.1, 
        exploration_final_eps=0.05
    )

    # [수정됨] MomentumSpatialCallback 적용
    beagt_callback = MomentumSpatialCallback(
        check_every=500,        # 반응 속도 빠르게 (500 step)
        start_after=10000,      # 초반 1만 스텝 후 바로 가동
        displacement_threshold=5.0,
        base_boost_steps=3000,  # 갇혔을 때 3000스텝 부스팅
        momentum_bonus=2000,    # 🔥 관성 보너스: 길 찾으면 2000스텝 추가 연장
        max_accumulated_boost=15000, # 최대 연장 한도
        verbose=1
    )
    
    validator = HypothesesValidator(run_name=run_name)

    try:
        model.learn(total_timesteps=100000, callback=[validator, beagt_callback])
    except KeyboardInterrupt:
        print("중지됨.")
    finally:
        validator.save_log()
        env.close()
