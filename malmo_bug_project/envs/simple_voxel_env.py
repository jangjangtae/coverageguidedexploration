import numpy as np
import gymnasium as gym
from gymnasium import spaces
import MalmoPython
import time
import json
import random

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
        if self.map_seed is None:
            seed = int(time.time() * 1000) % 10000
        else:
            seed = self.map_seed
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
        
        self.spawn_x, self.spawn_z = sx, sz
        self.goal_x, self.goal_z = gx, gz
        
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
        except RuntimeError as e:
            time.sleep(2)

        print("Generating new maze...", end="")
        while not self.agent_host.getWorldState().has_mission_begun:
            time.sleep(0.1)
        print(" Go!")
        
        time.sleep(1.0)

        while True:
            ws = self.agent_host.getWorldState()
            if ws.number_of_observations_since_last_state > 0:
                break
            time.sleep(0.1)
            
        self.agent_host.sendCommand(f"tp {self.spawn_x}.5 2 {self.spawn_z}.5")

        self.visited_cells.clear()
        return self._get_observation(ws), {}
    
    def close(self): pass
