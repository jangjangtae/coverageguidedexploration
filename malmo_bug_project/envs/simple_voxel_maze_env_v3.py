"""
simple_voxel_maze_env_v3.py

21x21 DFS-based Malmo maze environment for auxiliary pure-exploration experiments.

Key design for paper experiments:
- Maze size: 21x21, because DFS carving uses odd-sized wall-cell-wall grids.
- Episode horizon: max_episode_steps=4500 by default.
- Safety mission timeout: mission_time_limit_ms=120000 by default.
- Episode termination: diamond goal reached OR fixed step horizon reached.
- When the fixed horizon is reached, the mission is explicitly quit and the code waits briefly for Malmo to release the client.
- Coverage metric: visited path cells / reachable path cells.

Place at:
    envs/simple_voxel_maze_env_v3.py
"""

import json
import random
import time
from typing import Dict, Optional

import gymnasium as gym
import MalmoPython
import numpy as np
from gymnasium import spaces


class SimpleVoxelMazeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        port: int = 10006,
        map_seed: Optional[int] = None,
        maze_size: int = 21,
        max_episode_steps: int = 4500,
        mission_time_limit_ms: int = 120000,
        step_sleep: float = 0.02,
        reset_sleep: float = 1.0,
        step_penalty: float = 0.0,
        goal_reward: float = 100.0,
        mission_end_wait: float = 2.0,
    ):
        super().__init__()

        if maze_size % 2 == 0:
            raise ValueError("maze_size must be odd for DFS-based maze carving, e.g., 21.")

        self.grid_shape = (5, 3, 5)
        self.obs_dim = 5 * 3 * 5
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.obs_dim,), dtype=np.float32)

        # Walking-only action set, consistent with the original maze experiment.
        self.action_list = ["move 1", "turn 1", "turn -1", "move 0"]
        self.action_space = spaces.Discrete(len(self.action_list))

        self.port = int(port)
        self.agent_host = MalmoPython.AgentHost()
        self.client_pool = MalmoPython.ClientPool()
        self.client_pool.add(MalmoPython.ClientInfo("127.0.0.1", self.port))

        self.block_map: Dict[str, int] = {
            "air": 0,
            "stone": 1,
            "bedrock": 2,
            "gold_block": 3,
            "diamond_block": 4,
            "glass": 5,
            "glowstone": 6,
        }
        self.inv_block_map = {v: k for k, v in self.block_map.items()}

        self.maze_size = int(maze_size)
        self.offset = self.maze_size // 2
        self.max_episode_steps = int(max_episode_steps)
        self.mission_time_limit_ms = int(mission_time_limit_ms)
        self.step_sleep = float(step_sleep)
        self.reset_sleep = float(reset_sleep)
        self.step_penalty = float(step_penalty)
        self.goal_reward = float(goal_reward)
        self.mission_end_wait = float(mission_end_wait)

        self.base_seed = map_seed
        self.episode_index = 0
        self.current_seed = None

        self.visited_cells = set()
        self.reachable_cells = set()
        self.last_maze = None

        self.spawn_x = 0
        self.spawn_z = 0
        self.goal_x = 0
        self.goal_z = 0
        self.episode_step = 0

    # ------------------------------------------------------------------
    # Maze generation
    # ------------------------------------------------------------------
    def _episode_seed(self, seed=None) -> int:
        if seed is not None:
            return int(seed)
        if self.base_seed is not None:
            return int(self.base_seed) + int(self.episode_index)
        return int(time.time() * 1000) % 1_000_000

    def _generate_maze(self, seed: int):
        rng = random.Random(int(seed))
        maze = np.ones((self.maze_size, self.maze_size), dtype=int)  # 1=wall, 0=path

        start_x, start_z = 1, 1
        maze[start_x, start_z] = 0

        stack = [(start_x, start_z)]
        while stack:
            x, z = stack[-1]
            neighbors = []
            for dx, dz in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                nx, nz = x + dx, z + dz
                if 0 < nx < self.maze_size and 0 < nz < self.maze_size and maze[nx, nz] == 1:
                    neighbors.append((nx, nz))

            if neighbors:
                nx, nz = rng.choice(neighbors)
                maze[(x + nx) // 2, (z + nz) // 2] = 0
                maze[nx, nz] = 0
                stack.append((nx, nz))
            else:
                stack.pop()

        goal_x, goal_z = self.maze_size - 2, self.maze_size - 2
        maze[goal_x, goal_z] = 0
        return maze, (start_x, start_z), (goal_x, goal_z)

    def _get_mission_xml(self, seed: int) -> str:
        maze, start, goal = self._generate_maze(seed)
        self.last_maze = maze

        self.reachable_cells = set()
        for r in range(self.maze_size):
            for c in range(self.maze_size):
                if maze[r, c] == 0:
                    self.reachable_cells.add((r - self.offset, c - self.offset))

        draw_cmds = ""
        draw_cmds += (
            f'<DrawCuboid x1="-{self.offset+5}" y1="1" z1="-{self.offset+5}" '
            f'x2="{self.offset+5}" y2="10" z2="{self.offset+5}" type="air"/>'
        )
        draw_cmds += (
            f'<DrawCuboid x1="-{self.offset+5}" y1="1" z1="-{self.offset+5}" '
            f'x2="{self.offset+5}" y2="1" z2="{self.offset+5}" type="bedrock"/>'
        )

        rng = random.Random(int(seed) + 9999)
        for r in range(self.maze_size):
            for c in range(self.maze_size):
                x = r - self.offset
                z = c - self.offset
                if maze[r, c] == 1:
                    draw_cmds += (
                        f'<DrawCuboid x1="{x}" y1="2" z1="{z}" '
                        f'x2="{x}" y2="4" z2="{z}" type="stone"/>'
                    )
                else:
                    draw_cmds += f'<DrawBlock x="{x}" y="1" z="{z}" type="gold_block"/>'
                    if rng.random() < 0.2:
                        draw_cmds += f'<DrawBlock x="{x}" y="4" z="{z}" type="glowstone"/>'

        sx, sz = start[0] - self.offset, start[1] - self.offset
        gx, gz = goal[0] - self.offset, goal[1] - self.offset
        self.spawn_x, self.spawn_z = sx, sz
        self.goal_x, self.goal_z = gx, gz

        # Spawn safety area.
        draw_cmds += (
            f'<DrawCuboid x1="{sx-1}" y1="2" z1="{sz-1}" '
            f'x2="{sx+1}" y2="4" z2="{sz+1}" type="air"/>'
        )
        draw_cmds += f'<DrawBlock x="{sx}" y="1" z="{sz}" type="stone"/>'

        # Goal marker.
        draw_cmds += f'<DrawBlock x="{gx}" y="1" z="{gz}" type="diamond_block"/>'
        draw_cmds += f'<DrawBlock x="{gx}" y="2" z="{gz}" type="glowstone"/>'

        # Outer glass boundary.
        draw_cmds += (
            f'<DrawCuboid x1="-{self.offset+1}" y1="2" z1="-{self.offset+1}" '
            f'x2="{self.offset+1}" y2="4" z2="-{self.offset+1}" type="glass"/>'
        )
        draw_cmds += (
            f'<DrawCuboid x1="-{self.offset+1}" y1="2" z1="{self.offset+1}" '
            f'x2="{self.offset+1}" y2="4" z2="{self.offset+1}" type="glass"/>'
        )
        draw_cmds += (
            f'<DrawCuboid x1="-{self.offset+1}" y1="2" z1="-{self.offset+1}" '
            f'x2="-{self.offset+1}" y2="4" z2="{self.offset+1}" type="glass"/>'
        )
        draw_cmds += (
            f'<DrawCuboid x1="{self.offset+1}" y1="2" z1="-{self.offset+1}" '
            f'x2="{self.offset+1}" y2="4" z2="{self.offset+1}" type="glass"/>'
        )

        return f'''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <About><Summary>Safe Procedural Maze {seed}</Summary></About>
  <ServerSection>
    <ServerInitialConditions>
      <Time><StartTime>18000</StartTime><AllowPassageOfTime>false</AllowPassageOfTime></Time>
      <Weather>clear</Weather>
    </ServerInitialConditions>
    <ServerHandlers>
      <FlatWorldGenerator generatorString="3;7,2;1;"/>
      <DrawingDecorator>{draw_cmds}</DrawingDecorator>
      <ServerQuitFromTimeUp timeLimitMs="{self.mission_time_limit_ms}"/>
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
      <MissionQuitCommands/>
    </AgentHandlers>
  </AgentSection>
</Mission>'''


    def _request_mission_quit(self, reason="requested"):
        """Ask Malmo to end the current mission and wait briefly until it stops.

        This is important when the Gym episode is truncated by max_episode_steps
        before Malmo's ServerQuitFromTimeUp safety timeout fires. Without this,
        reset() can try to start a new mission while the previous one is still
        bound to the same port.
        """
        try:
            self.agent_host.sendCommand("move 0")
            self.agent_host.sendCommand("turn 0")
        except Exception:
            pass

        try:
            self.agent_host.sendCommand("quit")
        except Exception:
            pass

        t0 = time.time()
        while time.time() - t0 < self.mission_end_wait:
            try:
                if not self.agent_host.getWorldState().is_mission_running:
                    break
            except Exception:
                break
            time.sleep(0.1)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_index += 1
        self.episode_step = 0
        self.visited_cells.clear()

        # Make sure a previous truncated mission is no longer occupying this port.
        self._request_mission_quit(reason="reset_precheck")

        self.current_seed = self._episode_seed(seed)
        xml = self._get_mission_xml(self.current_seed)
        mission = MalmoPython.MissionSpec(xml, True)
        mission.forceWorldReset()

        last_error = None
        for _ in range(5):
            try:
                self.agent_host.startMission(
                    mission,
                    self.client_pool,
                    MalmoPython.MissionRecordSpec(),
                    0,
                    "procedural_maze",
                )
                last_error = None
                break
            except RuntimeError as e:
                last_error = e
                time.sleep(2.0)
        if last_error is not None:
            raise RuntimeError(f"Could not start mission after retries: {last_error}")

        print(f"Generating new maze seed={self.current_seed} ...", end="")
        t0 = time.time()
        while not self.agent_host.getWorldState().has_mission_begun:
            if time.time() - t0 > 30:
                raise RuntimeError("Mission start timed out")
            time.sleep(0.1)
        print(" Go!")

        time.sleep(self.reset_sleep)

        obs = np.zeros(self.obs_dim, dtype=np.float32)
        t0 = time.time()
        while True:
            ws = self.agent_host.getWorldState()
            if ws.number_of_observations_since_last_state > 0:
                obs = self._get_observation(ws)
                break
            if time.time() - t0 > 10:
                break
            time.sleep(0.1)

        try:
            self.agent_host.sendCommand(f"tp {self.spawn_x}.5 2 {self.spawn_z}.5")
        except RuntimeError:
            pass

        return obs, {
            "maze_seed": self.current_seed,
            "visited_count": 0,
            "reachable_count": len(self.reachable_cells),
            "coverage_ratio": 0.0,
        }

    def step(self, action_idx):
        self.episode_step += 1
        action = self.action_list[int(action_idx)]

        try:
            self.agent_host.sendCommand(action)
        except RuntimeError:
            pass

        time.sleep(self.step_sleep)
        ws = self.agent_host.getWorldState()
        obs = self._get_observation(ws)

        reward = self.step_penalty
        terminated = False
        truncated = False
        info = {
            "action_name": action,
            "maze_seed": self.current_seed,
            "episode_step": self.episode_step,
            "goal_x": self.goal_x,
            "goal_z": self.goal_z,
        }

        x, y, z = 0.0, 2.0, 0.0
        has_valid_data = False
        if ws.number_of_observations_since_last_state > 0:
            try:
                data = json.loads(ws.observations[-1].text)
                if "XPos" in data and "YPos" in data:
                    x = float(data.get("XPos"))
                    y = float(data.get("YPos"))
                    z = float(data.get("ZPos"))
                    has_valid_data = True
                    self.visited_cells.add((int(np.floor(x)), int(np.floor(z))))
                info["XPos"], info["YPos"], info["ZPos"] = x, y, z
                info["Yaw"] = float(data.get("Yaw", 0.0))
                info["Pitch"] = float(data.get("Pitch", 0.0))
            except Exception:
                pass

        if has_valid_data:
            dist_to_goal = abs(x - self.goal_x) + abs(z - self.goal_z)
            info["distance_to_goal"] = float(dist_to_goal)
            if dist_to_goal < 1.0:
                reward += self.goal_reward
                terminated = True
                info["success"] = True
                print(f"🎉 [Success] Maze solved! seed={self.current_seed}")
            else:
                info["success"] = False
        else:
            info["distance_to_goal"] = None
            info["success"] = False

        reachable_count = max(len(self.reachable_cells), 1)
        info["visited_count"] = int(len(self.visited_cells))
        info["reachable_count"] = int(reachable_count)
        info["coverage_ratio"] = float(len(self.visited_cells) / reachable_count)

        if self.episode_step >= self.max_episode_steps:
            truncated = True
            info["truncation_reason"] = "max_episode_steps"
            self._request_mission_quit(reason="max_episode_steps")

        if not ws.is_mission_running:
            truncated = True
            info.setdefault("truncation_reason", "mission_stopped")

        return obs, float(reward), terminated, truncated, info

    def _get_observation(self, ws):
        grid_vec = np.zeros(self.obs_dim, dtype=np.float32)
        if ws.number_of_observations_since_last_state > 0:
            try:
                data = json.loads(ws.observations[-1].text)
                if "surrounding_blocks" in data:
                    grid = data["surrounding_blocks"]
                    grid_vec = np.array([self.block_map.get(str(b), 0) for b in grid], dtype=np.float32)
            except Exception:
                pass
        return grid_vec

    def decode_blocks(self, obs):
        arr = np.asarray(obs, dtype=np.float32).astype(int).flatten()
        return [self.inv_block_map.get(int(v), "unknown") for v in arr]

    def close(self):
        self._request_mission_quit(reason="close")
