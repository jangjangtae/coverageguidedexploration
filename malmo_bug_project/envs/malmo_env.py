import os
import time
import json
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import MalmoPython

from envs.bug_detector import BugDetector

MAX_XZ, MIN_XZ = 19, 0

class InputFaultInjector:
    def __init__(self):
        self.last_action_type = None

    def reset(self):
        self.last_action_type = None

    def transform(self, action: str, obs_data: dict):
        flags = {}
        cheat_cmd = None
        effective_action = action
        
        if not action: return None, {}, None
        cmd_parts = action.split(" ")
        cmd_type = cmd_parts[0]
        is_wait_action = (cmd_type == "move" and len(cmd_parts) > 1 and cmd_parts[1] == "0")

        # 데이터 파싱
        x = obs_data.get("XPos", 0)
        z = obs_data.get("ZPos", 0)
        yaw = obs_data.get("Yaw", 0)
        pitch = obs_data.get("Pitch", 0)
        
        # 발밑 블록 (Grid Index 37)
        grid = obs_data.get("surrounding_blocks", [])
        block_under_feet = ""
        if len(grid) > 37:
            block_under_feet = grid[37]
        
        # 1x1 블록 판정
        is_on_sand = (block_under_feet == "sandstone")
        is_on_gold = (block_under_feet == "gold_block")
        is_on_redstone = (block_under_feet == "redstone_block")
        is_on_obsidian = (block_under_feet == "obsidian")
        is_on_clay = (block_under_feet == "clay")
        is_on_lapis = (block_under_feet == "lapis_block")
        is_on_quartz = (block_under_feet == "quartz_block")
        is_on_emerald = (block_under_feet == "emerald_block")

        los_raw = obs_data.get("LineOfSight", {})
        target_block = ""
        if isinstance(los_raw, dict):
            target_block = los_raw.get("type", "")
        
        # ==================== Extreme 10대 버그 로직 ====================
        # 1. REVERSE TURN
        if is_on_obsidian and cmd_type == "turn" and len(cmd_parts) > 1 and cmd_parts[1] == "1" and abs(yaw) > 135:
            flags["bug_reverse_turn"] = True
            effective_action = "turn -1" 
        # 2. TELEPORT TRAP
        elif is_on_redstone:
            flags["bug_teleport_trap"] = True
            cheat_cmd = "chat /tp @p 2.5 5 2.5" 
            effective_action = None
        # 3. DEAD ZONE
        elif is_on_gold and cmd_type in ["move", "turn"]:
            flags["bug_deadzone"] = True
            effective_action = None
        # 4. SUPER JUMP
        elif cmd_type == "jump" and is_on_sand:
            flags["bug_superjump"] = True
            cheat_cmd = "chat /effect @p levitation 1 10"
        # 5. SEQUENCE ERROR
        elif is_on_clay and cmd_type == "jump" and self.last_action_type and self.last_action_type.startswith("turn"):
            flags["bug_sequence"] = True
            effective_action = None
        # 6. INTERACT FAIL
        elif is_on_lapis and cmd_type == "use":
            flags["bug_interact_fail"] = True
            effective_action = None
        # 7. TRANSMUTATION (Web)
        elif cmd_type == "use":
            dist_to_web = ((x - 5.5)**2 + (z - 8.5)**2)**0.5
            if target_block == "web" or dist_to_web < 1.5:
                flags["bug_transmutation"] = True
                cheat_cmd = "chat /setblock 5 5 8 air"
        # 8. SKY FREEZE
        elif is_on_quartz and cmd_type == "move" and pitch < -85:
            flags["bug_sky_freeze"] = True
            effective_action = None
        # 9. HOTBAR ERROR
        elif is_on_emerald and cmd_type.startswith("hotbar"):
            flags["bug_hotbar_error"] = True
            effective_action = "hotbar.1 1"
        # 10. BREAK CRASH
        elif cmd_type == "attack":
            dist_to_glass = ((x - 19.5)**2 + (z - 9.5)**2)**0.5
            if target_block == "glass" or dist_to_glass < 2.0:
                flags["bug_break_crash"] = True
                cheat_cmd = "chat /setblock 19 6 9 air" 
                
        if effective_action and not is_wait_action:
            self.last_action_type = cmd_type
        
        return effective_action, flags, cheat_cmd


class MalmoEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, port=10000, log_root="runs", run_tag=None):
        super().__init__()
        self.run_id = run_tag or (time.strftime("%Y%m%d-%H%M%S") + f"-pid{os.getpid()}")
        self.log_dir = os.path.join(log_root, self.run_id)
        os.makedirs(self.log_dir, exist_ok=True)

        self.grid_key = "surrounding_blocks"
        self.grid_len = 125
        self.obs_dim = 3 + self.grid_len + 10
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        self.action_list = [
            "move 1", "turn 1", "turn -1", "move 0",
            "use 1", "attack 1", 
            "pitch 1", "pitch -1", "jump 1", "jump 0",
            "hotbar.1 1", "hotbar.2 1", "drop 1"
        ]
        self.action_space = spaces.Discrete(len(self.action_list))

        self.port = int(port)
        self.agent_host = MalmoPython.AgentHost()
        self.client_pool = MalmoPython.ClientPool()
        self.client_pool.add(MalmoPython.ClientInfo("127.0.0.1", self.port))
        self.exp_id = f"exp_{self.port}"
        self.role = 0
        self.mission_xml = self._get_mission_xml()

        self.bug_detector = BugDetector(log_dir=self.log_dir)
        self.fault = InputFaultInjector()
        self.visited_cells = set()
        self._last_obs_raw = {}
        
        self.obs_data = {} 
        self._last_msg_timestamp = 0

    def _get_mission_xml(self):
        try:
            with open("missions/bug_mission.xml", "r") as f:
                return f.read()
        except FileNotFoundError:
            return ""

    def _fix_inventory_obs(self, obs):
        if "inventory" in obs and isinstance(obs["inventory"], list) and len(obs["inventory"]) > 0:
            return obs
        reconstructed_inv = []
        for i in range(40):
            key_item = f"InventorySlot_{i}_item"
            key_size = f"InventorySlot_{i}_size"
            if key_item in obs:
                reconstructed_inv.append({
                    "slot": i,
                    "type": obs[key_item],
                    "quantity": obs.get(key_size, 1)
                })
        obs["inventory"] = reconstructed_inv
        return obs

    def update_obs(self):
        ws = self.agent_host.getWorldState()
        if ws.number_of_observations_since_last_state > 0:
            try:
                self._last_msg_timestamp = ws.observations[-1].timestamp
                raw = json.loads(ws.observations[-1].text)
                self._last_obs_raw = self._fix_inventory_obs(raw)
                self.obs_data = self._last_obs_raw
            except: pass
        return self._last_obs_raw

    def _safe_wait_for_mission_end(self):
        world_state = self.agent_host.getWorldState()
        if world_state.is_mission_running:
            self.agent_host.sendCommand("quit")
            time.sleep(0.5)
        
        for _ in range(30):
            world_state = self.agent_host.getWorldState()
            if not world_state.is_mission_running:
                return
            time.sleep(0.1)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed); random.seed(seed)
        
        self._safe_wait_for_mission_end()
        time.sleep(0.5)

        mission_spec = MalmoPython.MissionSpec(self.mission_xml, True)
        mission_spec.forceWorldReset()
        mission_record = MalmoPython.MissionRecordSpec()
        
        for attempt in range(5):
            try:
                self.agent_host.startMission(mission_spec, self.client_pool, mission_record, self.role, self.exp_id)
                break
            except RuntimeError:
                time.sleep(2)

        print("Waiting for mission start...", end=' ')
        max_wait = 300  # 30초 대기
        mission_started = False
        
        for _ in range(max_wait):
            ws = self.agent_host.getWorldState()
            if ws.has_mission_begun: 
                mission_started = True
                break
            print(".", end="")
            time.sleep(0.1)
            
        if not mission_started:
            print("\n❌ Error: Mission start timed out! (XML Error or Client Freeze)")
            raise RuntimeError("Mission Start Timeout")

        print(" Start!")

        self.update_obs()
        print("Waiting for inventory sync...", end=' ')
        for _ in range(50):
            obs = self.update_obs()
            if obs.get("inventory"):
                print(f" Synced! (Items: {len(obs['inventory'])})")
                break
            time.sleep(0.1)
        else:
            print(" Warning: Inventory empty after wait.")

        self.bug_detector.reset()
        self.fault.reset()
        self.visited_cells.clear()
        
        x = self._last_obs_raw.get("XPos", 0)
        z = self._last_obs_raw.get("ZPos", 0)
        init_inv = sum(i.get("quantity", 0) for i in self._last_obs_raw.get("inventory", []))
        
        self.bug_detector.prev_pos = (x, z)
        self.bug_detector.prev_inv_total = init_inv
        self._last_msg_timestamp = 0
        
        return self._get_observation(), {}

    def _get_observation(self):
        data = self.update_obs()
        x = float(data.get("XPos", 0.0))
        z = float(data.get("ZPos", 0.0))
        yaw = float(data.get("Yaw", 0.0))
        
        grid = data.get(self.grid_key, [])
        if len(grid) < self.grid_len: grid += [0]*(self.grid_len-len(grid))
        grid = grid[:self.grid_len]
        
        block_map = {
            "stone": 1, "planks": 2, "bedrock": 3, "apple": 4, "air": 0, 
            "sandstone": 5, "gold_block": 6, "web": 7, "iron_door": 8,
            "obsidian": 9, "clay": 10, "lapis_block": 11, "redstone_block": 12,
            "quartz_block": 13, "emerald_block": 14, "glass": 15
        }
        grid_vec = [block_map.get(b, 0) for b in grid]

        inv = data.get("inventory", [])
        inv_vec = [0]*10
        for it in inv:
            t = it.get("type")
            q = it.get("quantity", 0)
            if t=="apple": inv_vec[0] = q
            elif t=="stone": inv_vec[1] = q
        
        return np.array([x, z, yaw] + grid_vec + inv_vec, dtype=np.float32)

    def step(self, action_idx):
        self.update_obs()
        prev_timestamp = self._last_msg_timestamp
        
        action = self.action_list[action_idx]
        effective_action, flags, cheat_cmd = self.fault.transform(action, self._last_obs_raw)
        
        if flags:
            d = getattr(self.bug_detector, "env_injected", {})
            d.update(flags)
            self.bug_detector.env_injected = d
        
        # Action 실행
        if effective_action:
            self.agent_host.sendCommand(effective_action)
            time.sleep(0.1)

        # Cheat 실행
        if cheat_cmd:
            self.agent_host.sendCommand(cheat_cmd)
            time.sleep(1.0)
            if "bug_transmutation" in flags:
                self.agent_host.sendCommand("chat /give @p diamond 1")

        # Wait for obs
        wait_time = 1.0 if cheat_cmd else 0.5
        start_wait = time.time()
        while time.time() - start_wait < wait_time:
            self.update_obs()
            if not cheat_cmd and self._last_msg_timestamp != prev_timestamp and (time.time() - start_wait > 0.2):
                break
            time.sleep(0.05)

        # Auto-Stop
        if effective_action and ("move" in effective_action or "turn" in effective_action):
            cmd_type = effective_action.split()[0]
            val = effective_action.split()[1]
            if val != "0":
                self.agent_host.sendCommand(f"{cmd_type} 0")

        obs = self._get_observation()
        ws = self.agent_host.getWorldState()
        
        curr_x, curr_z = float(obs[0]), float(obs[1])
        cell_key = (int(curr_x), int(curr_z))
        
        # 탐험 보상
        exploration_reward = 0.0
        if MIN_XZ <= curr_x <= MAX_XZ and MIN_XZ <= curr_z <= MAX_XZ:
            if cell_key not in self.visited_cells:
                self.visited_cells.add(cell_key)
                exploration_reward = 1.0 
        
        # 버그 체크
        bug_reward, evidences = self.bug_detector.check_bugs(action, (curr_x, curr_z), ws)
        reward = exploration_reward + bug_reward
        
        if not (MIN_XZ <= curr_x <= MAX_XZ and MIN_XZ <= curr_z <= MAX_XZ):
            reward += -0.5

        done = not ws.is_mission_running
        
        safe_bug_ids = []
        if isinstance(evidences, dict) and "id" in evidences:
            safe_bug_ids.append(evidences["id"])
        elif isinstance(evidences, dict):
            safe_bug_ids = list(evidences.keys())
        elif isinstance(evidences, list):
            for item in evidences:
                if isinstance(item, str):
                    safe_bug_ids.append(item)
                elif isinstance(item, dict):
                    if "id" in item: safe_bug_ids.append(item["id"])
                    else: safe_bug_ids.extend(list(item.keys()))
                else:
                    safe_bug_ids.append(str(item))
        elif isinstance(evidences, str):
             safe_bug_ids = [evidences]

        new_bugs_found = (len(safe_bug_ids) > 0)

        info = {
            "bug_detected": new_bugs_found,
            "bug_ids": safe_bug_ids,
            "detected_bugs": list(self.bug_detector.detected_bugs),
            "XPos": self.obs_data.get("XPos", 0.0),
            "ZPos": self.obs_data.get("ZPos", 0.0),
            "Yaw": self.obs_data.get("Yaw", 0.0),
            "Pitch": self.obs_data.get("Pitch", 0.0),
            "action": self.action_list[action_idx] if action_idx < len(self.action_list) else "none"
        }
        
        return obs, float(reward), done, False, info

    def close(self):
        self._safe_wait_for_mission_end()

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'agent_host' in state:
            del state['agent_host']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        try:
            import MalmoPython
            self.agent_host = MalmoPython.AgentHost()
        except ImportError:
            print("Warning: MalmoPython not found during unpickling.")
            self.agent_host = None
