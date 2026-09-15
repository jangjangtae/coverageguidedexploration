import os
import time
import json
import random
import math
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import MalmoPython

from envs.bug_detector_issue10_consistent import BugDetector

MAX_XZ, MIN_XZ = 19, 0


ISSUE10_BUGS = [
    "BUG_HEADING_DYNAMICS_ANOMALY",
    "BUG_TRANSITION_TELEPORT",
    "BUG_MOVEMENT_LOCK_ZONE",
    "BUG_COLLISION_IMPULSE_GLITCH",
    "BUG_CONTEXTUAL_SEQUENCE_FAILURE",
    "BUG_CONTEXTUAL_INTERACTION_OMISSION",
    "BUG_WORLD_STATE_MUTATION",
    "BUG_VIEW_DEPENDENT_MOVEMENT_LOCK",
    "BUG_SLOT_SELECTION_DESYNC",
    "BUG_BREAK_EVENT_CORRUPTION",
]

SPECIAL_BLOCKS = {
    "obsidian", "redstone_block", "gold_block", "sandstone", "clay",
    "lapis_block", "web", "quartz_block", "emerald_block", "glass",
}


def _action_type(action: str) -> str:
    if not action:
        return "none"
    head = action.split(" ", 1)[0]
    if head.startswith("hotbar"):
        return "hotbar"
    return head


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


class InputFaultInjector:
    """
    Deterministic issue10 fault injector.

    Modified for distributed EP90 benchmark:
    - web target moved from (5, 8) to (5, 15)
    - glass target moved from (18, 9) to (18, 11)
    - contextual sequence uses a pending flag so turn-on-clay -> jump is stable.
    """

    def __init__(self):
        self.last_action_type = None
        self.last_turn_on_clay = False
        self.pending_sequence_bug = False

    def reset(self):
        self.last_action_type = None
        self.last_turn_on_clay = False
        self.pending_sequence_bug = False

    @staticmethod
    def _surface_and_target(obs_data: Dict[str, Any]) -> Tuple[str, str]:
        grid = obs_data.get("surrounding_blocks", []) or []
        block_under_feet = grid[37] if len(grid) > 37 else ""
        los_raw = obs_data.get("LineOfSight", {})
        target_block = los_raw.get("type", "") if isinstance(los_raw, dict) else ""
        return block_under_feet, target_block

    def transform(self, action: str, obs_data: dict):
        flags = {}
        cheat_cmd = None
        effective_action = action

        if not action:
            return None, {}, None

        cmd_parts = action.split(" ")
        cmd_type = _action_type(action)
        is_wait_action = (cmd_type == "move" and len(cmd_parts) > 1 and cmd_parts[1] == "0")

        x = _safe_float(obs_data.get("XPos", 0.0))
        z = _safe_float(obs_data.get("ZPos", 0.0))
        yaw = _safe_float(obs_data.get("Yaw", 0.0))
        pitch = _safe_float(obs_data.get("Pitch", 0.0))

        block_under_feet, target_block = self._surface_and_target(obs_data)

        is_on_sandstone = (block_under_feet == "sandstone")
        is_on_gold = (block_under_feet == "gold_block")
        is_on_redstone = (block_under_feet == "redstone_block")
        is_on_obsidian = (block_under_feet == "obsidian")
        is_on_clay = (block_under_feet == "clay")
        is_on_lapis = (block_under_feet == "lapis_block")
        is_on_quartz = (block_under_feet == "quartz_block")
        is_on_emerald = (block_under_feet == "emerald_block")

        if is_on_obsidian and cmd_type == "turn" and len(cmd_parts) > 1 and cmd_parts[1] == "1":
            flags["bug_heading_dynamics"] = {
                "surface": "obsidian",
                "original_action": action,
                "effective_action": "turn -1",
                "yaw_before": yaw,
            }
            effective_action = "turn -1"

        elif is_on_redstone and cmd_type == "move":
            flags["bug_transition_teleport"] = {
                "zone": "redstone_transition_zone",
                "original_action": action,
                "teleport_to": [2.5, 5.0, 2.5],
            }
            cheat_cmd = "chat /tp @p 2.5 5 2.5"
            effective_action = None

        elif is_on_gold and cmd_type in ["move", "turn"]:
            flags["bug_movement_lock_zone"] = {
                "zone": "gold_lock_zone",
                "suppressed_action": action,
            }
            effective_action = None

        elif cmd_type == "jump" and is_on_sandstone:
            flags["bug_collision_impulse"] = {
                "surface": "sandstone",
                "original_action": action,
                "effect": "levitation",
            }
            cheat_cmd = "chat /effect @p levitation 1 10"

        elif cmd_type == "turn" and is_on_clay:
            self.last_turn_on_clay = True
            self.pending_sequence_bug = True

        if cmd_type == "jump" and self.pending_sequence_bug:
            flags["bug_contextual_sequence_failure"] = {
                "surface": "clay",
                "failed_sequence": [self.last_action_type or "turn", action],
            }
            effective_action = None
            self.pending_sequence_bug = False
            self.last_turn_on_clay = False

        elif is_on_lapis and cmd_type == "use":
            flags["bug_contextual_interaction_omission"] = {
                "surface": "lapis_block",
                "omitted_action": action,
            }
            effective_action = None

        elif cmd_type == "use":
            # Distributed benchmark web target: block at (5, 5, 15), center approx (5.5, 15.5)
            dist_to_web = ((x - 5.5) ** 2 + (z - 15.5) ** 2) ** 0.5
            if target_block == "web" or dist_to_web < 1.5:
                flags["bug_world_state_mutation"] = {
                    "target_block": "web",
                    "mutation_result": "air",
                    "distance_to_target": round(dist_to_web, 3),
                }
                cheat_cmd = "chat /setblock 5 5 15 air"

        elif is_on_quartz and cmd_type == "move" and pitch < -85:
            flags["bug_view_dependent_lock"] = {
                "surface": "quartz_block",
                "pitch": pitch,
                "suppressed_action": action,
            }
            effective_action = None

        elif is_on_emerald and cmd_type == "hotbar":
            flags["bug_slot_selection_desync"] = {
                "surface": "emerald_block",
                "requested_action": action,
                "forced_action": "hotbar.1 1",
            }
            effective_action = "hotbar.1 1"

        elif cmd_type == "attack":
            # Distributed benchmark glass target: column at (18, 5/6, 11), center approx (18.5, 11.5)
            dist_to_glass = ((x - 18.5) ** 2 + (z - 11.5) ** 2) ** 0.5
            in_break_zone = (16.8 <= x <= 18.5) and (10.0 <= z <= 11.8)
            if target_block == "glass" or in_break_zone or dist_to_glass < 1.8:
                flags["bug_break_event_corruption"] = {
                    "target_block": "glass",
                    "corrupted_result": "air",
                    "distance_to_target": round(dist_to_glass, 3),
                    "trigger_mode": "zone_or_los",
                }
                cheat_cmd = "chat /setblock 18 6 11 air"

        if effective_action and not is_wait_action:
            self.last_action_type = cmd_type
            if cmd_type not in ["jump", "turn"]:
                self.last_turn_on_clay = False
                self.pending_sequence_bug = False

        return effective_action, flags, cheat_cmd


class MalmoEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        port=10000,
        log_root="runs",
        run_tag=None,
        bug_json_path="envs/bug_definitions_issue10_consistent.json",
        mission_xml_path="missions/bug_mission_issue10_distributed_ep90.xml",
        cae_intrinsic: bool = False,
        cae_config: Optional[Dict[str, float]] = None,
    ):
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
        self.mission_xml_path = mission_xml_path
        self.mission_xml = self._get_mission_xml()

        self.bug_detector = BugDetector(json_path=bug_json_path, log_dir=self.log_dir)
        self.fault = InputFaultInjector()
        self.visited_cells = set()
        self._last_obs_raw = {}
        self.obs_data = {}
        self._last_msg_timestamp = 0
        self.env_step_count = 0

        # CAE-v2 intrinsic interaction coverage. Disabled by default for non-CAE baselines.
        self.cae_intrinsic = bool(cae_intrinsic)
        cfg = {
            "cell": 0.02,
            "state_action": 0.08,
            "block_action": 0.10,
            "sequence": 0.05,
            "cap": 1.0,
        }
        if cae_config:
            cfg.update(cae_config)
        self.cae_config = cfg
        self.cae_counts = defaultdict(int)
        self.prev_cae_action_type = None
        self.last_cae_reward = 0.0
        self.last_cae_components = {}

        # Opportunity logging is independent of CAE and is useful for explaining hard bugs.
        self.opportunity_stats = self._init_opportunity_stats()
        self._sequence_pending_for_opportunity = False

    def _get_mission_xml(self):
        with open(self.mission_xml_path, "r", encoding="utf-8") as f:
            return f.read()

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
                    "quantity": obs.get(key_size, 1),
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
            except Exception:
                pass
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
            np.random.seed(seed)
            random.seed(seed)

        self._safe_wait_for_mission_end()
        time.sleep(0.5)

        mission_spec = MalmoPython.MissionSpec(self.mission_xml, True)
        mission_spec.forceWorldReset()
        mission_record = MalmoPython.MissionRecordSpec()

        for _ in range(5):
            try:
                self.agent_host.startMission(mission_spec, self.client_pool, mission_record, self.role, self.exp_id)
                break
            except RuntimeError:
                time.sleep(2)

        print("Waiting for mission start...", end=" ")
        mission_started = False
        for _ in range(300):
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
        print("Waiting for inventory sync...", end=" ")
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
        self._last_msg_timestamp = 0
        self.prev_cae_action_type = None
        self._sequence_pending_for_opportunity = False
        self.last_cae_reward = 0.0
        self.last_cae_components = {}
        # NOTE: CAE coverage counts and opportunity stats intentionally persist across episodes within one run.

        return self._get_observation(), {}

    def _get_observation(self):
        data = self.update_obs()
        x = float(data.get("XPos", 0.0))
        z = float(data.get("ZPos", 0.0))
        yaw = float(data.get("Yaw", 0.0))

        grid = data.get(self.grid_key, [])
        if len(grid) < self.grid_len:
            grid += [0] * (self.grid_len - len(grid))
        grid = grid[:self.grid_len]

        block_map = {
            "stone": 1, "planks": 2, "bedrock": 3, "apple": 4, "air": 0,
            "sandstone": 5, "gold_block": 6, "web": 7, "iron_door": 8,
            "obsidian": 9, "clay": 10, "lapis_block": 11, "redstone_block": 12,
            "quartz_block": 13, "emerald_block": 14, "glass": 15
        }
        grid_vec = [block_map.get(b, 0) for b in grid]

        inv = data.get("inventory", [])
        inv_vec = [0] * 10
        for it in inv:
            t = it.get("type")
            q = it.get("quantity", 0)
            if t == "apple":
                inv_vec[0] = q
            elif t == "stone":
                inv_vec[1] = q

        return np.array([x, z, yaw] + grid_vec + inv_vec, dtype=np.float32)

    def _surface_target(self, obs_data: Dict[str, Any]) -> Tuple[str, str]:
        grid = obs_data.get(self.grid_key, []) or []
        surface = grid[37] if len(grid) > 37 else ""
        los_raw = obs_data.get("LineOfSight", {})
        target = los_raw.get("type", "") if isinstance(los_raw, dict) else ""
        return surface, target

    def _cell(self, obs_data: Dict[str, Any]) -> Tuple[int, int]:
        return int(math.floor(_safe_float(obs_data.get("XPos", 0.0)))), int(math.floor(_safe_float(obs_data.get("ZPos", 0.0))))

    def _cae_novelty(self, key: Tuple[Any, ...]) -> float:
        n = self.cae_counts[key]
        self.cae_counts[key] += 1
        return 1.0 / math.sqrt(n + 1.0)

    def _compute_cae_intrinsic_reward(self, action: str, obs_data: Dict[str, Any]) -> float:
        if not self.cae_intrinsic:
            self.last_cae_reward = 0.0
            self.last_cae_components = {}
            self.prev_cae_action_type = _action_type(action)
            return 0.0

        action_t = _action_type(action)
        cell_x, cell_z = self._cell(obs_data)
        surface, target = self._surface_target(obs_data)

        cell_key = ("cell", cell_x, cell_z)
        state_action_key = ("state_action", cell_x, cell_z, surface, action_t)
        sequence_key = ("sequence", self.prev_cae_action_type or "none", action_t, surface)

        cell_nov = self._cae_novelty(cell_key)
        state_action_nov = self._cae_novelty(state_action_key)
        sequence_nov = self._cae_novelty(sequence_key)

        # Block-action novelty focuses on semantically meaningful objects, not bug IDs.
        if surface in SPECIAL_BLOCKS or target in SPECIAL_BLOCKS:
            block_action_key = ("block_action", surface, target, action_t)
            block_action_nov = self._cae_novelty(block_action_key)
        else:
            block_action_nov = 0.0

        r = (
            self.cae_config["cell"] * cell_nov
            + self.cae_config["state_action"] * state_action_nov
            + self.cae_config["block_action"] * block_action_nov
            + self.cae_config["sequence"] * sequence_nov
        )
        r = min(float(r), float(self.cae_config.get("cap", 1.0)))

        self.last_cae_reward = r
        self.last_cae_components = {
            "cell_novelty": float(cell_nov),
            "state_action_novelty": float(state_action_nov),
            "block_action_novelty": float(block_action_nov),
            "sequence_novelty": float(sequence_nov),
            "coverage_table_size": int(len(self.cae_counts)),
        }
        self.prev_cae_action_type = action_t
        return r

    def _init_opportunity_stats(self) -> Dict[str, Dict[str, Any]]:
        return {
            bug: {
                "zone_visit_count": 0,
                "first_zone_visit_step": None,
                "action_attempt_count": 0,
                "first_action_attempt_step": None,
                "trigger_opportunity_count": 0,
                "first_trigger_opportunity_step": None,
                "detected_after_opportunity": False,
            }
            for bug in ISSUE10_BUGS
        }

    def _mark_opp(self, bug_id: str, zone=False, action_attempt=False, trigger=False):
        s = self.opportunity_stats[bug_id]
        step = int(self.env_step_count)
        if zone:
            s["zone_visit_count"] += 1
            if s["first_zone_visit_step"] is None:
                s["first_zone_visit_step"] = step
        if action_attempt:
            s["action_attempt_count"] += 1
            if s["first_action_attempt_step"] is None:
                s["first_action_attempt_step"] = step
        if trigger:
            s["trigger_opportunity_count"] += 1
            if s["first_trigger_opportunity_step"] is None:
                s["first_trigger_opportunity_step"] = step

    def _update_opportunity_stats(self, action: str, obs_data: Dict[str, Any]):
        action_t = _action_type(action)
        x = _safe_float(obs_data.get("XPos", 0.0))
        z = _safe_float(obs_data.get("ZPos", 0.0))
        pitch = _safe_float(obs_data.get("Pitch", 0.0))
        surface, target = self._surface_target(obs_data)

        is_on_obsidian = surface == "obsidian"
        is_on_redstone = surface == "redstone_block"
        is_on_gold = surface == "gold_block"
        is_on_sandstone = surface == "sandstone"
        is_on_clay = surface == "clay"
        is_on_lapis = surface == "lapis_block"
        is_on_quartz = surface == "quartz_block"
        is_on_emerald = surface == "emerald_block"
        near_web = target == "web" or (((x - 5.5) ** 2 + (z - 15.5) ** 2) ** 0.5 < 1.5)
        near_glass = target == "glass" or (((x - 18.5) ** 2 + (z - 11.5) ** 2) ** 0.5 < 1.8)

        self._mark_opp("BUG_HEADING_DYNAMICS_ANOMALY", zone=is_on_obsidian, action_attempt=is_on_obsidian and action_t == "turn", trigger=is_on_obsidian and action_t == "turn")
        self._mark_opp("BUG_TRANSITION_TELEPORT", zone=is_on_redstone, action_attempt=is_on_redstone and action_t == "move", trigger=is_on_redstone and action_t == "move")
        self._mark_opp("BUG_MOVEMENT_LOCK_ZONE", zone=is_on_gold, action_attempt=is_on_gold and action_t in {"move", "turn"}, trigger=is_on_gold and action_t in {"move", "turn"})
        self._mark_opp("BUG_COLLISION_IMPULSE_GLITCH", zone=is_on_sandstone, action_attempt=is_on_sandstone and action_t == "jump", trigger=is_on_sandstone and action_t == "jump")
        self._mark_opp("BUG_CONTEXTUAL_INTERACTION_OMISSION", zone=is_on_lapis, action_attempt=is_on_lapis and action_t == "use", trigger=is_on_lapis and action_t == "use")
        self._mark_opp("BUG_WORLD_STATE_MUTATION", zone=near_web, action_attempt=near_web and action_t == "use", trigger=near_web and action_t == "use")
        self._mark_opp("BUG_VIEW_DEPENDENT_MOVEMENT_LOCK", zone=is_on_quartz, action_attempt=is_on_quartz and action_t == "move", trigger=is_on_quartz and action_t == "move" and pitch < -85)
        self._mark_opp("BUG_SLOT_SELECTION_DESYNC", zone=is_on_emerald, action_attempt=is_on_emerald and action_t == "hotbar", trigger=is_on_emerald and action_t == "hotbar")
        self._mark_opp("BUG_BREAK_EVENT_CORRUPTION", zone=near_glass, action_attempt=near_glass and action_t == "attack", trigger=near_glass and action_t == "attack")

        # Sequence opportunity is two-step: turn on clay followed by jump.
        if is_on_clay:
            self._mark_opp("BUG_CONTEXTUAL_SEQUENCE_FAILURE", zone=True)
            if action_t == "turn":
                self._sequence_pending_for_opportunity = True
                self._mark_opp("BUG_CONTEXTUAL_SEQUENCE_FAILURE", action_attempt=True)
            elif action_t == "jump" and self._sequence_pending_for_opportunity:
                self._mark_opp("BUG_CONTEXTUAL_SEQUENCE_FAILURE", action_attempt=True, trigger=True)
                self._sequence_pending_for_opportunity = False
        elif action_t not in {"turn", "jump"}:
            self._sequence_pending_for_opportunity = False

    def step(self, action_idx):
        self.env_step_count += 1
        self.update_obs()
        prev_timestamp = self._last_msg_timestamp

        action = self.action_list[action_idx]
        pre_obs_data = dict(self._last_obs_raw or {})

        self._update_opportunity_stats(action, pre_obs_data)
        cae_reward = self._compute_cae_intrinsic_reward(action, pre_obs_data)

        effective_action, flags, cheat_cmd = self.fault.transform(action, self._last_obs_raw)

        if flags:
            self.bug_detector.env_injected.update(flags)

        if effective_action:
            self.agent_host.sendCommand(effective_action)
            time.sleep(0.1)

        if cheat_cmd:
            self.agent_host.sendCommand(cheat_cmd)
            time.sleep(1.0)

        wait_time = 1.0 if cheat_cmd else 0.5
        start_wait = time.time()
        while time.time() - start_wait < wait_time:
            self.update_obs()
            if (not cheat_cmd) and self._last_msg_timestamp != prev_timestamp and (time.time() - start_wait > 0.2):
                break
            time.sleep(0.05)

        if effective_action and ("move" in effective_action or "turn" in effective_action or "pitch" in effective_action or "jump" in effective_action):
            parts = effective_action.split()
            if len(parts) == 2 and parts[1] != "0":
                self.agent_host.sendCommand(f"{parts[0]} 0")

        obs = self._get_observation()
        ws = self.agent_host.getWorldState()

        curr_x, curr_z = float(obs[0]), float(obs[1])
        cell_key = (int(curr_x), int(curr_z))

        exploration_reward = 0.0
        if MIN_XZ <= curr_x <= MAX_XZ and MIN_XZ <= curr_z <= MAX_XZ:
            if cell_key not in self.visited_cells:
                self.visited_cells.add(cell_key)
                exploration_reward = 1.0

        bug_reward, evidences = self.bug_detector.check_bugs(action, (curr_x, curr_z), ws)
        reward = exploration_reward + bug_reward + cae_reward

        if not (MIN_XZ <= curr_x <= MAX_XZ and MIN_XZ <= curr_z <= MAX_XZ):
            reward += -0.5

        done = not ws.is_mission_running
        safe_bug_ids = [ev["id"] for ev in evidences] if evidences else []
        for bug_id in safe_bug_ids:
            if bug_id in self.opportunity_stats:
                self.opportunity_stats[bug_id]["detected_after_opportunity"] = True
        new_bugs_found = len(safe_bug_ids) > 0

        info = {
            "bug_detected": new_bugs_found,
            "bug_ids": safe_bug_ids,
            "detected_bugs": list(self.bug_detector.detected_bugs),
            "bug_evidence": evidences,
            "XPos": self.obs_data.get("XPos", 0.0),
            "YPos": self.obs_data.get("YPos", 0.0),
            "ZPos": self.obs_data.get("ZPos", 0.0),
            "Yaw": self.obs_data.get("Yaw", 0.0),
            "Pitch": self.obs_data.get("Pitch", 0.0),
            "action": self.action_list[action_idx] if action_idx < len(self.action_list) else "none",
            "extrinsic_reward": float(exploration_reward + bug_reward),
            "bug_reward": float(bug_reward),
            "exploration_reward": float(exploration_reward),
            "cae_intrinsic_reward": float(cae_reward),
            "cae_components": dict(self.last_cae_components),
            "opportunity_stats": self.opportunity_stats,
        }

        return obs, float(reward), done, False, info

    def close(self):
        self._safe_wait_for_mission_end()

    def __getstate__(self):
        state = self.__dict__.copy()
        if "agent_host" in state:
            del state["agent_host"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        import MalmoPython
        self.agent_host = MalmoPython.AgentHost()
