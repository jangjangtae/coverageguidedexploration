import json
import os

class BugDetector:
    def __init__(self, json_path="envs/bug_definitions.json", log_dir="runs"):
        self.bugs = {}
        try:
            with open(json_path, "r") as f:
                self.bugs = {b["id"]: b for b in json.load(f)["bugs"]}
        except FileNotFoundError:
            pass
        
        self.detected_bugs = set()
        self.env_injected = {}
        self.prev_yaw = None
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def reset(self):
        self.detected_bugs.clear()
        self.env_injected = {}
        self.prev_yaw = None

    def _record(self, bug_id, evidence):
        if bug_id not in self.bugs: return 0, None
        if bug_id in self.detected_bugs: return 0, None
        print(f"🚨 BUG FOUND: {bug_id} | Evidence: {evidence}")
        self.detected_bugs.add(bug_id)
        return 100, {"id": bug_id, "evidence": evidence}

    def check_bugs(self, last_action, curr_pos, world_state):
        evidences = []
        total_reward = 0
        
        obs = {}
        try:
            if world_state.number_of_observations_since_last_state > 0:
                obs = json.loads(world_state.observations[-1].text)
        except: pass
        curr_yaw = obs.get("Yaw", 0)

        if self.env_injected.pop("bug_reverse_turn", False):
            delta = 0
            if self.prev_yaw is not None: delta = curr_yaw - self.prev_yaw
            r, e = self._record("BUG_REVERSE_TURN", {"delta_yaw": delta})
            if r: total_reward += r; evidences.append(e)

        if self.env_injected.pop("bug_teleport_trap", False):
            r, e = self._record("BUG_TELEPORT_TRAP", {"pos": curr_pos})
            if r: total_reward += r; evidences.append(e)

        if self.env_injected.pop("bug_deadzone", False):
            r, e = self._record("BUG_DEAD_ZONE", {"pos": curr_pos})
            if r: total_reward += r; evidences.append(e)

        if self.env_injected.pop("bug_superjump", False):
            r, e = self._record("BUG_SUPER_JUMP", {})
            if r: total_reward += r; evidences.append(e)

        if self.env_injected.pop("bug_sequence", False):
            r, e = self._record("BUG_SEQUENCE_ERROR", {})
            if r: total_reward += r; evidences.append(e)

        if self.env_injected.pop("bug_interact_fail", False):
            r, e = self._record("BUG_INTERACT_FAIL", {})
            if r: total_reward += r; evidences.append(e)

        if self.env_injected.pop("bug_transmutation", False):
            r, e = self._record("BUG_TRANSMUTATION", {})
            if r: total_reward += r; evidences.append(e)

        if self.env_injected.pop("bug_sky_freeze", False):
            r, e = self._record("BUG_SKY_FREEZE", {})
            if r: total_reward += r; evidences.append(e)

        if self.env_injected.pop("bug_hotbar_error", False):
            r, e = self._record("BUG_HOTBAR_ERROR", {})
            if r: total_reward += r; evidences.append(e)

        if self.env_injected.pop("bug_break_crash", False):
            r, e = self._record("BUG_BREAK_CRASH", {})
            if r: total_reward += r; evidences.append(e)

        self.prev_yaw = curr_yaw
        return total_reward, evidences
