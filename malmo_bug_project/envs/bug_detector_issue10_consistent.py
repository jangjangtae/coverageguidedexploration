import json
import os
from typing import Any, Dict, Optional, Tuple


class BugDetector:
    """
    Consistent issue-inspired bug detector (10-bug set).

    핵심 수정:
    - canonical 10-bug set만 사용
    - _pop_flag가 "키 존재 여부" 기준으로 동작해서 빈 dict도 정상 처리
    - yaw delta를 [-180, 180)로 정규화
    """

    def __init__(self, json_path="envs/bug_definitions_issue10_consistent.json", log_dir="runs"):
        self.bugs: Dict[str, Dict[str, Any]] = {}
        self.legacy_to_canonical: Dict[str, str] = {}

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for bug in data.get("bugs", []):
            bug_id = bug["id"]
            self.bugs[bug_id] = bug
            legacy_id = bug.get("legacy_id")
            if legacy_id:
                self.legacy_to_canonical[legacy_id] = bug_id

        self.detected_bugs = set()
        self.env_injected: Dict[str, Any] = {}
        self.prev_yaw: Optional[float] = None
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.log_dir, "bug_log.jsonl")

    def reset(self):
        self.detected_bugs.clear()
        self.env_injected = {}
        self.prev_yaw = None

    def _canonical_bug_id(self, bug_id: str) -> str:
        return self.legacy_to_canonical.get(bug_id, bug_id)

    def _pop_flag(self, key: str, legacy_key: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
        sentinel = object()
        raw = self.env_injected.pop(key, sentinel)
        if raw is sentinel and legacy_key:
            raw = self.env_injected.pop(legacy_key, sentinel)

        if raw is sentinel:
            return False, {}

        if isinstance(raw, dict):
            return True, raw
        return True, {}

    def _record(self, bug_id: str, evidence: Dict[str, Any]):
        bug_id = self._canonical_bug_id(bug_id)
        if bug_id not in self.bugs:
            return 0, None
        if bug_id in self.detected_bugs:
            return 0, None

        self.detected_bugs.add(bug_id)
        reward = int(self.bugs[bug_id].get("reward", 100))
        rec = {
            "id": bug_id,
            "description": self.bugs[bug_id].get("description", ""),
            "evidence": evidence,
        }

        print(f"🚨 BUG FOUND: {bug_id} | Evidence: {evidence}")

        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

        return reward, {"id": bug_id, "evidence": evidence}

    @staticmethod
    def _normalize_yaw_delta(curr_yaw: float, prev_yaw: Optional[float]) -> float:
        if prev_yaw is None:
            return 0.0
        delta = curr_yaw - prev_yaw
        return (delta + 180.0) % 360.0 - 180.0

    def check_bugs(self, last_action, curr_pos, world_state):
        evidences = []
        total_reward = 0

        obs = {}
        try:
            if world_state.number_of_observations_since_last_state > 0:
                obs = json.loads(world_state.observations[-1].text)
        except Exception:
            pass

        curr_yaw = float(obs.get("Yaw", 0.0))

        bug_specs = [
            ("BUG_HEADING_DYNAMICS_ANOMALY", "bug_heading_dynamics", "bug_reverse_turn"),
            ("BUG_TRANSITION_TELEPORT", "bug_transition_teleport", "bug_teleport_trap"),
            ("BUG_MOVEMENT_LOCK_ZONE", "bug_movement_lock_zone", "bug_deadzone"),
            ("BUG_COLLISION_IMPULSE_GLITCH", "bug_collision_impulse", "bug_superjump"),
            ("BUG_CONTEXTUAL_SEQUENCE_FAILURE", "bug_contextual_sequence_failure", "bug_sequence"),
            ("BUG_CONTEXTUAL_INTERACTION_OMISSION", "bug_contextual_interaction_omission", "bug_interact_fail"),
            ("BUG_WORLD_STATE_MUTATION", "bug_world_state_mutation", "bug_transmutation"),
            ("BUG_VIEW_DEPENDENT_MOVEMENT_LOCK", "bug_view_dependent_lock", "bug_sky_freeze"),
            ("BUG_SLOT_SELECTION_DESYNC", "bug_slot_selection_desync", "bug_hotbar_error"),
            ("BUG_BREAK_EVENT_CORRUPTION", "bug_break_event_corruption", "bug_break_crash"),
        ]

        for bug_id, new_flag, legacy_flag in bug_specs:
            fired, evidence = self._pop_flag(new_flag, legacy_flag)
            if not fired:
                continue

            base = {"pos": curr_pos}
            if bug_id == "BUG_HEADING_DYNAMICS_ANOMALY":
                base["delta_yaw"] = self._normalize_yaw_delta(curr_yaw, self.prev_yaw)

            merged = {**base, **evidence}
            r, e = self._record(bug_id, merged)
            if r:
                total_reward += r
                evidences.append(e)

        self.prev_yaw = curr_yaw
        return total_reward, evidences
