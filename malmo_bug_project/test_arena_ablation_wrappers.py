import math
import unittest
from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from agents.train_arena_issue10_final import (
    ABLATION_ALGO_ORDER,
    CAEConfig,
    HybridConfig,
    RNDCAEHybridRewardWrapper,
    cae_active_weights,
    cae_config_for_algorithm,
    extract_logged_position,
    extract_position,
    select_algorithms,
)


class DummyArenaEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        info = {"XPos": 1.5, "YPos": 5.0, "ZPos": 2.5}
        return np.zeros(4, dtype=np.float32), 0.0, False, False, info


class ArenaAblationTests(unittest.TestCase):
    def test_accepted_full_weights_are_unchanged(self):
        self.assertEqual(
            cae_active_weights(CAEConfig()),
            {
                "spatial": 0.005,
                "state_action": 0.05,
                "object_action": 0.10,
                "opportunity": 0.15,
                "room_action": 0.03,
                "sequence": 0.07,
            },
        )

    def test_ablation_selection_and_configs(self):
        args = SimpleNamespace(cae_config=vars(CAEConfig()).copy())
        spatial_cfg = cae_config_for_algorithm(args, "RND_CAE_SPATIAL")
        nostag_cfg = cae_config_for_algorithm(args, "RND_CAE_NO_STAG")

        self.assertEqual(select_algorithms("ablations"), ABLATION_ALGO_ORDER)
        self.assertEqual(select_algorithms("ablations_reverse"), list(reversed(ABLATION_ALGO_ORDER)))
        self.assertEqual(cae_active_weights(spatial_cfg), {"spatial": 0.405})
        self.assertTrue(spatial_cfg.spatial_only)
        self.assertTrue(spatial_cfg.enable_stagnation)
        self.assertFalse(nostag_cfg.spatial_only)
        self.assertFalse(nostag_cfg.enable_stagnation)

    def test_spatial_only_stagnation_uses_direct_uppercase_position_key(self):
        cfg = CAEConfig(
            spatial_only=True,
            spatial_only_weight=0.405,
            enable_stagnation=True,
            stagnation_threshold=1,
        )
        wrapper = RNDCAEHybridRewardWrapper(DummyArenaEnv(), HybridConfig(cae=cfg))
        obs = np.array([1.5, 2.5, 0.0, 1.0], dtype=np.float32)
        info = {"XPos": 1.5, "YPos": 5.0, "ZPos": 2.5}

        wrapper.global_step = 1
        first_reward, _, first_stag, first_new = wrapper._compute_cae(obs, info, 0)
        wrapper.global_step = 2
        second_reward, _, second_stag, second_new = wrapper._compute_cae(obs, info, 0)

        self.assertAlmostEqual(first_reward, 0.405)
        self.assertEqual(first_stag, 0)
        self.assertTrue(first_new)
        self.assertAlmostEqual(second_reward, 0.405 / math.sqrt(2.0) + 0.03)
        self.assertEqual(second_stag, 1)
        self.assertFalse(second_new)
        self.assertEqual(wrapper.stagnation_trigger_count, 1)
        self.assertEqual(len(wrapper.cae_counts["spatial"]), 1)
        self.assertIn((1, 2), wrapper.cae_counts["spatial"])
        self.assertTrue(all(len(wrapper.cae_counts[k]) == 0 for k in wrapper.cae_counts if k != "spatial"))

    def test_no_stag_disables_only_recovery_bonus(self):
        cfg = CAEConfig(
            spatial_only=True,
            spatial_only_weight=0.405,
            enable_stagnation=False,
            stagnation_threshold=1,
        )
        wrapper = RNDCAEHybridRewardWrapper(DummyArenaEnv(), HybridConfig(cae=cfg))
        obs = np.array([1.5, 2.5, 0.0, 1.0], dtype=np.float32)
        info = {"XPos": 1.5, "YPos": 5.0, "ZPos": 2.5}

        wrapper.global_step = 1
        wrapper._compute_cae(obs, info, 0)
        wrapper.global_step = 2
        reward, _, stagnated, _ = wrapper._compute_cae(obs, info, 0)

        self.assertAlmostEqual(reward, 0.405 / math.sqrt(2.0))
        self.assertEqual(stagnated, 0)
        self.assertEqual(wrapper.stagnation_trigger_count, 0)

    def test_logging_reads_uppercase_malmo_position_without_changing_cae_fallback(self):
        info = {"XPos": 1.5, "YPos": 5.0, "ZPos": 2.5}
        self.assertIsNone(extract_position(info))
        self.assertEqual(extract_logged_position(info), (1.5, 5.0, 2.5))


if __name__ == "__main__":
    unittest.main()
