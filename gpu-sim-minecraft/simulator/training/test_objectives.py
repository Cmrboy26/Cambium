import unittest

import numpy as np
import torch

from backend.fast_rl_model import BridgeActorCritic
from simulator.training.contract import bridge_observation_descriptor
from simulator.training.objectives import objective_observations, objective_progress_rewards


class ObjectiveGeometryChecks(unittest.TestCase):
    def test_actor_relative_angles_and_distance(self):
        positions = np.zeros((5, 3))
        targets = np.array([[0, 0, 10], [0, 0, -10], [10, 0, 0], [0, 10, 10], [0, 0, 0]])
        actual = objective_observations(positions, np.array([0, 180, 0, 0, 0]),
                                        np.zeros(5), targets, distance_scale=50)
        np.testing.assert_allclose(actual, [[0, 0, 0.2], [0, 0, 0.2], [-0.5, 0, 0.2],
                                            [0, -0.5, np.sqrt(200) / 50], [0, 0, 0]], atol=1e-6)
        self.assertEqual(actual.dtype, np.float32)

    def test_invalid_geometry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "geometry"):
            objective_observations([[0, 0, 0]], [0], [0], [[0, 0, float("nan")]])
        with self.assertRaisesRegex(ValueError, "geometry"):
            objective_observations([[0, 0, 0]], [0], [0], [[0, 0, 1]], distance_scale=0)
        with self.assertRaisesRegex(ValueError, "geometry"):
            objective_observations([[0, 0, 0]], [0], [0], [[0, 0, 1]], distance_scale="50")

    def test_progress_rewards_are_separate_and_do_not_reward_revisits(self):
        toward, away, best = objective_progress_rewards([10, 9, 10, 9], [9, 10, 9, 8], [10, 9, 9, 9],
                                                        distance_scale=10, toward_weight=0.2, away_weight=0.1)
        np.testing.assert_allclose(toward, [0.02, 0, 0, 0.02])
        np.testing.assert_allclose(away, [0, -0.01, 0, 0])
        np.testing.assert_allclose(best, [9, 9, 9, 8])
        positive, negative, _ = objective_progress_rewards([9], [10], [9], away_weight=0)
        np.testing.assert_array_equal(positive, [0])
        np.testing.assert_array_equal(negative, [0])

    def test_invalid_progress_state_is_rejected(self):
        for values in (([1], [1], [2]), ([1], [-1], [1]), ([1], [float("nan")], [1])):
            with self.subTest(values=values), self.assertRaises(ValueError):
                objective_progress_rewards(*values)
        with self.assertRaises(ValueError):
            objective_progress_rewards([1], [1], [1], away_weight=-1)
        with self.assertRaises(ValueError):
            objective_progress_rewards([1], [1], [1], toward_weight=1j)

    def test_versioned_bridge_model_shapes(self):
        old = BridgeActorCritic(1, bridge_observation_descriptor(), 8, {"movement": 3})
        new = BridgeActorCritic(1, bridge_observation_descriptor(2), 8, {"movement": 3})
        inputs = {"scalar": torch.zeros((2, 1)), "match": torch.zeros((2, 17)),
                  "voxels": torch.zeros((2, 5, 9, 9), dtype=torch.uint8)}
        logits, values = new(inputs)
        self.assertEqual(logits["movement"].shape, (2, 3))
        self.assertEqual(values.shape[0], 2)
        with self.assertRaises(ValueError):
            old(inputs)
        with self.assertRaises(RuntimeError):
            new.load_state_dict(old.state_dict())


if __name__ == "__main__":
    unittest.main()