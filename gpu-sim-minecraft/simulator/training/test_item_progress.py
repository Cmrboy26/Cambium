import os
import unittest

import numpy as np

from simulator.training.item_progress import item_progress_observations


class ItemProgressChecks(unittest.TestCase):
    def test_task_neutral_progress_and_cancellation(self):
        actual = item_progress_observations([0, 32, 24, 8, 0], 32, [0, 0.1, 0.5, 1, 0])
        np.testing.assert_allclose(actual, [[0, 0], [0, 0.1], [0.25, 0.5], [0.75, 1], [0, 0]])
        self.assertEqual(actual.dtype, np.float32)
        alternate_task = item_progress_observations([0, 10], [0, 20], [0, 0.25])
        np.testing.assert_allclose(alternate_task, [[0, 0], [0.5, 0.25]])

    def test_invalid_shapes_and_states_are_rejected(self):
        for remaining, duration, mining in (([-1], [32], [0]), ([33], [32], [0]),
                                           ([1], [0], [0]), ([0], [32], [float("nan")]),
                                           ([0], [32], [1.01]), ([True], [32], [0]),
                                           ([0, 1], [2, 2, 2], [0, 0])):
            with self.subTest(remaining=remaining, duration=duration, mining=mining), self.assertRaises(ValueError):
                item_progress_observations(remaining, duration, mining)

    @unittest.skipUnless(os.environ.get("MCSIM_TEST_CPU_LIBRARY"), "No explicit CPU library selected")
    def test_actor_progress_api_works_without_bridge(self):
        from simulator.training.environment import Environment, ROOT
        from simulator.training.objectives import actor_task_observations

        contract = {"format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3",
                    "ticks_per_action": 1, "observations": [{"feature": "relative.z"}],
                    "actions": [{"name": "movement", "values": ["neutral", "forward"]}]}
        with Environment(1, environment=ROOT / "simulator/tests/reset_environment.json", contract=contract,
                         library=os.environ["MCSIM_TEST_CPU_LIBRARY"]) as environment:
            np.testing.assert_array_equal(environment.item_progress(), np.zeros((1, 2, 2)))
            targets = np.array([[[0.5, 1, 2], [0.5, 1, -2]]])
            before = actor_task_observations(environment.actor_task_state(), targets)
            environment.step(np.ones((1, 2, 1), dtype=np.int32))
            after = actor_task_observations(environment.actor_task_state(), targets)
            self.assertTrue(np.all(after[..., 4] < before[..., 4]))
            np.testing.assert_array_equal(after[..., :2], np.zeros((1, 2, 2)))


if __name__ == "__main__":
    unittest.main()