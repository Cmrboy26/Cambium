import ctypes
import os
import unittest

import numpy as np
import torch

from simulator.training.environment import Environment


class CUDAObservationChecks(unittest.TestCase):
    def test_cpu_environment_rejects_cuda_observations(self):
        with Environment(1, library=os.environ.get("MCSIM_TEST_CPU_LIBRARY")) as environment:
            with self.assertRaisesRegex(ValueError, "require a CUDA environment"):
                environment.observe_base_cuda()
            function = environment.native.mcsim_observe_base_cuda_sync
            function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
            function.restype = ctypes.c_int
            self.assertEqual(function(environment._handle, None, 44, None), -1)

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_device_observations_match_host_after_steps_and_resets(self):
        library = os.environ.get("MCSIM_TEST_CUDA_LIBRARY")
        with Environment(3, device="cuda", library=library, max_ticks=20) as environment:
            def compare():
                expected = environment.observe()[:, :, :22]
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    actual = environment.observe_base_cuda()
                self.assertEqual(actual.device.type, "cuda")
                self.assertEqual(actual.dtype, torch.float32)
                self.assertEqual(tuple(actual.shape), (3, 2, 22))
                np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=0, atol=1e-6)

            compare()
            actions = np.zeros((3, 2, 6), dtype=np.int32)
            actions[:, :, 4] = 4
            actions[:, :, 5] = 2
            actions[:, 0, 0] = 1
            for _ in range(3):
                environment.step(actions)
                compare()
            environment.reset(np.array([True, False, True]))
            compare()

            function = environment.native.mcsim_observe_base_cuda_sync
            function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
            function.restype = ctypes.c_int
            host = np.empty((3, 2, 22), dtype=np.float32)
            self.assertEqual(function(environment._handle, host.ctypes.data, host.size, None), -1)
            self.assertIn(b"CUDA device", environment.native.mcsim_last_error())
            self.assertEqual(function(environment._handle, None, host.size, None), -1)
            device_output = torch.empty((3, 2, 22), device="cuda", dtype=torch.float32)
            self.assertEqual(function(environment._handle, device_output.data_ptr(), device_output.numel() - 1, None), -1)

        with self.assertRaisesRegex(RuntimeError, "closed"):
            environment.observe_base_cuda()

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_device_observations_are_not_policy_projected(self):
        contract = {"format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3",
                    "ticks_per_action": 1,
                    "observations": [{"feature": "opponent.health_fraction"},
                                     {"feature": "self.health_fraction"}],
                    "actions": [{"name": "jump", "values": [False, True]}]}
        with Environment(2, device="cuda", contract=contract,
                         library=os.environ.get("MCSIM_TEST_CUDA_LIBRARY")) as environment:
            players = [environment.episode_scenario(world)["players"] for world in range(2)]
            for definition in players:
                definition[0]["health"] = 9
                definition[1]["health"] = 17
            environment.reset_players(players, np.array([True, True]))
            projected = environment.observe()
            raw = environment.observe_base_cuda().cpu().numpy()
            self.assertEqual(projected.shape, (2, 2, 2))
            self.assertEqual(raw.shape, (2, 2, 22))
            self.assertNotEqual(projected[0, 0, 0], projected[0, 0, 1])
            np.testing.assert_array_equal(projected[:, :, 0], raw[:, :, 4])
            np.testing.assert_array_equal(projected[:, :, 1], raw[:, :, 0])


if __name__ == "__main__":
    unittest.main()
