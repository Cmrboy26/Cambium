import ctypes
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import weakref

import numpy as np
import torch

from simulator.training.contract import FEATURES
from simulator.training.environment import Environment, ROOT


class CUDAObservationChecks(unittest.TestCase):
    def configured_cuda(self, environment):
        prepare = environment.native.mcsim_prepare_cuda_observations
        prepare.argtypes = [ctypes.c_void_p]
        prepare.restype = ctypes.c_int
        self.assertEqual(prepare(environment._handle), 0, environment.native.mcsim_last_error())
        function = environment.native.mcsim_observe_cuda_sync
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        function.restype = ctypes.c_int
        output = torch.empty((environment.count, 2, environment.observation_dim), device="cuda", dtype=torch.float32)
        self.assertEqual(function(environment._handle, output.data_ptr(), output.numel(),
                                  torch.cuda.current_stream().cuda_stream), 0, environment.native.mcsim_last_error())
        return output.cpu().numpy()

    def compare_configured(self, environment):
        expected = environment.observe()
        np.testing.assert_allclose(self.configured_cuda(environment), expected, rtol=0, atol=1e-6)
        if environment._cuda_observations is None:
            environment.register_cuda_observations()
        output = environment.observe_cuda_async()
        np.testing.assert_allclose(output.cpu().numpy(), expected, rtol=0, atol=1e-6)

    def test_cpu_environment_rejects_cuda_observations(self):
        with Environment(1, library=os.environ.get("MCSIM_TEST_CPU_LIBRARY")) as environment:
            with self.assertRaisesRegex(ValueError, "require a CUDA environment"):
                environment.observe_base_cuda()
            function = environment.native.mcsim_observe_base_cuda_sync
            function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
            function.restype = ctypes.c_int
            self.assertEqual(function(environment._handle, None, 44, None), -1)
            with self.assertRaisesRegex(ValueError, "require a CUDA environment"):
                environment.register_cuda_observations()
            with self.assertRaisesRegex(RuntimeError, "Register a CUDA observation buffer"):
                environment.observe_cuda_async()

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

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_configured_layout_tracks_loadouts_bows_and_selective_resets(self):
        contract = {"format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3",
                    "ticks_per_action": 1, "requires": ["bow"],
                    "observations": [{"feature": name} for name in FEATURES] + [
                        {"feature": "self.weapon_slot", "encoding": "one_hot", "categories": [1, 0], "scale": -2},
                        {"feature": "self.health_fraction", "clip": [0.2, 0.6], "scale": 3}],
                    "actions": [{"name": "weapon", "values": [0, 1]},
                                {"name": "use_item", "values": [False, True]},
                                {"name": "attack", "values": [False, True]}]}
        path = ROOT / "simulator/tests/reset_environment.json"
        with Environment(3, device="cuda", environment=path, contract=contract, max_ticks=80,
                         library=os.environ.get("MCSIM_TEST_CUDA_LIBRARY")) as environment:
            self.compare_configured(environment)
            definitions = [environment.episode_scenario(world)["players"] for world in range(3)]
            definitions[0][0]["loadout"] = {"weapons": ["minecraft:bow", "minecraft:air"], "arrows": 2}
            definitions[1][0]["loadout"] = {"weapons": ["minecraft:air", "minecraft:air"], "selected": 1}
            definitions[2] = None
            environment.reset_players(definitions, np.array([True, True, False]))
            self.compare_configured(environment)
            actions = np.zeros((3, 2, 3), dtype=np.int32)
            actions[0, 0, 1] = 1
            for _ in range(12):
                environment.step(actions)
                self.compare_configured(environment)
            actions[0, 0, 1] = 0
            actions[1, 0, 0] = 0
            environment.step(actions)
            self.compare_configured(environment)
            environment.reset(np.array([False, True, False]))
            self.compare_configured(environment)

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_v2_layout_tracks_shield_disable(self):
        with tempfile.TemporaryDirectory() as directory:
            definition = json.loads((ROOT / "simulator/environments/duel.json").read_text())
            definition["players"][0]["position"][2] = -0.5
            definition["players"][1]["position"][2] = 1.5
            path = Path(directory) / "shield-duel.json"
            path.write_text(json.dumps(definition), encoding="utf-8")
            for prepared_first in (False, True):
                with self.subTest(prepared_first=prepared_first), Environment(
                        1, device="cuda", environment=path, schema="v2", max_ticks=120,
                        library=os.environ.get("MCSIM_TEST_CUDA_LIBRARY")) as environment:
                    if prepared_first:
                        self.compare_configured(environment)
                    actions = np.zeros((1, 2, 8), dtype=np.int32)
                    actions[:, :, 4] = 4
                    actions[:, :, 5] = 2
                    actions[0, 0, 6] = 1
                    actions[0, 1, 7] = 1
                    for _ in range(6):
                        environment.step(actions)
                        if prepared_first:
                            self.compare_configured(environment)
                    actions[0, 0, 3] = 1
                    observed, _, _, _ = environment.step(actions)
                    self.assertEqual(observed[0, 0, 27], 1.0)
                    self.compare_configured(environment)
                    actions[0, 0, 3] = 0
                    for _ in range(3):
                        environment.step(actions)
                        self.compare_configured(environment)
                    environment.reset()
                    self.compare_configured(environment)

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_async_observation_streams_lifetime_and_validation(self):
        library = os.environ.get("MCSIM_TEST_CUDA_LIBRARY")
        with Environment(3, device="cuda", library=library, max_ticks=30) as environment:
            with self.assertRaisesRegex(RuntimeError, "Register a CUDA observation buffer"):
                environment.observe_cuda_async()
            shape = (3, 2, environment.observation_dim)
            invalid = [torch.empty(shape), torch.empty(shape, device="cuda", dtype=torch.float64),
                       torch.empty((3, environment.observation_dim, 2), device="cuda").transpose(1, 2),
                       torch.empty((3, 2, environment.observation_dim + 1), device="cuda"),
                       torch.empty(shape, device="cuda", requires_grad=True)]
            for output in invalid:
                with self.assertRaisesRegex(ValueError, "contiguous float32"):
                    environment.register_cuda_observations(output)
            output = environment.register_cuda_observations()
            with ThreadPoolExecutor(max_workers=1) as executor:
                with self.assertRaisesRegex(RuntimeError, "registration thread"):
                    executor.submit(environment.observe_cuda_async).result()
                with self.assertRaisesRegex(RuntimeError, "registration thread"):
                    executor.submit(environment.reset).result()
            reference = weakref.ref(output)
            del output
            gc.collect()
            self.assertIsNotNone(reference())
            producer = torch.cuda.Stream()
            consumer = torch.cuda.Stream()
            expected = environment.observe()
            with torch.cuda.stream(producer):
                output = environment.observe_cuda_async()
                same_stream = output.clone()
            environment.wait_cuda_observations()
            with torch.cuda.stream(consumer):
                consumer.wait_stream(producer)
                cross_stream = output.clone()
            np.testing.assert_allclose(same_stream.cpu().numpy(), expected, rtol=0, atol=1e-6)
            np.testing.assert_allclose(cross_stream.cpu().numpy(), expected, rtol=0, atol=1e-6)
            with torch.cuda.stream(producer):
                output = environment.observe_cuda_async()
            actions = np.zeros((3, 2, 6), dtype=np.int32)
            actions[:, :, 4] = 4
            actions[:, :, 5] = 2
            actions[:, 0, 0] = 1
            environment.step(actions)
            np.testing.assert_allclose(output.cpu().numpy(), expected, rtol=0, atol=1e-6)
            expected = environment.observe()
            with torch.cuda.stream(producer):
                output = environment.observe_cuda_async()
            environment.reset(np.array([True, False, True]))
            np.testing.assert_allclose(output.cpu().numpy(), expected, rtol=0, atol=1e-6)
            output.resize_((output.numel() + 1,))
            with self.assertRaisesRegex(RuntimeError, "buffer changed"):
                environment.observe_cuda_async()
            replacement = environment.register_cuda_observations()
            np.testing.assert_allclose(environment.observe_cuda_async().cpu().numpy(), environment.observe(),
                                       rtol=0, atol=1e-6)
            self.assertEqual(tuple(replacement.shape), shape)
            graph = torch.cuda.CUDAGraph()
            capture_marker = torch.empty(1, device="cuda")
            with self.assertRaisesRegex(RuntimeError, "do not support graph capture"):
                with torch.cuda.graph(graph, stream=producer):
                    capture_marker.zero_()
                    environment.observe_cuda_async()
            expected = environment.observe()
            with torch.cuda.stream(producer):
                pending = environment.observe_cuda_async()
            environment.close()
            environment.close()
            np.testing.assert_allclose(pending.cpu().numpy(), expected, rtol=0, atol=1e-6)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            environment.observe_cuda_async()

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY")
                         and torch.cuda.device_count() >= 2, "Two CUDA devices and a CUDA library are required")
    def test_async_observation_rejects_another_cuda_device(self):
        library = os.environ.get("MCSIM_TEST_CUDA_LIBRARY")
        with torch.cuda.device(0):
            environment = Environment(2, device="cuda", library=library)
        wrong = torch.empty((2, 2, environment.observation_dim), device="cuda:1", dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "differs from batch device"):
            environment.register_cuda_observations(wrong)
        with torch.cuda.device(0):
            environment.register_cuda_observations()
            expected = environment.observe()
            other_stream = torch.cuda.Stream(device=1)
            function = environment.native.mcsim_observe_cuda_async
            function.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            function.restype = ctypes.c_int
            self.assertEqual(function(environment._handle, other_stream.cuda_stream), -1)
            self.assertIn(b"CUDA", environment.native.mcsim_last_error())
            output = environment.observe_cuda_async()
        with torch.cuda.device(1):
            environment.close()
        np.testing.assert_allclose(output.cpu().numpy(), expected, rtol=0, atol=1e-6)

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_async_observation_trace_has_no_synchronization_or_transfers(self):
        library = os.environ.get("MCSIM_TEST_CUDA_LIBRARY")
        with Environment(2, device="cuda", library=library) as environment:
            environment.register_cuda_observations()
            environment.observe_cuda_async()
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profiler:
                with torch.profiler.record_function("mcsim_async_observation"):
                    environment.observe_cuda_async()
            names = []
            for event in profiler.events():
                parent = event.cpu_parent
                while parent is not None:
                    if parent.name == "mcsim_async_observation":
                        names.append(event.name)
                        break
                    parent = parent.cpu_parent
            self.assertIn("cudaLaunchKernel", names, "Profiler must capture native CUDA runtime calls")
            self.assertFalse(any(name.startswith(("cudaDeviceSynchronize", "cudaStreamSynchronize",
                                                  "cudaEventSynchronize", "cudaMemcpy")) for name in names), names)

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_async_observation_supports_maximum_policy_width(self):
        contract = {"format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3",
                    "ticks_per_action": 1,
                    "observations": [{"feature": "self.health_fraction", "scale": (-1 if index % 2 else 1)}
                                     for index in range(128)],
                    "actions": [{"name": "jump", "values": [False, True]}]}
        with Environment(2, device="cuda", contract=contract,
                         library=os.environ.get("MCSIM_TEST_CUDA_LIBRARY")) as environment:
            output = environment.register_cuda_observations()
            self.assertEqual(tuple(output.shape), (2, 2, 128))
            self.compare_configured(environment)

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_async_observations_cover_non_aligned_world_batches(self):
        library = os.environ.get("MCSIM_TEST_CUDA_LIBRARY")
        with Environment(137, device="cuda", library=library, max_ticks=30) as environment:
            self.compare_configured(environment)
            actions = np.zeros((137, 2, 6), dtype=np.int32)
            actions[:, :, 4] = 4
            actions[:, :, 5] = 2
            actions[::2, 0, 0] = 1
            for _ in range(3):
                environment.step(actions)
                self.compare_configured(environment)
            environment.reset(np.arange(137) % 3 == 0)
            self.compare_configured(environment)

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_async_close_releases_resources_when_wait_fails(self):
        library = os.environ.get("MCSIM_TEST_CUDA_LIBRARY")
        environment = Environment(1, device="cuda", library=library)
        environment.register_cuda_observations()
        environment.observe_cuda_async()
        with mock.patch.object(environment, "wait_cuda_observations", side_effect=RuntimeError("wait failed")):
            with self.assertRaisesRegex(RuntimeError, "wait failed"):
                environment.close()
        self.assertIsNone(environment._handle)
        self.assertIsNone(environment._cuda_observations)
        environment.close()


if __name__ == "__main__":
    unittest.main()
