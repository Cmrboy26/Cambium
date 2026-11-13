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
from simulator.training.environment import Environment, REWARD_NAMES, ROOT


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


class CUDAActionChecks(unittest.TestCase):
    def test_cpu_environment_rejects_cuda_actions(self):
        with Environment(1, library=os.environ.get("MCSIM_TEST_CPU_LIBRARY")) as environment:
            with self.assertRaisesRegex(ValueError, "require a CUDA environment"):
                environment.register_cuda_actions()
            with self.assertRaisesRegex(RuntimeError, "Register a CUDA action buffer"):
                environment.decode_cuda_actions_async()

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_async_actions_validate_buffers_and_sticky_status(self):
        library = os.environ.get("MCSIM_TEST_CUDA_LIBRARY")
        with Environment(3, device="cuda", library=library) as environment:
            with self.assertRaisesRegex(RuntimeError, "observation buffer"):
                environment.register_cuda_actions()
            environment.register_cuda_observations()
            shape = (3, 2, environment.action_count)
            invalid = [torch.empty(shape), torch.empty(shape, device="cuda", dtype=torch.int64),
                       torch.empty((3, environment.action_count, 2), device="cuda", dtype=torch.int32).transpose(1, 2),
                       torch.empty((3, 2, environment.action_count + 1), device="cuda", dtype=torch.int32)]
            for actions in invalid:
                with self.assertRaisesRegex(ValueError, "contiguous int32"):
                    environment.register_cuda_actions(actions)
            actions = environment.register_cuda_actions()
            reference = weakref.ref(actions)
            del actions
            gc.collect()
            self.assertIsNotNone(reference())
            actions = reference()
            producer = torch.cuda.Stream()
            with torch.cuda.stream(producer):
                actions.zero_()
                actions[:, :, 4] = 4
                actions[:, :, 5] = 2
                environment.decode_cuda_actions_async()
            environment.wait_cuda_actions()
            with torch.cuda.stream(producer):
                actions[0, 0, 0] = -1
                environment.decode_cuda_actions_async()
            with self.assertRaisesRegex(ValueError, "Invalid CUDA action"):
                environment.wait_cuda_actions()
            with torch.cuda.stream(producer):
                actions[0, 0, 0] = 0
                environment.decode_cuda_actions_async()
            with self.assertRaisesRegex(ValueError, "Invalid CUDA action"):
                environment.wait_cuda_actions()
            environment.clear_cuda_action_status()
            with torch.cuda.stream(producer):
                environment.decode_cuda_actions_async()
            environment.wait_cuda_actions()
            actions.resize_((actions.numel() + 1,))
            with self.assertRaisesRegex(RuntimeError, "buffer changed"):
                environment.decode_cuda_actions_async()
            actions = environment.register_cuda_actions()
            actions.zero_()
            actions[:, :, 4] = 4
            actions[:, :, 5] = 2
            environment.decode_cuda_actions_async()
            environment.wait_cuda_actions()

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_cuda_actions_support_custom_head_order_and_selected_slots(self):
        contract = {"format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3",
                    "ticks_per_action": 1, "observations": [{"feature": "self.weapon_slot"}],
                    "actions": [{"name": "yaw", "values": [10, 0, -10]},
                                {"name": "weapon", "values": [1, 0]},
                                {"name": "sneak", "values": [True, False]}]}
        path = ROOT / "simulator/tests/reset_environment.json"
        with Environment(2, device="cuda", environment=path, contract=contract,
                         library=os.environ.get("MCSIM_TEST_CUDA_LIBRARY")) as environment:
            environment.register_cuda_observations()
            actions = environment.register_cuda_actions()
            actions.zero_()
            environment.decode_cuda_actions_async()
            environment.wait_cuda_actions()
            actions[1, 1, 1] = 2
            environment.decode_cuda_actions_async()
            with self.assertRaisesRegex(ValueError, "Invalid CUDA action"):
                environment.wait_cuda_actions()
            environment.clear_cuda_action_status()
            actions[1, 1, 1] = 1
            environment.decode_cuda_actions_async()
            environment.wait_cuda_actions()

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_cuda_action_trace_has_no_synchronization_or_transfers(self):
        library = os.environ.get("MCSIM_TEST_CUDA_LIBRARY")
        with Environment(2, device="cuda", library=library) as environment:
            environment.register_cuda_observations()
            actions = environment.register_cuda_actions()
            actions.zero_()
            environment.decode_cuda_actions_async()
            environment.wait_cuda_actions()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profiler:
                with torch.profiler.record_function("mcsim_async_actions"):
                    environment.decode_cuda_actions_async()
            names = []
            for event in profiler.events():
                parent = event.cpu_parent
                while parent is not None:
                    if parent.name == "mcsim_async_actions":
                        names.append(event.name)
                        break
                    parent = parent.cpu_parent
            self.assertIn("cudaLaunchKernel", names, "Profiler must capture native CUDA runtime calls")
            self.assertFalse(any(name.startswith(("cudaDeviceSynchronize", "cudaStreamSynchronize",
                                                  "cudaEventSynchronize", "cudaMemcpy")) for name in names), names)
            environment.wait_cuda_actions()


class CUDAResidentChecks(unittest.TestCase):
    def test_cpu_environment_rejects_resident_mode(self):
        with Environment(1, library=os.environ.get("MCSIM_TEST_CPU_LIBRARY")) as environment:
            with self.assertRaisesRegex(ValueError, "requires a CUDA environment"):
                environment.register_cuda_resident()
            with self.assertRaisesRegex(RuntimeError, "Register CUDA resident buffers"):
                environment.step_cuda_resident_async()

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_invalid_resident_action_and_reset_do_not_mutate_worlds(self):
        with Environment(3, device="cuda", library=os.environ["MCSIM_TEST_CUDA_LIBRARY"]) as environment:
            environment.register_cuda_observations()
            actions = environment.register_cuda_actions()
            outputs = environment.register_cuda_resident()
            initial = environment.observe_cuda_async().clone()
            actions.zero_()
            actions[:, :, 4] = 4
            actions[:, :, 5] = 2
            actions[1, 0, 0] = 999
            environment.step_cuda_resident_async()
            with self.assertRaisesRegex(ValueError, "Invalid CUDA action"):
                environment.wait_cuda_actions()
            environment.clear_cuda_action_status()
            torch.testing.assert_close(environment.observe_cuda_async(), initial)
            outputs["reset_mask"].fill_(2)
            environment.reset_cuda_resident_async()
            with self.assertRaisesRegex(ValueError, "Invalid CUDA reset mask"):
                environment.wait_cuda_actions()
            environment.clear_cuda_action_status()
            torch.testing.assert_close(environment.observe_cuda_async(), initial)

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_native_resident_episode_counters_round_trip(self):
        with Environment(3, device="cuda", library=os.environ["MCSIM_TEST_CUDA_LIBRARY"]) as environment:
            environment.register_cuda_observations()
            environment.register_cuda_actions()
            buffers = environment.register_cuda_resident()
            read = environment.native.mcsim_cuda_reset_episodes
            write = environment.native.mcsim_set_cuda_reset_episodes
            for function in (read, write):
                function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
                function.restype = ctypes.c_int
            before = np.empty(3, dtype=np.uint64)
            self.assertEqual(read(environment._handle, before.ctypes.data, 3), 0)
            buffers["reset_mask"].copy_(torch.tensor([1, 0, 1], device="cuda", dtype=torch.uint8))
            environment.reset_cuda_resident_async()
            after = np.empty_like(before)
            self.assertEqual(read(environment._handle, after.ctypes.data, 3), 0)
            np.testing.assert_array_equal(after, before + np.array([1, 0, 1], dtype=np.uint64))
            self.assertEqual(write(environment._handle, before.ctypes.data, 3), 0)
            self.assertEqual(read(environment._handle, after.ctypes.data, 3), 0)
            np.testing.assert_array_equal(after, before)
            self.assertEqual(read(environment._handle, after.ctypes.data, 2), -1)
            expected = torch.tensor([0, -9223372036854775808, -1], dtype=torch.int64)
            environment.set_cuda_reset_episodes(expected)
            torch.testing.assert_close(environment.cuda_reset_episodes(), expected)
            with self.assertRaisesRegex(ValueError, "one entry per world"):
                environment.set_cuda_reset_episodes(torch.zeros(2, dtype=torch.int64))

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_resident_step_matches_cpu_and_preserves_final_observations(self):
        cpu_library = os.environ.get("MCSIM_TEST_CPU_LIBRARY")
        cuda_library = os.environ.get("MCSIM_TEST_CUDA_LIBRARY")
        with Environment(3, device="cpu", library=cpu_library, max_ticks=3, seed=7) as reference, \
                Environment(3, device="cuda", library=cuda_library, max_ticks=3, seed=7) as environment:
            initial = reference.observe()
            environment.register_cuda_observations()
            actions = environment.register_cuda_actions()
            buffers = environment.register_cuda_resident()
            producer = torch.cuda.Stream()
            np.testing.assert_allclose(environment.observe_cuda_async().cpu().numpy(), initial, rtol=0, atol=1e-6)
            with self.assertRaisesRegex(RuntimeError, "Host state is unavailable"):
                environment.observe()
            with self.assertRaisesRegex(RuntimeError, "Host state is unavailable"):
                environment.step(np.zeros((3, 2, 6), dtype=np.int32))
            host_actions = np.zeros((3, 2, 6), dtype=np.int32)
            host_actions[:, :, 4] = 4
            host_actions[:, :, 5] = 2
            host_actions[::2, 0, 0] = 1
            for _ in range(3):
                expected, rewards, terminated, truncated = reference.step(host_actions)
                losses = reference.losses(np.ones(3, dtype=bool))
                with torch.cuda.stream(producer):
                    actions.copy_(torch.as_tensor(host_actions, device=actions.device))
                    observed, outputs = environment.step_cuda_resident_async()
                    captured = {name: value.clone() for name, value in outputs.items() if name != "reset_mask"}
                    final = observed.clone()
                producer.synchronize()
                environment.wait_cuda_actions()
                np.testing.assert_allclose(final.cpu().numpy(), expected, rtol=0, atol=1e-6)
                np.testing.assert_allclose(captured["rewards"].cpu().numpy(), rewards, rtol=0, atol=1e-6)
                np.testing.assert_array_equal(captured["terminated"].cpu().numpy(), terminated)
                np.testing.assert_array_equal(captured["truncated"].cpu().numpy(), truncated)
                np.testing.assert_array_equal(captured["losses"].cpu().numpy(), losses)
                for world in range(3):
                    for player in range(2):
                        selected = np.zeros((3, 2), dtype=bool)
                        selected[world, player] = True
                        totals = reference.step_reward_totals(selected)
                        np.testing.assert_allclose(captured["contributions"][world, player].cpu().numpy(),
                                                   [totals[name] for name in REWARD_NAMES], rtol=0, atol=1e-6)
            self.assertTrue(truncated.all())
            mask = np.array([True, False, True])
            expected_reset = reference.reset(mask)
            with torch.cuda.stream(producer):
                buffers["reset_mask"].copy_(torch.as_tensor(mask, device=actions.device, dtype=torch.uint8))
                reset = environment.reset_cuda_resident_async().clone()
            producer.synchronize()
            environment.wait_cuda_actions()
            np.testing.assert_allclose(reset.cpu().numpy(), expected_reset, rtol=0, atol=1e-6)
            np.testing.assert_array_equal(buffers["truncated"].cpu().numpy(), truncated)
            with torch.cuda.stream(producer):
                environment.step_cuda_resident_async()
            with self.assertRaisesRegex(ValueError, "finished world"):
                environment.wait_cuda_actions()
            environment.clear_cuda_action_status()
            mask = np.array([False, True, False])
            expected_reset = reference.reset(mask)
            with torch.cuda.stream(producer):
                buffers["reset_mask"].copy_(torch.as_tensor(mask, device=actions.device, dtype=torch.uint8))
                reset = environment.reset_cuda_resident_async().clone()
            producer.synchronize()
            environment.wait_cuda_actions()
            np.testing.assert_allclose(reset.cpu().numpy(), expected_reset, rtol=0, atol=1e-6)
            buffers["rewards"] = torch.empty_like(buffers["rewards"])
            with self.assertRaisesRegex(RuntimeError, "resident buffer changed"):
                environment.step_cuda_resident_async()

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_resident_configured_projection_and_loss_draw_match_cpu(self):
        definition = json.loads((ROOT / "simulator/tests/reset_environment.json").read_text())
        definition["blocks"] = [{"from": [0, 0, -2], "to": [0, 0, -2], "block": "minecraft:stone"}]
        definition["policy"] = {"format": "minecraft_sim_policy", "version": 1, "minecraft": "26.3",
                                "ticks_per_action": 1,
                                "observations": [{"feature": "self.weapon_slot", "encoding": "one_hot",
                                                  "categories": [1, 0]},
                                                 {"feature": "opponent.health_fraction"}],
                                "actions": [{"name": "yaw", "values": [-5, 0, 5]},
                                            {"name": "jump", "values": [False, True]}]}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fall.json"
            path.write_text(json.dumps(definition), encoding="utf-8")
            with Environment(2, "cpu", path, os.environ.get("MCSIM_TEST_CPU_LIBRARY"), max_ticks=100) as reference, \
                    Environment(2, "cuda", path, os.environ["MCSIM_TEST_CUDA_LIBRARY"], max_ticks=100) as device:
                players = [reference.episode_scenario(index)["players"] for index in range(2)]
                players[0][0]["position"] = [-1.5, 1, -1.5]
                players[0][1]["position"] = [1.5, 1, 1.5]
                players[1][1]["position"] = [1.5, 1, 1.5]
                mask = np.ones(2, dtype=bool)
                reference.reset_players(players, mask)
                device.reset_players(players, mask)
                device.register_cuda_observations()
                actions = device.register_cuda_actions()
                device.register_cuda_resident()
                host_actions = np.zeros((2, 2, 2), dtype=np.int32)
                host_actions[:, :, 0] = 1
                actions.copy_(torch.as_tensor(host_actions, device="cuda"))
                for _ in range(30):
                    expected, rewards, terminated, truncated = reference.step(host_actions)
                    actual, outputs = device.step_cuda_resident_async()
                    device.wait_cuda_actions()
                    np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=0, atol=1e-6)
                    np.testing.assert_allclose(outputs["rewards"].cpu().numpy(), rewards, rtol=0, atol=1e-6)
                    np.testing.assert_array_equal(outputs["terminated"].cpu().numpy(), terminated)
                    np.testing.assert_array_equal(outputs["truncated"].cpu().numpy(), truncated)
                    if terminated.all():
                        break
                self.assertTrue(terminated.all())
                np.testing.assert_array_equal(reference.losses(mask), [[True, True], [False, True]])
                np.testing.assert_array_equal(outputs["losses"].cpu().numpy(), [[1, 1], [0, 1]])

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_invalid_resident_registration_preserves_host_path(self):
        library = os.environ.get("MCSIM_TEST_CUDA_LIBRARY")
        with Environment(2, device="cuda", library=library) as environment:
            environment.register_cuda_observations()
            environment.register_cuda_actions()
            function = environment.native.mcsim_register_cuda_step_outputs
            function.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_size_t]
            function.restype = ctypes.c_int
            host = np.zeros(2, dtype=np.float32)
            self.assertEqual(function(environment._handle, host.ctypes.data, host.ctypes.data,
                                      host.ctypes.data, host.ctypes.data, host.ctypes.data, 2), -1)
            self.assertEqual(environment.observe().shape, (2, 2, 22))

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_resident_step_and_reset_trace_has_no_sync_or_transfers(self):
        library = os.environ.get("MCSIM_TEST_CUDA_LIBRARY")
        with Environment(2, device="cuda", library=library, max_ticks=1) as environment:
            environment.register_cuda_observations()
            actions = environment.register_cuda_actions()
            buffers = environment.register_cuda_resident()
            actions.zero_()
            actions[:, :, 4] = 4
            actions[:, :, 5] = 2
            buffers["reset_mask"].fill_(1)
            environment.step_cuda_resident_async()
            environment.wait_cuda_actions()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profiler:
                with torch.profiler.record_function("mcsim_resident_step_reset"):
                    environment.reset_cuda_resident_async()
                    environment.step_cuda_resident_async()
            names = []
            for event in profiler.events():
                parent = event.cpu_parent
                while parent is not None:
                    if parent.name == "mcsim_resident_step_reset":
                        names.append(event.name)
                        break
                    parent = parent.cpu_parent
            self.assertGreaterEqual(names.count("cudaLaunchKernel"), 8,
                                    "Profiler must capture native CUDA step and reset kernels")
            self.assertFalse(any(name.startswith(("cudaDeviceSynchronize", "cudaStreamSynchronize",
                                                  "cudaEventSynchronize", "cudaMemcpy")) for name in names), names)
            environment.wait_cuda_actions()

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_resident_collector_matches_host_across_timeout_and_rollouts(self):
        from copy import deepcopy
        from backend.fast_rl_model import ActorCritic
        from simulator.training._test_plans import example_plan
        from simulator.training.plans import resolve_plan
        from simulator.training.policies import fixed_indices
        from simulator.training.ppo import optimize
        from simulator.training.research import OpponentPool, ResidentCollector, SampledWorlds, collect

        plan = resolve_plan(example_plan())
        stage = deepcopy(plan["stages"][0])
        stage["scenario"].pop("spawns", None)
        stage["scenario"].pop("equipment", None)
        count = 4
        cpu = SampledWorlds(stage["scenario"], plan["policy"], count, "cpu",
                            os.environ.get("MCSIM_TEST_CPU_LIBRARY"), 3, 42)
        gpu = SampledWorlds(stage["scenario"], plan["policy"], count, "cuda",
                            os.environ["MCSIM_TEST_CUDA_LIBRARY"], 3, 42)
        try:
            cpu.configure(stage)
            gpu.configure(stage)
            torch.manual_seed(7)
            sizes = dict(zip(cpu.action_names, plan["policy"]["action_sizes"]))
            host_model = ActorCritic(cpu.observation_dim, 16, action_sizes=sizes)
            device_model = ActorCritic(cpu.observation_dim, 16, action_sizes=sizes).cuda()
            device_model.load_state_dict(host_model.state_dict())
            host_pool = OpponentPool(plan["opponents"], plan["policy"], count, "cpu", 42, learner=host_model)
            device_pool = OpponentPool(plan["opponents"], plan["policy"], count, "cuda", 42, learner=device_model)
            neutral = host_pool.controllers["stationary"]._neutral
            controls = {"fixed": {head["name"]: head["values"][int(index)]
                                  for head, index in zip(plan["policy"]["actions"], neutral)}}
            host_pool.configure(stage["opponents"], controls)
            device_pool.configure(stage["opponents"], controls)
            fixed = fixed_indices(plan["policy"], controls)
            host_observations = cpu.observe()
            resident = ResidentCollector(gpu, device_pool, 42)
            host_pool.assignments[:] = resident.pool.assignments.cpu().numpy()
            host_pool.learner_slots[:] = resident.pool.learner_slots.cpu().numpy()
            for steps in (2, 1):
                expected, host_observations, endings, timeouts = collect(
                    host_model, cpu, host_pool, host_observations, steps, torch.device("cpu"), fixed)
                actual, _, device_endings, device_timeouts = resident.collect(device_model, steps, fixed)
                self.assertEqual((endings, timeouts), (device_endings, device_timeouts))
                for name in ("observations", "actions", "log_probs", "values", "next_values", "rewards",
                             "terminated", "truncated", "advantages", "returns"):
                    torch.testing.assert_close(actual[name].cpu(), expected[name], rtol=0, atol=2e-6)
                for name in REWARD_NAMES:
                    self.assertAlmostEqual(actual["reward_types"][name], expected["reward_types"][name], places=5)
                self.assertEqual(actual["episode_metrics"], expected["episode_metrics"])
            self.assertEqual(device_timeouts, count)
            host_optimizer = torch.optim.Adam(host_model.parameters(), lr=3e-4, eps=1e-5)
            device_optimizer = torch.optim.Adam(device_model.parameters(), lr=3e-4, eps=1e-5)
            host_stats = optimize(host_model, host_optimizer, expected, 1, count)
            device_stats = optimize(device_model, device_optimizer, actual, 1, count)
            for name in host_stats:
                self.assertAlmostEqual(host_stats[name], device_stats[name], places=5)
            for host_parameter, device_parameter in zip(host_model.parameters(), device_model.parameters(), strict=True):
                torch.testing.assert_close(device_parameter.cpu(), host_parameter, rtol=2e-5, atol=2e-6)
        finally:
            gpu.close()
            cpu.close()

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_resident_training_checkpoint_replay_stage_and_resume(self):
        import contextlib
        from copy import deepcopy
        import io
        from simulator.training._test_plans import example_plan
        from simulator.training.plans import resolve_plan
        from simulator.training.train import main

        plan = resolve_plan(example_plan())
        suite = plan["evaluations"]["held_out_combat"]
        suite.update(every_updates=1, episodes=2, worlds=2)
        plan["opponents"]["self"] = {"type": "self_play", "deterministic": False}
        plan["stages"] = [plan["stages"][0], deepcopy(plan["stages"][0])]
        for index, stage in enumerate(plan["stages"]):
            stage["name"] = f"resident-{index}"
            stage["advance"] = {"updates": 1 if index == 0 else 3}
            stage["opponents"] = [{"id": "stationary", "weight": 1}, {"id": "self", "weight": 1}]
            stage["scenario"].pop("spawns", None)
            stage["scenario"].pop("equipment", None)
            stage["evaluations"] = {}
        plan["scenario"] = deepcopy(plan["stages"][0]["scenario"])
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            path = root / "plan.json"
            path.write_text(json.dumps(plan), encoding="utf-8")
            output = root / "run"
            library = os.environ["MCSIM_TEST_CUDA_LIBRARY"]
            flags = ["--device", "cuda", "--rollout-backend", "cuda-resident", "--library", library,
                     "--envs", "4", "--rollout-steps", "3", "--max-ticks", "3", "--hidden-dim", "16",
                     "--epochs", "1", "--minibatch-size", "12", "--updates", "1",
                     "--replay-every-updates", "1"]
            self.assertEqual(main(["--plan", str(path), "--output", str(output), *flags]), 0)
            checkpoint = torch.load(output / "latest.pt", weights_only=True)
            self.assertEqual(checkpoint["research"]["rollout_backend"], "cuda-resident")
            self.assertEqual(checkpoint["research"]["stage_index"], 1)
            self.assertIn("resident_rng", checkpoint["research"])
            self.assertIn("resident_episodes", checkpoint["research"])
            self.assertEqual(len(list((output / "replays").glob("*.jsonl"))), 1)
            first_metrics = json.loads((output / "metrics.jsonl").read_text().splitlines()[0])
            self.assertEqual(first_metrics["evaluations"]["held_out_combat"]["episodes"], 2)
            with self.assertRaisesRegex(ValueError, "rollout backend differs"):
                main(["--resume", str(output / "latest.pt"), "--device", "cuda", "--library", library,
                      "--updates", "1"])
            self.assertEqual(main(["--resume", str(output / "latest.pt"), "--device", "cuda",
                                   "--rollout-backend", "cuda-resident", "--library", library,
                                   "--updates", "1", "--replay-every-updates", "1"]), 0)
            resumed = torch.load(output / "latest.pt", weights_only=True)
            self.assertEqual(resumed["update"], 2)
            self.assertEqual(resumed["research"]["rollout_backend"], "cuda-resident")
            torch.testing.assert_close(resumed["research"]["resident_episodes"],
                                       checkpoint["research"]["resident_episodes"] + 2)
            self.assertEqual(len(list((output / "replays").glob("*.jsonl"))), 2)

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_resident_collector_decisions_have_no_host_transfers_or_waits(self):
        from backend.fast_rl_model import ActorCritic
        from simulator.training._test_plans import example_plan
        from simulator.training.plans import resolve_plan
        from simulator.training.research import OpponentPool, ResidentCollector, SampledWorlds

        plan = resolve_plan(example_plan())
        stage = plan["stages"][0]
        stage["scenario"].pop("spawns", None)
        stage["scenario"].pop("equipment", None)
        definitions = {"target": {"type": "scripted", "controller": "stationary_target"},
                       "self": {"type": "self_play"}}
        stage["opponents"] = [{"id": "target", "weight": 1}, {"id": "self", "weight": 1}]
        worlds = SampledWorlds(stage["scenario"], plan["policy"], 4, "cuda",
                               os.environ["MCSIM_TEST_CUDA_LIBRARY"], 2, 42)
        try:
            worlds.configure(stage)
            model = ActorCritic(worlds.observation_dim, 16,
                                action_sizes=dict(zip(worlds.action_names, plan["policy"]["action_sizes"]))).cuda()
            pool = OpponentPool(definitions, plan["policy"], 4, "cuda", 42, learner=model)
            pool.configure(stage["opponents"], {"fixed": {}})
            resident = ResidentCollector(worlds, pool, 42)
            resident.collect(model, 2, {})
            original_act, original_reset = resident.pool.act, resident.pool.reset
            scopes = []

            def start_scope(*args):
                scope = torch.profiler.record_function("mcsim_resident_decision")
                scope.__enter__()
                scopes.append(scope)
                return original_act(*args)

            def end_scope(*args):
                result = original_reset(*args)
                scopes.pop().__exit__(None, None, None)
                return result

            with mock.patch.object(resident.pool, "act", side_effect=start_scope), \
                    mock.patch.object(resident.pool, "reset", side_effect=end_scope), \
                    torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as profiler:
                resident.collect(model, 2, {})
            names = []
            operations = []
            for event in profiler.events():
                parent = event.cpu_parent
                chain = []
                while parent is not None:
                    if parent.name == "mcsim_resident_decision":
                        names.append(event.name)
                        if event.name.startswith(("cudaDeviceSynchronize", "cudaStreamSynchronize",
                                                  "cudaEventSynchronize", "aten::item",
                                                  "aten::_local_scalar_dense")):
                            operations.append((event.name, chain))
                        if event.name.startswith("cudaMemcpy"):
                            kernels = event.cpu_parent.kernels
                            if not kernels or any(not kernel.name.startswith("Memcpy DtoD") for kernel in kernels):
                                operations.append((event.name, chain, kernels))
                        break
                    chain.append(parent.name)
                    parent = parent.cpu_parent
            self.assertIn("cudaLaunchKernel", names, "Profiler must capture native CUDA calls")
            self.assertFalse(operations, operations)
        finally:
            worlds.close()

    @unittest.skipUnless(torch.cuda.is_available() and os.environ.get("MCSIM_TEST_CUDA_LIBRARY"),
                         "CUDA and MCSIM_TEST_CUDA_LIBRARY are required")
    def test_resident_collector_rejects_nonfinite_policy_before_optimization(self):
        from backend.fast_rl_model import ActorCritic
        from simulator.training._test_plans import example_plan
        from simulator.training.plans import resolve_plan
        from simulator.training.research import OpponentPool, ResidentCollector, SampledWorlds

        plan = resolve_plan(example_plan())
        stage = plan["stages"][0]
        stage["scenario"].pop("spawns", None)
        stage["scenario"].pop("equipment", None)
        worlds = SampledWorlds(stage["scenario"], plan["policy"], 2, "cuda",
                               os.environ["MCSIM_TEST_CUDA_LIBRARY"], 2, 42)
        try:
            worlds.configure(stage)
            model = ActorCritic(worlds.observation_dim, 16,
                                action_sizes=dict(zip(worlds.action_names, plan["policy"]["action_sizes"]))).cuda()
            pool = OpponentPool(plan["opponents"], plan["policy"], 2, "cuda", 42, learner=model)
            pool.configure(stage["opponents"], stage["controls"])
            resident = ResidentCollector(worlds, pool, 42)
            with torch.no_grad():
                next(model.parameters()).fill_(float("nan"))
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                resident.collect(model, 1, {})
        finally:
            worlds.close()


if __name__ == "__main__":
    unittest.main()
