import json
import unittest
from unittest.mock import Mock

from tools.rtsp_scene.launch import PR_SET_THP_DISABLE, build_command, disable_process_thp, select_devices


class LaunchTest(unittest.TestCase):
    def test_full_prefill_requires_the_context_token_budget(self):
        original = build_command("/model", 8003, 8, True, 2, max_batched_tokens=8192, kv_block_size=128)
        trial = build_command(
            "/model", 8003, 8, True, 2, max_batched_tokens=8192, kv_block_size=128, chunked_prefill=False
        )
        index = trial.index("--no-enable-chunked-prefill")
        self.assertEqual(trial[:index] + trial[index + 1 :], original)
        with self.assertRaisesRegex(ValueError, "token budget"):
            build_command("/model", 8003, 8, True, 2, chunked_prefill=False)
        with self.assertRaisesRegex(ValueError, "explicit"):
            build_command("/model", 8003, 8, True, 2, max_batched_tokens=8192, chunked_prefill=False)

    def test_async_scheduler_requires_explicit_opt_in(self):
        original = build_command("/model", 8003, 8, True, 2, decode_graph_sizes=[1, 8])
        trial = build_command("/model", 8003, 8, True, 2, decode_graph_sizes=[1, 8], async_scheduling=True)
        self.assertIn("--no-async-scheduling", original)
        self.assertNotIn("--no-async-scheduling", trial)
        index = original.index("--no-async-scheduling")
        self.assertEqual(trial[index], "--async-scheduling")
        self.assertEqual(trial[:index] + trial[index + 1 :], original[:index] + original[index + 1 :])

    def test_kv_geometry_is_explicit_and_preserves_other_settings(self):
        original = build_command("/model", 8003, 8, True, 2, decode_graph_sizes=[1, 8])
        for size in (64, 128):
            trial = build_command("/model", 8003, 8, True, 2, decode_graph_sizes=[1, 8], kv_block_size=size)
            index = trial.index("--block-size")
            self.assertEqual(trial[index + 1], str(size))
            self.assertEqual(trial[:index] + trial[index + 2 :], original)
        with self.assertRaisesRegex(ValueError, "KV block"):
            build_command("/model", 8003, 8, True, 2, kv_block_size=32)

    def test_vision_data_parallelism_preserves_text_and_graph_configuration(self):
        baseline = build_command("/models/candidate", 8003, 8, True, 2, decode_graph_sizes=[1, 8], dense=True)
        candidate = build_command(
            "/models/candidate", 8003, 8, True, 2, decode_graph_sizes=[1, 8], dense=True, vision_parallel_mode="data"
        )
        self.assertNotIn("--mm-encoder-tp-mode", baseline)
        index = candidate.index("--mm-encoder-tp-mode")
        self.assertEqual(candidate[index + 1], "data")
        self.assertEqual(candidate[:index] + candidate[index + 2 :], baseline)
        with self.assertRaisesRegex(ValueError, "vision encoder"):
            build_command("/models/candidate", 8003, 8, True, 2, vision_parallel_mode="invalid")

    def test_native_graph_capture_has_a_bounded_size_budget(self):
        command = build_command("/models/candidate", 8003, 8, True, 2, decode_graph_sizes=[4, 1])
        self.assertNotIn("--enforce-eager", command)
        config = json.loads(command[command.index("--compilation-config") + 1])
        self.assertEqual(config["mode"], 0)
        self.assertEqual(config["cudagraph_mode"], "FULL_DECODE_ONLY")
        self.assertEqual(config["cudagraph_capture_sizes"], [1, 4])
        for sizes in ([1, 2, 4], [0], [9]):
            with self.assertRaises(ValueError):
                build_command("/models/candidate", 8003, 8, True, 2, decode_graph_sizes=sizes)

    def test_tensor_parallel_service_uses_exactly_the_selected_devices(self):
        command = build_command("/models/candidate", 8003, 8, True, tensor_parallel_size=2, max_batched_tokens=2048)
        self.assertEqual(command[command.index("--tensor-parallel-size") + 1], "2")
        self.assertEqual(command[command.index("--max-num-batched-tokens") + 1], "2048")
        self.assertEqual(select_devices("0,1", 0, 2), [0, 1])
        self.assertEqual(select_devices(None, 1, 1), [1])
        for devices in ("0,0", "0,-1", "0", ""):
            with self.assertRaises(ValueError):
                select_devices(devices, 0, 2)
        profiled = build_command("/models/candidate", 8003, 8, True, 2, 2048, "/private/profile")
        profiler = json.loads(profiled[profiled.index("--profiler-config") + 1])
        self.assertTrue(profiler["ignore_frontend"])
        self.assertEqual(profiler["torch_profiler_dir"], "/private/profile")

    def test_baseline_and_candidate_have_explicit_eager_execution(self):
        baseline = build_command("/models/original", 8003, 2, False)
        candidate = build_command("/models/candidate", 8004, 4, True)
        self.assertIn("--enforce-eager", baseline)
        self.assertNotIn("--quantization", baseline)
        self.assertEqual(candidate[candidate.index("--quantization") + 1], "ascend")
        self.assertEqual(candidate[candidate.index("--safetensors-load-strategy") + 1], "eager")
        self.assertEqual(candidate[candidate.index("--max-num-seqs") + 1], "4")
        self.assertEqual(candidate[candidate.index("--host") + 1], "127.0.0.1")
        with self.assertRaises(ValueError):
            build_command("/models/original", 65536, 2, False)

    def test_thp_setting_is_process_scoped_and_failure_stops_launch(self):
        libc = Mock()
        libc.prctl.return_value = 0
        disable_process_thp(libc)
        libc.prctl.assert_called_once_with(PR_SET_THP_DISABLE, 1, 0, 0, 0)
        libc.prctl.return_value = -1
        with self.assertRaisesRegex(OSError, "loader"):
            disable_process_thp(libc)
