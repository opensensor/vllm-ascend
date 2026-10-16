import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from tools.rtsp_scene.multi_stream import choose_endpoint, has_budget
from tools.rtsp_scene.quantize import runtime_description, selected_weight, symmetric_int8


class QuantizationTest(unittest.TestCase):
    def test_conversion_keeps_source_and_vision_weights_and_loadable_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output = Path(temporary) / "source", Path(temporary) / "output"
            source.mkdir()
            text_name = "model.language_model.layers.0.self_attn.q_proj.weight"
            vision_name = "model.visual.blocks.0.attn.qkv.weight"
            weights = {
                text_name: torch.tensor([[-2, 0, 2]], dtype=torch.float16),
                vision_name: torch.tensor([[3, 4, 5]], dtype=torch.float16),
                "lm_head.weight": torch.tensor([[1, 2, 3]], dtype=torch.float16),
            }
            shard = source / "model-00001-of-00001.safetensors"
            save_file(weights, shard)
            (source / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))
            (source / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": dict.fromkeys(weights, shard.name)})
            )
            digest = hashlib.sha256(shard.read_bytes()).hexdigest()
            command = [
                sys.executable,
                "-m",
                "tools.rtsp_scene.quantize",
                "--source",
                str(source),
                "--output",
                str(output),
            ]
            subprocess.run(command, check=True, capture_output=True)
            converted = load_file(output / shard.name)
            self.assertEqual(converted[text_name].dtype, torch.int8)
            torch.testing.assert_close(converted[vision_name], weights[vision_name])
            torch.testing.assert_close(converted["lm_head.weight"], weights["lm_head.weight"])
            configuration = json.loads((output / "config.json").read_text())["quantization_config"]
            self.assertEqual(configuration["visual.blocks.0.attn.qkv.weight"], "FLOAT")
            self.assertEqual(configuration["language_model.model.layers.0.self_attn.q_proj.weight"], "W8A8_DYNAMIC")
            index = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
            self.assertEqual(set(index), set(converted))
            self.assertEqual(hashlib.sha256(shard.read_bytes()).hexdigest(), digest)
            rerun = subprocess.run(command, capture_output=True)
            self.assertNotEqual(rerun.returncode, 0)
            self.assertEqual(hashlib.sha256(shard.read_bytes()).hexdigest(), digest)

    def test_only_text_projections_are_selected_by_default(self):
        self.assertTrue(selected_weight("model.language_model.layers.12.mlp.down_proj.weight"))
        self.assertTrue(selected_weight("model.language_model.layers.0.self_attn.q_proj.weight"))
        self.assertFalse(selected_weight("model.visual.blocks.0.attn.qkv.weight"))
        self.assertFalse(selected_weight("model.language_model.layers.0.self_attn.q_norm.weight"))
        self.assertFalse(selected_weight("model.language_model.embed_tokens.weight"))
        self.assertFalse(selected_weight("lm_head.weight"))
        self.assertTrue(selected_weight("lm_head.weight", quantize_head=True))

    def test_quantization_preserves_zero_rows_and_per_channel_scale(self):
        weight = torch.tensor([[0, 0, 0], [-1, 0, 1], [-10, 0, 10]], dtype=torch.float16)
        quantized, scale, offset = symmetric_int8(weight)
        self.assertEqual(quantized.dtype, torch.int8)
        self.assertEqual(scale.shape, (3, 1))
        self.assertTrue(torch.all(scale > 0))
        self.assertTrue(torch.all(offset == 0))
        torch.testing.assert_close(quantized.float() * scale, weight.float(), atol=1e-6, rtol=1e-6)
        with self.assertRaisesRegex(ValueError, "finite weight matrix"):
            symmetric_int8(torch.tensor([[float("nan")]]))

    def test_vision_and_text_namespace_aliases_keep_quantization_types(self):
        original = {
            "model.visual.blocks.0.attn.qkv.weight": "FLOAT",
            "model.language_model.layers.0.self_attn.q_proj.weight": "W8A8_DYNAMIC",
            "lm_head.weight": "FLOAT",
        }
        mapped = runtime_description(original)
        self.assertEqual(mapped["visual.blocks.0.attn.qkv.weight"], "FLOAT")
        self.assertEqual(mapped["language_model.model.layers.0.self_attn.q_proj.weight"], "W8A8_DYNAMIC")
        self.assertEqual(mapped["language_model.lm_head.weight"], "FLOAT")
        self.assertEqual(runtime_description(mapped), mapped)

    def test_capture_budget_rejects_new_work_without_blocking_capture(self):
        pending = type("Pending", (), {"done": lambda self: False})()
        completed = type("Completed", (), {"done": lambda self: True})()
        self.assertFalse(has_budget([pending, completed], 1))
        self.assertTrue(has_budget([pending, completed], 2))
        self.assertEqual(choose_endpoint(3, ["baseline", "candidate"]), "candidate")
        with self.assertRaisesRegex(ValueError, "endpoint"):
            choose_endpoint(0, [])
