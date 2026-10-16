"""Opt-in Qwen3-VL worker with a load-time numerical gate and memory accounting."""

import json
from functools import partial
from pathlib import Path

import torch
from transformers import AutoImageProcessor
from vllm.distributed import tensor_model_parallel_all_reduce

from tools.rtsp_scene.dense_runtime import NZMLP, DecodeLinearMethod, DenseProjection, KernelBackend
from tools.rtsp_scene.dense_setup import RUNTIME_CONFIG, checked_qkv_build
from tools.rtsp_scene.input_quant import MAX_ROWS as INPUT_QUANT_MAX_ROWS
from tools.rtsp_scene.input_quant import InputQuantizer
from tools.rtsp_scene.vision_norm import VISION_WIDTH, VisionZeroAddNorm
from tools.rtsp_scene.vision_norm import eligible as eligible_vision_norm
from tools.rtsp_scene.worker_image_prepare import WorkerImagePreparer
from vllm_ascend._310p.worker_310p import NPUWorker310

GATE_ROWS = (1, 8)
GATE_MAGNITUDES = (0.1, 1.0, 3.0)
VISION_GATE_MAGNITUDES = (0.0001, 0.1, 1.0, 3.0)
VISION_GATE_ROWS = (1760, 2640, 3520, 3688)
VISION_SMALL_GATE_ROWS = 8


def check_pair(name, rows, magnitude, expected, actual):
    expected, actual = expected.float(), actual.float()
    finite = bool(torch.isfinite(actual).all().cpu())
    relative = (
        float((torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)).cpu())
        if finite
        else None
    )
    return {
        "module": name,
        "rows": rows,
        "magnitude": magnitude,
        "finite": finite,
        "relative_l2": relative,
        "max_abs": float((actual - expected).abs().max().cpu()) if finite else None,
        "bitwise_equal": bool(torch.equal(actual, expected)),
        "passed": finite and bool(torch.allclose(actual, expected, rtol=0.001, atol=0.0001)),
    }


def input_cases(device, inner, max_rows=GATE_ROWS[-1]):
    for rows in (1, max_rows):
        for magnitude in GATE_MAGNITUDES:
            generator = torch.Generator().manual_seed(20261012 + rows)
            yield rows, magnitude, (torch.randn(rows, inner, generator=generator).half() * magnitude).to(device)


class SceneDenseWorker(NPUWorker310):
    def check_image_transport(self, processor, overlay, rgb, shapes, output):
        sizes = [width * height * 3 for width, height, _, _ in shapes]
        # This transfer is limited to the one-shot real-request correctness gate.
        frames = rgb.cpu().split(sizes)
        images = [
            frame.reshape(shape[1], shape[0], 3).permute(2, 0, 1) for frame, shape in zip(frames, shapes, strict=True)
        ]
        expected = processor(images=images, return_tensors="pt")["pixel_values"].half()
        actual = output.cpu()
        passed = torch.equal(actual, expected)
        (overlay / f"image-transport-gate-rank{self.rank}.json").write_text(
            json.dumps(
                {
                    "rank": self.rank,
                    "frames": len(images),
                    "geometries": shapes,
                    "shape": list(actual.shape),
                    "bitwise_equal": passed,
                    "max_abs_error": float((expected - actual).abs().max()),
                    "source": "First real serialized RGB byte request",
                    "passed": passed,
                },
                indent=2,
            )
            + "\n"
        )
        if not passed:
            raise ValueError("Real request image transport changed the model pixels")

    def compile_or_warm_up_model(self):
        timings = super().compile_or_warm_up_model()
        # Startup invokes the encoder on dummy inputs. Re-arm the one-shot
        # gate after warmup so the first request checks real activations.
        if hasattr(self, "vision_actual_gate_records"):
            overlay = Path(self.model_config.model)
            previous = overlay / f"vision-gate-rank{self.rank}-actual.json"
            if previous.exists():
                previous.rename(overlay / f"vision-gate-rank{self.rank}-warmup.json")
            self.vision_actual_gate_records = []
            norms = [
                (name, module)
                for name, module in self.model_runner.model.named_modules()
                if isinstance(module, VisionZeroAddNorm)
            ]
            for name, candidate in norms:
                candidate.verify = partial(self.check_vision_activation, name, candidate.original, overlay, len(norms))
        return timings

    def check_vision_activation(self, name, original, overlay, expected_checks, x, output):
        record = check_pair(name, x.shape[0], None, original(x), output)
        record["shape"] = list(x.shape)
        record["source"] = "First eligible model activation after loaded-weight checks"
        self.vision_actual_gate_records.append(record)
        passed = all(item["passed"] and item["bitwise_equal"] for item in self.vision_actual_gate_records)
        (overlay / f"vision-gate-rank{self.rank}-actual.json").write_text(
            json.dumps(
                {
                    "records": self.vision_actual_gate_records,
                    "passed": passed,
                    "complete": len(self.vision_actual_gate_records) == expected_checks,
                },
                indent=2,
            )
            + "\n"
        )
        return record["passed"] and record["bitwise_equal"]

    def install_vision_norms(self, model, modules, backend, overlay):
        norms = [
            (name, module)
            for name, module in modules
            if name.startswith("visual.blocks.") and name.endswith((".norm1", ".norm2"))
        ]
        if len(norms) != 2 * self.model_config.hf_config.vision_config.depth:
            raise ValueError("Could not find every Qwen3-VL vision block normalization")
        max_rows = self.scheduler_config.max_num_batched_tokens * model.visual.spatial_merge_size**2
        zero = torch.zeros(max_rows, 1, VISION_WIDTH, dtype=torch.float16, device=self.device)
        self.vision_actual_gate_records = []
        records = []
        for index, (name, original) in enumerate(norms):
            candidate = VisionZeroAddNorm(original, zero, backend.add_layer_norm)
            rows_to_check = (
                sorted({1, min(VISION_SMALL_GATE_ROWS, max_rows), max_rows, *VISION_GATE_ROWS})
                if index == 0
                else [min(VISION_SMALL_GATE_ROWS, max_rows)]
            )
            for rows in rows_to_check:
                if rows > max_rows:
                    continue
                for magnitude in VISION_GATE_MAGNITUDES:
                    generator = torch.Generator().manual_seed(20261013 + rows)
                    x = (torch.randn(rows, 1, VISION_WIDTH, generator=generator).half() * magnitude).to(self.device)
                    if not eligible_vision_norm(x, zero, original):
                        raise ValueError(f"Loaded vision normalization cannot use the gated path: {name}")
                    record = check_pair(name, rows, magnitude, original(x), candidate(x))
                    records.append(record)
                    if not record["passed"] or not record["bitwise_equal"]:
                        (overlay / f"vision-gate-rank{self.rank}.json").write_text(
                            json.dumps({"records": records, "passed": False}, indent=2) + "\n"
                        )
                        raise ValueError(f"Loaded vision normalization gate failed: {name}, rows={rows}")
            candidate.verify = partial(self.check_vision_activation, name, original, overlay, len(norms))
            parent_name, attribute = name.rsplit(".", 1)
            setattr(model.get_submodule(parent_name), attribute, candidate)
        (overlay / f"vision-gate-rank{self.rank}.json").write_text(
            json.dumps(
                {
                    "records": records,
                    "passed": True,
                    "max_rows": max_rows,
                    "zero_bytes": zero.numel() * zero.element_size(),
                },
                indent=2,
            )
            + "\n"
        )

    def load_model(self) -> None:
        super().load_model()
        overlay = Path(self.model_config.model)
        config = json.loads((overlay / RUNTIME_CONFIG).read_text())
        if self.parallel_config.tensor_parallel_size != 2 or self.model_config.enable_sleep_mode:
            raise ValueError("The scene dense worker requires TP2 with sleep mode disabled")
        backend = KernelBackend(Path(config["build"]))
        if backend.manifest != config["manifest"]:
            raise ValueError("Overlay and native build fingerprints differ")
        qkv_backend = backend
        if "qkv_projection_build" in config:
            if config["mode"] != "all":
                raise ValueError("A QKV-specific build requires dense mode all")
            qkv_config = config["qkv_projection_build"]
            if checked_qkv_build(Path(qkv_config["build"]), backend.manifest) != qkv_config:
                raise ValueError("Overlay and QKV build fingerprints differ")
            qkv_backend = KernelBackend(Path(qkv_config["build"]))
        # The graph wrapper delegates reads, but not attribute assignments.
        # Install per-model helpers on the module that embed_multimodal uses.
        model = self.model_runner.get_model()
        modules = list(model.named_modules())
        mlps = [
            (name, module)
            for name, module in modules
            if name.startswith("language_model.")
            and name.endswith(".mlp")
            and hasattr(module, "gate_up_proj")
            and hasattr(module, "down_proj")
        ]
        if len(mlps) != self.model_config.hf_config.text_config.num_hidden_layers:
            raise ValueError("Could not find every Qwen3-VL text FFN")
        before = torch.npu.memory_allocated()
        if "input_quant" in config:
            if backend.max_decode_rows != INPUT_QUANT_MAX_ROWS:
                raise ValueError("NZ input quantization requires an eight-row matrix build")
            backend.input_quant = InputQuantizer(Path(config["input_quant"]["build"]), backend, self.device)
            if backend.input_quant.manifest != config["input_quant"]["manifest"]:
                raise ValueError("Overlay and input quantization build fingerprints differ")
            qkv_backend.input_quant = backend.input_quant
        records = []
        # Validate every loaded FFN with the real TP reduction before serving.
        for name, original in mlps:
            candidate = NZMLP(original, backend, tensor_model_parallel_all_reduce)
            for rows, magnitude, x in input_cases(self.device, 4096, backend.max_decode_rows):
                record = check_pair(name, rows, magnitude, original(x), candidate(x))
                records.append(record)
                if not record["passed"]:
                    (overlay / f"dense-gate-rank{self.rank}.json").write_text(json.dumps(records, indent=2) + "\n")
                    raise ValueError(f"Loaded FFN numerical gate failed: {name}, rows={rows}")
            parent_name, attribute = name.rsplit(".", 1)
            setattr(model.get_submodule(parent_name), attribute, candidate)
        if config["mode"] == "all":
            for name, layer in modules:
                if name.startswith("language_model.") and name.endswith((".qkv_proj", ".o_proj")):
                    selected_backend = qkv_backend if name.endswith(".qkv_proj") else backend
                    projection = DenseProjection(layer, selected_backend)
                    for rows, magnitude, x in input_cases(self.device, projection.inner, backend.max_decode_rows):
                        record = check_pair(name, rows, magnitude, layer.quant_method.apply(layer, x), projection(x))
                        records.append(record)
                        if not record["passed"]:
                            (overlay / f"dense-gate-rank{self.rank}.json").write_text(
                                json.dumps(records, indent=2) + "\n"
                            )
                            raise ValueError(f"Loaded projection numerical gate failed: {name}, rows={rows}")
                    layer.add_module("scene_dense_projection", projection)
                    layer.quant_method = DecodeLinearMethod(layer.quant_method, projection)
        if config.get("vision_zero_add_norm", False):
            self.install_vision_norms(model, modules, backend, overlay)
        if "device_image_prepare" in config:
            image_config = config["device_image_prepare"]
            processor = AutoImageProcessor.from_pretrained(
                str(overlay), **(self.model_config.multimodal_config.mm_processor_kwargs or {})
            )
            model.scene_image_preparer = WorkerImagePreparer(
                image_config["build"],
                backend,
                processor,
                self.device,
                verify=partial(self.check_image_transport, processor, overlay),
            )
            if any(p.manifest != image_config["manifest"] for p in model.scene_image_preparer.preparers.values()):
                raise ValueError("Overlay and device image build fingerprints differ")
        torch.npu.synchronize()
        # The baseline weights remain for prefill. Charge all resident additions
        # before determine_available_memory profiles activation/cache headroom.
        added = torch.npu.memory_allocated() - before
        self.model_runner.model_memory_usage += added
        report = {
            "mode": config["mode"],
            "rank": self.rank,
            "added_resident_bytes": added,
            "model_memory_usage": self.model_runner.model_memory_usage,
            "records": records,
            "qkv_projection_build": config.get("qkv_projection_build"),
        }
        (overlay / f"dense-gate-rank{self.rank}.json").write_text(json.dumps(report, indent=2) + "\n")
