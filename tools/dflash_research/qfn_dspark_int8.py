# SPDX-License-Identifier: Apache-2.0
"""Explicit, reversible INT8 projections for the eager QFN DSpark experiment.

The target and checkpoint weights remain untouched. Context K/V precompute,
feature projection, embeddings and norms retain their existing precision.
"""

import torch

CHUNK_ROWS = 4096


class PreparedLinear:
    def __init__(self, weight):
        from vllm_ascend.models.qwen4_exp.lm_head_w8a8 import quantize_linear_weight

        if weight.ndim != 2 or weight.dtype != torch.float16 or any(n % 16 for n in weight.shape):
            raise ValueError("INT8 probe requires aligned FP16 matrices")
        self.shape = tuple(weight.shape)
        packed = torch.empty(self.shape, dtype=torch.int8, device="cpu")
        scales = torch.empty(self.shape[0], dtype=torch.float32, device="cpu")
        for start in range(0, self.shape[0], CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, self.shape[0])
            q, s = quantize_linear_weight(weight[start:stop].detach().cpu())
            packed[start:stop].copy_(q)
            scales[start:stop].copy_(s)
        if weight.device.type == "npu":
            import torch_npu

            self.weight = torch_npu.npu_format_cast(packed.to(weight.device), 29).transpose(0, 1)
        else:
            self.weight = packed.t().contiguous()
        self.scale = scales.to(weight.device)

    def apply(self, layer, x, bias=None):
        from vllm_ascend.models.qwen4_exp.lm_head_w8a8 import dynamic_w8a8_linear

        if bias is not None:
            raise ValueError("INT8 experiment excludes biased projections")
        return dynamic_w8a8_linear(x, self.weight, self.scale)


class DenseInt8Control:
    def __init__(self, worker):
        runner = worker.model_runner
        spec = runner.speculative_config
        if spec.method != "dspark" or spec.draft_tensor_parallel_size != 1 or not spec.enforce_eager:
            raise ValueError("requires eager singleton DSpark draft")
        if runner.drafter.use_cuda_graph or runner.use_async_scheduling:
            raise ValueError("requires synchronous, uncaptured draft execution")
        model = runner.drafter.model.unwrap()
        if len(model.model.layers) != 5 or model.config.hidden_size != 2560:
            raise ValueError("requires the qualified five-layer QFN draft")
        if model.draft_id_to_target_id is not None:
            raise ValueError("vocabulary remapping is outside this experiment")
        self.rank = worker.rank
        self.entries = []
        self.private_checks = []
        self.enabled = False
        selected = [("lm_head", model.lm_head), ("markov", model.model.markov_head.markov_w2)]
        for i, layer in enumerate(model.model.layers):
            selected.extend(
                [
                    (f"layers.{i}.qkv", layer.self_attn.qkv_proj),
                    (f"layers.{i}.gate_up", layer.mlp.gate_up_proj),
                    (f"layers.{i}.down", layer.mlp.down_proj),
                ]
            )
        # Validate the entire scope before preparing or publishing anything.
        for name, layer in selected:
            if getattr(layer, "tp_size", 1) != 1 or getattr(layer, "bias", None) is not None:
                raise ValueError(f"unsupported parallel/bias contract: {name}")
            if "Unquantized" not in type(layer.quant_method).__name__:
                raise ValueError(f"already quantized projection: {name}")
        with torch.inference_mode():
            for name, layer in selected:
                method = PreparedLinear(layer.weight)
                self.entries.append((name, layer, layer.quant_method, method))

    def validate_private(self):
        """Exercise prepared methods before publication; this is not an acceptance test."""
        generator = torch.Generator().manual_seed(7319)
        checks = []
        with torch.inference_mode():
            for name, layer, original, prepared in self.entries:
                for rows in (7, 28):
                    x = torch.randn(rows, prepared.shape[1], generator=generator).half().to(layer.weight.device)
                    reference = original.apply(layer, x, None).float()
                    actual = prepared.apply(layer, x, None).float()
                    relative_rms = float(
                        (actual - reference).square().mean().sqrt() / reference.square().mean().sqrt().clamp_min(1e-12)
                    )
                    if not bool(torch.isfinite(actual).all()) or relative_rms > 0.1:
                        raise RuntimeError(f"private INT8 projection check failed: {name}, {relative_rms}")
                    checks.append(dict(name=name, rows=rows, relative_rms=relative_rms))
        self.private_checks = checks
        return checks

    def configure(self, enabled):
        if type(enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if enabled and len(self.private_checks) != 2 * len(self.entries):
            raise RuntimeError("private projection checks must pass before enabling INT8")
        for name, layer, original, prepared in self.entries:
            expected = prepared if self.enabled else original
            if layer.quant_method is not expected:
                raise RuntimeError(f"projection owner changed: {name}")
        for _, layer, original, prepared in self.entries:
            layer.quant_method = prepared if enabled else original
        self.enabled = enabled
        return self.status()

    def status(self):
        return dict(
            rank=self.rank,
            passed=True,
            enabled=self.enabled,
            projections=[dict(name=name, shape=list(method.shape)) for name, _, _, method in self.entries],
            int8_bytes=sum(method.weight.numel() + method.scale.numel() * 4 for _, _, _, method in self.entries),
            private_checks=self.private_checks,
        )
