# SPDX-License-Identifier: Apache-2.0
import torch


def replacements(native_resources):
    changes = serving_replacements(native_resources)
    operation = native_resources["sinkhorn_normalize_v1"]
    counts = torch.zeros(4, dtype=torch.float32, device="npu")

    def shadow(mix):
        actual = operation(mix)
        expected = mix / (mix.sum(dim=-2, keepdim=True) + 1e-6)
        for _ in range(19):
            expected = expected / (expected.sum(dim=-1, keepdim=True) + 1e-6)
            expected = expected / (expected.sum(dim=-2, keepdim=True) + 1e-6)
        counts[0].add_((actual != expected).float().sum())
        counts[1].add_(float(actual.numel()))
        counts[2].copy_(torch.maximum(counts[2], (actual - expected).abs().max()))
        counts[3].add_((actual.half() != expected.half()).float().sum())
        return expected

    resources = dict(native_resources, sinkhorn_normalize_v1=shadow)
    changes.update(sinkhorn_replacements(resources))
    target = "vllm_ascend._310p.worker_310p:NPUWorker310.resident_status"
    original = changes[target]

    def status(self):
        result = original(self)
        result["sinkhorn_audit"] = dict(zip(("fp32_mismatches", "elements", "max_abs", "fp16_mismatches"), counts.cpu().tolist()))
        return result

    changes[target] = status
    return changes
