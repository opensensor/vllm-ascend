# SPDX-License-Identifier: Apache-2.0
def prepare():
    import importlib.util

    root = "/home/matteius/experiments/glm-sinkhorn-resident-20261005"
    spec = importlib.util.spec_from_file_location("glm_sinkhorn_normalize_v1", root + "/sinkhorn_native.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SinkhornNormalize(root + "/normalize-v1.bin", order=2)


def validate(operation):
    import torch

    def reference(mix):
        mix = mix / (mix.sum(-2, keepdim=True) + 1e-6)
        for _ in range(19):
            mix = mix / (mix.sum(-1, keepdim=True) + 1e-6)
            mix = mix / (mix.sum(-2, keepdim=True) + 1e-6)
        return mix

    gen = torch.Generator().manual_seed(310)
    for rows in (1, 2, 4, 8):
        for scale in (0, .1, 1, 10, 100):
            mix = torch.softmax((torch.randn(rows, 4, 4, generator=gen) * scale).npu(), -1) + 1e-6
            torch.testing.assert_close(operation(mix), reference(mix), rtol=0, atol=0)
    return {"passed": True, "exact_fp32": True, "cases": 20, "order": 2}
