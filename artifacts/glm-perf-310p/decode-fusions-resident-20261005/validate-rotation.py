# SPDX-License-Identifier: Apache-2.0
def prepare():
    import importlib.util

    root = "/home/matteius/experiments/glm-decode-fusions-resident-20261005"
    spec = importlib.util.spec_from_file_location("glm_rotation_candidate_v1", root + "/rotation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Rotation(root + "/rotation-v1.bin")


def validate(op):
    import torch

    from vllm_ascend.models.glm5next.kpool_ops import hadamard128

    gen = torch.Generator().manual_seed(310)
    for rows in (1, 2, 4, 8):
        query = torch.randn(rows, 32, 128, generator=gen).bfloat16().npu()
        expected = hadamard128(query).bfloat16().half()
        torch.testing.assert_close(op(query), expected, rtol=0, atol=0)
    return {"passed": True, "exact": True, "rows": [1, 2, 4, 8], "input": "BF16", "output": "FP16"}
