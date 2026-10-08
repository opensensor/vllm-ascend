# SPDX-License-Identifier: Apache-2.0
"""Source embedded in the versioned manifest, executed only while paused."""


def prepare():
    from tools.glm_perf.direct_prefill import DirectPrefillScore

    return DirectPrefillScore("/home/matteius/experiments/glm-native-resident-20261005/prefill-v1.bin")


def validate(op):
    import torch

    query = torch.full((4, 32, 128), 0.25, dtype=torch.float16, device="npu")
    weights = torch.tensor([1, 2, -1, 0.5], dtype=torch.float32, device="npu")[:, None].expand(4, 32).contiguous()
    keys = torch.ones((2, 160, 128), dtype=torch.float16, device="npu")
    keys[0].mul_(2)
    table = torch.tensor([[1, 0]], dtype=torch.int32, device="npu")
    ends = torch.tensor([4], dtype=torch.int32, device="npu")
    positions = torch.tensor([3, 255, 639, 643], dtype=torch.int32, device="npu")
    output = op(query, weights, keys.flatten(), table, ends, positions, 168, 2, 160, 160 * 128, 128, 0)
    expected = torch.full((4, 168), -torch.inf)
    for row, (count, weight) in enumerate(zip([1, 64, 160, 161], [1, 2, -1, 0.5])):
        expected[row, : min(count, 160)] = 1024 * weight
        if count > 160:
            expected[row, 160:count] = 2048 * weight
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    return {"passed": True, "rows": 4, "pools": 168, "page_boundary": 160, "exact": True}
