# SPDX-License-Identifier: Apache-2.0
"""Exercise packed-byte reuse with an exact, varying-K one-hot oracle."""

import torch

from vllm_ascend._310p.ops.minimax_m3 import pack_w4_nz256

PROBE_OUTPUTS = 256
PROBE_WIDTH = 4096
PROBE_POSITIONS = (0, 1, 32, 33, 127, 128, 129, 255, 256, 257, 1023, 4095)


def byte_reuse_probe(kernel, device):
    rows = torch.arange(PROBE_OUTPUTS, dtype=torch.int32)[:, None]
    columns = torch.arange(PROBE_WIDTH // 2, dtype=torch.int32)[None, :]
    canonical = ((rows + columns * 37) & 255).to(torch.uint8)
    codes = pack_w4_nz256(canonical).unsqueeze(0).to(device)
    scales = torch.ones(
        (1, PROBE_OUTPUTS // 32, PROBE_WIDTH // 32), dtype=getattr(kernel, "scale_dtype", torch.float32), device=device
    )
    ends = torch.ones(1, dtype=torch.int64, device=device)
    records = []
    for repeat in range(2):
        for position in PROBE_POSITIONS:
            x = torch.zeros((1, PROBE_WIDTH), dtype=torch.float16)
            x[0, position] = 1
            actual = kernel(x.to(device), codes, scales, ends).cpu().flatten()
            byte = canonical[:, position // 2].to(torch.int32)
            unsigned = (byte >> (4 * (position % 2))) & 15
            expected = (((unsigned + 8) & 15) - 8).half()
            records.append({"repeat": repeat, "position": position, "passed": torch.equal(actual, expected)})
    return {"passed": all(row["passed"] for row in records), "cases": records}
