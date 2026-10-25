# SPDX-License-Identifier: Apache-2.0
"""The one-hot oracle covers both fields and catches stale-byte corruption."""

import pytest

from tools.minimax_m3 import byte_reuse_probe as probe
from vllm_ascend._310p.ops.minimax_m3 import reference_grouped_mm


@pytest.mark.parametrize("corrupt", [False, True])
def test_byte_reuse_oracle(monkeypatch, corrupt):
    # CPU reference accepts canonical bytes; packing has separate layout tests.
    monkeypatch.setattr(probe, "pack_w4_nz256", lambda codes: codes)

    def kernel(x, codes, scales, ends):
        result = reference_grouped_mm(x, codes, scales, ends)
        if corrupt:
            result[:, 16:32] = result[:, 80:96]
        return result

    result = probe.byte_reuse_probe(kernel, "cpu")
    assert result["passed"] is (not corrupt)
    assert len(result["cases"]) == 2 * len(probe.PROBE_POSITIONS)
    assert {row["position"] % 2 for row in result["cases"]} == {0, 1}
