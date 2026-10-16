"""Check the restored QSA ABI and TP6 query geometry against CPU reference."""

import json
from pathlib import Path

import torch
import torch_npu

from vllm_ascend.models.qwen4_exp.ops.qsa_indexer import QSAGroupSelection
from vllm_ascend.models.qwen4_exp.ops.qsa_sparse_attention_310 import (
    qsa_sparse_attention_310,
    qsa_sparse_attention_310_reference,
)
from vllm_ascend.utils import enable_custom_op

enable_custom_op()
torch_npu.npu.set_compile_mode(jit_compile=False)
torch.manual_seed(31)
records = []
for rows in (3, 18, 65):
    query = torch.randn(rows, 4, 256).half()
    key = torch.randn(2, 16, 128, 16).half() * 0.1
    value = torch.randn_like(key) * 0.1
    indices = torch.arange(16, dtype=torch.int32).expand(rows, -1).contiguous()
    counts = torch.full((rows,), 16, dtype=torch.int32)
    starts = torch.full((rows,), 64, dtype=torch.int32)
    tails = torch.arange(rows, dtype=torch.int32) % 4
    table = torch.tensor([[0, 1]], dtype=torch.int32)
    bounds = torch.tensor([0, rows], dtype=torch.int32)
    selected = QSAGroupSelection(indices, counts, starts, tails)
    expected = qsa_sparse_attention_310_reference(query, key, value, selected, table, bounds)
    selected_npu = QSAGroupSelection(*(x.npu() for x in (indices, counts, starts, tails)))
    actual = qsa_sparse_attention_310(
        query.npu(), key.npu(), value.npu(), selected_npu, table.npu(), bounds.npu()
    ).cpu()
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.002)
    records.append(dict(rows=rows, passed=True, max_abs=float((actual - expected).abs().max())))
Path("/srv/ai/storage/artifacts/qfn-tp6-scaleout-20261016/qsa-gate.json").write_text(
    json.dumps(records, indent=2) + "\n"
)
print(records)
