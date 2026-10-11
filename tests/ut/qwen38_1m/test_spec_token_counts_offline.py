# SPDX-License-Identifier: Apache-2.0
"""Count narrowing must preserve next-token selection, discard and backups."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend._310p import spec_token_counts


@pytest.mark.parametrize("batch,width", [(0, 3), (1, 3), (6, 3), (6, 9), (6, 10), (1, 0)])
@pytest.mark.parametrize("pattern", ["none", "all", "mixed"])
def test_short_counts_are_exact_int64_and_wide_counts_keep_reference(monkeypatch, batch, width, pattern):
    monkeypatch.setattr(spec_token_counts, "_supports_device", lambda device: True)
    mask = torch.randint(2, (batch, width), generator=torch.Generator().manual_seed(310)).bool()
    if pattern == "none":
        mask.zero_()
    elif pattern == "all":
        mask.fill_(True)
    actual = spec_token_counts.count_valid_spec_tokens(mask)
    assert actual.dtype == torch.int64
    torch.testing.assert_close(actual, mask.sum(1), rtol=0, atol=0)


@pytest.mark.parametrize("is_310p", [False, True])
def test_actual_proposer_method_keeps_discarded_and_empty_row_backups(monkeypatch, is_310p):
    monkeypatch.setattr(spec_token_counts, "_supports_device", lambda device: True)
    root = Path(__file__).resolve().parents[3]
    path = root / "vllm_ascend/spec_decode/llm_base_proposer.py"
    tree = ast.parse(path.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendSpecDecodeBaseProposer")
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "prepare_next_token_ids_padded")
    namespace = {
        "torch": torch,
        "np": np,
        "ascend_utils": SimpleNamespace(is_310p=lambda: is_310p),
        "count_valid_spec_tokens": spec_token_counts.count_valid_spec_tokens,
        "DeviceOperator": SimpleNamespace(index_fill=lambda x, dim, indices, value: x.index_fill(dim, indices, value)),
    }
    code = ast.unparse(ast.Module(body=[method], type_ignores=[]))
    exec("from __future__ import annotations\n" + code, namespace)
    backup = SimpleNamespace(np=np.zeros(4, dtype=np.int64), gpu=torch.zeros(4, dtype=torch.int64))
    copies = []

    def copy_to_gpu(rows):
        backup.gpu[:rows].copy_(torch.from_numpy(backup.np[:rows]))
        copies.append(rows)

    backup.copy_to_gpu = copy_to_gpu
    proposer = SimpleNamespace(backup_next_token_ids=backup)
    batch = SimpleNamespace(
        num_reqs=4, req_ids=["a", "b", "c", "d"], vocab_size=100, num_tokens_no_spec=np.array([4, 5, 6, 7])
    )
    requests = {name: SimpleNamespace(get_token_id=lambda position: position + 70) for name in batch.req_ids}
    sampled = torch.tensor([[10, 11, -1], [-1, -1, -1], [99, 100, -1], [20, 21, 22]])
    original = sampled.clone()
    next_ids, counts = namespace[method.name](proposer, sampled, requests, batch, torch.tensor([3]), 1)
    assert copies == [4]
    assert next_ids.tolist() == [11, 74, 99, 76]
    assert counts.tolist() == [2, 0, 1, 0]
    assert counts.dtype == torch.int64
    assert torch.equal(sampled, original)
