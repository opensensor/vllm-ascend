# SPDX-License-Identifier: Apache-2.0
"""Check continued-prefill page addressing without importing an NPU runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def prefill_functions():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/_310p/attention/mla_v1_310.py"
    tree = ast.parse(path.read_text())
    names = {"_qsa_physical_cache_page", "_qsa_cache_block_table", "_forward_prefill_paged_latent"}
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
    module = ast.Module(body=functions, type_ignores=[])
    scope = {
        "torch": torch,
        "_NZ_INNER": 16,
        "_QSA_KERNEL_BLOCK_SIZE": 32,
        "_QSA_COMPRESS_RATIO": 4,
        "record_attention_compute_start": lambda: None,
    }
    exec(compile(module, str(path), "exec"), scope)
    return scope


@pytest.mark.parametrize("physical_channels", [1, 2, 4])
@pytest.mark.parametrize("offset", [0, 16])
def test_prefill_passes_physical_pages_and_logical_heads(physical_channels, offset):
    functions = prefill_functions()
    backing = torch.full((offset + 3 * physical_channels * 32 * 16,), -9.0, dtype=torch.float16)
    physical = backing[offset:].view(3, physical_channels, 32, 16)
    cache = physical[:, :1]
    for page in range(3):
        cache[page].fill_(page + 1)
    before = backing.clone()
    captured = []

    def kernel(*args):
        captured.append(args)
        passed = args[1]
        # Native QSA derives page stride from shape, not tensor strides.
        page_offset = 2 * passed.shape[1] * passed.shape[2] * passed.shape[3]
        assert backing[passed.storage_offset() + page_offset].item() == 3.0
        return args[0].clone()

    owner = SimpleNamespace(
        W_UK_T=torch.eye(16, dtype=torch.float16).expand(2, -1, -1).contiguous(),
        glm_indexer=None,
        host_kv_layer=None,
        scale=0.25,
        _get_decode_constant_buffers=lambda device, rows: (
            torch.full((rows, 1), -1, dtype=torch.int32),
            torch.zeros(rows, dtype=torch.int32),
            torch.full((rows,), -1, dtype=torch.int32),
            None,
        ),
        _get_paged_latent_op=lambda: kernel,
        _v_up_proj=lambda value: value,
    )
    metadata = SimpleNamespace(
        prefill=SimpleNamespace(
            input_positions=torch.arange(64, 68),
            actual_seq_lengths_q=[4],
            max_seq_lens=68,
            block_table=torch.tensor([[0, 1, 2]], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        )
    )
    functions["_forward_prefill_paged_latent"](
        owner, torch.ones(4, 2, 16, dtype=torch.float16), (cache, cache), metadata
    )
    args = captured[0]
    assert args[1].shape == physical.shape
    assert args[1].untyped_storage().data_ptr() == backing.untyped_storage().data_ptr()
    assert args[2].untyped_storage().data_ptr() == backing.untyped_storage().data_ptr()
    assert args[1].storage_offset() == offset
    assert args[11:] == (() if physical_channels == 1 else (1,))
    torch.testing.assert_close(backing, before, rtol=0, atol=0)


def test_physical_page_layout_rejects_overlapping_rows():
    functions = prefill_functions()
    cache = torch.empty(3, 2, 32, 16).transpose(1, 2)
    with pytest.raises(ValueError, match="physical page layout"):
        functions["_qsa_physical_cache_page"](cache)
