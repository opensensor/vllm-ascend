# SPDX-License-Identifier: Apache-2.0
"""NPU regression at the old 16K boundary and the configured context limit."""

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu

import vllm_ascend
from vllm_ascend.utils import enable_custom_op

# Extract the production metadata methods to avoid the model runner's import
# cycle in this standalone operator process. No substitute addressing formula.
source_root = Path(vllm_ascend.__file__).parent
tree = ast.parse((source_root / "spec_decode/multi_kv_cache_group_proposer.py").read_text())
cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
methods = [
    node
    for node in cls.body
    if isinstance(node, ast.FunctionDef)
    and node.name in {"_draft_block_table_width", "_common_attn_metadata_for_draft_group"}
]
subset = ast.ClassDef(name="Proposer", bases=[], keywords=[], body=methods, decorator_list=[])
mla_tree = ast.parse((source_root / "_310p/attention/mla_v1_310.py").read_text())
helper = next(
    node for node in mla_tree.body if isinstance(node, ast.FunctionDef) and node.name == "_qsa_cache_block_table"
)
future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
scope = {"copy": copy, "torch": torch, "_QSA_KERNEL_BLOCK_SIZE": 32}
exec(
    compile(
        ast.fix_missing_locations(ast.Module(body=[future, subset, helper], type_ignores=[])),
        "production_metadata",
        "exec",
    ),
    scope,
)

torch.npu.set_device(0)
torch_npu.npu.set_compile_mode(jit_compile=False)
enable_custom_op()
context, page, kernel_page = 311040, 640, 32
blocks, split = context // page, page // kernel_page
proposer = scope["Proposer"]()
proposer.max_model_len = context
proposer.vllm_config = None
proposer.kv_cache_gid = 0
proposer._uses_multi_group_kv_cache = True
proposer.runner = SimpleNamespace(
    input_batch=SimpleNamespace(block_table=[SimpleNamespace(blocks_per_phys_block=split)])
)
builder = SimpleNamespace(kv_cache_spec=SimpleNamespace(max_num_blocks_per_req=lambda *args: blocks))
group = SimpleNamespace(kv_cache_group_id=0, get_metadata_builder=lambda: builder)
common = SimpleNamespace(num_reqs=1, block_table_tensor=torch.arange(blocks * split, dtype=torch.int32).npu()[None, :])
metadata = proposer._common_attn_metadata_for_draft_group(common, group, 4)
table = scope["_qsa_cache_block_table"](metadata.block_table_tensor, page)
assert table.shape == (1, blocks)
lengths_cpu = torch.tensor([16000, 16001, 16640, context], dtype=torch.int32)
page_values = (torch.arange(blocks) % 13).float() / 16
cache = page_values[:, None, None, None].expand(blocks, 32, page, 16).contiguous().half().npu()
query = torch.zeros(4, 8, 512, dtype=torch.float16).npu()
args = (
    query,
    cache,
    cache,
    torch.zeros(4, 1, dtype=torch.int32).npu(),
    lengths_cpu.npu(),
    torch.zeros(4, dtype=torch.int32).npu(),
    torch.full((4,), -1, dtype=torch.int32).npu(),
    table,
    torch.tensor([0, 4], dtype=torch.int32).npu(),
    0.01,
    4,
)
for repeat in range(3):
    result = torch.ops._C_ascend.npu_qsa_sparse_attention_310(*args)
    torch.npu.synchronize()
    expected = []
    for length in lengths_cpu.tolist():
        full, tail = divmod(length, page)
        total = page_values[:full].sum() * page
        if tail:
            total += page_values[full] * tail
        expected.append(total / length)
    expected = torch.stack(expected).half()[:, None, None].expand_as(result)
    torch.testing.assert_close(result.cpu(), expected, rtol=1e-3, atol=1e-3)
    print(
        json.dumps(
            {
                "repeat": repeat,
                "kernel_columns": metadata.block_table_tensor.shape[1],
                "physical_columns": table.shape[1],
                "visible_lengths": lengths_cpu.tolist(),
                "max_error": (result.cpu().float() - expected.float()).abs().max().item(),
            }
        ),
        flush=True,
    )
