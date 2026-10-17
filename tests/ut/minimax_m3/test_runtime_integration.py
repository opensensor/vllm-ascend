# SPDX-License-Identifier: Apache-2.0
"""Exercise production adapter methods with CPU distributed/runtime stand-ins.

AST extraction avoids importing torch_npu on the offline host. The method bodies
are compiled unchanged from the production files; these are wiring tests, not
evidence of vLLM/NPU startup or hardware collectives.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend._310p.attention.minimax_m3 import M3EagerMetadataBuilder
from vllm_ascend._310p.ops.minimax_m3 import PackedM3Experts, clipped_swiglu, paged_sparse_attention

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "vllm_ascend/models/minimax_m3/minimax_m3.py"
RUNNER = ROOT / "vllm_ascend/_310p/model_runner_310p.py"


def production_definition(path, class_name, namespace, method=None):
    tree = ast.parse(path.read_text())
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == class_name)
    if method:
        node = next(item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == method)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method or class_name]


def test_scheduler_metadata_uses_cpu_boundaries_and_separate_tables():
    builder = object.__new__(M3EagerMetadataBuilder)
    common = SimpleNamespace(
        query_start_loc_cpu=torch.tensor([0, 1, 4]),
        num_reqs=2,
        num_actual_tokens=4,
        seq_lens=torch.tensor([90, 3]),
        max_seq_len=90,
        block_table_tensor=torch.tensor([[7], [8]]),
        slot_mapping=torch.tensor([1, 2, 3, 4]),
    )
    metadata = builder.build(0, common)
    assert metadata.query_starts == (0, 1, 4)
    assert metadata.block_table is common.block_table_tensor
    assert metadata.seq_lens is common.seq_lens
    with pytest.raises(ValueError, match="cascade"):
        builder.build(1, common)
    common.num_actual_tokens = 3
    with pytest.raises(ValueError, match="count"):
        builder.build(0, common)


def test_actual_dense_activation_uses_fp32_clipped_oai_on_310p():
    cls = production_definition(
        MODEL,
        "MiniMaxM3SwiGLUOAI",
        {
            "torch": torch,
            "nn": torch.nn,
            "get_ascend_device_type": lambda: 1,
            "AscendDeviceType": SimpleNamespace(_310P=1),
            "clipped_swiglu": clipped_swiglu,
        },
    )
    gate = torch.tensor([-20, -1, 0, 1, 20], dtype=torch.float16)
    up = torch.tensor([-20, 0, 1, 2, 20], dtype=torch.float16)
    output = cls(alpha=1.702, beta=1, limit=7)(torch.cat((gate, up)))
    torch.testing.assert_close(output, clipped_swiglu(gate, up).half(), atol=0, rtol=0)


def test_actual_sparse_layer_bridge_uses_indexer_block_table():
    shape, dim = (2, 4, 2, 8), 8
    main = SimpleNamespace(
        query_starts=(0, 3),
        seq_lens=torch.tensor([3]),
        block_table=torch.tensor([[1]]),
        slot_mapping=torch.tensor([4, 5, 6]),
        max_seq_len=3,
    )
    index = SimpleNamespace(slot_mapping=torch.tensor([0, 1, 2]), block_table=torch.tensor([[0]]))
    context = SimpleNamespace(attn_metadata={"main": main, "index": index})
    function = production_definition(
        MODEL,
        "MiniMaxM3SparseAttention",
        {
            "torch": torch,
            "get_ascend_device_type": lambda: 1,
            "AscendDeviceType": SimpleNamespace(_310P=1),
            "get_forward_context": lambda: context,
            "paged_sparse_attention": paged_sparse_attention,
        },
        "_run_sparse_attention",
    )
    layer = SimpleNamespace(
        layer_name="main",
        num_heads=4,
        num_kv_heads=2,
        head_dim=dim,
        num_idx_heads=2,
        idx_head_dim=dim,
        topk_blocks=2,
        scaling=dim**-0.5,
        kv_cache=(torch.full(shape, torch.nan), torch.full(shape, torch.nan)),
        indexer=SimpleNamespace(
            index_cache=SimpleNamespace(prefix="index", kv_cache=(torch.full((2, 4, dim), torch.nan),)),
            impl=SimpleNamespace(init_blocks=0, local_blocks=1),
        ),
    )
    torch.manual_seed(5)
    q, k, v = torch.randn(3, 4 * dim), torch.randn(3, 2 * dim), torch.randn(3, 2 * dim)
    iq, ik, out = torch.randn(3, 2 * dim), torch.randn(3, dim), torch.empty_like(q)
    function(layer, q, k, v, iq, ik, out)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(layer.indexer.index_cache.kv_cache[0][0, :3], ik)
    assert torch.isnan(layer.indexer.index_cache.kv_cache[0][1]).all()
    torch.testing.assert_close(out[0].view(4, dim), v[0].view(2, dim).repeat_interleave(2, 0))


def test_actual_runner_allocates_nd_main_pair_and_single_index_cache():
    class Spec:
        def __init__(self, index):
            self.block_size, self.num_kv_heads, self.head_size, self.dtype = 128, 1 if index else 2, 128, torch.float16
            self.page_size_bytes = 128 * 128 * 2 * (1 if index else 4)

    class MainBackend:
        @staticmethod
        def get_name():
            return "ASCEND_MINIMAX_M3_SPARSE"

        @staticmethod
        def get_kv_cache_shape(blocks, block, heads, dim):
            return 2, blocks, block, heads, dim

    class IndexBackend(MainBackend):
        @staticmethod
        def get_name():
            return "ASCEND_MINIMAX_M3_SPARSE_INDEXER"

        @staticmethod
        def get_kv_cache_shape(blocks, block, heads, dim):
            return blocks, block, dim

    method = production_definition(
        RUNNER,
        "NPUModelRunner310",
        {
            "torch": torch,
            "UniformTypeKVCacheSpecs": type("Uniform", (), {}),
            "MambaSpec": type("Mamba", (), {}),
            "AttentionSpec": Spec,
            "_get_layer_attention_backends": lambda groups: {"main": MainBackend, "index": IndexBackend},
            "get_kv_cache_tensor_layers": lambda descriptor: descriptor.layers,
        },
        "_allocate_kv_cache_tensors",
    )
    specs = {name: Spec(name == "index") for name in ("main", "index")}
    config = SimpleNamespace(
        num_blocks=3,
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec, layer_names=[name]) for name, spec in specs.items()],
        kv_cache_tensors=[SimpleNamespace(layers=[name]) for name in specs],
    )
    runner = SimpleNamespace(runner_only_attn_layers=set(), device=torch.device("cpu"))
    caches = method(runner, config)
    assert len(caches["main"]) == 2 and len(caches["index"]) == 1
    for name, buffers in caches.items():
        assert all(tensor.is_contiguous() for tensor in buffers)
        assert sum(tensor.numel() * tensor.element_size() for tensor in buffers) == 3 * specs[name].page_size_bytes
    assert caches["main"][0].data_ptr() != caches["main"][1].data_ptr()


@pytest.mark.parametrize("rank", [0, 1])
def test_actual_moe_adapter_bypasses_fused_allocation_and_reduces_shared_once(rank):
    class Gate(torch.nn.Module):
        def __init__(self, hidden, experts, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(experts, hidden))

        def forward(self, x):
            return x.float() @ self.weight.T, None

    class Shared(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

        def forward(self, x):
            return x / 2

    calls = []
    group = SimpleNamespace(device_group=SimpleNamespace(size=lambda: 2), rank_in_group=rank)
    namespace = {
        "torch": torch,
        "nn": torch.nn,
        "get_tensor_model_parallel_world_size": lambda: 2,
        "get_ep_group": lambda: group,
        "AscendDeviceType": SimpleNamespace(_310P=1),
        "get_ascend_device_type": lambda: 1,
        "MiniMaxM3MLP": Shared,
        "GateLinear": Gate,
        "PackedM3Experts": PackedM3Experts,
        "tensor_model_parallel_all_reduce": lambda x: calls.append(x.clone()) or x,
    }
    cls = production_definition(MODEL, "MiniMaxM3MoE", namespace)
    config = SimpleNamespace(
        n_shared_experts=1,
        num_local_experts=4,
        hidden_size=32,
        intermediate_size=64,
        num_experts_per_tok=2,
        routed_scaling_factor=2,
        use_routing_bias=True,
        swiglu_alpha=1.702,
        swiglu_limit=7,
    )
    parallel = SimpleNamespace(enable_eplb=False, eplb_config=SimpleNamespace(num_redundant_experts=0))
    model = cls(config, parallel, prefix="model.layers.0.block_sparse_moe")
    assert isinstance(model.experts, PackedM3Experts)
    for name, buffer in model.experts.named_buffers():
        buffer.fill_(1 if name.endswith("scales") else 0)
    model.e_score_correction_bias.data.zero_()
    x = torch.randn(3, 32).half()
    result = model(x)
    assert len(calls) == 1 and calls[0].dtype == torch.float32
    torch.testing.assert_close(result, x / 2)
