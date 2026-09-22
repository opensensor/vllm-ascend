#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.config import CUDAGraphMode
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec, MLAAttentionSpec
from vllm.v1.worker.utils import copy_kv_cache_blocks_inplace

from tests.ut.base import TestBase
from vllm_ascend._310p.model_runner_310p import (
    NPUModelRunner310,
    _allocate_attention_cache_tensor,
    _get_attention_cache_tensor_shape,
    _get_layer_attention_backends,
    _iter_kv_cache_tensors,
)


def _prepare_inputs_source() -> str:
    source_path = Path(__file__).resolve().parents[3] / "vllm_ascend" / "_310p" / "model_runner_310p.py"
    source = source_path.read_text(encoding="utf-8")
    start = source.index("    def _prepare_inputs(")
    end = source.index("    @torch.inference_mode()", start)
    return source[start:end]


def test_prepare_inputs_keeps_aclgraph_metadata_on_cpu() -> None:
    source = _prepare_inputs_source()

    assert "block_table.compute_slot_mapping(" in source
    assert "req_indices," in source
    assert "positions_np[:total_num_scheduled_tokens]" in source

    assert "self.input_batch.block_table.compute_slot_mapping(" not in source
    assert "query_start_loc.gpu[: num_reqs + 1]" not in source
    assert "req_indices_gpu" not in source
    assert "self.num_computed_tokens[req_indices_gpu]" not in source

    assert "self.positions[:total_num_scheduled_tokens].copy_(" in source
    assert "self._positions_cpu_buf[:total_num_scheduled_tokens]" in source
    assert "self.seq_lens[:num_reqs].copy_(" in source
    assert "self.optimistic_seq_lens_cpu[:num_reqs]" in source
    assert "self._sync_num_accepted_tokens(" in source
    assert "self.input_batch.num_accepted_tokens_cpu[" not in source


def test_model_forward_updates_mtp_full_graph_params_before_replay() -> None:
    runner = object.__new__(NPUModelRunner310)
    runner.uses_mrope = False
    runner.enable_enpu = False
    runner.speculative_config = SimpleNamespace(method="mtp")
    runner.update_stream = MagicMock()
    runner._all_gather_hidden_states_and_aux = MagicMock()

    calls = []

    def fake_update(*args):
        calls.append("update")

    def fake_model(**kwargs):
        calls.append("model")
        return torch.ones(1)

    runner.model = fake_model
    runner._update_full_graph_params_if_needed = fake_update
    forward_context = SimpleNamespace(
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        capturing=False,
    )

    with patch(
        "vllm_ascend._310p.model_runner_310p.get_forward_context",
        return_value=forward_context,
    ):
        hidden_states = runner._model_forward(
            8,
            input_ids=torch.tensor([1]),
            positions=torch.tensor([0]),
        )

    assert calls == ["update", "model"]
    torch.testing.assert_close(hidden_states, torch.ones(1))


def test_310p_runner_does_not_advertise_standardized_shared_kv_backing() -> None:
    assert NPUModelRunner310.supports_standardized_shared_kv_backing is False
    assert NPUModelRunner310.supports_compact_mamba_state is False


def test_iter_kv_cache_tensors_flattens_hybrid_layout() -> None:
    attention_k = torch.empty(2)
    attention_v = torch.empty(2)
    mamba_conv = torch.empty(3)
    mamba_ssm = torch.empty(4)

    flattened = list(_iter_kv_cache_tensors([(attention_k, attention_v), [mamba_conv, mamba_ssm]]))

    assert flattened == [attention_k, attention_v, mamba_conv, mamba_ssm]


def test_mla_cache_shape_uses_divisible_kernel_page_and_keeps_blocks() -> None:
    backend = SimpleNamespace(
        get_supported_kernel_block_sizes=lambda: [32, 16],
        get_kv_cache_shape=lambda blocks, block_size, heads, head_size: (
            blocks,
            block_size,
            heads,
            head_size,
        ),
    )
    cache_spec = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.float16,
    )

    shape = _get_attention_cache_tensor_shape(backend, 64, cache_spec)

    assert shape == (64, 16, 1, 512)


def test_dense_cache_shape_splits_leading_key_value_axis() -> None:
    backend = SimpleNamespace(
        get_supported_kernel_block_sizes=lambda: [128, 64],
        get_kv_cache_shape=lambda blocks, block_size, heads, head_size: (
            2,
            blocks,
            heads * head_size // 16,
            block_size,
            16,
        ),
    )
    cache_spec = AttentionSpec(
        block_size=128,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.float16,
    )

    shape = _get_attention_cache_tensor_shape(backend, 64, cache_spec)

    assert shape == (64, 8, 128, 16)


def test_layer_attention_backends_preserve_hybrid_group_selection() -> None:
    linear_backend = object()
    mla_backend = object()
    attn_groups = [
        [
            SimpleNamespace(backend=linear_backend, layer_names=["layers.0.attn"]),
            SimpleNamespace(backend=mla_backend, layer_names=["layers.3.attn"]),
        ]
    ]

    layer_backends = _get_layer_attention_backends(attn_groups)

    assert layer_backends == {
        "layers.0.attn": linear_backend,
        "layers.3.attn": mla_backend,
    }


def test_mla_cache_allocation_keeps_nd_layout() -> None:
    cache_spec = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.float16,
    )

    with patch(
        "vllm_ascend._310p.model_runner_310p.torch_npu.empty_with_format",
        create=True,
    ) as empty_nz:
        cache = _allocate_attention_cache_tensor(
            (4, 16, 1, 8),
            torch.float16,
            torch.device("cpu"),
            cache_spec,
        )

    assert cache.shape == (4, 16, 1, 8)
    assert cache.is_contiguous()
    empty_nz.assert_not_called()


def test_nested_hybrid_cache_copy_uses_scheduler_block_geometry() -> None:
    attention = (torch.arange(128).reshape(64, 2), torch.arange(128, 256).reshape(64, 2))
    mamba = [torch.arange(192).reshape(64, 3)]

    with patch(
        "vllm.v1.worker.utils.async_tensor_h2d",
        return_value=torch.tensor([[1, 5]], dtype=torch.int64),
    ):
        copy_kv_cache_blocks_inplace(
            _iter_kv_cache_tensors([attention, mamba]),
            num_blocks=64,
            kv_cache_block_copies=[(1, 5)],
        )

    for cache in (*attention, *mamba):
        torch.testing.assert_close(cache[5], cache[1])


def test_update_states_copies_nested_hybrid_cache_once() -> None:
    runner = object.__new__(NPUModelRunner310)
    caches = [(torch.empty(2), torch.empty(2)), [torch.empty(3)]]
    runner.kv_caches = caches
    runner.kv_cache_config = SimpleNamespace(num_blocks=8)
    block_copies = [(1, 2)]
    scheduler_output = SimpleNamespace(
        kv_cache_block_copies=block_copies,
        finished_req_ids=set(),
    )
    deferred = object()
    base_observed_copies = []

    def fake_base_update_states(_runner, output):
        base_observed_copies.append(output.kv_cache_block_copies)
        return deferred

    with (
        patch(
            "vllm_ascend._310p.model_runner_310p.NPUModelRunner._update_states",
            new=fake_base_update_states,
        ),
        patch("vllm.v1.worker.utils.copy_kv_cache_blocks_inplace") as copy_blocks,
    ):
        result = runner._update_states(scheduler_output)

    assert result is deferred
    assert base_observed_copies == [None]
    assert scheduler_output.kv_cache_block_copies == block_copies
    copied_caches = list(copy_blocks.call_args.args[0])
    assert copied_caches == [caches[0][0], caches[0][1], caches[1][0]]
    assert copy_blocks.call_args.args[1:] == (8, block_copies)


def test_single_request_runner_compacts_mamba_allocation() -> None:
    runner = object.__new__(NPUModelRunner310)
    runner.device = torch.device("cpu")
    runner.runner_only_attn_layers = set()
    runner.max_num_reqs = 1
    runner.supports_compact_mamba_state = True
    spec = MambaSpec(
        block_size=128,
        shapes=((4, 8), (2, 4)),
        dtypes=(torch.float16, torch.float16),
    )
    layer_name = "model.layers.0.linear_attn"
    kv_cache_config = SimpleNamespace(
        num_blocks=128,
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec, layer_names=[layer_name])],
        kv_cache_tensors=[
            SimpleNamespace(
                size=128 * spec.page_size_bytes,
                layers=[layer_name],
                shared_by=[layer_name],
            )
        ],
    )

    caches = runner._allocate_kv_cache_tensors(kv_cache_config)

    assert caches[layer_name][0].shape == (1, 4, 8)
    assert caches[layer_name][1].shape == (1, 2, 4)
    assert caches[layer_name][0].untyped_storage().nbytes() == spec.page_size_bytes


def test_single_request_runner_remaps_only_mamba_device_table() -> None:
    runner = object.__new__(NPUModelRunner310)
    runner.supports_compact_mamba_state = True
    attention_gpu = torch.tensor([[17, 18]], dtype=torch.int32)
    mamba_gpu = torch.tensor([[91, 92]], dtype=torch.int32)
    attention_table = SimpleNamespace(
        is_mamba_group=False,
        block_table=SimpleNamespace(gpu=attention_gpu),
    )
    mamba_table = SimpleNamespace(
        is_mamba_group=True,
        block_table=SimpleNamespace(gpu=mamba_gpu),
    )
    runner.input_batch = SimpleNamespace(block_table=SimpleNamespace(block_tables=[attention_table, mamba_table]))

    runner._remap_compact_mamba_block_tables(num_reqs=1)

    torch.testing.assert_close(attention_gpu, torch.tensor([[17, 18]], dtype=torch.int32))
    torch.testing.assert_close(mamba_gpu, torch.zeros_like(mamba_gpu))


class TestNPUModelRunner310(TestBase):
    def test_may_reinitialize_input_batch_expands_prefix_mamba_block_table(self):
        runner = object.__new__(NPUModelRunner310)
        runner.max_num_reqs = 8
        runner.max_model_len = 512
        runner.max_encoder_len = 0
        runner.max_num_tokens = 1024
        runner.device = torch.device("cpu")
        runner.pin_memory = False
        runner.is_pooling_model = False
        runner.model_config = SimpleNamespace(max_model_len=512, get_vocab_size=lambda: 32000)
        runner.cache_config = SimpleNamespace(block_size=128, enable_prefix_caching=True)
        runner.parallel_config = SimpleNamespace(cp_kv_cache_interleave_size=4)
        runner.vllm_config = SimpleNamespace(speculative_config=None)
        runner.offload_config = SimpleNamespace(uva=SimpleNamespace(cpu_offload_gb=0))
        runner.input_batch = SimpleNamespace(logitsprocs=MagicMock())
        attention_backend = SimpleNamespace(get_supported_kernel_block_sizes=lambda: [128, 64])
        runner.attn_groups = [[SimpleNamespace(backend=attention_backend)]]

        attention_spec = AttentionSpec(
            block_size=128,
            num_kv_heads=2,
            head_size=64,
            dtype=torch.float16,
        )
        mamba_spec = MambaSpec(
            block_size=128,
            shapes=((16,),),
            dtypes=(torch.float16,),
            mamba_cache_mode="align",
            num_speculative_blocks=2,
        )
        kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=attention_spec),
                SimpleNamespace(kv_cache_spec=mamba_spec),
            ]
        )

        with (
            patch("vllm_ascend._310p.model_runner_310p.NPUInputBatch") as mock_input_batch,
            patch(
                "vllm_ascend._310p.model_runner_310p.get_decode_context_model_parallel_world_size",
                return_value=1,
            ),
        ):
            runner.may_reinitialize_input_batch(kv_cache_config)

        kwargs = mock_input_batch.call_args.kwargs
        self.assertEqual(kwargs["block_sizes"], [128, 128])
        self.assertEqual(kwargs["kernel_block_sizes"], [[128, 64], [0]])
        self.assertEqual(kwargs["max_num_blocks_per_req"], [4, 6])
        self.assertIs(kwargs["kv_cache_groups"], kv_cache_config.kv_cache_groups)
        self.assertEqual(kwargs["cp_kv_cache_interleave_size"], 4)
