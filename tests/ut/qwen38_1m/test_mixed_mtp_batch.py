# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute production batching methods with CPU tensors and kernel doubles."""

import ast
from copy import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def load_method(path, cls, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls)
    method = next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.parse("from __future__ import annotations")
    module.body.append(method)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("compact_argmax", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 6])
@pytest.mark.parametrize("full_graph", [False, True])
@pytest.mark.parametrize("k", [1, 2, 4])
def test_drafts_compact_after_first_pass_but_preserve_graph_bucket(batch_size, full_graph, k, compact_argmax):
    """Prefill 256 rows -> B rows; a captured verify bucket stays padded."""
    first_rows = batch_size * (k + 1) if full_graph else 256
    context = SimpleNamespace(cudagraph_runtime_mode="full" if full_graph else "none", attn_metadata=None)
    namespace = {
        "torch": torch,
        "lmhead_tp_enable": lambda: False,
        "try_qwen4exp_draft_argmax": lambda owner, hidden, **kwargs: hidden[:, 0].long() if compact_argmax else None,
        "get_ascend_config": lambda: SimpleNamespace(enable_reduce_sample=True),
        "get_forward_context": lambda: context,
        "CUDAGraphMode": SimpleNamespace(FULL="full"),
        "_EXTRA_CTX": SimpleNamespace(),
        "_split_draft_outputs": lambda x: (x, x),
    }
    method = load_method(
        "vllm_ascend/spec_decode/llm_base_proposer.py", "AscendSpecDecodeBaseProposer", "_run_merged_draft", namespace
    )
    calls = []

    def model(**kwargs):
        expected = first_rows if not calls or full_graph else batch_size
        assert kwargs["input_ids"].shape[0] == expected
        assert kwargs["positions"].shape[0] == expected
        assert kwargs["hidden_states"].shape == (expected, 8)
        calls.append(expected)
        return kwargs["hidden_states"] + 1

    def logits(hidden):
        assert not compact_argmax, "compact proposals must skip full-vocabulary logits"
        return hidden

    model.compute_logits = logits

    proposer = SimpleNamespace(
        runner=None,
        method="mtp",
        model=model,
        _share_mtp_indices=False,
        input_ids=torch.arange(first_rows),
        positions=torch.arange(first_rows),
        hidden_states=torch.zeros(first_rows, 8),
        pass_hidden_states_to_model=True,
        num_speculative_tokens=k,
        parallel_drafting=False,
        device="cpu",
        uses_mrope=False,
        use_cuda_graph=True,
        use_compress=False,
        supports_mm_inputs=False,
        arange=torch.arange(first_rows),
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=4096)),
        compute_draft_token_ids=lambda hidden, sampling: (hidden[:, 0].long(), None),
        _sample_draft_from_logits=lambda logits, sampling: (logits[:, 0].long(), None),
    )
    proposer._get_positions = lambda n: proposer.positions[:n]
    proposer._set_positions = lambda n, p: proposer.positions[:n].copy_(p)
    indices = torch.arange(batch_size) * (first_rows // batch_size) + first_rows // batch_size - 1
    result = method(proposer, first_rows, batch_size, indices, None, None, [object() for _ in range(k)], first_rows)
    assert result.shape == (batch_size, k)
    assert result.is_contiguous(), "draft rows must be packed before asynchronous CPU readback"
    assert calls == [first_rows] + [first_rows if full_graph else batch_size] * (k - 1)
    assert torch.equal(result, torch.arange(1, k + 1).expand(batch_size, -1))


@pytest.mark.parametrize("spec_indices", [[0, 1, 2], [3, 4, 5], [0, 2, 4]])
@pytest.mark.parametrize("has_state", [False, True])
def test_mixed_gdn_routes_and_merges_without_cross_request_state(spec_indices, has_state):
    method = load_method(
        "vllm_ascend/models/qwen4_exp/model.py",
        "_GDNAttention",
        "_native_mixed_attention",
        {"torch": torch, "copy": copy},
    )
    spec_indices = torch.tensor(spec_indices)
    non_indices = torch.tensor([i for i in range(6) if i not in spec_indices.tolist()])
    metadata = SimpleNamespace(
        num_prefills=1,
        num_prefill_tokens=3,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=1,
        num_spec_decode_tokens=3,
        num_actual_tokens=6,
        spec_sequence_masks=torch.tensor([True, False]),
        spec_decode_metadata=object(),
        spec_token_indx=spec_indices,
        non_spec_token_indx=non_indices,
        spec_state_indices_tensor=torch.tensor([[7, 8, 9]]),
        non_spec_state_indices_tensor=torch.tensor([21]),
        spec_query_start_loc=torch.tensor([0, 3]),
        non_spec_query_start_loc=torch.tensor([0, 3]),
        has_initial_state=torch.tensor([has_state]),
    )
    mixed = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    a, b = torch.arange(6.0).reshape(6, 1), torch.arange(6.0).reshape(6, 1) + 20
    calls = []

    def conv(x, states, query, initial, *, metadata):
        spec = metadata.spec_sequence_masks is not None
        assert (metadata.spec_decode_metadata is not None) == spec
        assert metadata.num_prefills == (0 if spec else 1)
        assert metadata.num_spec_decodes == (1 if spec else 0)
        assert metadata.num_actual_tokens == 3
        torch.testing.assert_close(states, torch.tensor([[7, 8, 9]]) if spec else torch.tensor([21]))
        assert initial is None if spec else initial.tolist() == [has_state]
        calls.append(spec)
        return x

    def delta(q, k, v, g, beta, view, states, query, initial):
        indices = spec_indices if view.spec_sequence_masks is not None else non_indices
        torch.testing.assert_close(g.squeeze(0), a[indices])
        torch.testing.assert_close(beta.squeeze(0), b[indices])
        return v + (100 if view.spec_sequence_masks is not None else 200)

    layer = SimpleNamespace(
        params=SimpleNamespace(head_k_dim=1, head_v_dim=1),
        num_k_heads=1,
        num_v_heads=1,
        key_dim=1,
        value_dim=1,
        compute_dtype=torch.float32,
        _native_gating=lambda a, b: (a.unsqueeze(0), b.unsqueeze(0)),
        _stateful_short_conv=conv,
        _native_delta_rule=delta,
    )
    expected = mixed[:, 2].reshape(6, 1, 1).clone()
    expected[spec_indices] += 100
    expected[non_indices] += 200
    torch.testing.assert_close(method(layer, mixed, a, b, metadata), expected)
    views = metadata._qwen4exp_mixed_views
    torch.testing.assert_close(method(layer, mixed, a, b, metadata), expected)
    assert metadata._qwen4exp_mixed_views is views  # shared across layers, not requests/steps
    assert metadata.num_prefills == 1 and metadata.spec_sequence_masks is not None
    assert calls == [True, False, True, False]
