# SPDX-License-Identifier: Apache-2.0
"""Offline gates for the actual M3 eager arithmetic and checkpoint path."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from vllm.config.load import LoadConfig

from tools.minimax_m3.attention_candidate import paged_sparse_attention_batched_heads
from tools.minimax_m3.checkpoint import dense_weights, load_expert_banks, read_manifest
from tools.minimax_m3.export_w4 import MANIFEST, export_checkpoint
from tools.minimax_m3.probe_w4 import quantize_chunk
from vllm_ascend._310p.ops.minimax_m3 import (
    W4_LAYOUT,
    PackedM3Experts,
    clipped_swiglu,
    paged_sparse_attention,
    reference_grouped_mm,
    route_experts,
    unpack_w4,
    validate_runtime,
)
from vllm_ascend.model_loader.minimax_m3_w4 import MiniMaxM3W4Loader


@pytest.fixture(autouse=True)
def small_cpu_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def make_bank(ep_size=1, ep_rank=0):
    bank = PackedM3Experts(32, 64, 4, ep_size, ep_rank, 2, prefix="model.layers.0.block_sparse_moe.experts")
    generator = torch.Generator().manual_seed(15)
    for projection, shape in (("w1", (64, 32)), ("w3", (64, 32)), ("w2", (32, 64))):
        for expert in range(4):
            weight = torch.randn(shape, generator=generator) * 0.1
            if bank.local_expert_offset <= expert < bank.local_expert_offset + bank.num_local_experts:
                codes, scales, _ = quantize_chunk(weight, "minmax")
                slot = expert - bank.local_expert_offset
                getattr(bank, projection + "_codes")[slot].copy_(codes)
                getattr(bank, projection + "_scales")[slot].copy_(scales)
    return bank


def expert_oracle(bank, x, logits, bias):
    scores = torch.sigmoid(logits.float())
    ids = (scores + bias).argsort(dim=-1, descending=True)[:, : bank.top_k]
    weights = scores.gather(1, ids)
    weights /= weights.sum(1, keepdim=True)
    result = torch.zeros_like(x, dtype=torch.float32)
    for token in range(len(x)):
        for route in range(bank.top_k):
            expert = int(ids[token, route])
            gate = (x[token].float() @ unpack_w4(bank.w1_codes[expert], bank.w1_scales[expert]).T).half().float()
            up = (x[token].float() @ unpack_w4(bank.w3_codes[expert], bank.w3_scales[expert]).T).half().float()
            gate = gate.clamp(max=7)
            act = (gate / (1 + torch.exp(-1.702 * gate)) * (up.clamp(-7, 7) + 1)).half()
            down = (act.float() @ unpack_w4(bank.w2_codes[expert], bank.w2_scales[expert]).T).half()
            result[token] += down.float() * weights[token, route] * 2
    return result


def test_activation_is_clipped_oai_not_silu():
    gate = torch.tensor([-20.0, -1, 0, 1, 20])
    up = torch.tensor([-20.0, 0, 1, 2, 20])
    expected_gate = gate.clamp(max=7)
    expected = expected_gate / (1 + torch.exp(-1.702 * expected_gate)) * (up.clamp(-7, 7) + 1)
    torch.testing.assert_close(clipped_swiglu(gate, up), expected)
    assert clipped_swiglu(gate, up)[0] > 0


def test_router_bias_only_changes_selection():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    weights, ids = route_experts(logits, torch.tensor([10.0, 9.0, 0, 0]), 2)
    assert ids.tolist() == [[0, 1]]
    expected = logits.sigmoid()[:, :2]
    torch.testing.assert_close(weights, expected / expected.sum(1, keepdim=True))


@pytest.mark.parametrize("route_cap", [128, 512, 2048])
@pytest.mark.parametrize("tokens", [0, 1, 17, 65, 257, 1025])
def test_packed_moe_ep_sum_and_chunk_boundaries(tokens, route_cap):
    torch.manual_seed(7)
    x = torch.randn(tokens, 32).half()
    logits = torch.randn(tokens, 4)
    bias = torch.tensor([0.03, -0.04, 0.07, 0.01])
    full = make_bank()
    full.max_grouped_routes = route_cap
    actual = full(x, logits, bias)
    if tokens:
        torch.testing.assert_close(actual, expert_oracle(full, x, logits, bias), atol=0.003, rtol=0.003)
    shards = [make_bank(2, rank) for rank in range(2)]
    for shard in shards:
        shard.max_grouped_routes = route_cap
    distributed = sum(shard(x, logits, bias) for shard in shards)
    torch.testing.assert_close(actual, distributed, atol=1e-6, rtol=1e-6)


def test_no_local_routes_and_empty_groups():
    x = torch.randn(3, 32).half()
    logits = torch.tensor([[10.0, 9.0, -10.0, -9.0]]).expand(3, -1)
    assert torch.count_nonzero(make_bank(2, 1)(x, logits)) == 0


@pytest.mark.parametrize("rank", [0, 1])
def test_route_count_reductions_preserve_int64_native_boundaries(rank):
    bank = make_bank(2, rank)
    bank.max_grouped_routes = 2048
    boundaries = []

    def project(x, codes, scales, ends):
        assert ends.dtype == torch.int64 and ends.is_contiguous()
        boundaries.append(ends.tolist())
        return x.new_zeros((x.shape[0], codes.shape[1]))

    bank.projector = project
    logits = torch.tensor([[10.0, 9.0, -10.0, -9.0]]).expand(1025, -1)
    bank(torch.zeros(1025, 32).half(), logits)
    assert boundaries == ([[1024, 2048]] * 3 + [[1, 2]] * 3 if rank == 0 else [[0, 0]] * 6)


def attention_oracle(q, k, v, iq, ik, *, prefix, block_size, topk, local, init):
    """Independent scalar token/head implementation, no paged runtime helpers."""
    result = torch.empty_like(q)
    repeats = q.shape[1] // k.shape[1]
    for token in range(q.shape[0]):
        pos = prefix + token
        for head in range(k.shape[1]):
            scores = []
            for block in range(pos // block_size + 1):
                end = min((block + 1) * block_size, pos + 1)
                score = (ik[block * block_size : end].float() @ iq[token, head].float()).max().item()
                if block > pos // block_size - local:
                    score = 1e29
                if block < init:
                    score = 1e30
                scores.append(score)
            # Use torch topk for the same tie contract, including forced blocks.
            chosen = torch.tensor(scores).topk(min(topk, len(scores))).indices.tolist()
            positions = [
                i for block in chosen for i in range(block * block_size, min((block + 1) * block_size, pos + 1))
            ]
            for qhead in range(head * repeats, (head + 1) * repeats):
                logits = k[positions, head].float() @ q[token, qhead].float() / q.shape[-1] ** 0.5
                result[token, qhead] = logits.softmax(0) @ v[positions, head].float()
    return result


@pytest.mark.parametrize(
    "prefix,length,topk,local,init",
    [
        (0, 35, 2, 1, 0),
        (19, 18, 2, 0, 0),
        (71, 1, 3, 1, 1),
        (0, 2, 16, 1, 0),
        (0, 129, 64, 1, 0),
        (93, 17, 32, 1, 1),
        (127, 2, 32, 1, 0),
        (19, 129, 4, 1, 1),
    ],
)
@pytest.mark.parametrize("attention_impl", [paged_sparse_attention, paged_sparse_attention_batched_heads])
def test_paged_sparse_attention_causality_and_head_selection(prefix, length, topk, local, init, attention_impl):
    torch.manual_seed(44)
    block, dim, heads = 4, 8, 2
    total = prefix + length
    pages = (total + block - 1) // block
    k, v = torch.randn(total, heads, dim).half(), torch.randn(total, heads, dim).half()
    ik = torch.randn(total, dim).half()
    q, iq = torch.randn(length + 2, 4, dim).half(), torch.randn(length + 2, heads, dim).half()
    # Main and index caches intentionally use different physical page orders.
    order, index_order = torch.randperm(pages), torch.randperm(pages)
    kc = torch.full((pages, block, heads, dim), torch.nan, dtype=torch.float16)
    vc = torch.full_like(kc, torch.nan)
    ic = torch.full((pages, block, dim), torch.nan, dtype=torch.float16)
    logical = torch.arange(total)
    slots = order[logical // block] * block + logical % block
    islots = index_order[logical // block] * block + logical % block
    kc.flatten(0, 1)[slots[:prefix]] = k[:prefix]
    vc.flatten(0, 1)[slots[:prefix]] = v[:prefix]
    ic.flatten(0, 1)[islots[:prefix]] = ik[:prefix]
    actual = attention_impl(
        q,
        k[prefix:],
        v[prefix:],
        iq,
        ik[prefix:],
        kc,
        vc,
        ic,
        query_starts=(0, length),
        seq_lens=torch.tensor([total]),
        block_table=order[None],
        index_block_table=index_order[None],
        slot_mapping=slots[prefix:],
        index_slot_mapping=islots[prefix:],
        max_seq_len=total,
        topk_blocks=topk,
        scale=dim**-0.5,
        local_blocks=local,
        init_blocks=init,
    )
    expected = attention_oracle(
        q[:length], k, v, iq[:length], ik, prefix=prefix, block_size=block, topk=topk, local=local, init=init
    )
    torch.testing.assert_close(actual[:length], expected, atol=0.002, rtol=0.002)
    assert torch.count_nonzero(actual[length:]) == 0
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(ic.flatten(0, 1)[islots], ik)


@pytest.mark.parametrize("attention_impl", [paged_sparse_attention, paged_sparse_attention_batched_heads])
def test_mixed_decode_prefill_batch_has_no_cross_request_cache_leak(attention_impl):
    torch.manual_seed(18)
    block, dim = 4, 8
    lengths, prefixes = (6, 7), (5, 0)
    main_pages = torch.tensor([[3, 0], [1, 2]])
    index_pages = torch.tensor([[2, 1], [0, 3]])
    kc = torch.full((4, block, 2, dim), torch.nan)
    vc, ic = torch.full_like(kc, torch.nan), torch.full((4, block, dim), torch.nan)
    keys, values, index_keys, queries, index_queries, slots, index_slots, references = ([] for _ in range(8))
    for request, (length, prefix) in enumerate(zip(lengths, prefixes)):
        k, v, ik = torch.randn(length, 2, dim), torch.randn(length, 2, dim), torch.randn(length, dim)
        q, iq = torch.randn(length - prefix, 4, dim), torch.randn(length - prefix, 2, dim)
        positions = torch.arange(length)
        loc = main_pages[request, positions // block] * block + positions % block
        iloc = index_pages[request, positions // block] * block + positions % block
        kc.flatten(0, 1)[loc[:prefix]], vc.flatten(0, 1)[loc[:prefix]] = k[:prefix], v[:prefix]
        ic.flatten(0, 1)[iloc[:prefix]] = ik[:prefix]
        for values_list, tensor in (
            (keys, k[prefix:]),
            (values, v[prefix:]),
            (index_keys, ik[prefix:]),
            (queries, q),
            (index_queries, iq),
            (slots, loc[prefix:]),
            (index_slots, iloc[prefix:]),
        ):
            values_list.append(tensor)
        references.append(attention_oracle(q, k, v, iq, ik, prefix=prefix, block_size=block, topk=1, local=0, init=0))
    output = paged_sparse_attention(
        torch.cat(queries),
        torch.cat(keys),
        torch.cat(values),
        torch.cat(index_queries),
        torch.cat(index_keys),
        kc,
        vc,
        ic,
        query_starts=(0, 1, 8),
        seq_lens=torch.tensor(lengths),
        block_table=main_pages,
        index_block_table=index_pages,
        slot_mapping=torch.cat(slots),
        index_slot_mapping=torch.cat(index_slots),
        max_seq_len=7,
        topk_blocks=1,
        scale=dim**-0.5,
        local_blocks=0,
    )
    torch.testing.assert_close(output, torch.cat(references), atol=1e-6, rtol=1e-5)


@pytest.fixture
def tiny_checkpoint(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    tensors = {}
    torch.manual_seed(22)
    for expert in range(4):
        for projection, shape in (("w1", (64, 32)), ("w2", (32, 64)), ("w3", (64, 32))):
            name = f"language_model.model.layers.0.block_sparse_moe.experts.{expert}.{projection}.weight"
            tensors[name] = (torch.randn(shape) * 0.1).bfloat16()
    tensors["language_model.model.layers.0.block_sparse_moe.gate.weight"] = torch.randn(4, 32)
    tensors["language_model.model.embed_tokens.weight"] = torch.randn(16, 32).bfloat16()
    tensors["language_model.lm_head.weight"] = torch.randn(16, 32).bfloat16()
    save_file(tensors, source / "model.safetensors")
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "model.safetensors")})
    )
    (source / "config.json").write_text(json.dumps({"text_config": {"moe_layer_freq": [1], "num_local_experts": 4}}))
    (source / "tokenizer_config.json").write_text("{}")
    return source, tmp_path / "export", tensors


def test_export_resume_load_inference_end_to_end(tiny_checkpoint):
    source, target, tensors = tiny_checkpoint
    assert not export_checkpoint(source, target, rows=32, method="minmax", max_tensors=2)["complete"]
    assert not (target / MANIFEST).exists()
    assert not (target / "config.json").exists()
    report = export_checkpoint(source, target, rows=32, method="minmax")
    assert report["complete"]
    manifest = read_manifest(target)
    assert (target / "tokenizer_config.json").exists()
    assert not (target / "model.safetensors.index.json").exists()
    bank = make_bank()
    assert load_expert_banks(bank, target, manifest) == 24
    dense = dict(dense_weights(target, manifest, start_layer=0, end_layer=1, first_stage=True, last_stage=True))
    embeddings = dense["language_model.model.embed_tokens.weight"]
    gate = dense["language_model.model.layers.0.block_sparse_moe.gate.weight"]
    assert embeddings.dtype == torch.float16 and gate.dtype == torch.float32
    x = embeddings[torch.tensor([1, 2, 3])]
    logits = x.float() @ gate.T
    bias = torch.tensor([0.1, -0.1, 0.2, 0.0])
    actual = (x.float() + bank(x, logits, bias)) @ dense["language_model.lm_head.weight"].float().T
    reference = (x.float() + expert_oracle(bank, x, logits, bias)) @ tensors["language_model.lm_head.weight"].float().T
    torch.testing.assert_close(actual, reference, atol=0.02, rtol=0.003)
    shards = [make_bank(2, rank) for rank in range(2)]
    for shard in shards:
        assert load_expert_banks(shard, target, manifest) == 12
    torch.testing.assert_close(bank(x, logits, bias), sum(s(x, logits, bias) for s in shards), atol=1e-6, rtol=1e-6)
    assert export_checkpoint(source, target, rows=32, method="minmax") == report


def test_corrupt_bundle_and_incomplete_expert_set_rejected(tiny_checkpoint):
    source, target, _ = tiny_checkpoint
    export_checkpoint(source, target, method="minmax")
    manifest = read_manifest(target)
    broken = copy.deepcopy(manifest)
    del broken["weights"][next(name for name, rec in broken["weights"].items() if rec["expert"])]
    broken["tensor_count"] -= 1
    (target / MANIFEST).write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="source index"):
        read_manifest(target)
    record = next(rec for rec in manifest["weights"].values() if rec["expert"])
    payload = target / record["file"]
    data = bytearray(payload.read_bytes())
    data[-1] ^= 1
    payload.write_bytes(data)
    with pytest.raises(ValueError, match="digest"):
        load_expert_banks(make_bank(), target, manifest)
    with pytest.raises(ValueError, match="Corrupt"):
        export_checkpoint(source, target, method="minmax")


def test_registered_loader_checks_dense_parameters(tiny_checkpoint, monkeypatch):
    source, target, _ = tiny_checkpoint
    export_checkpoint(source, target, method="minmax")

    class MiniMaxM3Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.start_layer, self.end_layer = 0, 1
            self.experts = make_bank()
            self.router = torch.nn.Parameter(torch.empty(4, 32))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = MiniMaxM3Model()
            self.report_loaded = True

        def load_weights(self, iterator):
            for name, tensor in iterator:
                if name.endswith("gate.weight"):
                    self.model.router.data.copy_(tensor)
            return {"model.router"} if self.report_loaded else set()

    monkeypatch.setattr(
        "vllm_ascend.model_loader.minimax_m3_w4.get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    loader = MiniMaxM3W4Loader(LoadConfig(load_format="minimax_m3_w4"))
    model, config = Model(), SimpleNamespace(model=str(target))
    loader.download_model(config)
    loader.load_weights(model, config)
    assert torch.isfinite(model.model.router).all()
    model.report_loaded = False
    with pytest.raises(ValueError, match="not initialized"):
        loader.load_weights(model, config)


def test_export_rejects_source_change_and_dense_overflow(tiny_checkpoint):
    source, target, tensors = tiny_checkpoint
    export_checkpoint(source, target, method="minmax", max_tensors=1)
    tensors["language_model.lm_head.weight"].fill_(1e10)
    save_file(tensors, source / "model.safetensors")
    with pytest.raises(ValueError, match="different source"):
        export_checkpoint(source, target, method="minmax")
    with pytest.raises(ValueError, match="overflows"):
        export_checkpoint(source, target.parent / "overflow", method="minmax")


def test_pipeline_dense_filter_and_nonlocal_payloads_are_not_read(tiny_checkpoint):
    source, target, _ = tiny_checkpoint
    export_checkpoint(source, target, method="minmax")
    manifest = read_manifest(target)
    for record in manifest["weights"].values():
        # Delete every peer file. A rank-local load must never touch these.
        if ".experts.2." in record["name"] or ".experts.3." in record["name"]:
            (target / record["file"]).unlink()
    assert load_expert_banks(make_bank(2, 0), target, manifest) == 12
    assert not list(dense_weights(target, manifest, start_layer=1, end_layer=2, first_stage=False, last_stage=False))


def runtime_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.float16,
            enforce_eager=True,
            hf_text_config=SimpleNamespace(
                ascend_minimax_m3_w4=W4_LAYOUT,
                num_local_experts=128,
                scoring_func="sigmoid",
                hidden_act="swigluoai",
                num_key_value_heads=4,
                sparse_attention_config={"sparse_block_size": 128, "sparse_num_index_heads": 4},
            ),
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=1, tensor_parallel_size=2, enable_expert_parallel=True, enable_eplb=False
        ),
        cache_config=SimpleNamespace(block_size=128, cache_dtype="auto"),
        load_config=SimpleNamespace(load_format="minimax_m3_w4"),
        quant_config=None,
        speculative_config=None,
        lora_config=None,
    )


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("model_config", "dtype", torch.bfloat16),
        ("model_config", "enforce_eager", False),
        ("parallel_config", "tensor_parallel_size", 6),
        ("parallel_config", "enable_expert_parallel", False),
        ("parallel_config", "enable_eplb", True),
        ("parallel_config", "data_parallel_size", 2),
        ("parallel_config", "use_sequence_parallel_moe", True),
        ("cache_config", "block_size", 16),
        ("load_config", "load_format", "auto"),
    ],
)
def test_runtime_rejects_unsupported_modes(section, key, value):
    config = runtime_config()
    validate_runtime(config)
    setattr(getattr(config, section), key, value)
    with pytest.raises(ValueError):
        validate_runtime(config)


@pytest.mark.parametrize("int8_layouts", [(), ("w1",), ("w1", "w2", "w3")])
def test_int8_scale_metadata_reuses_original_buffers_without_changing_other_kernels(int8_layouts):
    bank = make_bank()
    bank._m3_int8_layouts = set(int8_layouts)
    seen = []

    def project(x, codes, scales, ends):
        seen.append((codes.data_ptr(), scales.data_ptr(), scales.dtype))
        return reference_grouped_mm(x, codes, scales, ends)

    bank.projector = project
    x = torch.randn(1, 32, generator=torch.Generator().manual_seed(61)).half()
    logits = torch.tensor([[1.0, -2.0, 3.0, 4.0]])
    bias = torch.zeros(4)
    actual = bank(x, logits, bias)
    torch.testing.assert_close(actual, expert_oracle(bank, x, logits, bias), atol=0.003, rtol=0.003)
    assert len(seen) == 3
    by_codes = {pointer: (scale_pointer, dtype) for pointer, scale_pointer, dtype in seen}
    for name in ("w1", "w2", "w3"):
        pointer, dtype = by_codes[getattr(bank, name + "_codes").data_ptr()]
        assert dtype == (torch.float16 if name in int8_layouts else torch.float32)
        if name in int8_layouts:
            assert pointer == getattr(bank, name + "_scales").data_ptr()
