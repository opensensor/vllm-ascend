# SPDX-License-Identifier: Apache-2.0
"""Offline rank ownership, persistent state, and exact result transport gates."""

import ast
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace

import pytest
import torch
from test_pipeline_all_gather import Backend, Network

from tools.glm_perf import indexer_owner


def config(*, tp=2, eager=False, sizes=(1, 4), mode="FULL_DECODE_ONLY", **parallel):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=tp, **parallel),
        speculative_config=None,
        model_config=SimpleNamespace(enforce_eager=eager),
        compilation_config=SimpleNamespace(cudagraph_mode=mode, cudagraph_capture_sizes=list(sizes)),
    )


@pytest.mark.parametrize("tp,owner", [(2, 0), (2, 1), (4, 3)])
def test_valid_local_owners_and_decode_config(tp, owner):
    indexer_owner.validate_config(config(tp=tp), owner)
    indexer_owner.validate_config(config(tp=tp, eager=True, sizes=(), mode="NONE"), owner)


@pytest.mark.parametrize("tp,owner", [(6, 0), (1, 0), (2, 2), (2, -1), (2, True)])
def test_owner_admission_does_not_imply_tp6_support(tp, owner):
    with pytest.raises(ValueError, match="local rank"):
        indexer_owner.validate_config(config(tp=tp), owner)


@pytest.mark.parametrize(
    "updates",
    [
        dict(use_sequence_parallel_moe=True),
        dict(data_parallel_size=2),
        dict(decode_context_parallel_size=2),
        dict(prefill_context_parallel_size=2),
    ],
)
def test_partitioned_input_modes_rejected(updates):
    with pytest.raises(ValueError, match="replicated token"):
        indexer_owner.validate_config(config(**updates), 0)


@pytest.mark.parametrize("field", ["speculative_config", "kv_transfer_config"])
def test_external_state_consumers_rejected(field):
    value = config()
    setattr(value, field, object())
    with pytest.raises(ValueError, match="KV transfer"):
        indexer_owner.validate_config(value, 0)


@pytest.mark.parametrize(
    "sizes,mode",
    [((), "FULL_DECODE_ONLY"), ((1, 9), "FULL_DECODE_ONLY"), ((1,), "FULL"), ((True,), "FULL_DECODE_ONLY")],
)
def test_asymmetric_eager_graph_breaks_rejected(sizes, mode):
    with pytest.raises(ValueError, match="capture sizes"):
        indexer_owner.validate_config(config(sizes=sizes, mode=mode), 0)


@pytest.mark.parametrize(
    "options",
    [
        {"indexer_owner_rank": 0},
        {"indexer_exchange_budget_bytes": 100},
        {"indexer_owner_rank": False, "indexer_exchange_budget_bytes": 100},
        {"indexer_owner_rank": 0, "indexer_exchange_budget_bytes": 0},
        {"indexer_overlap_prefill": True},
    ],
)
def test_explicit_owner_and_memory_reservation_required(options):
    with pytest.raises(ValueError, match="requires|require"):
        indexer_owner.read_options(options)
    assert indexer_owner.read_options({}) is None


def test_overlap_options_are_explicit_and_typed():
    options = dict(indexer_owner_rank=0, indexer_exchange_budget_bytes=12 * 1024**2, indexer_overlap_prefill=True)
    assert indexer_owner.read_options(options) == dict(owner_rank=0, budget_bytes=12 * 1024**2, overlap_prefill=True)
    options["indexer_overlap_prefill"] = 1
    with pytest.raises(ValueError, match="boolean"):
        indexer_owner.read_options(options)


class Indexer(torch.nn.Module):
    def __init__(self, buffer):
        super().__init__()
        self.topk_indices_buffer = buffer
        self.indexer_op = SimpleNamespace(topk_indices_buffer=buffer, capture_safe_selection=True)
        self.calls = 0
        self.cache_state = 0

    def forward(self, hidden_states, qr, positions, rotary_emb):
        self.calls += 1
        self.cache_state += int(positions[-1]) + 1
        rows = hidden_states.shape[0]
        # State-dependent outputs, including an INT32 value inexact in FP32.
        self.topk_indices_buffer[:rows, 0] = positions + self.cache_state
        self.topk_indices_buffer[:rows, 1] = 16_777_217 + self.cache_state
        self.topk_indices_buffer[:rows, 2:] = -1
        return self.topk_indices_buffer


@pytest.mark.parametrize(
    "groups,owner", [(((0, 1), (2, 3), (4, 5)), 0), (((0, 1), (2, 3), (4, 5)), 1), (((5, 1, 4, 2),), 3)]
)
def test_only_one_indexer_executes_per_group_with_persistent_state_and_exact_bytes(groups, owner):
    network = Network()
    workers = []
    reference_states = {group: 0 for group in groups}
    for group in groups:
        for rank, global_rank in enumerate(group):
            backend = Backend(network, global_rank)
            exchange = indexer_owner.IndexerResultExchange(
                group=group,
                rank=rank,
                size=len(group),
                owner_rank=owner,
                max_tokens=640,
                width=4,
                device="cpu",
                budget_bytes=640 * 4 * 4 * len(group),
                backend=backend,
            )
            instance = Indexer(torch.full((640, 4), -987, dtype=torch.int32))
            indexer_owner.bind_indexer(instance, exchange)
            workers.append((group, rank, backend, exchange, instance))

    def warm(worker):
        worker[3].warmup()

    with ThreadPoolExecutor(max_workers=len(workers)) as pool:
        for future in [pool.submit(warm, worker) for worker in workers]:
            future.result(timeout=15)
        pointers = [worker[3].storage.data_ptr() for worker in workers]
        for step, tokens in enumerate((1, 4, 37, 640, 1)):

            def run(worker, tokens=tokens, step=step):
                group, rank, _, _, instance = worker
                # Different stages have distinct positions and indexer state.
                positions = torch.arange(tokens, dtype=torch.int32) + step * 640 + group[0] * 100
                x = torch.full((tokens, 8), float(step))
                before = instance.topk_indices_buffer[tokens:].clone()
                result = instance(x, x, positions, None)
                assert result is instance.topk_indices_buffer
                assert torch.equal(result[tokens:], before)
                return group, result[:tokens].clone(), positions

            outputs = [future.result(timeout=15) for future in [pool.submit(run, worker) for worker in workers]]
            for group in groups:
                members = [(value, positions) for own_group, value, positions in outputs if own_group == group]
                reference_states[group] += int(members[0][1][-1]) + 1
                for value, positions in members:
                    assert torch.equal(value[:, 0], positions + reference_states[group])
                    assert torch.all(value[:, 1] == 16_777_217 + reference_states[group])
                    assert torch.all(value[:, 2:] == -1)
            assert pointers == [worker[3].storage.data_ptr() for worker in workers]
    for _, rank, backend, _, instance in workers:
        assert instance.calls == (5 if rank == owner else 0)
        assert instance.indexer_op.capture_safe_selection is True
        assert all(count == 6 for count in backend.rounds.values())  # warmup plus five forwards
        if rank != owner:
            assert instance.cache_state == 0
        with pytest.raises(ValueError, match="cold-load"):
            indexer_owner.bind_indexer(instance, instance._glm_indexer_result_owner)


def test_stage_install_uses_tp_local_rank_and_one_preallocated_slab(monkeypatch):
    buffer = torch.empty(640, 2176, dtype=torch.int32)
    model = torch.nn.Sequential(Indexer(buffer), Indexer(buffer))
    model.vllm_config = config()
    stub = ModuleType("vllm.distributed")
    stub.get_tp_group = lambda: SimpleNamespace(device_group="stage1_tp", rank_in_group=1, world_size=2)
    monkeypatch.setitem(sys.modules, "vllm.distributed", stub)
    attention = ModuleType("vllm_ascend.models.glm5next.attention")
    attention.Indexer = Indexer
    monkeypatch.setitem(sys.modules, attention.__name__, attention)
    warmups = []
    monkeypatch.setattr(
        indexer_owner.IndexerResultExchange, "warmup", lambda self, **kwargs: warmups.append(self.group)
    )
    report = indexer_owner.install(model, owner_rank=0, budget_bytes=12 * 1024**2)
    assert report["exchange_bytes"] == 11_141_120
    assert report["tp_rank"] == 1 and report["local_owner_rank"] == 0
    assert report["hardware_qualified"] is False
    assert warmups == ["stage1_tp"]
    assert model[0]._glm_indexer_result_owner is model[1]._glm_indexer_result_owner
    assert model[0].indexer_op.capture_safe_selection is True
    with pytest.raises(ValueError, match="twice"):
        indexer_owner.install(model, owner_rank=0, budget_bytes=12 * 1024**2)


@pytest.mark.parametrize("failure", ["cache_buffer", "unprepared_graph", "budget", "different_buffers"])
def test_preflight_rejects_incompatible_stage_without_allocation_or_binding(monkeypatch, failure):
    buffer = torch.empty(640, 2176, dtype=torch.int32)
    model = torch.nn.Sequential(Indexer(buffer), Indexer(buffer))
    model.vllm_config = config()
    attention = ModuleType("vllm_ascend.models.glm5next.attention")
    attention.Indexer = Indexer
    monkeypatch.setitem(sys.modules, attention.__name__, attention)
    if failure == "cache_buffer":
        model[1].indexer_op.topk_indices_buffer = buffer.clone()
    elif failure == "unprepared_graph":
        model[1].indexer_op.capture_safe_selection = False
    elif failure == "different_buffers":
        model[1].topk_indices_buffer = buffer.clone()
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: pytest.fail("allocated during host preflight"))
    with pytest.raises(ValueError):
        indexer_owner.plan_install(model, owner_rank=0, budget_bytes=1 if failure == "budget" else 12 * 1024**2)
    assert all(not hasattr(module, "_glm_indexer_result_owner") for module in model)


@pytest.mark.parametrize("failure", ["shape", "dtype", "capacity", "contiguous"])
def test_exchange_rejects_invalid_input_before_collective(failure):
    backend = SimpleNamespace(all_gather_into_tensor=lambda *args, **kwargs: pytest.fail("invalid collective"))
    exchange = indexer_owner.IndexerResultExchange(
        group=None,
        rank=0,
        size=2,
        owner_rank=0,
        max_tokens=4,
        width=4,
        device="cpu",
        budget_bytes=128,
        backend=backend,
    )
    buffer = torch.zeros(4, 4, dtype=torch.int32)
    tokens = 4
    if failure == "shape":
        buffer = buffer[:, :3]
    elif failure == "dtype":
        buffer = buffer.float()
    elif failure == "capacity":
        tokens = 5
    else:
        buffer = buffer.t()
    with pytest.raises(ValueError, match="contiguous INT32"):
        exchange.share(buffer, tokens)


def test_audit_static_counts_and_per_stage_scratch_are_not_per_layer_allocations():
    from tools.glm_perf.rank_work_audit import audit

    text = dict(
        num_hidden_layers=45,
        layer_types=["deepseek_sparse_attention" if n % 4 == 3 else "linear_attention" for n in range(45)],
        hidden_size=4096,
        q_lora_rank=1536,
        index_n_heads=32,
        index_head_dim=128,
        index_kpool=4,
        index_topk=2048,
    )
    report = audit({"text_config": text}, tp_size=2, pp_size=3, tokens=640)
    assert report["selection_width"] == 2176 and report["indexer_projection_weights"] == 7_471_104
    assert [len(stage["sparse_layers"]) for stage in report["stages"]] == [3, 4, 4]
    assert sum(stage["replicated_indexer_executions"] for stage in report["stages"]) == 22
    assert sum(stage["owner_indexer_executions"] for stage in report["stages"]) == 11
    assert all(stage["exchange_bytes_per_rank"] == 11_141_120 for stage in report["stages"])
    two_card = audit(text, tp_size=4, pp_size=1, tokens=640)
    assert two_card["stages"][0]["replicated_indexer_executions"] == 44
    assert two_card["stages"][0]["exchange_bytes_per_rank"] == 22_282_240


def test_exchange_memory_admission_before_allocation(monkeypatch):
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: pytest.fail("allocated before budget check"))
    with pytest.raises(ValueError, match="reservation"):
        indexer_owner.IndexerResultExchange(
            group=None,
            rank=0,
            size=2,
            owner_rank=0,
            max_tokens=640,
            width=2176,
            device="cpu",
            budget_bytes=1,
        )


@pytest.mark.parametrize("rank", [0, 1])
def test_prefill_async_join_follows_useful_preprocess_and_precedes_indices_read(rank):
    operations = []
    selected = torch.arange(37 * 4, dtype=torch.int32).reshape(37, 4)
    buffer = selected.clone() if rank == 0 else torch.full((37, 4), -999, dtype=torch.int32)

    def gather(output, source, *, group, async_op=False):
        operations.append("post_async" if async_op else "synchronous")

        def wait():
            operations.append("join")
            owner_bytes = selected.view(torch.int8).reshape(-1)[: source.numel()]
            output[: owner_bytes.numel()].copy_(owner_bytes)
            return True

        if async_op:
            return SimpleNamespace(wait=wait)
        wait()

    exchange = indexer_owner.IndexerResultExchange(
        group=None,
        rank=rank,
        size=2,
        owner_rank=0,
        max_tokens=37,
        width=4,
        device="cpu",
        budget_bytes=37 * 4 * 4 * 2,
        backend=SimpleNamespace(all_gather_into_tensor=gather),
        overlap_prefill=True,
    )
    # Use a preselected owner input to inspect publish timing independently of
    # indexer math. Reuse while a previous collective is pending is forbidden.
    assert exchange.share(buffer, 37) is buffer
    if rank == 1:
        assert torch.all(buffer == -999)
    with pytest.raises(RuntimeError, match="next layer"):
        exchange.share(buffer, 37)

    def local_preprocess(*args, **kwargs):
        operations.append("local_Q_KV_preprocess")
        return "local_results"

    frontend = SimpleNamespace(_mla_preprocess=local_preprocess)
    indexer_owner.bind_preprocess(frontend, exchange)
    assert frontend._mla_preprocess() == "local_results"
    assert operations == ["post_async", "local_Q_KV_preprocess", "join"]
    assert torch.equal(buffer, selected) and exchange.pending is None
    exchange.finish()  # no extra wait or collective
    operations.clear()
    exchange.share(buffer, 4)
    assert operations == ["synchronous", "join"] and exchange.pending is None


def test_prefill_overlap_admits_only_complete_310p_frontend_mapping(monkeypatch):
    class Frontend:
        def __init__(self, indexer):
            self.glm_indexer = indexer

        def _mla_preprocess(self):
            return "local"

    class Wrapper(torch.nn.Module):
        def __init__(self, indexer):
            super().__init__()
            self.indexer = indexer
            self.impl = Frontend(indexer)

    attention = ModuleType("vllm_ascend.models.glm5next.attention")
    attention.Indexer = Indexer
    backend_module = ModuleType("vllm_ascend._310p.attention.mla_v1_310")
    backend_module.AscendMLAImpl310 = Frontend
    monkeypatch.setitem(sys.modules, attention.__name__, attention)
    monkeypatch.setitem(sys.modules, backend_module.__name__, backend_module)
    stub = ModuleType("vllm.distributed")
    stub.get_tp_group = lambda: SimpleNamespace(device_group="stage_tp", rank_in_group=0, world_size=2)
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    monkeypatch.setattr(indexer_owner.IndexerResultExchange, "warmup", lambda *args, **kwargs: None)
    buffer = torch.empty(640, 2176, dtype=torch.int32)
    model = torch.nn.Sequential(Wrapper(Indexer(buffer)), Wrapper(Indexer(buffer)))
    model.vllm_config = config()
    model.vllm_config.scheduler_config = SimpleNamespace(max_num_seqs=4)
    report = indexer_owner.install(model, owner_rank=0, budget_bytes=12 * 1024**2, overlap_prefill=True)
    assert report["overlap_prefill"] is True
    assert all(
        wrapper.impl._glm_indexer_prefill_exchange is wrapper.indexer._glm_indexer_result_owner for wrapper in model
    )
    assert model[0].impl._glm_indexer_prefill_exchange is model[1].impl._glm_indexer_prefill_exchange
    invalid = torch.nn.Sequential(Indexer(buffer))
    invalid.vllm_config = model.vllm_config
    with pytest.raises(ValueError, match="frontend consumer"):
        indexer_owner.plan_install(invalid, owner_rank=0, budget_bytes=12 * 1024**2, overlap_prefill=True)
    invalid.vllm_config.scheduler_config.max_num_seqs = 9
    with pytest.raises(ValueError, match="eight decode"):
        indexer_owner.plan_install(invalid, owner_rank=0, budget_bytes=12 * 1024**2, overlap_prefill=True)


def test_production_310p_frontend_completes_local_projections_before_owned_indices_publish():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/_310p/attention/mla_v1_310.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendMLAImpl310")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_mla_preprocess")
    operations = []
    namespace = dict(
        torch=torch,
        wait_for_kv_layer_from_connector=lambda *args: operations.append("cache_wait"),
        notify_kv_cache_written=lambda *args: operations.append("KV_written"),
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    selected = torch.arange(37 * 4, dtype=torch.int32).reshape(37, 4)
    buffer = torch.full_like(selected, -1)

    def gather(output, source, *, group, async_op):
        assert async_op
        operations.append("result_posted")

        def wait():
            operations.append("result_joined")
            output[: selected.numel() * 4].copy_(selected.view(torch.int8).reshape(-1))

        return SimpleNamespace(wait=wait)

    exchange = indexer_owner.IndexerResultExchange(
        group=None,
        rank=1,
        size=2,
        owner_rank=0,
        max_tokens=37,
        width=4,
        device="cpu",
        budget_bytes=37 * 4 * 4 * 2,
        backend=SimpleNamespace(all_gather_into_tensor=gather),
        overlap_prefill=True,
    )

    def local_projection(label):
        operations.append(label)
        assert torch.all(buffer == -1)  # current result has not been published
        return label

    frontend = SimpleNamespace(
        glm_indexer=lambda *args: exchange.share(buffer, 37),
        fused_qkv_a_proj=lambda x: (torch.zeros(37, 2), None),
        q_a_layernorm=lambda x: x,
        q_lora_rank=1,
        kv_lora_rank=1,
        qk_rope_head_dim=0,
        mla_preprocess_decode=lambda *args: local_projection("decode_Q_KV"),
        mla_preprocess_prefill=lambda *args: local_projection("prefill_Q_KV"),
    )
    frontend._mla_preprocess = MethodType(namespace["_mla_preprocess"], frontend)
    indexer_owner.bind_preprocess(frontend, exchange)
    metadata = SimpleNamespace(
        decode=SimpleNamespace(input_positions=torch.arange(1)),
        prefill=SimpleNamespace(input_positions=torch.arange(1, 37)),
        num_actual_tokens=37,
        num_decodes=1,
        num_prefills=1,
    )
    result = frontend._mla_preprocess("layer", torch.zeros(37, 1), None, metadata)
    assert result == ("decode_Q_KV", "prefill_Q_KV")
    assert operations == ["result_posted", "cache_wait", "decode_Q_KV", "prefill_Q_KV", "KV_written", "result_joined"]
    assert torch.equal(buffer, selected) and exchange.pending is None
