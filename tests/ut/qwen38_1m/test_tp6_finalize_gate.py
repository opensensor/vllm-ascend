# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from tools.qwen4exp import tp6_finalize_gate as gate
from tools.qwen4exp import tp6_resident_fusion_gate as fusion_gate


def test_snapshot_tracks_graph_cache_storage_and_checkpoint_ownership():
    tensor = torch.zeros(2, 3)
    tier = SimpleNamespace(
        _resident={7: 2},
        _device_archive_resident={9: 0},
        _host={},
        layer_states=((tensor,),),
        _device_archive=(),
        cache_status=lambda: {"resident": 1},
    )
    wrapper = SimpleNamespace(entries={"bucket": object()}, graph_pool=(1, 2))
    runner = SimpleNamespace(kv_caches=[tensor], _prefix_mamba_tiers={0: tier})
    worker = SimpleNamespace(model_runner=runner, _resident_wrappers=lambda: [wrapper])
    before = gate.worker_snapshot(worker)
    assert gate.worker_snapshot(worker) == before
    tier._resident[7] = 3
    assert gate.worker_snapshot(worker) != before
    tier._resident[7] = 2
    runner.kv_caches = [tensor.clone()]
    assert gate.worker_snapshot(worker) != before
    runner.kv_caches = [tensor]
    wrapper.entries["bucket"] = object()
    assert gate.worker_snapshot(worker) != before


def test_snapshot_does_not_claim_tensor_content_verification():
    tensor = torch.zeros(2, 3)
    metadata = gate.storage_metadata(tensor)
    tensor.add_(1)
    assert gate.storage_metadata(tensor) == metadata
    with pytest.raises(TypeError, match="unsupported cache metadata"):
        gate.storage_metadata(object())


def test_bitwise_comparison_detects_signed_zero():
    actual, expected = torch.tensor([0.0]), torch.tensor([-0.0])
    result = gate.compare(actual, expected)
    assert not result["bitwise"]
    assert result["unequal"] == 0


@pytest.mark.parametrize("kind,entry", [("hc", "make_mix"), ("gate_pack", "GatePack")])
def test_followup_bundles_run_without_frozen_runtime_imports(kind, entry):
    namespace = {}
    exec(fusion_gate.engine_source(dict(kind=kind)), namespace)
    resource = {}
    exec(namespace["RESOURCE_SOURCE"], resource)
    assert entry in resource
    worker = {}
    exec(namespace["GATE_SOURCE"], worker)
    assert "run_gate" in worker and "worker_snapshot" in worker
    assert worker["run_gate"].__globals__ is worker


def test_pair_comparison_includes_dtype_shape_and_all_components():
    value = torch.tensor([1.0, 2.0])
    assert fusion_gate.exact_pair((value, value.half()), (value.clone(), value.half()))
    assert not fusion_gate.exact_pair((value, value), (value, value + 1))
    assert not fusion_gate.exact_pair(value, value.half())


def test_dynamic_method_metadata_does_not_abort_completed_hardware_gate():
    namespace = {}
    exec(compile("def mix(): pass", "<resident-candidate>", "exec"), namespace)
    assert fusion_gate.function_metadata(namespace["mix"]) == dict(name="mix", file="<resident-candidate>", source=None)


def test_projection_clone_requires_exact_live_source():
    def unqualified(self, inputs, weights, ids):
        return inputs

    with pytest.raises(ValueError, match="unqualified live"):
        gate.projection_only(unqualified)


def test_private_bundle_reuses_resource_without_loading_kernels():
    namespace = {}
    exec(gate.resource_source(), namespace)
    assert "Finalize" in namespace
    assert "GatePack" not in namespace
    assert "load_resources" not in namespace


def test_engine_candidate_cannot_apply_and_requires_draining():
    namespace = {}
    exec(gate.engine_source({}), namespace)
    candidate = namespace["Candidate"]()
    with pytest.raises(RuntimeError, match="cannot install"):
        candidate.apply(None)
    with pytest.raises(RuntimeError, match="drain engine"):
        candidate.validate(SimpleNamespace(_pause_complete=lambda: False))


def test_completed_numerical_rejection_allows_discard_and_resume():
    # A clean arithmetic mismatch is distinct from an uncertain native failure.
    pool = SimpleNamespace(
        cached_block_hash_to_block=SimpleNamespace(_cache={b"hash": SimpleNamespace(block_id=2)}),
        cached_block_hashes_by_block={2: {b"hash"}},
        blocks=[SimpleNamespace(block_id=2, ref_cnt=0, block_hash=b"hash")],
    )
    workers = [dict(rank=i, passed=True, retained_exact=False) for i in range(6)]
    engine = SimpleNamespace(
        _pause_complete=lambda: True,
        batch_queue=[],
        scheduler=SimpleNamespace(
            get_num_unfinished_requests=lambda: 0, kv_cache_manager=SimpleNamespace(block_pool=pool)
        ),
        model_executor=SimpleNamespace(collective_rpc=lambda operation: workers),
        vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=6)),
    )
    namespace = {}
    exec(gate.engine_source({}), namespace)
    result = namespace["Candidate"]().validate(engine)
    assert result["passed"] and not result["installed"]
    assert result["before"] == result["after"]
    workers[0]["passed"] = False
    assert not namespace["Candidate"]().validate(engine)["passed"]
