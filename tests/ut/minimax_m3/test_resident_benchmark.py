# SPDX-License-Identifier: Apache-2.0
"""Exercise bounded benchmarking and cache-preserving restoration."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from tools.glm_perf.resident_control import Control
from tools.minimax_m3.resident import candidate_source
from tools.minimax_m3.resident_benchmark import benchmark_replacements, run_benchmark
from vllm_ascend._310p.ops.minimax_m3 import PackedM3Experts


class Client:
    def __init__(self, failure=None):
        control = Control("0" * 32, "direct-target", "m3_candidate", candidate_source(native_name="base"))
        self.state = dict(
            rank=0,
            pid=123,
            generation=control.generation,
            candidate=control.candidate,
            digest=control.digest,
            native_failed=False,
            graphs_dirty=False,
            weight_storage_digest="weights",
        )
        self.initial = copy.deepcopy(self.state)
        self.cache = ["prefix and physical cache"]
        self.calls = []
        self.failure = failure
        self.paused = False

    def rpc(self, method, *args):
        self.calls.append(method)
        if method == "resident_status":
            return [copy.deepcopy(self.state)]
        if method == "resident_native_prepare":
            return [{"native_digest": "native"}]
        if method == "resident_native_load":
            if self.failure == "native":
                self.state["native_failed"] = True
                raise RuntimeError("native load failed")
            return [{"validation": {"passed": True}}]
        if method == "resident_prepare":
            self.pending = Control.from_dict(json.loads(args[0]))
            return [{"digest": self.pending.digest}]
        if method == "resident_apply":
            self.state.update(
                generation=self.pending.generation, candidate=self.pending.candidate, digest=self.pending.digest
            )
            return [copy.deepcopy(self.state)]
        if method == "resident_capture":
            if self.failure:
                raise RuntimeError("benchmark RPC failed")
            return [{"benchmark": {"passed": True, "kv_storage_unchanged": True}}]
        raise AssertionError(method)

    def request(self, path, method="POST"):
        self.calls.append(path)
        if path == "/is_paused":
            return {"is_paused": self.paused}
        assert path == "/pause?mode=wait&clear_cache=false"
        self.paused = True
        return {"status": "paused"}

    def resume(self):
        self.paused = False


@pytest.mark.parametrize("failure", [None, "capture"])
def test_restoration_preserves_generation_and_cache_even_on_failure(failure):
    client = Client(failure)
    manifest = SimpleNamespace(name="candidate", digest="native", payload="manifest")
    if failure:
        with pytest.raises(RuntimeError, match="benchmark RPC failed"):
            run_benchmark(client, manifest, "base", 2)
    else:
        assert run_benchmark(client, manifest, "base", 2)["passed"]
    assert client.state == client.initial
    assert client.cache == ["prefix and physical cache"]
    assert not client.paused
    assert "resident_reset" not in client.calls
    assert "/pause?mode=wait&clear_cache=false" in client.calls


def test_unknown_installed_generation_is_not_replaced():
    client = Client()
    client.state["digest"] = "unrelated implementation"
    with pytest.raises(ValueError, match="native-only baseline"):
        run_benchmark(client, SimpleNamespace(name="candidate"), "base", 1)
    assert not client.paused


@pytest.mark.parametrize("different", [False, True])
def test_worker_benchmark_compares_outputs_without_touching_kv(monkeypatch, different):
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            synchronize=lambda: None,
            mem_get_info=lambda: (100, 200),
            memory_allocated=lambda: 50,
            memory_reserved=lambda: 70,
        ),
        raising=False,
    )
    monkeypatch.setattr("tools.minimax_m3.byte_reuse_probe.byte_reuse_probe", lambda *args: {"passed": True})
    bank = PackedM3Experts(32, 32, 2, 1, 0, 2, prefix="layer")
    cache = torch.ones(3, 5)
    worker = SimpleNamespace(
        _m3_idle=lambda: None,
        model_runner=SimpleNamespace(model=bank, kv_caches=[cache]),
        resident_status=lambda: {"rank": 0},
    )

    def original(x, codes, scales, ends):
        return x.mean(1, keepdim=True).expand(x.shape[0], codes.shape[1]).contiguous()

    def proposed(*args):
        return original(*args) + (1 if different else 0)

    changes = benchmark_replacements({"base": original, "new": proposed}, "base", "new", 1)
    report = changes["vllm_ascend._310p.worker_310p:NPUWorker310.resident_capture"](worker)["benchmark"]
    assert report["passed"] is (not different)
    assert report["kv_storage_unchanged"]
    assert len(report["cases"]) == 15
    torch.testing.assert_close(cache, torch.ones_like(cache))


def test_native_failure_keeps_server_paused():
    client = Client("native")
    manifest = SimpleNamespace(name="candidate", digest="native", payload="manifest")
    with pytest.raises(RuntimeError, match="native state is unhealthy"):
        run_benchmark(client, manifest, "base", 2)
    assert client.paused


def qualified_fixture():
    client = Client()
    client.state["native_loaded"] = {"candidate": {"native_digest": "native", "validation": {"passed": True}}}
    proof = {
        "passed": True,
        "kv_storage_unchanged": True,
        "byte_reuse": {"passed": True},
        "cases": [{"passed": True, "max_abs": 0.0}],
    }
    evidence = {
        "passed": True,
        "baseline": "base",
        "candidate": "candidate",
        "after": [copy.deepcopy(client.state)],
        "workers": [{"rank": 0, "benchmark": proof}],
    }
    return client, SimpleNamespace(name="candidate", digest="native"), evidence


def test_exact_native_activation_preserves_caches():
    from tools.minimax_m3.activate_equivalent import activate

    client, manifest, evidence = qualified_fixture()
    report = activate(client, manifest, evidence)
    assert report["passed"] and not report["cache_reset"]
    assert client.cache == ["prefix and physical cache"]
    assert "resident_reset" not in client.calls
    assert client.state["generation"] != evidence["after"][0]["generation"]
    assert not client.paused


@pytest.mark.parametrize("damage", ["digest", "pid", "output", "byte_probe"])
def test_equivalent_activation_rejects_stale_or_inexact_evidence(damage):
    from tools.minimax_m3.activate_equivalent import activate

    client, manifest, evidence = qualified_fixture()
    if damage == "digest":
        manifest.digest = "different binary"
    elif damage == "pid":
        client.state["pid"] += 1
    elif damage == "output":
        evidence["workers"][0]["benchmark"]["cases"][0]["max_abs"] = 0.01
    else:
        evidence["workers"][0]["benchmark"]["byte_reuse"]["passed"] = False
    with pytest.raises(ValueError):
        activate(client, manifest, evidence)
    assert not client.paused


def test_attention_experiment_rejects_tight_memory_before_allocations(monkeypatch):
    from tools.minimax_m3.resident_attention_benchmark import benchmark_attention

    monkeypatch.setattr(torch, "npu", SimpleNamespace(mem_get_info=lambda: (1024**3, 46 * 1024**3)), raising=False)
    with pytest.raises(ValueError, match="at least 2 GiB"):
        benchmark_attention("npu:0", 1)


def test_native_activation_restores_original_after_partial_apply():
    from tools.minimax_m3.activate_equivalent import activate

    client, manifest, evidence = qualified_fixture()
    rpc = client.rpc
    failed = False

    def fail_once(method, *args):
        nonlocal failed
        result = rpc(method, *args)
        if method == "resident_apply" and not failed:
            failed = True
            raise RuntimeError("lost activation acknowledgment")
        return result

    client.rpc = fail_once
    with pytest.raises(RuntimeError, match="lost activation"):
        activate(client, manifest, evidence)
    assert client.state == evidence["after"][0]
    assert not client.paused
