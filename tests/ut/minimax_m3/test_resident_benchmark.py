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
