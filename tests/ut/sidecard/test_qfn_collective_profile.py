# SPDX-License-Identifier: Apache-2.0
import json
import sys
from types import SimpleNamespace

import pytest

from tools.sidecard import qfn_collective_benchmark as benchmark
from tools.sidecard import qfn_collective_profile as profile


def test_profile_preserves_qualified_execution_and_disables_replay_observers(tmp_path):
    config = dict(execution_mode="graph", graph_replay_tracking=False)
    options = profile.engine_options(config, tmp_path)
    assert options.pop("worker_extension_cls").endswith("QfnProfileWorkerExtension")
    recorder = options.pop("profiler_config")
    expected = benchmark.engine_options(config, graph_profile="c1-c2")
    expected.pop("worker_extension_cls")
    assert options == expected
    assert recorder["delay_iterations"] == 8 and recorder["max_iterations"] == 8
    assert recorder["ignore_frontend"] is True


@pytest.mark.parametrize(
    "config", [{}, {"execution_mode": "eager"}, {"execution_mode": "graph", "graph_replay_tracking": True}]
)
def test_rejects_wrong_execution_mode(config, tmp_path):
    with pytest.raises(ValueError, match="unobserved"):
        profile.engine_options(config, tmp_path)


def test_graph_inventory_reads_only_owned_captures(monkeypatch):
    capture = SimpleNamespace(num_graphs=2, num_eager_breaks=1, segments=[lambda: None])
    entry = SimpleNamespace(capture=capture, batch_descriptor="tokens=3")
    owned = SimpleNamespace(_collective_graph_observer=object(), entries={3: entry})
    unrelated = SimpleNamespace(entries={6: entry})
    monkeypatch.setitem(
        sys.modules,
        "vllm.compilation.breakable_cudagraph",
        SimpleNamespace(BreakableCUDAGraphWrapper=SimpleNamespace(_all_instances=[owned, unrelated])),
    )
    records = profile.QfnProfileWorkerExtension().qfn_graph_inventory()
    assert len(records) == 1
    assert records[0]["graphs"] == 2 and records[0]["eager_breaks"] == 1
    assert records[0]["descriptor"] == "tokens=3"


def test_cpu_profile_requires_explicit_lifecycle_and_dumps_rank_file(tmp_path):
    worker = profile.QfnProfileWorkerExtension()
    worker.rank = 2
    with pytest.raises(RuntimeError, match="not started"):
        worker.qfn_cpu_profile_stop(str(tmp_path))
    assert worker.qfn_cpu_profile_start()["started"]
    with pytest.raises(RuntimeError, match="already exists"):
        worker.qfn_cpu_profile_start()
    assert worker.qfn_cpu_profile_stop(str(tmp_path))["filename"] == "rank2.pstats"
    assert (tmp_path / "rank2.pstats").stat().st_size > 0
    assert worker._qfn_cpu_profile is None


@pytest.mark.parametrize("failure,diagnostics", [(None, True), ("npu", True), ("cpu", True), (None, False)])
def test_profile_lifecycle_keeps_timing_separate_and_retains_owners_on_failure(
    tmp_path, monkeypatch, failure, diagnostics
):
    events = []
    monkeypatch.setattr(benchmark, "verify_workers", lambda *a, **kw: None)
    monkeypatch.setattr(benchmark, "coverage_deltas", lambda *a, **kw: [])
    monkeypatch.setattr(benchmark, "replay_deltas", lambda *a, **kw: [])

    class FakeLLM:
        def __init__(self, **options):
            self.active = None
            self.llm_engine = SimpleNamespace(engine_core=SimpleNamespace(shutdown=lambda: events.append("shutdown")))

        def collective_rpc(self, name, **kwargs):
            events.append(name)
            if name == "qfn_cpu_profile_start":
                self.active = "cpu"
            elif name == "qfn_cpu_profile_stop":
                self.active = None
            return []

        def start_profile(self, prefix):
            events.append("npu-start")
            self.active = "npu"

        def stop_profile(self):
            events.append("npu-stop")
            self.active = None

        def get_tokenizer(self):
            return SimpleNamespace(apply_chat_template=lambda *a, **kw: "prompt")

        def generate(self, prompts, params, **kwargs):
            events.append(("generate", self.active, params.max_tokens))
            if failure is not None and self.active == failure:
                raise RuntimeError("diagnostic generation failed")
            return [
                SimpleNamespace(
                    prompt_token_ids=[1],
                    outputs=[SimpleNamespace(text="output", token_ids=[2] * params.max_tokens, finish_reason="length")],
                )
            ]

    def run():
        return profile.run(
            tmp_path / "run",
            dict(execution_mode="graph"),
            llm_factory=FakeLLM,
            sampling_factory=lambda **kwargs: SimpleNamespace(**kwargs),
            collect_diagnostics=diagnostics,
        )

    if failure is not None:
        with pytest.raises(RuntimeError, match="diagnostic generation"):
            run()
        assert events[-1] == ("npu-stop" if failure == "npu" else "qfn_cpu_profile_stop")
        assert "npu_collective_close" not in events and "shutdown" not in events
        assert not (tmp_path / "run/result.json").exists()
    else:
        result = run()
        assert events[-2:] == ["npu_collective_close", "shutdown"]
        assert [m["label"] for m in result["measurements"] if m["profiled"]] == (["npu", "cpu"] if diagnostics else [])
        if not diagnostics:
            assert "npu-start" not in events and "qfn_cpu_profile_start" not in events
        assert all(m["generated_tokens"] == 512 for m in result["measurements"] if m["label"].startswith("before"))
        assert result["ownership_released"]
        assert json.loads((tmp_path / "run/result.json").read_text()) == result
