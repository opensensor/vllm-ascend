# SPDX-License-Identifier: Apache-2.0
import asyncio
import hashlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tools.sidecard.qfn_engine_control import (
    EngineControlSession,
    engine_entrypoint,
    install_engine_entrypoint,
    install_engine_launch_hook,
    install_engine_route,
)


def engine():
    return SimpleNamespace(
        is_scheduler_paused=lambda: True,
        batch_queue=None,
        scheduler=SimpleNamespace(get_num_unfinished_requests=lambda: 0),
        _pause_complete=lambda: True,
        async_scheduling=False,
        batch_queue_size=1,
    )


SOURCE = """
class Candidate:
    def validate(self, engine):
        return dict(passed=not engine.async_scheduling)
    def apply(self, engine):
        engine.applied_marker = 17
        return dict(passed=True)
def prepare(engine):
    return Candidate()
"""


def prepare(source=SOURCE, revision="test_v1"):
    return dict(
        operation="prepare", source=source, revision=revision, sha256=hashlib.sha256(source.encode()).hexdigest()
    )


def apply(request):
    return {key: value for key, value in request.items() if key != "source"} | {"operation": "apply"}


def test_two_phase_admission_does_not_mutate_until_apply():
    owner, session, request = engine(), EngineControlSession(), prepare()
    assert session.request(owner, request)["passed"]
    assert not hasattr(owner, "applied_marker")
    receipt = session.request(owner, apply(request))
    assert receipt["passed"] and owner.applied_marker == 17
    assert not owner.async_scheduling  # never silently turns on async mode
    assert receipt["status"]["retained_owners"] == 2


@pytest.mark.parametrize("dirty", ["running", "outputs", "requests", "pause_pending"])
def test_busy_engine_rejected_before_source_execution(dirty):
    owner = engine()
    if dirty == "running":
        owner.is_scheduler_paused = lambda: False
    elif dirty == "outputs":
        owner.batch_queue = [object()]
    elif dirty == "requests":
        owner.scheduler.get_num_unfinished_requests = lambda: 1
    else:
        owner._pause_complete = lambda: False
    session = EngineControlSession()
    assert not session.request(owner, prepare())["passed"]
    assert session.owners == []


def test_digest_and_revision_must_match_before_apply():
    owner, session, request = engine(), EngineControlSession(), prepare()
    assert not session.request(owner, request | {"sha256": "0" * 64})["passed"]
    assert session.request(owner, request)["passed"]
    assert not session.request(owner, apply(request) | {"revision": "other"})["passed"]
    assert not hasattr(owner, "applied_marker")


def test_failed_mutation_is_quarantined_without_automatic_retry():
    owner, session = engine(), EngineControlSession()
    request = prepare(SOURCE.replace("return dict(passed=True)", "raise RuntimeError('partial change')"))
    assert session.request(owner, request)["passed"]
    result = session.request(owner, apply(request))
    assert not result["passed"] and result["failed"]
    assert owner.applied_marker == 17 and len(session.owners) == 2
    assert not session.request(owner, prepare(revision="other"))["passed"]
    assert session.request(owner, {"operation": "status"})["failed"]


def test_discard_retires_prepared_revision_without_unloading_owners():
    owner, session = engine(), EngineControlSession()
    assert session.request(owner, prepare())["passed"]
    assert session.request(owner, dict(operation="discard", revision="test_v1"))["passed"]
    assert session.pending is None and len(session.owners) == 2


@pytest.mark.parametrize("forked", [False, True])
def test_entrypoint_installs_control_under_spawn_and_fork(monkeypatch, forked):
    calls = []

    class Core:
        pass

    class Proc(Core):
        @staticmethod
        def run_engine_core(*args, **kwargs):
            calls.append((args, kwargs, hasattr(Core, "qfn_engine_control")))

    module = ModuleType("vllm.v1.engine.core")
    module.EngineCore, module.EngineCoreProc = Core, Proc
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.core", module)
    platform = ModuleType("vllm_ascend.patch.platform")
    platform.patch_engine_core = SimpleNamespace(_run_engine_core_patch_func=Proc.run_engine_core)
    monkeypatch.setitem(sys.modules, "vllm_ascend.patch.platform", platform)
    if forked:
        install_engine_entrypoint()
        install_engine_entrypoint()  # same owner is idempotent
        assert Proc.run_engine_core is engine_entrypoint
    engine_entrypoint(1, key=2)
    assert calls == [((1,), {"key": 2}, True)]


def test_launch_hook_installs_after_platform_replaces_entrypoint(monkeypatch):
    calls = []

    class Proc:
        run_engine_core = staticmethod(lambda: None)

    class Manager:
        def __init__(self, marker):
            calls.append((Proc.run_engine_core, marker))

    core = ModuleType("vllm.v1.engine.core")
    core.EngineCoreProc = Proc
    utils = ModuleType("vllm.v1.engine.utils")
    utils.CoreEngineProcManager = Manager
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.core", core)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.utils", utils)
    install_engine_launch_hook()
    initialize = Manager.__init__
    install_engine_launch_hook()
    assert Manager.__init__ is initialize
    platform_entrypoint = lambda: None
    Proc.run_engine_core = staticmethod(platform_entrypoint)
    Manager(17)
    assert calls == [(engine_entrypoint, 17)]
    assert Proc._qfn_original_engine_entrypoint is platform_entrypoint


def test_engine_route_targets_engine_utility_not_worker_rpc():
    routes, calls = {}, []

    def post(path):
        def register(function):
            routes[path] = function
            return function

        return register

    async def utility(method, payload):
        calls.append((method, payload))
        return dict(passed=True)

    async def request_json():
        return dict(operation="status")

    app = SimpleNamespace(
        post=post,
        state=SimpleNamespace(engine_client=SimpleNamespace(engine_core=SimpleNamespace(call_utility_async=utility))),
    )
    install_engine_route(app)
    result = asyncio.run(routes["/qfn_engine_control"](SimpleNamespace(app=app, json=request_json)))
    assert result["passed"] and calls == [("qfn_engine_control", dict(operation="status"))]


def test_real_spawn_transmits_top_level_entrypoint(tmp_path):
    script = tmp_path / "engine_spawn_probe.py"
    script.write_text(
        textwrap.dedent("""
        import multiprocessing as mp
        import sys
        from types import ModuleType, SimpleNamespace
        from tools.sidecard.qfn_engine_control import install_engine_entrypoint
        class Core:
            pass
        class Proc(Core):
            @staticmethod
            def run_engine_core(queue):
                queue.put(hasattr(Core, "qfn_engine_control"))
        module = ModuleType("vllm.v1.engine.core")
        module.EngineCore, module.EngineCoreProc = Core, Proc
        sys.modules["vllm.v1.engine.core"] = module
        platform = ModuleType("vllm_ascend.patch.platform")
        platform.patch_engine_core = SimpleNamespace(_run_engine_core_patch_func=Proc.run_engine_core)
        sys.modules["vllm_ascend.patch.platform"] = platform
        if __name__ == "__main__":
            install_engine_entrypoint()
            context = mp.get_context("spawn")
            queue = context.Queue()
            process = context.Process(target=Proc.run_engine_core, args=(queue,))
            process.start()
            assert queue.get(timeout=10) is True
            process.join(timeout=10)
            assert process.exitcode == 0
    """)
    )
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[3]))
    subprocess.run([sys.executable, str(script)], env=environment, check=True, timeout=25, capture_output=True)
