# SPDX-License-Identifier: Apache-2.0
"""Authenticated resident engine control, installed at engine process startup.

The existing worker RPC cannot replace the engine's synchronous scheduler.
This separate, cold-only utility admits reviewed engine changes after requests
and queued outputs drain. It never enables async scheduling merely by changing
a flag, restarts workers, or automatically recovers uncertain mutations.
"""

import hashlib
import json
from functools import wraps
from types import ModuleType

MAX_SOURCE_BYTES = 128 * 1024


def require_drained_engine(engine):
    if not engine.is_scheduler_paused():
        raise RuntimeError("pause and drain the engine before configuration")
    if engine.batch_queue or engine.scheduler.get_num_unfinished_requests():
        raise RuntimeError("engine still owns requests or queued model outputs")
    if hasattr(engine, "_pause_complete") and not engine._pause_complete():
        raise RuntimeError("engine pause has not finished")


class EngineControlSession:
    """Engine-owned transaction state; no inference-path polling or hooks."""

    def __init__(self):
        self.pending = None
        self.current = None
        self.failed = False
        self.owners = []

    def status(self, engine):
        return dict(
            passed=True,
            failed=self.failed,
            current=None if self.current is None else self.current[:2],
            pending=None if self.pending is None else self.pending[:2],
            paused=engine.is_scheduler_paused(),
            async_scheduling=engine.async_scheduling,
            scheduler_type=type(engine.scheduler).__name__,
            batch_queue_size=engine.batch_queue_size,
            retained_owners=len(self.owners),
        )

    def request(self, engine, payload):
        try:
            if not isinstance(payload, dict):
                raise ValueError("engine control requires a JSON object")
            operation = payload.get("operation")
            if operation == "status" and set(payload) == {"operation"}:
                return self.status(engine)
            require_drained_engine(engine)
            if self.failed:
                raise RuntimeError("failed engine mutation retains ownership; inspection required")
            if operation == "prepare":
                if set(payload) != {"operation", "revision", "sha256", "source"}:
                    raise ValueError("prepare requires revision, sha256, source")
                source, revision = payload["source"], payload["revision"]
                if not isinstance(source, str) or not isinstance(revision, str) or not revision.isidentifier():
                    raise ValueError("source text and identifier revision required")
                encoded = source.encode()
                digest = hashlib.sha256(encoded).hexdigest()
                if len(encoded) > MAX_SOURCE_BYTES or digest != payload["sha256"]:
                    raise ValueError("engine source length or digest differs")
                if self.pending is not None:
                    raise RuntimeError("retire the existing prepared revision first")
                if self.current is not None and self.current[0] == revision:
                    raise ValueError("engine revision must be fresh")
                module = ModuleType(f"qfn_engine_{revision}")
                exec(compile(source, f"<qfn-engine-{revision}>", "exec"), module.__dict__)
                factory = getattr(module, "prepare", None)
                if not callable(factory):
                    raise ValueError("source must define prepare(engine)")
                # Preparation may allocate owners but must not change engine
                # behavior. Keep its module alive even if the factory fails.
                self.owners.append(module)
                candidate = factory(engine)
                self.owners.append(candidate)
                if not callable(getattr(candidate, "apply", None)) or not callable(
                    getattr(candidate, "validate", None)
                ):
                    raise ValueError("candidate requires apply(engine) and validate(engine)")
                validation = candidate.validate(engine)
                json.dumps(validation, allow_nan=False)
                if not isinstance(validation, dict) or validation.get("passed") is not True:
                    raise RuntimeError("engine candidate admission failed")
                self.pending = (revision, digest, candidate)
                return dict(passed=True, revision=revision, sha256=digest, validation=validation)
            if operation == "apply" and set(payload) == {"operation", "revision", "sha256"}:
                if self.pending is None or tuple(self.pending[:2]) != (payload["revision"], payload["sha256"]):
                    raise ValueError("apply must identify the prepared revision")
                candidate = self.pending[2]
                validation = candidate.validate(engine)
                if not isinstance(validation, dict) or validation.get("passed") is not True:
                    raise RuntimeError("engine candidate no longer admitted")
                # Never fall back or resume after an uncertain mutation.
                self.failed = True
                receipt = candidate.apply(engine)
                json.dumps(receipt, allow_nan=False)
                if not isinstance(receipt, dict) or receipt.get("passed") is not True:
                    raise RuntimeError("engine candidate did not confirm application")
                require_drained_engine(engine)
                self.current, self.pending = self.pending, None
                self.failed = False
                return dict(passed=True, applied=receipt, status=self.status(engine))
            if operation == "discard" and set(payload) == {"operation", "revision"}:
                if self.pending is None or self.pending[0] != payload["revision"]:
                    raise ValueError("discard must identify the prepared revision")
                self.pending = None
                # Retain owners until explicit lifecycle retirement is provided.
                return self.status(engine)
            raise ValueError("invalid engine control operation or fields")
        except Exception as error:
            return dict(passed=False, error=str(error), failed=self.failed)


def engine_control(engine, payload):
    session = getattr(engine, "_qfn_engine_control_session", None)
    if session is None:
        session = engine._qfn_engine_control_session = EngineControlSession()
    return session.request(engine, payload)


def engine_entrypoint(*args, **kwargs):
    """Top-level multiprocessing target; works under spawn as well as fork."""
    from vllm.v1.engine.core import EngineCore, EngineCoreProc

    # Import the Ascend wrapper in the spawned child as well. Its platform
    # configuration/profiling setup must remain ahead of engine construction.
    from vllm_ascend.patch.platform import patch_engine_core

    original = getattr(EngineCoreProc, "_qfn_original_engine_entrypoint", patch_engine_core._run_engine_core_patch_func)
    if original is engine_entrypoint:
        raise RuntimeError("engine entrypoint lost original ownership")
    EngineCore.qfn_engine_control = engine_control
    return original(*args, **kwargs)


def install_engine_entrypoint():
    """Call from the resident server before vLLM starts its engine process."""
    from vllm.v1.engine.core import EngineCoreProc

    if EngineCoreProc.run_engine_core is engine_entrypoint:
        return
    if hasattr(EngineCoreProc, "_qfn_original_engine_entrypoint"):
        raise RuntimeError("engine entrypoint already has a different owner")
    EngineCoreProc._qfn_original_engine_entrypoint = staticmethod(EngineCoreProc.run_engine_core)
    EngineCoreProc.run_engine_core = staticmethod(engine_entrypoint)


def install_engine_launch_hook():
    """Wrap the process factory after platform configuration has installed patches.

    Ascend replaces run_engine_core during argument/config initialization.
    Installing our entrypoint earlier loses it under spawn: the imported Ascend
    wrapper binds its own original in the fresh child. Select our top-level
    target immediately before the process factory constructs its Process.
    """
    from vllm.v1.engine.utils import CoreEngineProcManager

    original = CoreEngineProcManager.__init__
    if getattr(original, "_qfn_engine_launch_hook", False):
        return

    @wraps(original)
    def initialize(manager, *args, **kwargs):
        install_engine_entrypoint()
        return original(manager, *args, **kwargs)

    initialize._qfn_engine_launch_hook = True
    CoreEngineProcManager.__init__ = initialize


def install_engine_route(app):
    """Uses the resident server's existing loopback/token middleware."""
    from starlette.requests import Request

    @app.post("/qfn_engine_control")
    async def control(request: Request):
        payload = await request.json()
        engine = request.app.state.engine_client
        return await engine.engine_core.call_utility_async("qfn_engine_control", payload)
