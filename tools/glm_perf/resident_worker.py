"""Worker RPCs for paused GLM diagnostics; never reload or modify weights."""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import os
from typing import Any

import torch
from vllm.compilation.monitor import set_cudagraph_capturing_enabled

from tools.glm_perf.resident_control import PatchSession
from tools.glm_perf.resident_native import NativeSession
from vllm_ascend.compilation.acl_graph import (
    get_draft_graph_params,
    get_draft_graph_prefill_params,
    get_graph_params,
)
from vllm_ascend.compilation.breakable_aclgraph import BreakableACLGraphWrapper


def renew_graph_pool(wrappers: list[Any]) -> None:
    # Destroying the final graph retires its allocator pool. The numeric handle
    # remains on the wrappers, but torch-npu cannot capture into that retired
    # pool (NPUCachingAllocator use_count assertion). Share a fresh handle across
    # target/draft captures; model weights are outside this graph allocation pool.
    pool = torch.npu.graph_pool_handle()
    for wrapper in wrappers:
        wrapper.graph_pool = pool


def clear_graph_state(wrappers: list[Any], parameters: list[Any]) -> None:
    for wrapper in wrappers:
        wrapper.clear_graphs()
        wrapper.__dict__.pop("_metadata_audit_snapshots", None)
        wrapper.__dict__.pop("_metadata_audit_count", None)
    # Attention task handles and events belong to the discarded captures.
    # Preserve the configured capture-size keys for the new captures.
    for params in parameters:
        if params is None:
            continue
        for field in dataclasses.fields(params):
            values = getattr(params, field.name)
            for size in values:
                values[size] = None if field.name == "workspaces" else []
    renew_graph_pool(wrappers)


def zero_cache(value: Any) -> None:
    if isinstance(value, torch.Tensor):
        value.zero_()
    elif isinstance(value, dict):
        for child in value.values():
            zero_cache(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            zero_cache(child)
    else:
        raise TypeError(f"unsupported resident cache entry: {type(value)}")


def resident_weight_tensors(model):
    yield from model.named_parameters()
    # Packed experts are deliberately held outside nn.Parameter registration.
    # Include both grouped banks and their per-expert views without touching data.
    for name, module in model.named_modules():
        bank = getattr(module, "w2_experts", None)
        if bank is None:
            continue
        for key, value in sorted(vars(bank).items()) if hasattr(bank, "__dict__") else ():
            if isinstance(value, torch.Tensor) and key.endswith("_bank"):
                yield f"{name}.w2_experts.{key}", value
        for expert_id, expert in enumerate(bank):
            for projection in ("gate", "up", "down"):
                for kind in ("packed", "scale"):
                    key = f"{projection}_{kind}"
                    value = getattr(expert, key, None)
                    if isinstance(value, torch.Tensor):
                        yield f"{name}.w2_experts.{expert_id}.{key}", value


class ResidentWorkerExtension:
    """Call through the harness, which drains the scheduler before these RPCs."""

    def _resident_session(self) -> PatchSession:
        if "_glm_resident_session" not in self.__dict__:
            self._glm_resident_session = PatchSession(self._resident_native_session().resources)
        return self._glm_resident_session

    def _resident_native_session(self) -> NativeSession:
        if "_glm_native_session" not in self.__dict__:
            self._glm_native_session = NativeSession()
        return self._glm_native_session

    @staticmethod
    def _resident_operator_exists(name: str) -> bool:
        try:
            torch._C._dispatch_find_schema_or_throw(name, "")
            return True
        except RuntimeError:
            return False

    def resident_native_prepare(self, payload: str) -> dict[str, Any]:
        result = {"rank": torch.distributed.get_rank(), "pid": os.getpid()}
        try:
            self._resident_wrappers()
            result.update(self._resident_native_session().prepare(json.loads(payload), self._resident_operator_exists))
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    @torch.inference_mode()
    def resident_native_load(self, digest: str) -> dict[str, Any]:
        result = {"rank": torch.distributed.get_rank(), "pid": os.getpid()}
        try:
            if getattr(self.model_runner, "execute_model_state", None) is not None:
                raise RuntimeError("worker still has a pending model execution")
            result.update(
                self._resident_native_session().load(
                    digest, torch.ops.load_library, self._resident_operator_exists, torch.npu.synchronize
                )
            )
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    def _resident_wrappers(self) -> list[Any]:
        runner = self.model_runner
        wrappers = [runner.model]
        drafter = getattr(runner, "drafter", None)
        if drafter is not None:
            wrappers.append(drafter.model)
        if not all(isinstance(wrapper, BreakableACLGraphWrapper) for wrapper in wrappers):
            raise ValueError("resident GLM diagnostics require breakable target and draft graphs")
        if getattr(runner, "use_async_scheduling", False):
            raise ValueError("resident GLM diagnostics require synchronous scheduling")
        return wrappers

    def resident_prepare(self, payload: str) -> dict[str, Any]:
        try:
            self._resident_wrappers()
            return self._resident_session().prepare(json.loads(payload))
        except Exception as exc:
            return self._resident_error(exc)

    def _resident_error(self, error: Exception) -> dict[str, Any]:
        # Never let an RPC exception abandon the remaining ranks' responses.
        return {"rank": torch.distributed.get_rank(), "pid": os.getpid(), "error": f"{type(error).__name__}: {error}"}

    def resident_apply(self, generation: str) -> dict[str, Any]:
        try:
            wrappers = self._resident_wrappers()
            torch.npu.synchronize()
            session = self._resident_session()
            if session.apply(generation):
                clear_graph_state(
                    wrappers, [get_graph_params(), get_draft_graph_params(), get_draft_graph_prefill_params()]
                )
                gc.collect()
            self._resident_set_mode()
            return self.resident_status()
        except Exception as exc:
            return self._resident_error(exc)

    def _resident_set_mode(self) -> None:
        session = self._resident_session()
        mode = session.current.mode if session.current else "graph"
        for index, wrapper in enumerate(self._resident_wrappers()):
            wrapper._resident_direct = mode == "direct-both" or mode == (
                "direct-target" if index == 0 else "direct-draft"
            )

    @torch.inference_mode()
    def resident_capture(self) -> dict[str, Any]:
        session = self._resident_session()
        error = None
        if session.graphs_dirty:
            # Build both graph sets even when the comparison selects direct
            # execution, so the following mode change needs no recapture.
            for wrapper in self._resident_wrappers():
                wrapper._resident_direct = False
            try:
                self.model_runner.capture_model()
                session.graphs_dirty = False
            except Exception as exc:
                # Return every rank's acknowledgment. Raising through the
                # executor can leave other ranks' replies queued for the next RPC.
                error = f"{type(exc).__name__}: {exc}"
            finally:
                set_cudagraph_capturing_enabled(False)
                self._resident_set_mode()
        result = self.resident_status()
        if error is not None:
            result["error"] = error
        return result

    @torch.inference_mode()
    def resident_reset(self) -> dict[str, Any]:
        runner = self.model_runner
        if getattr(runner, "execute_model_state", None) is not None:
            raise RuntimeError("worker still has a pending model execution")
        for req_id in tuple(runner.input_batch.req_ids):
            if req_id is not None:
                runner.input_batch.remove_request(req_id)
        runner.requests.clear()
        zero_cache(runner.kv_caches)
        torch.npu.synchronize()
        return self.resident_status()

    def resident_status(self) -> dict[str, Any]:
        session = self._resident_session()
        setting = session.current
        storage = hashlib.sha256()
        for index, wrapper in enumerate(self._resident_wrappers()):
            for name, parameter in resident_weight_tensors(wrapper.runnable):
                record = (
                    index,
                    name,
                    parameter.untyped_storage().data_ptr(),
                    tuple(parameter.shape),
                    str(parameter.dtype),
                    tuple(parameter.stride()),
                    parameter.storage_offset(),
                    str(parameter.device),
                )
                storage.update(repr(record).encode())
        return {
            "pid": os.getpid(),
            "rank": torch.distributed.get_rank(),
            "generation": setting.generation if setting else None,
            "mode": setting.mode if setting else "graph",
            "candidate": setting.candidate if setting else "baseline",
            "digest": setting.digest if setting else None,
            "graphs_dirty": session.graphs_dirty,
            "weight_storage_digest": storage.hexdigest(),
            "native_failed": self._resident_native_session().failed,
            "native_loaded": self._resident_native_session().loaded,
        }
