# SPDX-License-Identifier: Apache-2.0
"""Eager M3 resident controls; all mutations run through a drained scheduler."""

from __future__ import annotations

import hashlib
import json
import os

import torch

from tools.glm_perf.resident_control import PatchSession
from tools.glm_perf.resident_native import NativeSession


def storage_identity(model):
    """Include registered packed expert buffers as well as dense parameters."""
    digest = hashlib.sha256()
    count = 0
    for name, tensor in (*model.named_parameters(), *model.named_buffers()):
        record = (
            name,
            tensor.untyped_storage().data_ptr(),
            tuple(tensor.shape),
            str(tensor.dtype),
            tuple(tensor.stride()),
            tensor.storage_offset(),
            str(tensor.device),
        )
        digest.update(repr(record).encode())
        count += 1
    return digest.hexdigest(), count


def clear_cache(value):
    if isinstance(value, torch.Tensor):
        value.zero_()
    elif isinstance(value, dict):
        for child in value.values():
            clear_cache(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            clear_cache(child)
    else:
        raise TypeError(f"unsupported cache entry: {type(value)}")


class MiniMaxResidentExtension:
    """Opt in using --worker-extension-cls; never reload checkpoint shards."""

    def _m3_sessions(self):
        if "_m3_native" not in self.__dict__:
            self._m3_native = NativeSession()
            self._m3_patch = PatchSession(self._m3_native.resources)
        return self._m3_patch, self._m3_native

    def _m3_idle(self):
        runner = self.model_runner
        if not runner.vllm_config.model_config.enforce_eager or getattr(runner, "use_async_scheduling", False):
            raise ValueError("M3 resident controls require eager, synchronous serving")
        if getattr(runner, "execute_model_state", None) is not None:
            raise RuntimeError("worker still has pending model execution")

    def _m3_error(self, exc):
        return {"rank": torch.distributed.get_rank(), "pid": os.getpid(), "error": f"{type(exc).__name__}: {exc}"}

    @staticmethod
    def _m3_operator_exists(name):
        try:
            torch._C._dispatch_find_schema_or_throw(name, "")
            return True
        except RuntimeError:
            return False

    def resident_status(self):
        patch, native = self._m3_sessions()
        setting = patch.current
        digest, count = storage_identity(self.model_runner.model)
        return {
            "rank": torch.distributed.get_rank(),
            "pid": os.getpid(),
            "generation": setting.generation if setting else None,
            "mode": setting.mode if setting else "direct-target",
            "candidate": setting.candidate if setting else "baseline",
            "digest": setting.digest if setting else None,
            "graphs_dirty": patch.graphs_dirty,
            "weight_storage_digest": digest,
            "weight_tensor_count": count,
            "native_failed": native.failed,
            "native_loaded": native.loaded,
        }

    def resident_prepare(self, payload):
        try:
            self._m3_idle()
            control = json.loads(payload)
            if control.get("mode") != "direct-target":
                raise ValueError("M3 eager controls support only direct-target mode")
            return self._m3_sessions()[0].prepare(control)
        except Exception as exc:
            return self._m3_error(exc)

    def resident_apply(self, generation):
        try:
            self._m3_idle()
            torch.npu.synchronize()
            patch, _ = self._m3_sessions()
            patch.apply(generation)
            # Eager execution has no captures that can retain stale functions.
            patch.graphs_dirty = False
            return self.resident_status()
        except Exception as exc:
            return self._m3_error(exc)

    def resident_capture(self):
        return self.resident_status()

    @torch.inference_mode()
    def resident_reset(self):
        try:
            self._m3_idle()
            runner = self.model_runner
            for request_id in tuple(runner.input_batch.req_ids):
                if request_id is not None:
                    runner.input_batch.remove_request(request_id)
            runner.requests.clear()
            clear_cache(runner.kv_caches)
            torch.npu.synchronize()
            return self.resident_status()
        except Exception as exc:
            return self._m3_error(exc)

    def resident_native_prepare(self, payload):
        try:
            self._m3_idle()
            receipt = self._m3_sessions()[1].prepare(json.loads(payload), self._m3_operator_exists)
            return {"rank": torch.distributed.get_rank(), "pid": os.getpid(), **receipt}
        except Exception as exc:
            return self._m3_error(exc)

    @torch.inference_mode()
    def resident_native_load(self, digest):
        try:
            self._m3_idle()
            receipt = self._m3_sessions()[1].load(
                digest, torch.ops.load_library, self._m3_operator_exists, torch.npu.synchronize
            )
            return {"rank": torch.distributed.get_rank(), "pid": os.getpid(), **receipt}
        except Exception as exc:
            return self._m3_error(exc)
