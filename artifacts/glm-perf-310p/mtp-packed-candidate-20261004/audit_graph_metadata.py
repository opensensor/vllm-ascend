"""Bounded serving diagnostic, loaded only by an experimental worker extension."""

import dataclasses
import json
import os

import torch
from vllm.forward_context import get_forward_context

from tools.glm_perf.resident_worker import ResidentWorkerExtension
from vllm_ascend.compilation.breakable_aclgraph import BreakableACLGraphWrapper


def tensors(value, path="", depth=0):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif depth < 5:
        if isinstance(value, dict):
            items = value.items()
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            items = ((field.name, getattr(value, field.name)) for field in dataclasses.fields(value))
        elif isinstance(value, (list, tuple)):
            items = enumerate(value)
        else:
            return
        for key, child in items:
            yield from tensors(child, f"{path}.{key}", depth + 1)


def install():
    original_call = BreakableACLGraphWrapper.__call__
    original_capture = BreakableACLGraphWrapper._capture
    original_replay = BreakableACLGraphWrapper._replay

    def call(self, *args, **kwargs):
        if self.__dict__.get("_resident_direct", False):
            return self.runnable(*args, **kwargs)
        return original_call(self, *args, **kwargs)

    def capture(self, entry, args, kwargs):
        if not hasattr(self, "_metadata_audit_snapshots"):
            self._metadata_audit_snapshots = {}
            self._metadata_audit_count = 0
        self._metadata_audit_snapshots[id(entry)] = dict(tensors(get_forward_context().attn_metadata))
        return original_capture(self, entry, args, kwargs)

    def replay(self, entry, args, kwargs):
        if self._metadata_audit_count < 24 and torch.distributed.get_rank() == 0:
            self._metadata_audit_count += 1
            captured = self._metadata_audit_snapshots.get(id(entry), {})
            changes, samples, seen = [], [], set()
            for path, live in tensors(get_forward_context().attn_metadata):
                old = captured.get(path)
                if old is None:
                    continue
                pair = (old.data_ptr(), live.data_ptr())
                if pair in seen:
                    continue
                seen.add(pair)
                lhs, rhs = old.cpu(), live.cpu()
                record = dict(
                    path=path,
                    same_pointer=old.data_ptr() == live.data_ptr(),
                    old_shape=list(lhs.shape),
                    new_shape=list(rhs.shape),
                    captured=lhs.flatten()[:16].tolist(),
                    live=rhs.flatten()[:16].tolist(),
                )
                samples.append(record)
                if lhs.shape != rhs.shape or not torch.equal(lhs, rhs):
                    changes.append(record)
            print(
                "GLM_METADATA_AUDIT "
                + json.dumps(
                    dict(
                        pid=os.getpid(),
                        step=self._metadata_audit_count,
                        model=type(self.runnable).__name__,
                        descriptor=str(entry.batch_descriptor),
                        input_addresses_match=entry.input_addresses == self._collect_tensor_addresses(args, kwargs),
                        mismatches=changes,
                        samples=samples,
                    )
                ),
                flush=True,
            )
        return original_replay(self, entry, args, kwargs)

    BreakableACLGraphWrapper.__call__ = call
    BreakableACLGraphWrapper._capture = capture
    BreakableACLGraphWrapper._replay = replay


class MetadataAuditExtension(ResidentWorkerExtension):
    pass


install()
