# SPDX-License-Identifier: Apache-2.0
"""Bounded host timing spans for a drained, temporary mixed-prefill diagnostic.

Times measure host call duration, including existing waits, without adding any
device synchronization. Nested spans overlap and must not be summed. The caller
owns this recorder and must restore hooks while the engine is drained.
"""

import inspect
import time
from collections import deque
from functools import wraps

MAX_RECORDS = 16384


class Recorder:
    def __init__(self, limit=MAX_RECORDS):
        if type(limit) is not int or limit < 1:
            raise ValueError("positive record limit required")
        self.records = deque(maxlen=limit)
        self.hooks = []
        self.total = 0

    def wrap(self, owner, name, label, metadata=None):
        original = getattr(owner, name)
        previous = vars(owner).get(name)
        owned = name in vars(owner)
        if isinstance(inspect.getattr_static(owner, name), (staticmethod, classmethod)):
            raise ValueError("wrap an ordinary method or an instance-bound method")

        @wraps(original)
        def measured(*args, **kwargs):
            details = {} if metadata is None else metadata(*args, **kwargs)
            started = time.monotonic_ns()
            failed = True
            try:
                result = original(*args, **kwargs)
                failed = False
                return result
            finally:
                self.total += 1
                self.records.append(
                    dict(
                        label=label,
                        start_ns=started,
                        duration_ns=time.monotonic_ns() - started,
                        failed=failed,
                        **details,
                    )
                )

        self.hooks.append((owner, name, original, previous, owned, measured))
        setattr(owner, name, measured)

    def restore(self):
        if any(getattr(owner, name) is not wrapper for owner, name, _, _, _, wrapper in self.hooks):
            raise RuntimeError("diagnostic hook ownership changed; retain recorder for inspection")
        for owner, name, _, previous, owned, _ in reversed(self.hooks):
            if owned:
                setattr(owner, name, previous)
            else:
                delattr(owner, name)
        self.hooks.clear()

    def snapshot(self):
        return dict(total=self.total, dropped=self.total - len(self.records), records=list(self.records))


def install_worker(worker):
    # These imports belong to the live worker; host tests do not initialize NPU.
    from vllm.forward_context import get_forward_context

    from vllm_ascend._310p.prefix_mamba_state import PrefixMambaStateTier
    from vllm_ascend.models.qwen4_exp.model import _PLEInjection

    if getattr(worker, "_qfn_mixed_profile", None) is not None:
        raise RuntimeError("mixed profiler already installed")
    recorder = worker._qfn_mixed_profile = Recorder()
    runner = worker.model_runner

    def schedule_metadata(schedule, *args, **kwargs):
        return dict(
            tokens=dict(schedule.num_scheduled_tokens),
            drafts={key: len(value) for key, value in schedule.scheduled_spec_decode_tokens.items()},
        )

    def forward_metadata(num_tokens_padded, *args, **kwargs):
        context = get_forward_context()
        return dict(
            tokens=num_tokens_padded,
            mode=str(context.cudagraph_runtime_mode),
            requests=list(runner.input_batch.req_ids),
        )

    def ple_metadata(layer, input_ids, *args, **kwargs):
        return dict(tokens=input_ids.shape[0])

    recorder.wrap(runner, "execute_model", "execute_model", schedule_metadata)
    recorder.wrap(runner, "_model_forward", "model_forward", forward_metadata)
    recorder.wrap(_PLEInjection, "_forward_eager", "ple", ple_metadata)
    for method in ("_spill_to_host", "_admit", "remap_rows"):
        recorder.wrap(PrefixMambaStateTier, method, "checkpoint" + method)
    return dict(rank=worker.rank, passed=True)


def stop_worker(worker):
    recorder = worker._qfn_mixed_profile
    recorder.restore()
    result = dict(rank=worker.rank, passed=True, **recorder.snapshot())
    worker._qfn_mixed_profile = None
    return result
