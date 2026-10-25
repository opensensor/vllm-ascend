# SPDX-License-Identifier: Apache-2.0
"""Opt-in resident PLE decode transport; explicit Torch stream join required.

Prefill and unsupported layouts retain the existing checkpoint reader. Native
owners survive graph recapture and failed calls; no destructor frees uncertain
DMA resources. This is host-assisted serving, not device-initiated host access.
"""

import ctypes
import hashlib
import json
import os
import time
from collections import Counter, deque
from pathlib import Path

from tools.qwen4exp.ple_request import MAX_REQUESTS as MAX_NATIVE_REQUESTS
from tools.qwen4exp.ple_request import PLERequestGeometry

MAX_REQUESTS = 4
TOKENS_PER_REQUEST = 3
PRODUCER_POLL_US = 40


def open_table(metadata):
    fd = os.open(metadata["fd_path"], os.O_RDONLY | os.O_CLOEXEC)
    info = os.fstat(fd)
    if (info.st_dev, info.st_ino, info.st_size) != (metadata["device"], metadata["inode"], metadata["bytes"]):
        os.close(fd)
        raise ValueError("shared PLE owner or table identity changed")
    return fd


def request_count(
    tokens, starts, history, starts_cpu, *, max_requests=MAX_REQUESTS, tokens_per_request=TOKENS_PER_REQUEST
):
    """Admit metadata without a device readback; reject excess padding groups."""
    if type(max_requests) is not int or not 1 <= max_requests <= MAX_NATIVE_REQUESTS:
        raise ValueError("native PLE supports one to six requests")
    if (
        type(tokens_per_request) is not int
        or tokens_per_request not in (3, 4, 8)
        or tokens_per_request * max_requests > 32
    ):
        raise ValueError("native PLE serving admits three, four or eight tokens per request")
    if not 1 <= tokens <= tokens_per_request * max_requests or starts is None or history is None or starts_cpu is None:
        return None
    count = min(max_requests, history.shape[0])
    if history.ndim != 2 or history.shape[1] != 2 or starts.numel() < count + 1:
        return None
    if int(starts_cpu[count]) != tokens:
        return None
    return count


def validate_capacity(lib, required_tokens):
    # The retained six-request adapter predates this query and admits 18 rows.
    query = getattr(lib, "ple_host_max_tokens", None)
    capacity = 18
    if query is not None:
        query.argtypes, query.restype = [], ctypes.c_size_t
        capacity = query()
    if required_tokens > capacity:
        raise ValueError(f"native PLE capacity {capacity} is below required {required_tokens}; rebuild adapter")
    return capacity


def load_library(path, digest, required_tokens=TOKENS_PER_REQUEST * MAX_REQUESTS):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
        raise ValueError("native PLE library digest differs")
    lib = ctypes.CDLL(str(path))
    validate_capacity(lib, required_tokens)
    pointer, integer, timings = ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint64)
    for name, arguments in (
        ("ple_host_create", [integer, integer, ctypes.c_size_t, pointer, pointer, ctypes.POINTER(pointer)]),
        ("ple_host_create_shared", [pointer, ctypes.c_size_t, pointer, pointer, ctypes.POINTER(pointer)]),
        ("ple_host_set_producer", [pointer, ctypes.c_char_p, ctypes.POINTER(pointer), ctypes.c_uint]),
        ("ple_host_produce_fetch_inputs", [pointer, pointer, pointer, pointer, timings, ctypes.c_uint]),
        ("ple_host_close", [pointer]),
    ):
        function = getattr(lib, name)
        function.argtypes, function.restype = arguments, integer
    return lib


def checked(status, operation):
    if status:
        raise RuntimeError(f"native PLE {operation} failed: {status}; retain owners")


class Lookup:
    def __init__(self, manager, tokens, requests, device):
        import torch

        self.manager = manager
        self.geometry = PLERequestGeometry(tokens=tokens, requests=requests, **manager.metadata["geometry"])
        self.packet = torch.empty(tokens * 160, dtype=torch.uint8, device=device)
        self.rows = self.packet[: tokens * 128].view(torch.int64).view(tokens, 16)
        self.receipt = self.packet[tokens * 128 :].view(torch.int32).view(tokens, 8)
        self.offsets = torch.empty_like(self.rows)
        self.config = torch.tensor(self.geometry.values(), dtype=torch.int64, device=device)
        self.output = torch.empty((tokens, 2560), dtype=torch.float16, device=device)
        self.handle = ctypes.c_void_p()
        self.timing = (ctypes.c_uint64 * 8)()
        self.ready = False
        # The manager retains this owner even if admission or a later DMA fails.
        manager.retained.append(self)
        torch.npu.current_stream(device).synchronize()
        parent = manager.mapping_parent
        if parent is None:
            status = manager.lib.ple_host_create(
                device.index,
                manager.fd,
                tokens,
                self.packet.data_ptr(),
                self.output.data_ptr(),
                ctypes.byref(self.handle),
            )
        else:
            if parent.output.device != device:
                raise ValueError("shared PLE mapping belongs to another device")
            status = manager.lib.ple_host_create_shared(
                parent.handle, tokens, self.packet.data_ptr(), self.output.data_ptr(), ctypes.byref(self.handle)
            )
        checked(status, "create")
        if parent is None:
            manager.mapping_parent = self

    def fetch(self, tokens, starts, history):
        import torch

        inputs = (
            tokens.to(torch.int64).contiguous(),
            starts.to(torch.int32).contiguous(),
            history.to(torch.int64).contiguous(),
        )
        # Flush queued Torch commands and finish their stream before launching
        # the private native producer. This also retires previous output users.
        torch.npu.current_stream(tokens.device).synchronize()
        self.inputs = inputs
        if not self.ready:
            arguments = (ctypes.c_void_p * 7)(
                *(
                    tensor.data_ptr()
                    for tensor in (
                        *inputs,
                        self.config,
                        self.rows,
                        self.offsets,
                        self.receipt,
                    )
                )
            )
            checked(
                self.manager.lib.ple_host_set_producer(
                    self.handle,
                    os.fsencode(self.manager.binary),
                    arguments,
                    min(8, self.geometry.tokens),
                ),
                "producer admission",
            )
            self.ready = True
        checked(
            self.manager.lib.ple_host_produce_fetch_inputs(
                self.handle,
                *(tensor.data_ptr() for tensor in inputs),
                self.timing,
                PRODUCER_POLL_US,
            ),
            "produce/fetch",
        )
        return self.output


class ServingPLE:
    def __init__(self, config):
        self.max_requests = config.get("max_requests", MAX_REQUESTS)
        if type(self.max_requests) is not int or not 1 <= self.max_requests <= MAX_NATIVE_REQUESTS:
            raise ValueError("native PLE supports one to six requests")
        self.tokens_per_request = config.get("tokens_per_request", TOKENS_PER_REQUEST)
        if (
            type(self.tokens_per_request) is not int
            or self.tokens_per_request not in (3, 4, 8)
            or self.tokens_per_request * self.max_requests > 32
        ):
            raise ValueError("native PLE serving admits three, four or eight tokens per request")
        self.metadata = json.loads(Path(config["table"]).read_text())
        self.lib = load_library(
            config["library"], config["library_sha256"], self.tokens_per_request * self.max_requests
        )
        self.fd = open_table(self.metadata)
        self.binary = config["binary"]
        if hashlib.sha256(Path(self.binary).read_bytes()).hexdigest() != config["binary_sha256"]:
            raise ValueError("native selector digest differs")
        self.lookups, self.retained = {}, []
        self.mapping_parent = None
        self.calls, self.fallbacks = Counter(), Counter()
        self.checked = 0
        self.enabled = self.verify = True
        self.failed = self.busy = False
        self.elapsed = deque(maxlen=1024)
        self.native_elapsed = deque(maxlen=1024)

    def status(self):
        def median(values):
            return sorted(values)[len(values) // 2] / 1000 if values else None

        return dict(
            enabled=self.enabled,
            failed=self.failed,
            verify=self.verify,
            checked=self.checked,
            native_calls=dict(self.calls),
            fallbacks=dict(self.fallbacks),
            table_inode=self.metadata["inode"],
            table_bytes=self.metadata["bytes"],
            owners=len(self.retained),
            tokens_per_request=self.tokens_per_request,
            shared_mapping=self.mapping_parent is not None,
            host_assisted=True,
            torch_stream_join=True,
            lookup_median_us=median(self.elapsed),
            native_median_us=median(self.native_elapsed),
        )

    def lookup(self, injection, tokens, starts, history, requests, host_inputs):
        import torch

        if self.failed or self.busy:
            raise RuntimeError("native PLE failed or reentered; retain owners")
        if Path(injection.checkpoint_dir).resolve() != Path(self.metadata["model"]):
            raise ValueError("PLE source checkpoint differs from loaded model")
        self.busy = self.failed = True
        key = (tokens.shape[0], requests, tokens.device.index)
        try:
            if key not in self.lookups:
                self.lookups[key] = Lookup(self, *key[:2], tokens.device)
            owner = self.lookups[key]
            started = time.perf_counter_ns()
            output = owner.fetch(tokens, starts[: requests + 1], history[:requests])
            self.elapsed.append(time.perf_counter_ns() - started)
            self.native_elapsed.append(owner.timing[5])
            if self.verify:
                if any(value is None for value in host_inputs):
                    raise ValueError("verification requires the serving host history mirror")
                expected = injection._real_ngram_ids(*host_inputs, reduce=False)
                torch.testing.assert_close(owner.rows.cpu(), expected, rtol=0, atol=0)
                rows = injection.ple.ple_method.gather_rows(expected.reshape(-1))
                torch.testing.assert_close(output.cpu(), rows.reshape(output.shape), rtol=0, atol=0)
                self.checked += 1
            self.calls[str(tokens.shape[0])] += 1
            self.failed = False
            return owner.rows, output
        finally:
            self.busy = False


def make_replacements(config, *, manager=None):
    # Lazy model imports: standalone admission/unit tests do not initialize CANN.
    from vllm_ascend._310p.worker_310p import NPUWorker310
    from vllm_ascend.models.qwen4_exp.model import _PLEInjection
    from vllm_ascend.models.qwen4_exp.ple_layer import AscendQwen4ExpPLELayer

    manager = ServingPLE(config) if manager is None else manager
    original = _PLEInjection._forward_eager
    gather_base = AscendQwen4ExpPLELayer.gather_embeddings
    status_base = NPUWorker310.resident_status

    def forward(
        self,
        hidden_states,
        input_ids,
        query_start_loc=None,
        ngram_context=None,
        output_buffer=None,
        input_ids_cpu=None,
        query_start_loc_cpu=None,
        ngram_context_cpu=None,
    ):
        count = (
            request_count(
                input_ids.shape[0],
                query_start_loc,
                ngram_context,
                query_start_loc_cpu,
                max_requests=manager.max_requests,
                tokens_per_request=manager.tokens_per_request,
            )
            if manager.enabled
            else None
        )
        if count is None:
            manager.fallbacks[str(input_ids.shape[0])] += 1
            return original(
                self,
                hidden_states,
                input_ids,
                query_start_loc,
                ngram_context,
                output_buffer,
                input_ids_cpu,
                query_start_loc_cpu,
                ngram_context_cpu,
            )
        self._ensure_ple_method(hidden_states.device)
        rows, output = manager.lookup(
            self,
            input_ids,
            query_start_loc,
            ngram_context,
            count,
            (input_ids_cpu, query_start_loc_cpu, ngram_context_cpu),
        )
        self.ple._native_host_ple_active = (rows, output)
        try:
            return self.ple(hidden_states, rows, output_buffer=output_buffer)
        finally:
            self.ple._native_host_ple_active = None

    def gather(self, ngram_ids, output_dtype, output_device=None):
        active = getattr(self, "_native_host_ple_active", None)
        if active is None:
            return gather_base(self, ngram_ids, output_dtype, output_device)
        if active[0] is not ngram_ids or active[1].dtype != output_dtype or active[1].device != output_device:
            raise ValueError("native PLE consumer differs from prepared lookup")
        return active[1]

    def status(worker):
        result = status_base(worker)
        result["native_ple"] = manager.status()
        return result

    return {
        "vllm_ascend.models.qwen4_exp.model:_PLEInjection._forward_eager": forward,
        "vllm_ascend.models.qwen4_exp.ple_layer:AscendQwen4ExpPLELayer.gather_embeddings": gather,
        "vllm_ascend._310p.worker_310p:NPUWorker310.resident_status": status,
    }
