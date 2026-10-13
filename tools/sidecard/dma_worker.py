# SPDX-License-Identifier: Apache-2.0
"""Privileged Ascend expert actor owning both direct DMA endpoints."""

import argparse
import ctypes
import json
import math
import os
import signal
import socket
import struct
import sys
import time
from collections import deque
from pathlib import Path

from .cold_expert_probe import receive
from .cold_placement import load
from .completion_protocol import (
    CONTROL_CAPACITY,
    INPUT_SEQUENCE_OFFSET,
    RESULT_OFFSET,
    SEQUENCE_OFFSET,
    RemoteResult,
    epoch,
    input_record,
    receive_fd,
    receive_result,
    result_header,
)
from .completion_protocol import (
    enabled as completion_enabled,
)
from .dma_protocol import (
    CLOSE_TAG,
    COMMAND,
    INPUT_ROW_BYTES,
    OUTPUT_ROW_BYTES,
    PROTOCOL,
    RELEASE,
    AdmittedCudaSource,
    dma_capacity,
    receive_control,
    send_control,
    validate_command,
)
from .expert_decode_graph import ExpertDecodeGraphs, prefill_graph_rows
from .expert_prefill_gate import validate as validate_prefill_graphs
from .layer_weights import MAX_CHUNK, NPU_LAYERS, TOP_K, WIDTH, NpuExperts, expert_batch_size
from .pcie.mailbox import RECORD, Invocation
from .pcie.peer import ASCEND_TO_CUDA, CUDA_TO_ASCEND, Driver, KernelPeer
from .pcie.runtime import Runtime
from .pcie.segmented import export_allocation, make_peer, mapping_counts
from .single_route_gate import validate as validate_single_route
from .single_route_graph import SingleRouteDecodeGraphs, single_route_enabled

GUARD_POISON = 0xD3
PROFILE_RECORD_LIMIT = 4096
RECEIPT_INTERVAL = 1024
EXPERT_COUNT = 288
EXPERTS_PER_RANK = EXPERT_COUNT // 2


def validate_routes(metadata, rows, hot_ids, *, max_chunk=MAX_CHUNK):
    """Check bounded CPU metadata without launching small Torch operations."""
    if type(rows) is not int or not 1 <= rows <= max_chunk:
        raise ValueError("route metadata exceeds its row capacity")
    count = rows * TOP_K
    values = struct.unpack_from(f"<{count}i{count}f", metadata)
    owned = [0, 0]
    for index in range(count):
        expert, weight = values[index], values[count + index]
        if (
            not 0 <= expert < EXPERT_COUNT
            or not math.isfinite(weight)
            or weight < 0
            or (expert in hot_ids and weight != 0)
        ):
            raise ValueError("invalid or hot-owned route in direct expert request")
        if weight > 0:
            owned[expert // EXPERTS_PER_RANK] += 1
    return tuple(owned)


class TensorAllocation:
    """Retain a worker-owned contiguous Torch allocation during native DMA."""

    def __init__(self, runtime, tensor):
        if (
            runtime.kind != "ascend"
            or tensor.device.type != "npu"
            or tensor.device.index != runtime.device
            or not tensor.is_contiguous()
        ):
            raise ValueError("direct HBM endpoint requires a contiguous NPU tensor")
        self.runtime, self.tensor = runtime, tensor
        self.size = tensor.numel() * tensor.element_size()
        self.host, self.leases = False, 0

    @property
    def pointer(self):
        self.runtime.check_owner()
        if self.tensor is None:
            raise RuntimeError("direct HBM tensor endpoint is closed")
        return self.tensor.data_ptr()

    def close(self):
        if self.leases:
            raise RuntimeError("HBM tensor is retained by uncertain DMA")
        self.tensor = None


class TorchSourceEvent:
    def __init__(self, runtime, event):
        self.runtime, self.event, self.leases = runtime, event, 0

    def require_recorded(self):
        self.runtime.check_owner()

    def wait(self):
        self.require_recorded()
        self.event.synchronize()


class WorkerSlots:
    def __init__(self, spec, rank, torch):
        self.torch, self.rank = torch, rank
        self.max_chunk = expert_batch_size(spec)
        self.capacity = dma_capacity(spec)
        self.cuda = Runtime("cuda", spec["cuda_backend"]["cuda_runtime"])
        library = spec.get("direct_dma", {}).get(
            "ascend_library", "/usr/local/Ascend/cann-9.1.0/x86_64-linux/lib64/libascendcl.so"
        )
        self.ascend = Runtime("ascend", library, rank, initialize=False)
        self.driver = Driver()
        self.completion_contract = completion_enabled(spec)
        self.input_record = input_record(spec)
        self.control_export = self.notification_peer = None
        self.gpu_input = self.cuda.allocate(self.capacity)
        self.gpu_output = None if self.completion_contract else self.cuda.allocate(self.capacity)
        self.input_export = export_allocation(self.driver, self.gpu_input)
        self.output_export = None if self.completion_contract else export_allocation(self.driver, self.gpu_output)
        self.packed = torch.empty(self.capacity, dtype=torch.uint8, device=f"npu:{rank}")
        self.npu_input = TensorAllocation(self.ascend, self.packed)
        self.guard = self.ascend.allocate(self.capacity, host=True)
        ctypes.memset(self.guard.pointer, GUARD_POISON, self.capacity)
        self.stream, self.event = self.ascend.stream(), self.ascend.event()
        self.peer = make_peer(
            rank + 2,
            self.capacity,
            persistent_gpu_mappings=spec.get("direct_dma", {}).get("persistent_gpu_mappings", False),
        )
        self.source_event = AdmittedCudaSource(self.cuda)
        self.metadata_host = self.ascend.allocate(
            self.max_chunk * TOP_K * 8 + (RECORD.size if self.input_record else 0), host=True
        )
        storage = (ctypes.c_uint8 * self.metadata_host.size).from_address(self.metadata_host.pointer)
        self.metadata = storage
        self.hot_ids = {int(layer): frozenset(ids) for layer, ids in load(spec).items()}
        self.calls, self.upload_bytes, self.download_bytes = 0, 0, 0
        self.transfer_calls = 0
        self.single_route_calls = 0
        self.mapping_creations, self.mapping_hits = 0, 0
        self.profile = deque(maxlen=PROFILE_RECORD_LIMIT)
        if self.completion_contract and not self.input_record:
            self.result_packet = torch.empty(self.capacity, dtype=torch.uint8, device=f"npu:{rank}")
            self.result_header_host = self.ascend.allocate(RESULT_OFFSET, host=True)
            self.epoch = None

    def register_result(self, connection, reply, nonce):
        expected = {
            "source_readiness": "sender-event-synchronized",
            "nonce": nonce,
            "result_owner": "coordinator",
            "native_owner_ordering": True,
            "epoch": epoch(nonce),
            "capacity": self.capacity,
        }
        if self.input_record:
            expected.update(input_record=True, control_capacity=CONTROL_CAPACITY)
        if reply != expected:
            raise ValueError("coordinator result ownership contract differs")
        self.output_export = receive_result(self.cuda, connection, self.capacity)
        self.epoch = epoch(nonce)
        if self.input_record:
            self.control_export = RemoteResult(self.cuda, receive_fd(connection), size=CONTROL_CAPACITY)
            self.notification_peer = KernelPeer(self.rank + 2, persistent_gpu_mappings=True)

    def release_result_mapping(self):
        self.peer.close()
        if self.output_export is not None:
            self.output_export.close()
        if self.notification_peer is not None:
            self.notification_peer.close()
            self.control_export.close()

    def receipt(self, state):
        return {
            "state": state,
            "pid": os.getpid(),
            "rank": self.rank,
            "transport": "direct-dma",
            "payload_host_staging": False,
            "calls": self.calls,
            "single_route_calls": self.single_route_calls,
            "verified_direct_transfers": self.transfer_calls,
            "upload_bytes": self.upload_bytes,
            "download_bytes": self.download_bytes,
            "gpu_mapping_creations": self.mapping_creations,
            "gpu_mapping_hits": self.mapping_hits,
            "persistent_gpu_mappings": self.peer.persistent_gpu_mappings,
            "profile": list(self.profile),
            "profile_record_limit": PROFILE_RECORD_LIMIT,
            "receipt_interval_calls": RECEIPT_INTERVAL,
            "max_chunk_rows": self.max_chunk,
            "cuda_completion_contract": self.completion_contract,
            "gpu_result_owner": "coordinator" if self.completion_contract else "independent-ascend-worker",
            "completion_input_record": self.input_record,
        }

    def compute(self, actor, sequence, layer, rows):
        """Transfer directly into Torch HBM and out of the native output tensor."""
        t = self.torch
        input_contract = getattr(self, "input_record", False)
        start = time.perf_counter()
        upload = self.peer.copy(
            CUDA_TO_ASCEND,
            self.input_export,
            self.npu_input,
            self.guard,
            rows * INPUT_ROW_BYTES + (RECORD.size if input_contract else 0),
            source_ready=self.source_event,
        )
        uploaded = time.perf_counter()
        x_bytes, id_bytes = rows * WIDTH * 2, rows * TOP_K * 4
        # Validate only routing metadata on CPU; activations never leave HBM.
        self.ascend.copy(
            self.metadata_host.pointer,
            self.npu_input.pointer + x_bytes,
            rows * TOP_K * 8 + (RECORD.size if input_contract else 0),
            "d2h",
            self.stream,
        )
        self.event.record(self.stream)
        self.event.wait()
        x = self.packed[:x_bytes].view(t.float16).reshape(rows, WIDTH)
        ids = self.packed[x_bytes : x_bytes + id_bytes].view(t.int32).reshape(rows, TOP_K)
        weights = self.packed[x_bytes + id_bytes : rows * INPUT_ROW_BYTES].view(t.float32).reshape(rows, TOP_K)
        route_counts = validate_routes(self.metadata, rows, self.hot_ids[layer], max_chunk=self.max_chunk)
        contract = getattr(self, "completion_contract", False)
        if input_contract:
            metadata_offset = rows * TOP_K * 8
            record = Invocation.decode(
                bytes(self.metadata[metadata_offset : metadata_offset + RECORD.size]),
                epoch=self.epoch,
                sequence=sequence + 1,
                slots=2,
                rank=self.rank,
            )
            if (record.layer, record.rows, record.slot) != (layer, rows, self.rank):
                raise ValueError("GPU input record does not identify the requested cold computation")
        if contract and not input_contract:
            header = result_header(self.epoch, sequence, layer, rows, self.rank)
            ctypes.memmove(self.result_header_host.pointer, header, len(header))
            self.ascend.copy(
                self.result_packet.data_ptr(), self.result_header_host.pointer, len(header), "h2d", self.stream
            )
            self.event.record(self.stream)
            self.event.wait()
        native_start = time.perf_counter()
        single = rows == 1 and route_counts[self.rank] == 1 and isinstance(actor, SingleRouteDecodeGraphs)
        partial = actor(layer, x, ids, weights, owned_routes=1) if single else actor(layer, x, ids, weights)
        if contract and not input_contract:
            length = RESULT_OFFSET + rows * OUTPUT_ROW_BYTES
            self.result_packet[RESULT_OFFSET:length].view(t.float32).reshape(rows, WIDTH).copy_(partial)
        complete = t.npu.Event()
        complete.record()
        complete.synchronize()
        native_end = time.perf_counter()
        packed_result = contract and not input_contract
        returned = TensorAllocation(self.ascend, self.result_packet[:length] if packed_result else partial)
        download = self.peer.copy(
            ASCEND_TO_CUDA,
            self.output_export,
            returned,
            self.guard,
            length if packed_result else rows * OUTPUT_ROW_BYTES,
            source_ready=TorchSourceEvent(self.ascend, complete),
        )
        returned.close()
        if contract:
            # The sequence already resides inside the completed packet header.
            # Copy it to the completion word only after the entire packet DMA.
            source = (
                self.packed[rows * INPUT_ROW_BYTES + INPUT_SEQUENCE_OFFSET : rows * INPUT_ROW_BYTES + RECORD.size]
                if input_contract
                else self.result_packet[SEQUENCE_OFFSET:RESULT_OFFSET]
            )
            signal_source = TensorAllocation(self.ascend, source)
            signal_peer = self.notification_peer if input_contract else self.peer
            signal_export = self.control_export if input_contract else self.output_export
            signal_peer.copy(
                ASCEND_TO_CUDA,
                signal_export,
                signal_source,
                self.guard,
                8,
                source_ready=TorchSourceEvent(self.ascend, complete),
            )
            signal_source.close()
        else:
            # Legacy results are consumed through IPC in another CUDA context.
            self.driver.flush(all_devices=True)
        self.calls += 1
        self.single_route_calls += int(single)
        for receipt in (upload, download):
            creations, hits = mapping_counts(receipt)
            self.mapping_hits += hits
            self.mapping_creations += creations
            self.transfer_calls += creations + hits
        self.transfer_calls += int(contract)
        self.source_event.admitted = False
        self.upload_bytes += rows * INPUT_ROW_BYTES
        self.download_bytes += rows * OUTPUT_ROW_BYTES
        self.profile.append(
            {
                "sequence": sequence,
                "layer": layer,
                "rows": rows,
                "owned_routes": route_counts[self.rank],
                "single_route_graph": single,
                "upload_ms": (uploaded - start) * 1000,
                "unpack_and_validation_ms": (native_start - uploaded) * 1000,
                "native_ms": (native_end - native_start) * 1000,
                "pack_and_download_ms": (time.perf_counter() - native_end) * 1000,
            }
        )

    def close(self):
        self.peer.close()
        self.event.close()
        self.stream.close()
        self.input_export.close()
        if self.output_export is not None:
            self.output_export.close()
        if self.notification_peer is not None:
            self.notification_peer.close()
            self.control_export.close()
        for allocation in (self.npu_input, self.metadata_host, self.guard, self.gpu_input, self.gpu_output):
            if allocation is not None:
                allocation.close()
        if self.completion_contract and not self.input_record:
            self.result_header_host.close()
            self.result_packet = None


def quarantine(slots, output, rank, error):
    """Preserve original CUDA/HBM ownership independently of API process lifetime."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    receipt = {
        "pid": os.getpid(),
        "physical_chip": rank + 2,
        "state": "quarantined",
        "error": {"type": type(error).__name__, "message": str(error)},
        "automatic_kill": False,
        "gpu_and_hbm_owners_alive": True,
        "native_dma_uncertain": slots.peer.failed
        or bool(getattr(slots, "notification_peer", None) and slots.notification_peer.failed),
    }
    (output / f"dma-quarantine-rank{rank}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    while True:
        time.sleep(1)


def worker(args, spec):
    """Authenticate the private owner, gate banks, and serialize DMA invocations."""
    if os.geteuid() != 0 or os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "2,3":
        raise ValueError("direct expert worker requires root and assigned physical chips 2,3")
    prefill_rows = prefill_graph_rows(spec)
    single_route = single_route_enabled(spec)
    import torch
    import torch_npu

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.npu.set_device(args.rank)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    output = Path(spec["output"])
    layers = tuple(spec.get("cold_gate_layers", NPU_LAYERS))
    actor = NpuExperts(
        Path(spec["checkpoint"]),
        spec["checkpoint_sha256"],
        args.rank,
        torch,
        gate_override=spec["native_gate_override"],
        reduce_results=False,
        layers=layers,
    )
    actor.validate_repeat(output / f"expert-repeat-rank{args.rank}.json", combine_results=False)
    if spec.get("direct_dma", {}).get("npu_expert_graphs", False):
        with torch.inference_mode():
            actor = (SingleRouteDecodeGraphs if single_route else ExpertDecodeGraphs)(
                actor, torch, prefill_rows=prefill_rows
            )
            if single_route:
                validate_single_route(actor, torch, output / f"single-route-parity-rank{args.rank}.json")
            if prefill_rows is not None:
                validate_prefill_graphs(
                    actor.actor, actor, torch, output=output / f"expert-prefill-parity-rank{args.rank}.json"
                )
    slots = WorkerSlots(spec, args.rank, torch)
    # Ordinary termination is deferred to private socket EOF; never abandon in-flight DMA.
    signal.signal(signal.SIGTERM, lambda signum, frame: None)
    signal.signal(signal.SIGINT, lambda signum, frame: None)
    expected = 0
    released_cleanly = False
    try:
        with socket.socket(socket.AF_UNIX) as connection, torch.inference_mode():
            connection.connect(args.socket)
            pid, uid, _ = struct.unpack("3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
            if (pid, uid) != (args.parent_pid, args.parent_uid):
                raise ValueError("direct expert socket belongs to another coordinator")
            send_control(
                connection,
                {
                    "protocol": PROTOCOL,
                    "nonce": args.nonce,
                    "rank": args.rank,
                    "physical_chip": args.rank + 2,
                    "capacity": slots.capacity,
                    "checkpoint_sha256": spec["checkpoint_sha256"],
                    "placement_sha256": spec["expert_placement_sha256"],
                    "pid": os.getpid(),
                    "input": slots.gpu_input.export(),
                    **({} if slots.completion_contract else {"output": slots.gpu_output.export()}),
                    "cuda_completion_contract": slots.completion_contract,
                },
            )
            reply = receive_control(connection)
            if slots.completion_contract:
                slots.register_result(connection, reply, args.nonce)
            elif reply != {"source_readiness": "sender-event-synchronized", "nonce": args.nonce}:
                raise ValueError("direct source-event handshake differs")
            send_control(connection, {"ready": True, "nonce": args.nonce})
            while True:
                tag = bytearray(COMMAND.size)
                try:
                    receive(connection, memoryview(tag))
                except EOFError:
                    if slots.calls:
                        raise RuntimeError("CUDA IPC owner disappeared without a mapping-release acknowledgement")
                    break
                if tag == CLOSE_TAG:
                    if slots.completion_contract:
                        # Release foreign mappings/fd before the GPU owner is
                        # permitted to free its result allocation.
                        slots.release_result_mapping()
                    connection.sendall(tag)
                    released = bytearray(len(RELEASE))
                    receive(connection, memoryview(released))
                    if released != RELEASE:
                        raise ValueError("direct CUDA mapping release acknowledgement differs")
                    released_cleanly = True
                    break
                sequence, layer, rows = validate_command(tag, expected, layers, max_chunk=slots.max_chunk)
                slots.source_event.admitted = True
                slots.compute(actor, sequence, layer, rows)
                connection.sendall(tag)
                expected += 1
                if expected % RECEIPT_INTERVAL == 0:
                    path = output / f"dma-worker-rank{args.rank}.json"
                    temporary = path.with_suffix(".json.tmp")
                    temporary.write_text(json.dumps(slots.receipt("running")) + "\n")
                    temporary.replace(path)
    except BaseException as error:
        if slots.peer.failed or ((slots.calls or slots.source_event.admitted) and not released_cleanly):
            quarantine(slots, output, args.rank, error)
        raise
    finally:
        if not slots.peer.failed:
            guard_exact = (
                ctypes.string_at(slots.guard.pointer, slots.capacity) == bytes([GUARD_POISON]) * slots.capacity
            )
            receipt = dict(slots.receipt("closed"), guard_poison_exact=guard_exact)
            if isinstance(actor, ExpertDecodeGraphs):
                actor.close()
            slots.close()
            (output / f"dma-worker-rank{args.rank}.json").write_text(json.dumps(receipt, indent=2) + "\n")
            if not guard_exact:
                raise AssertionError("unused HBM lifetime guard was modified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--parent-uid", type=int, required=True)
    parser.add_argument("--bootstrap", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.config.read_bytes())
    if args.bootstrap:
        environment = dict(os.environ)
        environment.update(json.loads(Path(spec["npu_environment"]).read_bytes()))
        environment["ASCEND_RT_VISIBLE_DEVICES"] = "2,3"
        environment["PYTHONPATH"] = (
            str(args.config.resolve().parent / "source") + ":" + environment.get("PYTHONPATH", "")
        )
        command = [
            spec["npu_python"],
            "-m",
            "tools.sidecard.dma_worker",
            *[v for v in sys.argv[1:] if v != "--bootstrap"],
        ]
        os.execve(command[0], command, environment)
    worker(args, spec)


if __name__ == "__main__":
    main()
