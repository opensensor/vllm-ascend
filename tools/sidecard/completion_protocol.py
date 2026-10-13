# SPDX-License-Identifier: Apache-2.0
"""Private one-time registration of coordinator-owned CUDA DMA-BUF results."""

import array
import os
import socket

from .dma_protocol import CAPACITY, OUTPUT_ROW_BYTES, dma_capacity
from .layer_weights import expert_batch_size
from .pcie.mailbox import RECORD, Invocation

RESULT_OFFSET = 8 + RECORD.size
SEQUENCE_OFFSET = 8 + 8 + 8
FD_MARKER = b"R"
CONTROL_CAPACITY = 65536
INPUT_SEQUENCE_OFFSET = 16


def enabled(spec):
    options = spec.get("direct_dma", {})
    value = options.get("cuda_completion_contract", False)
    if type(value) is not bool:
        raise ValueError("CUDA completion contract selection must be boolean")
    separate = options.get("completion_input_record", False)
    if type(separate) is not bool:
        raise ValueError("input-record completion selection must be boolean")
    header_bytes = 0 if separate else RESULT_OFFSET
    if value and expert_batch_size(spec) * OUTPUT_ROW_BYTES + header_bytes > dma_capacity(spec):
        raise ValueError("completion header and prefill results exceed their dedicated allocation")
    return value


def input_record(spec):
    value = spec.get("direct_dma", {}).get("completion_input_record", False)
    if type(value) is not bool or (value and not enabled(spec)):
        raise ValueError("input-record completion requires the CUDA completion contract")
    return value


def epoch(nonce):
    if not isinstance(nonce, str) or len(nonce) != 64:
        raise ValueError("completion contract requires an authenticated session nonce")
    return int(nonce[:16], 16) or 1


def result_header(generation, sequence, layer, rows, rank):
    return bytes(8) + Invocation(generation, sequence + 1, layer, rows, rank, rank).encode(slots=2)


def send_fd(connection, fd):
    descriptors = array.array("i", [fd])
    sent = connection.sendmsg([FD_MARKER], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, descriptors)])
    if sent != len(FD_MARKER):
        raise RuntimeError("result DMA-BUF registration was incomplete")


def receive_fd(connection):
    descriptors = array.array("i")
    body, ancillary, flags, _ = connection.recvmsg(
        len(FD_MARKER), socket.CMSG_SPACE(descriptors.itemsize * 2), socket.MSG_CMSG_CLOEXEC
    )
    for level, kind, data in ancillary:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            descriptors.frombytes(data[: len(data) - len(data) % descriptors.itemsize])
    if body != FD_MARKER or flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC) or len(descriptors) != 1:
        for fd in descriptors:
            os.close(fd)
        raise ValueError("result registration requires exactly one complete DMA-BUF descriptor")
    return descriptors[0]


class RemoteAllocation:
    """Worker-side lifetime token; GPU addresses are never imported or consumed."""

    def __init__(self, runtime, size):
        self.runtime, self.size, self.leases = runtime, size, 1


class RemoteResult:
    """Hold a duplicate fd until completed DMA and bridge mappings are released."""

    def __init__(self, runtime, fd, *, size=CAPACITY):
        self.allocation = RemoteAllocation(runtime, size)
        self.fd = fd
        self.driver = self

    def flush(self, *, all_devices=False):
        self.allocation.runtime.check_owner()
        if all_devices:
            raise ValueError("remote results qualify only the coordinator's owner context")
        # The authenticated receiver prevalidated native owner-context ordering.
        # This worker never consumes the remote GPU allocation itself.

    def close(self):
        self.allocation.runtime.check_owner()
        if self.fd is None:
            return
        if self.allocation.leases != 1:
            raise RuntimeError("remote result is retained by an active or uncertain DMA mapping")
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
            self.allocation.leases = 0
