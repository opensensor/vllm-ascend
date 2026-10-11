# SPDX-License-Identifier: Apache-2.0
"""Opt-in 310P pipeline transport with one collective per tensor dictionary."""

import math

import torch

from vllm_ascend.distributed.device_communicators.pipeline_all_gather import PipelineAllGatherTransport

PACKET_VERSION = 1
PACKET_ALIGNMENT = 32
DEFAULT_PAYLOAD_LIMIT = 256 * 1024**2


def aligned_bytes(size):
    return (size + PACKET_ALIGNMENT - 1) // PACKET_ALIGNMENT * PACKET_ALIGNMENT


def validate_header(header, limit):
    """Validate CPU metadata before allocating device storage or entering HCCL."""
    if (
        not isinstance(header, dict)
        or type(header.get("version")) is not int
        or header.get("version") != PACKET_VERSION
    ):
        raise ValueError("unsupported packed pipeline protocol")
    size, entries = header.get("bytes"), header.get("entries")
    if type(size) is not int or not 0 <= size <= limit or size % PACKET_ALIGNMENT or not isinstance(entries, list):
        raise ValueError("invalid packed pipeline payload size")
    keys, previous_end = set(), 0
    for entry in entries:
        if not isinstance(entry, tuple) or len(entry) < 2:
            raise ValueError("invalid packed pipeline field")
        key, kind = entry[:2]
        if not isinstance(key, str) or key in keys:
            raise ValueError("duplicate or invalid packed pipeline key")
        keys.add(key)
        if kind == "value" and len(entry) == 3:
            continue
        if kind != "tensor" or len(entry) != 6:
            raise ValueError("invalid packed pipeline tensor field")
        dtype, shape, offset, length = entry[2:]
        if not isinstance(dtype, torch.dtype) or not isinstance(shape, tuple):
            raise ValueError("invalid packed pipeline tensor metadata")
        if any(type(dim) is not int or dim < 0 for dim in shape):
            raise ValueError("invalid packed pipeline tensor shape")
        try:
            element_size = torch.empty((), dtype=dtype, device="cpu").element_size()
        except (RuntimeError, TypeError) as error:
            raise ValueError("unsupported packed pipeline tensor dtype") from error
        if (
            type(offset) is not int
            or type(length) is not int
            or offset < previous_end
            or offset % PACKET_ALIGNMENT
            or length != math.prod(shape) * element_size
            or offset + length > size
        ):
            raise ValueError("overlapping or out-of-bounds packed pipeline field")
        previous_end = offset + length
    if aligned_bytes(previous_end) != size:
        raise ValueError("packed pipeline payload has unaccounted trailing bytes")
    return size, entries


class PackedPipelineAllGatherTransport(PipelineAllGatherTransport):
    """Preserve mixed dtypes and independent packet lifetimes in one byte slab.

    This candidate adds device-to-device packing copies, but reduces a four-field
    GLM mHC handoff from four HCCL collectives to one. No device values are read
    on CPU. Every receive owns its gathered slab; subsequent tokens cannot reuse
    or overwrite it. Hardware latency and memory qualification remain required.
    """

    def __init__(self, *args, max_payload_bytes=DEFAULT_PAYLOAD_LIMIT, **kwargs):
        if type(max_payload_bytes) is not int or max_payload_bytes <= 0:
            raise ValueError("packed pipeline payload limit must be positive")
        super().__init__(*args, **kwargs)
        self.max_payload_bytes = max_payload_bytes

    @classmethod
    def take_ownership(cls, original, *, max_payload_bytes=DEFAULT_PAYLOAD_LIMIT):
        """Upgrade a cold communicator using its already warmed process groups.

        No additional group is created or warmed. The caller replaces the
        communicator's transport immediately; all ranks must choose this same
        protocol before any model request or graph capture.
        """
        if type(original) is not PipelineAllGatherTransport or not original.groups:
            raise ValueError("requires an owned, initialized legacy pipeline transport")
        if type(max_payload_bytes) is not int or max_payload_bytes <= 0:
            raise ValueError("packed pipeline payload limit must be positive")
        result = object.__new__(cls)
        result.__dict__.update(original.__dict__)
        result.max_payload_bytes = max_payload_bytes
        original.groups = {}  # exactly one owner destroys these process groups
        return result

    def send_tensor_dict(self, packet, peer=None):
        peer = self.rank + 1 if peer is None else peer
        self._pair(peer)
        entries, payload, size = [], [], 0
        for key, value in packet.items():
            if isinstance(value, torch.Tensor) and value.device.type == self.device.type:
                if value.layout != torch.strided or value.is_quantized:
                    raise ValueError("packed pipeline tensors must have strided byte-addressable storage")
                offset = aligned_bytes(size)
                size = offset + value.numel() * value.element_size()
                entries.append((key, "tensor", value.dtype, tuple(value.shape), offset, size - offset))
                payload.append((offset, value))
            else:
                if isinstance(value, torch.Tensor) and value.device.type != "cpu":
                    raise ValueError("pipeline packet contains a tensor on an unexpected device")
                entries.append((key, "value", value))
        header = {"version": PACKET_VERSION, "bytes": aligned_bytes(size), "entries": entries}
        size, _ = validate_header(header, self.max_payload_bytes)
        # Padding is deterministic; no uninitialized bytes enter the collective.
        packed = torch.zeros(size, dtype=torch.int8, device=self.device)
        for offset, value in payload:
            raw = value.contiguous().reshape(-1).view(torch.int8)
            packed[offset : offset + raw.numel()].copy_(raw)
        self.backend.send_object_list([header], dst=self.ranks[peer], group=self.cpu_group)
        if size:
            self._exchange(packed, peer, self.global_rank)

    def recv_tensor_dict(self, peer=None):
        peer = self.rank - 1 if peer is None else peer
        self._pair(peer)
        objects = [None]
        self.backend.recv_object_list(objects, src=self.ranks[peer], group=self.cpu_group)
        size, entries = validate_header(objects[0], self.max_payload_bytes)
        packed = torch.empty(size, dtype=torch.int8, device=self.device)
        gathered = self._exchange(packed, peer, self.ranks[peer]) if size else packed
        packet = {}
        for key, kind, *value in entries:
            if kind == "value":
                packet[key] = value[0]
            else:
                dtype, shape, offset, length = value
                packet[key] = gathered[offset : offset + length].view(dtype).reshape(shape)
        return packet
