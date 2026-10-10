# SPDX-License-Identifier: Apache-2.0
"""310P pipeline packets through supported pairwise AllGather collectives."""

import torch
import torch.distributed as dist

HCCL_WARMUP_BYTES = 32


class PipelineAllGatherTransport:
    """Preserve tensor bytes without HcclSend/Recv or activation host copies.

    Adjacent pipeline stages share two-rank HCCL groups. CPU communication
    carries shapes and non-device control values only. Receive storage has
    exactly twice the incoming payload size and is owned by the returned
    tensors, so a later request cannot overwrite a still-live packet.
    """

    def __init__(self, cpu_group, ranks, global_rank, device, backend=dist):
        if len(ranks) < 2 or len(set(ranks)) != len(ranks) or global_rank not in ranks:
            raise ValueError("pipeline transport requires distinct ranks and a local member")
        self.cpu_group = cpu_group
        self.ranks = tuple(ranks)
        self.global_rank = global_rank
        self.rank = self.ranks.index(global_rank)
        self.device = torch.device(device)
        self.backend = backend
        self.groups = {}
        # Forward/backward transfers use a line. Retain the bootstrap PG creation
        # order used by the qualified transport; the closing PG remains dormant.
        pairs = {tuple(sorted((self.ranks[index], self.ranks[index + 1]))) for index in range(len(ranks) - 1)}
        self.pairs = tuple(sorted(pairs))
        creation_pairs = pairs | {tuple(sorted((self.ranks[0], self.ranks[-1])))}
        # The same ordering on all members avoids overlapping-group deadlocks.
        for pair in sorted(creation_pairs):
            group = backend.new_group(list(pair), backend="hccl", use_local_synchronization=True)
            if global_rank in pair:
                self.groups[pair] = group

    def _pair(self, peer):
        if type(peer) is not int or not 0 <= peer < len(self.ranks) or peer == self.rank:
            raise ValueError("invalid pipeline peer")
        pair = tuple(sorted((self.global_rank, self.ranks[peer])))
        if pair not in self.pairs:
            raise ValueError("pipeline transport supports adjacent stages only")
        return pair, self.groups[pair]

    def warmup(self, synchronize=None):
        """Initialize lazy HCCL storage before model/cache memory profiling."""
        for pair in self.pairs:
            # All stage members observe the same initialization order, including
            # stages outside this pair. Overlapping lazy HCCL initializations
            # must not race one another through their root-info bootstrap.
            self.backend.barrier(group=self.cpu_group)
            if self.global_rank in pair:
                packet = torch.empty(HCCL_WARMUP_BYTES, dtype=torch.int8, device=self.device)
                gathered = torch.empty(HCCL_WARMUP_BYTES * 2, dtype=torch.int8, device=self.device)
                self.backend.all_gather_into_tensor(gathered, packet, group=self.groups[pair])
                if synchronize is not None:
                    synchronize()
            self.backend.barrier(group=self.cpu_group)

    def _exchange(self, tensor, peer, source):
        pair, group = self._pair(peer)
        packed = tensor.contiguous().reshape(-1).view(torch.int8)
        gathered = torch.empty(packed.numel() * len(pair), dtype=torch.int8, device=self.device)
        if packed.numel():
            self.backend.all_gather_into_tensor(gathered, packed, group=group)
        offset = pair.index(source) * packed.numel()
        return gathered[offset : offset + packed.numel()].view(tensor.dtype).reshape(tensor.shape)

    def send_tensor_dict(self, packet, peer=None):
        peer = (self.rank + 1) % len(self.ranks) if peer is None else peer
        self._pair(peer)
        metadata = []
        payload = []
        for key, value in packet.items():
            if isinstance(value, torch.Tensor) and value.device.type == self.device.type:
                metadata.append((key, "tensor", value.dtype, tuple(value.shape)))
                payload.append(value)
            else:
                if isinstance(value, torch.Tensor) and value.device.type != "cpu":
                    raise ValueError("pipeline packet contains a tensor on an unexpected device")
                metadata.append((key, "value", value))
        self.backend.send_object_list([metadata], dst=self.ranks[peer], group=self.cpu_group)
        for tensor in payload:
            self._exchange(tensor, peer, self.global_rank)

    def recv_tensor_dict(self, peer=None):
        peer = (self.rank - 1) % len(self.ranks) if peer is None else peer
        self._pair(peer)
        objects = [None]
        self.backend.recv_object_list(objects, src=self.ranks[peer], group=self.cpu_group)
        packet = {}
        for key, kind, *value in objects[0]:
            if kind == "value":
                packet[key] = value[0]
            elif kind == "tensor":
                dtype, shape = value
                # Receiver input bytes are irrelevant; select the sender's slice.
                tensor = torch.empty(shape, dtype=dtype, device=self.device)
                packet[key] = self._exchange(tensor, peer, self.ranks[peer])
            else:
                raise ValueError("unknown pipeline packet metadata")
        return packet

    def destroy(self):
        for group in self.groups.values():
            self.backend.destroy_process_group(group)
        self.groups.clear()
