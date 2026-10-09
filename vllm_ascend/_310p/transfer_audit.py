# SPDX-License-Identifier: Apache-2.0
"""Host-only accounting of submitted payloads and explicit host barriers.

These are logical copy bytes, not measured DDR/HCCL bus traffic. Device events
are never queried, tensor values are never read, and no synchronization is
introduced. Bounded timestamp records are optional; cumulative counts remain.
"""

from collections import Counter, deque
from time import monotonic_ns, time_ns


class TransferLedger:
    def __init__(self, event_limit: int = 0) -> None:
        if type(event_limit) is not int or not 0 <= event_limit <= 4096:
            raise ValueError("event_limit must be an integer in [0, 4096]")
        self._counts = Counter()
        self._events = deque(maxlen=event_limit)
        self._sequence = 0

    def enable_events(self, event_limit: int = 256) -> None:
        if type(event_limit) is not int or not 1 <= event_limit <= 4096:
            raise ValueError("event_limit must be an integer in [1, 4096]")
        if self._events.maxlen != event_limit:
            self._events = deque(self._events, maxlen=event_limit)

    def record(self, kind: str, reason: str, *, nbytes: int = 0, elapsed_ns: int = 0) -> None:
        if nbytes < 0 or elapsed_ns < 0:
            raise ValueError("negative transfer bytes or barrier duration")
        self._counts[f"{kind}_calls"] += 1
        self._counts[f"{kind}_bytes"] += nbytes
        self._counts[f"{kind}_host_ns"] += elapsed_ns
        self._counts[f"reason:{reason}:calls"] += 1
        self._sequence += 1
        if self._events.maxlen:
            self._events.append(
                {
                    "sequence": self._sequence,
                    "wall_ns": time_ns(),
                    "monotonic_ns": monotonic_ns(),
                    "kind": kind,
                    "reason": reason,
                    "bytes": nbytes,
                    "host_ns": elapsed_ns,
                }
            )

    def snapshot(self) -> dict:
        return {
            "counts": dict(self._counts),
            "events": [dict(event) for event in self._events],
            "sequence": self._sequence,
            "dropped_events": max(0, self._sequence - len(self._events)) if self._events.maxlen else 0,
            "measured_bus_bytes": False,
        }


def copy_direction(source_device, target_device) -> str:
    source_cpu, target_cpu = source_device.type == "cpu", target_device.type == "cpu"
    if source_cpu and target_cpu:
        return "host_copy"
    if source_cpu:
        return "h2d"
    return "d2h" if target_cpu else "d2d"
