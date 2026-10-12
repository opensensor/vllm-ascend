# SPDX-License-Identifier: Apache-2.0
"""Object-local socket and device-event observation at existing sync boundaries."""

import json
import time
from pathlib import Path


class TimedPeer:
    def __init__(self, connection, rank, clock=time.perf_counter, *, header_size=0, header_decoder=None):
        self.connection, self.rank, self.clock = connection, rank, clock
        self.record = None
        self.header_size, self.header_decoder = header_size, header_decoder

    def observe(self, direction, operation, data):
        begin = self.clock()
        result = operation(data)
        end = self.clock()
        if self.record is not None:
            record = dict(
                rank=self.rank,
                direction=direction,
                begin=begin,
                end=end,
                bytes=result if direction == "receive" else len(data),
            )
            if direction == "send" and self.header_decoder is not None and len(data) == self.header_size:
                record["header"] = self.header_decoder(data)
            self.record.append(record)
        return result

    def sendall(self, data):
        return self.observe("send", self.connection.sendall, data)

    def recv_into(self, data):
        return self.observe("receive", self.connection.recv_into, data)

    def close(self):
        self.connection.close()


class WorkerTimers:
    def __init__(self, output, rank, backend, clock=time.perf_counter):
        self.output, self.rank, self.backend, self.clock = Path(output), rank, backend, clock
        self.records = []
        self.current = None

    def begin(self, sequence, layer, rows):
        if self.current is not None:
            raise RuntimeError("overlapping expert profile invocation")
        self.current = dict(sequence=sequence, layer=layer, rows=rows, rank=self.rank, begin=self.clock(), events=[])

    def mark(self, name):
        event = self.backend.Event(enable_timing=True)
        event.record()
        self.current["events"].append((name, event))

    def finish(self):
        # Caller has already completed its original NPU synchronization.
        # Reading event durations introduces no additional synchronization.
        record, self.current = self.current, None
        events = record.pop("events")
        if [name for name, _ in events] != ["upload", "native", "download", "complete"]:
            raise ValueError("incomplete worker timing boundaries")
        record["end"] = self.clock()
        record["stream_ms"] = {
            name: start.elapsed_time(end) for (name, start), (_, end) in zip(events[:-1], events[1:], strict=True)
        }
        self.records.append(record)

    def close(self):
        self.output.write_text("".join(json.dumps(record) + "\n" for record in self.records))
