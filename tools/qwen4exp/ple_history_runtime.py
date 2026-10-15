# SPDX-License-Identifier: Apache-2.0
"""Device-owned PLE history for admitted, unchanged MTP2 request rows.

The existing host-paged PLE table still needs a host history mirror. This owner
does not enable async scheduling: it first qualifies history continuity without
using sampled-token placeholders or rejected drafts as authoritative history.
"""

import numpy as np
import torch

from tools.qwen4exp.iteration_transition import advance_ple_history


def uniform_history_rows(runner, padded):
    """Check shape/identity metadata without reading a device decision."""
    batch = runner.input_batch
    n = batch.num_reqs
    return (
        1 <= n <= 4
        and not runner.with_prefill
        and padded == n * 3
        and np.array_equal(runner.query_start_loc.np[: n + 1], np.arange(n + 1) * 3)
    )


class ResidentPLEHistory:
    """Retain history graphs and bootstrap only at request-layout boundaries."""

    def __init__(self, transition):
        self.transition, self.runner = transition, transition.runner
        self.enabled = self.verify = self.failed = False
        self.staged = self.pending = self.continuation = None
        self.checked_histories = 0
        self.history = torch.full_like(self.runner._ple_context_gpu[:4], self.runner._ple_eos_token_id)
        self.history_width = self.history.shape[1]
        self.packet = torch.zeros((4, self.history_width + 3), dtype=torch.int64, device=self.history.device)
        self.host_packet = torch.empty(self.packet.shape, dtype=torch.int64, pin_memory=True)
        self.first_input = torch.zeros(4, dtype=torch.int64, device=self.history.device)
        self.graphs = {}
        self.done = torch.npu.Event()
        self.early_readback = False
        self.packet_done = torch.npu.Event()
        self.packet_requests = None
        self.early_packets = self.ready_packets = 0
        stream = torch.npu.Stream(device=self.history.device)
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for n, record in transition.records.items():
                graph = torch.npu.NPUGraph()
                self.graphs[n] = graph

                def advance(n=n, record=record):
                    history = advance_ple_history(
                        self.history[:n], self.first_input[:n], transition.samples[:n], record["accepted"]
                    )
                    self.history[:n].copy_(history)
                    self.packet[:n, : self.history_width].copy_(history)
                    self.packet[:n, self.history_width :].copy_(record["next"].input_ids)

                for _ in range(3):
                    advance()
                torch.npu.synchronize()
                with torch.npu.graph(graph, stream=stream):
                    advance()
                torch.npu.synchronize()
        torch.npu.current_stream().wait_stream(stream)
        torch.npu.synchronize()

    def status(self):
        return dict(
            enabled=self.enabled,
            verify=self.verify,
            failed=self.failed,
            checked_histories=self.checked_histories,
            pending=self.pending is not None,
            host_paged_lookup_retained=True,
            host_hash_inputs_from_device=True,
            host_scheduler_retained=True,
            early_readback=self.early_readback,
            early_packets=self.early_packets,
            ready_packets=self.ready_packets,
        )

    def input_result(self, used):
        self.continuation = self.pending if used else None
        self.pending = None

    def stage(self, padded, original):
        if self.failed:
            raise RuntimeError("PLE history submission failed; retain graph owners")
        runner, batch = self.runner, self.runner.input_batch
        requests = tuple(batch.req_ids)
        continuation, self.continuation = self.continuation, None
        if not self.enabled or not self.transition.enabled or not uniform_history_rows(runner, padded):
            self.staged = None
            return original(runner, padded)
        n = batch.num_reqs
        self.failed = True
        if continuation == requests:
            torch.npu.current_stream().wait_event(self.done)
            if self.verify:
                original(runner, padded)
                torch.testing.assert_close(self.history[:n].cpu(), runner._ple_context_cpu[:n], rtol=0, atol=0)
                torch.testing.assert_close(
                    self.packet[:n, self.history_width :].cpu().to(runner.input_ids.cpu.dtype),
                    runner.input_ids.cpu[:padded].reshape(n, 3),
                    rtol=0,
                    atol=0,
                )
                self.checked_histories += n
            else:
                # A temporary adapter for the host-paged table. This explicit
                # readback remains a dependency; do not label it autonomous.
                if self.early_readback:
                    if self.packet_requests != requests:
                        raise RuntimeError("PLE readback packet belongs to a different iteration layout")
                    self.ready_packets += self.packet_done.query()
                    self.packet_done.synchronize()
                    self.packet_requests = None
                else:
                    self.host_packet[:n].copy_(self.packet[:n])
                runner._ple_context_cpu.fill_(runner._ple_eos_token_id)
                runner._ple_context_cpu[:n].copy_(self.host_packet[:n, : self.history_width])
                # The host-paged hasher must use the same device-owned token
                # IDs as the target; CPU token placeholders are never valid.
                runner.input_ids.cpu[:padded].copy_(self.host_packet[:n, self.history_width :].reshape(-1))
                runner._ple_query_start_loc_cpu.fill_(padded)
                runner._ple_query_start_loc_cpu[: n + 1].copy_(
                    torch.arange(n + 1, dtype=runner._ple_query_start_loc_cpu.dtype) * 3
                )
                runner._ple_query_start_loc_gpu.copy_(runner._ple_query_start_loc_cpu, non_blocking=True)
            runner._ple_context_gpu.fill_(runner._ple_eos_token_id)
            runner._ple_context_gpu[:n].copy_(self.history[:n])
            result = runner._ple_query_start_loc_gpu, runner._ple_context_gpu
        else:
            result = original(runner, padded)
            self.history[:n].copy_(runner._ple_context_gpu[:n])
        self.first_input[:n].copy_(runner.input_ids.gpu[: n * 3].reshape(n, 3)[:, 0])
        self.staged = requests
        self.failed = False
        return result

    def advance(self):
        next_pending = self.transition.next_pending
        self.pending = None
        if not self.enabled or self.staged is None or next_pending is None:
            return
        requests, _ = next_pending
        if requests != self.staged:
            return
        self.failed = True
        torch.npu.current_stream().wait_event(self.transition.next_done)
        self.graphs[len(requests)].replay()
        self.done.record()
        if self.early_readback and not self.verify:
            # Queue on the producer stream before next-step metadata uploads.
            # This single buffer is consumed by stage() before the next advance;
            # abandoned layouts may overwrite it in the same stream order.
            self.host_packet[: len(requests)].copy_(self.packet[: len(requests)], non_blocking=True)
            self.packet_done.record()
            self.packet_requests = requests
            self.early_packets += 1
        self.pending = requests
        self.staged = None
        self.failed = False
