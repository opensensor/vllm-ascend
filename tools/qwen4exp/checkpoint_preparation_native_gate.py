# SPDX-License-Identifier: Apache-2.0
"""Private 72-bank numerical and paired graph timing gate for fused pre-copy."""

import statistics
import time

import torch

from tools.qwen4exp.checkpoint_preparation import prepare_checkpoint
from tools.qwen4exp.checkpoint_preparation_native import CheckpointPrecopyGroup
from tools.qwen4exp.mamba_commit import MambaCommitGroup


def oracle(state, table, previous, computed, accepted, conv):
    result = state.clone()
    receipt = torch.zeros((2, 8), dtype=torch.int32)
    receipt[0].fill_(1)
    for row, (source, position, count) in enumerate(zip(previous.tolist(), computed.tolist(), accepted.tolist())):
        if source < -1 or position < 0 or count < 1 or count > 3:
            receipt[1, row] = 1
            continue
        destination = (position + 2) // 128
        copy = source >= 0 and source != destination
        receipt[0, row] = 1 if copy else count
        if not copy:
            continue
        src_col = source if conv else source + count - 1
        if src_col >= table.shape[1] or destination >= table.shape[1]:
            receipt[1, row] = 2
            continue
        src, dst = int(table[row, src_col]), int(table[row, destination])
        if min(src, dst) < 0 or max(src, dst) >= state.shape[0]:
            receipt[1, row] = 3
            continue
        selected = result[src, count - 1 :] if conv else result[src]
        result[dst].reshape(-1)[: selected.numel()].copy_(selected.reshape(-1).clone())
    return result, receipt


@torch.inference_mode()
def gate(owner, kernel, launch, commit, device):
    """Caller retains owner even on failure; release only after synchronized pass."""
    generator = torch.Generator().manual_seed(1417)
    host = []
    for _ in range(36):
        host.extend(
            (
                torch.randint(-32768, 32767, (4, 5, 2560), dtype=torch.int16, generator=generator).view(torch.float16),
                torch.randint(
                    -(1 << 31), (1 << 31) - 1, (4, 12, 128, 128), dtype=torch.int32, generator=generator
                ).view(torch.float32),
            )
        )
    conv = (True, False) * 36
    owner["banks"] = banks = [bank.to(device) for bank in host]
    owner["values"] = values = [torch.zeros(4, dtype=torch.int32, device=device) for _ in range(3)]
    values[1].fill_(10)
    values[2].fill_(1)
    owner["table"] = table = torch.arange(32, dtype=torch.int32, device=device).reshape(4, 8) % 4
    owner["records"] = records = {}
    owner["kernel"] = kernel
    stream = torch.npu.Stream(device=device)
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for n in range(1, 5):
            resource = CheckpointPrecopyGroup(kernel, launch, banks, [table[:n]] * 72, conv, block_size=128)
            record = dict(resource=resource, graph=torch.npu.NPUGraph())
            records[n] = record
            resource(*(value[:n] for value in values))
            torch.npu.synchronize()
            with torch.npu.graph(record["graph"], stream=stream):
                resource(*(value[:n] for value in values))
            torch.npu.synchronize()
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    checked = 0
    for n, record in records.items():
        for revision in range(10):
            previous = torch.tensor([0, 1, 0, -1], dtype=torch.int32)[:n]
            computed = torch.tensor([126, 126, 10, 254], dtype=torch.int32)[:n]
            accepted = (torch.arange(n, dtype=torch.int32) + revision) % 3 + 1
            ht = (torch.arange(n * 8, dtype=torch.int32).reshape(n, 8) + revision) % 4
            if revision == 4:
                previous[0] = -2
            elif revision == 5:
                computed[0] = -1
            elif revision == 6:
                accepted[0] = 4
            elif revision == 7:
                previous[0] = 8
            elif revision == 8:
                ht.fill_(-1)
            elif revision == 9:
                ht.fill_(4)
            for dest, src in zip(values, (previous, computed, accepted)):
                dest[:n].copy_(src)
            table[:n].copy_(ht)
            for bank, initial in zip(banks, host):
                bank.copy_(initial)
            record["graph"].replay()
            receipts = record["resource"].receipt.cpu()
            for index, (bank, initial, convolution) in enumerate(zip(banks, host, conv)):
                expected, receipt = oracle(initial, ht, previous, computed, accepted, convolution)
                torch.testing.assert_close(bank.cpu().view(torch.int32), expected.view(torch.int32), rtol=0, atol=0)
                torch.testing.assert_close(receipts[index], receipt, rtol=0, atol=0)
                checked += 1
    # Paired timing of the previous tensor-decision graph and fused graph.
    values[0].zero_()
    values[1].fill_(10)
    values[2].fill_(1)
    table.copy_(torch.arange(32, dtype=torch.int32, device=device).reshape(4, 8) % 4)
    owner["inputs"] = inputs = torch.zeros((5, 8), dtype=torch.int32, device=device)
    owner["old_resource"] = old = MambaCommitGroup(
        commit.records[4][0].kernel, commit.records[4][0].launch, banks, [table] * 72, conv, block_size=128
    )
    owner["old_graph"] = old_graph = torch.npu.NPUGraph()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        prepare_checkpoint(*values, inputs, block_size=128)
        old(inputs)
        torch.npu.synchronize()
        with torch.npu.graph(old_graph, stream=stream):
            prepare_checkpoint(*values, inputs, block_size=128)
            old(inputs)
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    timing = {}
    for mode in ("no_copy", "boundary_copy"):
        values[1].fill_(10 if mode == "no_copy" else 126)
        values[2].fill_(3)
        samples = {"tensor_graph": [], "fused_graph": []}
        for repeat in range(4):
            order = [("tensor_graph", old_graph), ("fused_graph", records[4]["graph"])]
            for name, graph in order if repeat % 2 == 0 else reversed(order):
                torch.npu.synchronize()
                start = time.perf_counter()
                for _ in range(5):
                    graph.replay()
                torch.npu.synchronize()
                samples[name].append((time.perf_counter() - start) * 1000 / 5)
        timing[mode] = dict(samples_ms=samples, median_ms={k: statistics.median(v) for k, v in samples.items()})
    torch.npu.synchronize()
    for record in records.values():
        record["graph"].reset()
    old_graph.reset()
    owner.clear()
    return dict(passed=True, exact_bank_checks=checked, banks=72, slots=4, timing=timing)
