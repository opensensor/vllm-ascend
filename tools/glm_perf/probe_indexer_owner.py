# SPDX-License-Identifier: Apache-2.0
"""torchrun protocol gate; Gloo is CPU-only, HCCL requires explicit reservation.

Synthetic indices test collective ordering, bytes, and asymmetric graph work.
This does not qualify the real indexer, model quality, or serving performance.
"""

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from .indexer_owner import IndexerResultExchange, bind_indexer


class ProbeIndexer(torch.nn.Module):
    def __init__(self, buffer):
        super().__init__()
        self.topk_indices_buffer = buffer
        self.indexer_op = SimpleNamespace(topk_indices_buffer=buffer)

    def forward(self, hidden_states, qr, positions, rotary_emb):
        rows = hidden_states.shape[0]
        self.topk_indices_buffer[:rows].fill_(-1)
        self.topk_indices_buffer[:rows, 0].copy_(positions)
        self.topk_indices_buffer[:rows, 1].fill_(16_777_217)
        return self.topk_indices_buffer


def run(*, backend, tp_size, owner_rank, tokens, output, graphs=False, overlap_prefill=False):
    if backend not in ("gloo", "hccl") or tp_size not in (2, 4) or not 0 <= owner_rank < tp_size:
        raise ValueError("protocol gate requires Gloo/HCCL and a valid TP2/TP4 local owner")
    if not tokens or any(type(rows) is not int or rows < 1 for rows in tokens) or graphs and backend != "hccl":
        raise ValueError("positive row counts and HCCL-only graph gate required")
    device = torch.device("cpu")
    if backend == "hccl":
        import torch_npu  # noqa: F401

        device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
        torch.npu.set_device(device)
    # CPU bootstrap avoids a redundant world-sized device communicator. The
    # production candidate reuses existing TP groups instead of creating any.
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    rank, world = dist.get_rank(), dist.get_world_size()
    groups = []
    try:
        if world % tp_size:
            raise ValueError("torchrun world size must divide into complete TP groups")
        local_group = None
        for start in range(0, world, tp_size):
            members = list(range(start, start + tp_size))
            group = dist.new_group(members, backend=backend, timeout=timedelta(seconds=60))
            if rank in members:
                local_group = group
                groups.append(group)
        stage, local_rank = divmod(rank, tp_size)
        width, capacity = 2176, max(tokens)
        buffer = torch.full((capacity, width), -987, dtype=torch.int32, device=device)
        exchange = IndexerResultExchange(
            group=local_group,
            rank=local_rank,
            size=tp_size,
            owner_rank=owner_rank,
            max_tokens=capacity,
            width=width,
            device=device,
            budget_bytes=tp_size * capacity * width * 4,
            overlap_prefill=overlap_prefill,
        )
        instance = ProbeIndexer(buffer)
        bind_indexer(instance, exchange)
        exchange.warmup(synchronize=torch.npu.synchronize if backend == "hccl" else None)
        cases = []
        for rows in tokens:
            x = torch.zeros(rows, 1, dtype=torch.float16, device=device)
            positions = (torch.arange(rows, dtype=torch.int32) + stage * 10_000).to(device)
            for step in range(2):
                positions.add_(100)
                instance(x, x, positions, None)
                # Synthetic independent local preprocessing between post and
                # join. This checks protocol ordering, not an overlap speedup.
                local_preprocess = x + 1
                exchange.finish()
                check(buffer[:rows], positions, backend)
                del local_preprocess
            graph_passed = None
            if graphs and rows <= 4:
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    instance(x, x, positions, None)
                positions.add_(700)
                graph.replay()
                check(buffer[:rows], positions, backend)
                graph_passed = True
                del graph
            cases.append(dict(tokens=rows, changed_input_byte_parity=True, graph_parity=graph_passed))
        report = dict(
            backend=backend,
            rank=rank,
            stage=stage,
            tp_local_rank=local_rank,
            owner_rank=owner_rank,
            overlap_prefill=overlap_prefill,
            exchange_bytes=exchange.required_bytes,
            cases=cases,
            evidence="synthetic transport/capture gate only; real indexer and serving remain unqualified",
        )
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output / f"rank-{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        for group in groups:
            dist.destroy_process_group(group)
        dist.destroy_process_group()


def check(result, positions, backend):
    if backend == "hccl":
        torch.npu.synchronize()
    actual, position_cpu = result.cpu(), positions.cpu()
    if not torch.equal(actual[:, 0], position_cpu) or not torch.all(actual[:, 1] == 16_777_217):
        raise AssertionError("owned indexer result changed INT32 values or returned stale positions")
    if not torch.all(actual[:, 2:] == -1):
        raise AssertionError("owned indexer result changed padding sentinels")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("gloo", "hccl"), default="gloo")
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--owner-rank", type=int, default=0)
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 4, 37, 640])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graphs", action="store_true")
    parser.add_argument("--overlap-prefill", action="store_true")
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
