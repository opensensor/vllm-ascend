# SPDX-License-Identifier: Apache-2.0
"""Six-process Gloo gate for packed experts and production pipeline packets.

Synthetic small weights; no attention, scheduler, HCCL, or NPU qualification.
"""

import argparse
import datetime
import importlib.util
import json
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tools.minimax_m3.probe_w4 import quantize_chunk
from vllm_ascend._310p.ops.minimax_m3 import PackedM3Experts

WORLD_SIZE = 6
TP_SIZE = 2
STAGES = 3
HIDDEN = 32
EXPERTS = 8
TOP_K = 4
TOKEN_CASES = (1, 33, 129)


class GlooBackend:
    """Use real CPU collectives with the unchanged pipeline packet implementation."""

    def new_group(self, ranks, *, backend, use_local_synchronization):
        assert backend == "hccl"
        return dist.new_group(
            ranks,
            backend="gloo",
            use_local_synchronization=use_local_synchronization,
            timeout=datetime.timedelta(seconds=60),
        )

    def __getattr__(self, name):
        return getattr(dist, name)


def pipeline_class():
    path = Path(__file__).resolve().parents[2] / "vllm_ascend/distributed/device_communicators/pipeline_all_gather.py"
    spec = importlib.util.spec_from_file_location("m3_cpu_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PipelineAllGatherTransport


def bank(stage, ep_size, ep_rank):
    result = PackedM3Experts(
        HIDDEN, 64, EXPERTS, ep_size, ep_rank, TOP_K, prefix=f"model.layers.{stage}.block_sparse_moe.experts"
    )
    generator = torch.Generator().manual_seed(100 + stage)
    for name, shape in (("w1", (64, HIDDEN)), ("w3", (64, HIDDEN)), ("w2", (HIDDEN, 64))):
        for expert in range(EXPERTS):
            weight = torch.randn(shape, generator=generator) * 0.05
            local = expert - result.local_expert_offset
            if 0 <= local < result.num_local_experts:
                codes, scales, _ = quantize_chunk(weight, "minmax")
                getattr(result, name + "_codes")[local].copy_(codes)
                getattr(result, name + "_scales")[local].copy_(scales)
    return result


def logits(x, stage):
    generator = torch.Generator().manual_seed(200 + stage)
    return x.float() @ torch.randn(HIDDEN, EXPERTS, generator=generator)


def worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=WORLD_SIZE, timeout=datetime.timedelta(seconds=90)
    )
    groups = [dist.new_group([2 * s, 2 * s + 1]) for s in range(STAGES)]
    pp_groups = [dist.new_group(list(range(t, WORLD_SIZE, TP_SIZE))) for t in range(TP_SIZE)]
    stage, tp_rank = divmod(rank, TP_SIZE)
    transport = pipeline_class()(
        pp_groups[tp_rank],
        tuple(range(tp_rank, WORLD_SIZE, TP_SIZE)),
        rank,
        "cpu",
        backend=GlooBackend(),
        world_size=WORLD_SIZE,
    )
    transport.warmup()
    local_bank = bank(stage, TP_SIZE, tp_rank)
    full_banks = [bank(s, 1, 0) for s in range(STAGES)] if stage == STAGES - 1 else []
    results = []
    for tokens in TOKEN_CASES:
        initial = torch.randn(tokens, HIDDEN, generator=torch.Generator().manual_seed(tokens)).half()
        if stage:
            packet = transport.recv_tensor_dict()
            x, residual = packet["hidden_states"], packet["residual"]
            assert packet["tokens"] == tokens
            torch.testing.assert_close(residual, initial, rtol=0, atol=0)
        else:
            x, residual = initial, initial.clone()
        partial = local_bank(x, logits(x, stage))
        dist.all_reduce(partial, group=groups[stage])
        result = (x.float() + partial).half()
        if stage < STAGES - 1:
            transport.send_tensor_dict(dict(hidden_states=result, residual=residual, tokens=tokens))
        else:
            reference = initial
            for s, full in enumerate(full_banks):
                reference = (reference.float() + full(reference, logits(reference, s))).half()
            error = (result.float() - reference.float()).abs().max().item()
            torch.testing.assert_close(result, reference, rtol=0, atol=0.002)
            results.append(dict(tokens=tokens, max_abs_error=error))
        dist.barrier()
    Path(output, f"rank-{rank}.json").write_text(
        json.dumps(dict(rank=rank, stage=stage, passed=True, comparisons=results))
    )
    transport.destroy()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="m3-gloo-") as temporary:
        mp.spawn(
            worker,
            args=("file://" + temporary + "/rendezvous", str(args.output.resolve())),
            nprocs=WORLD_SIZE,
            join=True,
        )
    ranks = [json.loads((args.output / f"rank-{rank}.json").read_text()) for rank in range(WORLD_SIZE)]
    report = dict(
        passed=all(r["passed"] for r in ranks),
        backend="gloo",
        hardware_validated=False,
        production_components=["PackedM3Experts", "PipelineAllGatherTransport"],
        excludes=[
            "full model",
            "attention",
            "vLLM scheduler",
            "HCCL backend",
            "Cube kernels",
            "real weights",
        ],
        ranks=ranks,
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
