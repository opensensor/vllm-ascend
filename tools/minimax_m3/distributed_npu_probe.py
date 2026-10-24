# SPDX-License-Identifier: Apache-2.0
"""Six-chip TP2/PP3 transport gate using production pipeline packet code."""

import argparse
import datetime
import importlib.util
import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD_SIZE = 6
TP_SIZE = 2
STAGES = 3
HIDDEN_SIZE = 6144
TOKEN_CASES = (1, 3, 128)


def worker(rank, output, transport_path, port):
    torch.set_num_threads(1)
    torch.npu.set_device(rank)
    device = torch.device("npu", rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=WORLD_SIZE,
        timeout=datetime.timedelta(seconds=180),
    )
    tp_groups = [dist.new_group([2 * stage, 2 * stage + 1], backend="hccl") for stage in range(STAGES)]
    pp_groups = [dist.new_group(list(range(tp, WORLD_SIZE, TP_SIZE)), backend="gloo") for tp in range(TP_SIZE)]
    spec = importlib.util.spec_from_file_location("m31_pipeline_transport", transport_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stage, tp_rank = divmod(rank, TP_SIZE)
    transport = module.PipelineAllGatherTransport(
        pp_groups[tp_rank], tuple(range(tp_rank, WORLD_SIZE, TP_SIZE)), rank, device, world_size=WORLD_SIZE
    )
    transport.warmup(synchronize=torch.npu.synchronize)
    comparisons = []
    for tokens in TOKEN_CASES:
        initial = (torch.arange(tokens * HIDDEN_SIZE).reshape(tokens, HIDDEN_SIZE) % 16).half()
        if stage:
            packet = transport.recv_tensor_dict()
            x, residual = packet["hidden_states"], packet["residual"]
            assert packet["tokens"] == tokens
            torch.testing.assert_close(residual.cpu(), initial, atol=0, rtol=0)
        else:
            x = initial.to(device)
            residual = x.clone()
        partial = x.float() + (rank + 1)
        dist.all_reduce(partial, group=tp_groups[stage])
        result = partial.half()
        if stage < STAGES - 1:
            transport.send_tensor_dict({"hidden_states": result, "residual": residual, "tokens": tokens})
        else:
            expected = initial.float()
            for s in range(STAGES):
                expected = (expected * TP_SIZE + (2 * s + 1) + (2 * s + 2)).half().float()
            torch.testing.assert_close(result.cpu().float(), expected, atol=0, rtol=0)
            comparisons.append({"tokens": tokens, "exact": True})
        dist.barrier()
    torch.npu.synchronize()
    Path(output, f"rank-{rank}.json").write_text(
        json.dumps({"rank": rank, "passed": True, "comparisons": comparisons}, indent=2) + "\n"
    )
    transport.destroy()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--transport", type=Path, default=Path("vllm_ascend/distributed/device_communicators/pipeline_all_gather.py")
    )
    parser.add_argument("--port", type=int, default=29681)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    mp.spawn(
        worker,
        args=(str(args.output.resolve()), str(args.transport.resolve()), args.port),
        nprocs=WORLD_SIZE,
        join=True,
    )
    ranks = [json.loads((args.output / f"rank-{rank}.json").read_text()) for rank in range(WORLD_SIZE)]
    report = {
        "passed": all(r["passed"] for r in ranks),
        "hardware_validated": True,
        "scope": "TP2 sums and PP3 hidden/residual packet transport, not full model",
        "ranks": ranks,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
