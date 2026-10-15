# SPDX-License-Identifier: Apache-2.0
"""Qualify acceptance -> native state commit -> next inputs in one private graph.

The model service remains running. This is a device-chain correctness gate,
not evidence that its host scheduler or eager callbacks have been removed.
"""

import argparse
import json
from pathlib import Path

import torch

from tools.qwen4exp.benchmark_mamba_state_commit_310 import load
from tools.qwen4exp.iteration_transition import accept_step, prepare_next_step
from tools.qwen4exp.mamba_commit import MambaCommitGroup, reference


def cases():
    for revision in range(16):
        metadata = torch.zeros((5, 8), dtype=torch.int32)
        metadata[:, :4] = torch.tensor(
            [[0, 1, 2, 0], [126 + revision % 3, 254 + revision % 3, 382, 10], [3] * 4, [2] * 4, [0] * 4]
        )
        samples = torch.full((4, 3), -1, dtype=torch.int64)
        for row in range(4):
            count = (revision + row) % 4
            samples[row, :count] = torch.arange(count) + 7 + row
        if revision == 14:
            samples[0] = torch.tensor([7, -1, 9])
        if revision == 15:
            samples[0] = torch.tensor([7, -2, -1])
        table = ((torch.arange(32).reshape(4, 8) + revision) % 4).int()
        drafts = torch.arange(8).reshape(4, 2).long() + 20
        yield metadata, samples, table, drafts


@torch.inference_mode()
def gate(build, device, emit):
    load(build)
    kernel = torch.classes.qwen_mamba_commit_v1.Kernel(
        str(Path(build) / "native_mamba_commit.bin"), "qwen_mamba_commit_group_v1"
    )
    generator = torch.Generator().manual_seed(1418)
    # One full-size conv/recurrent pair is enough to prove producer/consumer
    # ordering here. All 72-bank copy geometry has a separate qualification.
    host_banks = [
        torch.randint(-32768, 32767, (4, 5, 2560), dtype=torch.int16, generator=generator).view(torch.float16),
        torch.randint(-(1 << 31), (1 << 31) - 1, (4, 12, 128, 128), dtype=torch.int32, generator=generator).view(
            torch.float32
        ),
    ]
    banks = [value.to(device) for value in host_banks]
    host_case = next(cases())
    metadata, samples, table, drafts = [value.to(device) for value in host_case]
    commit = MambaCommitGroup(
        kernel, torch.ops.qwen_mamba_commit_v1.launch, banks, [table] * 2, [True, False], block_size=128
    )

    def chain():
        accepted = accept_step(samples, metadata, block_size=128)
        receipt = commit(metadata)
        next_step = prepare_next_step(samples, drafts, metadata, accepted, max_model_len=262144, eos_token_id=8)
        # Consumer clones prove the native copy completed before dependent work.
        consumed = [bank.clone() for bank in banks]
        return accepted, receipt, next_step, consumed

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            chain()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        output = chain()
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    emit(dict(stage="captured", device=device, host_decisions_inside_chain=0))
    addresses = [tensor.data_ptr() for tensor in (metadata, samples, table, drafts)]
    try:
        for index, host_case in enumerate(cases()):
            for destination, source in zip((metadata, samples, table, drafts), host_case):
                destination.copy_(source)
            for bank, host in zip(banks, host_banks):
                bank.copy_(host)
            graph.replay()
            host_metadata, host_samples, host_table, host_drafts = host_case
            expected_acceptance = accept_step(host_samples, host_metadata, block_size=128)
            expected_next = prepare_next_step(
                host_samples, host_drafts, host_metadata, expected_acceptance, max_model_len=262144, eos_token_id=8
            )
            accepted, receipt, next_step, consumed = output
            for actual, expected in zip((*accepted, *next_step), (*expected_acceptance, *expected_next)):
                torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
            for row, (bank, original) in enumerate(zip(consumed, host_banks)):
                expected_bank, expected_receipt = reference(
                    original, host_metadata, host_table, conv=row == 0, block_size=128
                )
                torch.testing.assert_close(
                    bank.cpu().view(torch.int32), expected_bank.view(torch.int32), rtol=0, atol=0
                )
                torch.testing.assert_close(receipt[row].cpu(), expected_receipt, rtol=0, atol=0)
        assert addresses == [tensor.data_ptr() for tensor in (metadata, samples, table, drafts)]
        emit(dict(stage="passed", changing_replays=index + 1, exact_bank_checks=2 * (index + 1), scope="private_chain"))
    finally:
        torch.npu.synchronize()
        graph.reset()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new output path")
    import torch_npu

    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    if "310" not in torch_npu.npu.get_device_name(args.device):
        raise RuntimeError("requires Ascend 310P")
    free, total = torch.npu.mem_get_info(args.device)
    if free < 768 * 1024 * 1024:
        raise RuntimeError("insufficient private gate memory")
    torch.npu.set_per_process_memory_fraction(128 * 1024 * 1024 / total, args.device)
    with args.output.open("x") as stream:

        def emit(value):
            line = json.dumps(value)
            print(line, flush=True)
            stream.write(line + "\n")
            stream.flush()

        gate(args.build, f"npu:{args.device}", emit)


if __name__ == "__main__":
    main()
