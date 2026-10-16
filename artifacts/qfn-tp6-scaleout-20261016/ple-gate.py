"""Private device hash/offset graph gate using real checkpoint parameters."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from safetensors import safe_open

sys.path.insert(0, "/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609")

from ple_hash_reference import NGramHashConfig, compute_ngram_ids

from tools.qwen4exp.ple_request import PLEDeviceRequest, PLERequestGeometry, load

ROOT = Path("/srv/ai/storage/artifacts/qfn-tp6-scaleout-20261016")
MODEL = Path("/srv/ai/models/Qwen3.8-Flash-Next-W4A16-G128-300i")


def checkpoint_parameters():
    cfg = json.loads((MODEL / "config.json").read_text())["text_config"]
    reference = NGramHashConfig(
        ngram_size=cfg["ngram_size"],
        heads_per_ngram=cfg["heads_per_ngram"],
        eos_token_id=cfg["eos_token_id"],
        unigram_vocab_size=cfg["vocab_size"],
        ngram_vocab_size_base=cfg["ngram_vocab_size_base"],
        ple_dense_layer_id=0,
    )
    weights = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    for suffix, values in (
        ("layer_multipliers", reference.multipliers),
        ("ngram_heads_offsets", reference.offsets),
        ("ngram_heads_vocab_sizes", reference.sizes),
    ):
        key = next(key for key in weights if key.endswith("ple_embedding." + suffix))
        with safe_open(MODEL / weights[key], framework="pt") as file:
            assert file.get_tensor(key).reshape(-1).tolist() == values, suffix
    return reference


@torch.inference_mode()
def main(rank, output):
    if output.exists():
        raise FileExistsError(output)
    reference = checkpoint_parameters()
    device = torch.device(f"npu:{rank}")
    torch.npu.set_device(device)
    kernel, launch = load(ROOT / "ple-selector-six-v2")
    generator = torch.Generator().manual_seed(1421)
    checked = invalid_checks = 0
    shapes = []
    for lengths in (
        [1],
        [3],
        [3, 3],
        [3, 3, 3],
        [3, 3, 3, 3],
        [3, 3, 3, 3, 3],
        [3, 3, 3, 3, 3, 3],
        [0, 3, 0, 4, 0, 2],
        [0, 3, 0, 4],
        [1, 0, 16, 0],
        [128],
        [2048],
    ):
        count, requests = sum(lengths), len(lengths)
        geometry = PLERequestGeometry(
            count,
            requests,
            reference.eos_token_id,
            reference.unigram_vocab_size,
            tuple(reference.multipliers),
            tuple(reference.sizes),
            tuple(reference.offsets),
            (reference.total_vocab_size + 127) // 128 * 128,
        )
        owner = PLEDeviceRequest(kernel, launch, geometry, device)
        tokens = torch.zeros(count, dtype=torch.int64, device=device)
        history = torch.full((requests, 2), reference.eos_token_id, dtype=torch.int64, device=device)
        boundaries = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
        starts = boundaries.to(device)
        graph = torch.npu.NPUGraph()
        stream = torch.npu.Stream(device=device)
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(3):
                owner(tokens, starts, history)
            torch.npu.synchronize()
            with torch.npu.graph(graph, stream=stream):
                owner(tokens, starts, history)
            torch.npu.synchronize()
        torch.npu.current_stream().wait_stream(stream)
        revisions = 10 if count < 128 else 3
        for revision in range(revisions):
            inputs = torch.randint(reference.unigram_vocab_size, (count,), generator=generator)
            past = torch.randint(reference.unigram_vocab_size, (requests, 2), generator=generator)
            inputs[revision % count] = reference.eos_token_id if revision % 2 else reference.unigram_vocab_size - 1
            past[revision % requests, revision % 2] = reference.eos_token_id
            if revision == 0:
                inputs.zero_()
                past.fill_(reference.eos_token_id)
            expected = torch.cat(
                [
                    compute_ngram_ids(inputs[boundaries[row] : boundaries[row + 1]], reference, past[row])
                    for row in range(requests)
                ]
            )
            tokens.copy_(inputs)
            history.copy_(past)
            graph.replay()
            actual, offsets, receipt = owner.rows.cpu(), owner.offsets.cpu(), owner.receipt.cpu()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(offsets, expected * 320, rtol=0, atol=0)
            assert not receipt.any()
            checked += actual.numel()
        # Malformed packed boundaries must produce no usable DMA offsets.
        starts[0] = 1
        graph.replay()
        assert torch.all(owner.rows.cpu() == -1) and torch.all(owner.offsets.cpu() == -1)
        assert torch.all(owner.receipt.cpu()[:, 0] == 1)
        invalid_checks += 1
        starts.copy_(boundaries)
        tokens[0] = -1
        graph.replay()
        assert owner.receipt.cpu()[0, 0] == 2 and torch.all(owner.offsets.cpu()[0] == -1)
        invalid_checks += 1
        tokens.fill_(7)
        history.fill_(reference.eos_token_id)
        elapsed = []
        for _ in range(5):
            torch.npu.synchronize()
            begin = time.perf_counter()
            for _ in range(10):
                graph.replay()
            torch.npu.synchronize()
            elapsed.append((time.perf_counter() - begin) * 1e6 / 10)
        shapes.append(dict(lengths=lengths, graph_replay_us=elapsed, median_us=statistics.median(elapsed)))
        graph.reset()
    report = dict(
        passed=True,
        rank=rank,
        checked_row_ids=checked,
        checked_ram_offsets=checked,
        invalid_input_checks=invalid_checks,
        checkpoint_hash_parameters_verified=True,
        graph_shapes=shapes,
        device_hash=True,
        device_dma=False,
        server_modified=False,
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.rank, args.output)
