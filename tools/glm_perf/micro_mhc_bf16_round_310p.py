"""Compare bitwise and native BF16 mHC rounding on one Ascend 310P."""

import json
import statistics
import time

import torch
import torch_npu

SHAPES = ((512, 4, 1), (512, 4, 4), (512, 4096), (512, 4, 4096))
WARMUPS = 3
REPEATS = 12


def bitwise_round(value: torch.Tensor) -> torch.Tensor:
    bits = value.float().view(torch.int32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) & -65536).view(torch.float32)


def cast_round(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.bfloat16).to(torch.float32)


def measure(fn, value: torch.Tensor) -> tuple[float, torch.Tensor]:
    for _ in range(WARMUPS):
        output = fn(value)
    torch.npu.synchronize()
    samples = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        output = fn(value)
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples), output


def check_graph(value: torch.Tensor) -> bool:
    graph = torch.npu.NPUGraph()
    stream = torch.npu.Stream()
    with torch.npu.graph(graph, stream=stream):
        output = cast_round(value)
    value.add_(0.125)
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()
    return torch.equal(output.cpu().view(torch.int32), bitwise_round(value).cpu().view(torch.int32))


def main() -> None:
    torch.manual_seed(20261003)
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    finite_bits = torch.randint(0, 2**31, (1_000_000,), dtype=torch.int64).to(torch.int32)
    finite_bits &= 0x7F7FFFFF
    finite_values = finite_bits.view(torch.float32)
    finite_values[1::2].neg_()
    finite_values = finite_values.npu()
    random_bits_equal = torch.equal(
        bitwise_round(finite_values).cpu().view(torch.int32),
        cast_round(finite_values).cpu().view(torch.int32),
    )
    print(json.dumps({"random_finite_bits_equal": random_bits_equal, "samples": finite_values.numel()}), flush=True)
    for shape in SHAPES:
        sample = torch.randn(shape, dtype=torch.float32)
        # Include exact half-way BF16 ties and values outside FP16's range.
        sample.view(-1)[:8] = torch.tensor([0.0, -0.0, 1.00390625, -1.00390625, 65504.0, 70000.0, 1e-30, -1e-30])
        value = sample.npu()
        bitwise_ms, bitwise = measure(bitwise_round, value)
        try:
            cast_ms, cast = measure(cast_round, value)
            equal = torch.equal(bitwise.cpu().view(torch.int32), cast.cpu().view(torch.int32))
            result = {
                "shape": shape,
                "bitwise_ms": bitwise_ms,
                "cast_ms": cast_ms,
                "bitwise_equal": equal,
                "graph_replay_equal": check_graph(value),
            }
        except Exception as error:
            result = {"shape": shape, "bitwise_ms": bitwise_ms, "cast_error": str(error)}
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
