"""Compare GLM KDA prefill carry zeroing on one Ascend device.

This is a same-process operator probe, not a full-model TTFT measurement.
"""

import argparse
import json
import statistics
import time

import torch
import torch_npu  # noqa: F401  # Registers the NPU device and kernels.


def _boolean_index(state: torch.Tensor, indices: torch.Tensor, has_initial_state: torch.Tensor) -> torch.Tensor:
    initial_state = state[indices].float().contiguous()
    initial_state[~has_initial_state] = 0
    return initial_state


def _broadcast_mask(state: torch.Tensor, indices: torch.Tensor, has_initial_state: torch.Tensor) -> torch.Tensor:
    initial_state = state[indices].float().contiguous()
    fresh = ~has_initial_state.reshape((-1,) + (1,) * (initial_state.ndim - 1))
    initial_state.masked_fill_(fresh, 0)
    return initial_state


def _measure(fn, state, indices, has_initial_state, repeats):
    times_ms = []
    for _ in range(repeats):
        torch.npu.synchronize()
        start = time.perf_counter()
        fn(state, indices, has_initial_state)
        torch.npu.synchronize()
        times_ms.append((time.perf_counter() - start) * 1000)
    return statistics.median(times_ms)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")

    device = torch.device("npu:0")
    results = []
    for num_requests in (1, 4):
        state = torch.randn(8, 16, 128, 128, dtype=torch.float16, device=device)
        indices = torch.arange(num_requests, dtype=torch.int32, device=device)
        has_initial_state = torch.tensor(
            [request % 2 == 1 for request in range(num_requests)], dtype=torch.bool, device=device
        )
        # Cold state rows may contain stale NaNs. The new path must clear them,
        # unlike multiplication by a zero mask.
        state[0].fill_(float("nan"))
        reference = _boolean_index(state, indices, has_initial_state)
        candidate = _broadcast_mask(state, indices, has_initial_state)
        torch.testing.assert_close(reference.cpu(), candidate.cpu(), atol=0, rtol=0, equal_nan=True)

        for _ in range(4):
            _boolean_index(state, indices, has_initial_state)
            _broadcast_mask(state, indices, has_initial_state)
        torch.npu.synchronize()
        baseline_ms = _measure(_boolean_index, state, indices, has_initial_state, args.repeats)
        candidate_ms = _measure(_broadcast_mask, state, indices, has_initial_state, args.repeats)
        results.append(
            {
                "requests": num_requests,
                "state_shape": list(state.shape),
                "baseline_median_ms": baseline_ms,
                "candidate_median_ms": candidate_ms,
                "speedup": baseline_ms / candidate_ms,
                "bitwise_parity": True,
            }
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
