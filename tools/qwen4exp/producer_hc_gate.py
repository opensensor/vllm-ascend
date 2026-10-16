# SPDX-License-Identifier: Apache-2.0
"""Private eager lifetime gate, called only on a drained NPU worker."""

import torch


@torch.inference_mode()
def validate_lifetime(op, *, repeats=64):
    """Churn operands on a different stream after the launch returns.

    Waiting for operand readiness and recording their consumer stream are
    separate obligations. Poison allocations exercise the latter; exact output
    checks also establish completion before the private resources are retired.
    """
    producer = torch.npu.Stream()
    consumer = torch.npu.current_stream()
    cases = 0
    try:
        for revision in range(repeats):
            rows = (128, 256, 640, 512)[revision % 4]
            value = (revision % 7) / 8
            with torch.npu.stream(producer):
                hyper = torch.full((rows, 10240), value, dtype=torch.float16, device="npu")
                block = torch.full((rows, 2560), -0.5, dtype=torch.float16, device="npu")
                injection = torch.ones(rows, 4, dtype=torch.float32, device="npu")
            consumer.wait_stream(producer)
            actual = op(hyper, block, injection)
            del hyper, block, injection
            with torch.npu.stream(producer):
                poison = (
                    torch.full((rows, 10240), 31, dtype=torch.float16, device="npu"),
                    torch.full((rows, 2560), 37, dtype=torch.float16, device="npu"),
                    torch.full((rows, 4), 41, dtype=torch.float32, device="npu"),
                )
            host = actual.cpu()
            expected = torch.full_like(host, value - 0.5)
            torch.testing.assert_close(host.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0)
            del actual, poison
            cases += 1
        return dict(passed=True, cross_stream_churn_cases=cases, exact=True)
    finally:
        torch.npu.synchronize()
