# SPDX-License-Identifier: Apache-2.0
"""Hardware gate callable for a prepared GateBeta resource; not run in this queue.

Require exact operator output before any serving experiment. If it fails,
record errors and revisit arithmetic; do not silently loosen tolerances.
Graph/recurrent-carry validation follows isolated parity, before promotion.
"""


def validate(op):
    import torch

    from tools.glm_perf.resident_candidates.kda_input_preparation import gate_beta_reference

    generator = torch.Generator().manual_seed(310)
    cases = 0
    for rows in (1, 2, 4, 8):
        for gate_dtype in (torch.float16, torch.float32):
            for beta_dtype in (torch.float16, torch.float32):
                for extreme in (False, True):
                    raw = torch.randn(1, rows, op.heads, 128, generator=generator)
                    beta = torch.randn(1, rows, op.heads, generator=generator)
                    if extreme:
                        raw.flatten()[:8] = torch.tensor([-1000, -100, -1, 0, 1, 100, 1000, -0.0])
                        beta.flatten()[:8] = raw.flatten()[:8]
                    raw = raw.to(device=op.lower.device, dtype=gate_dtype)
                    beta = beta.to(device=op.lower.device, dtype=beta_dtype)
                    scale = torch.randn(1, 1, op.heads, 1, generator=generator).exp().to(op.lower.device)
                    bias = torch.randn(1, 1, op.heads, 128, generator=generator).to(op.lower.device)
                    args = (raw, beta, scale, bias, op.lower_bound)
                    actual = op(*args)
                    expected = gate_beta_reference(*args)
                    for a, b in zip(actual, expected):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                    cases += 1
    return {"passed": True, "exact": True, "cases": cases}
