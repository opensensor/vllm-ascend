# DeepSeek V4.1 W2 quality / mixed-bit host proxy

> Host-side float SwiGLU proxy. Identical seeded activations run through the
> FP4 source expert and the converted W2 artifact expert; the output delta is
> the weight-quantisation error. **Not** the real Ascend fused-MoE kernel —
> final quality still needs a runtime/rental.

## Headline

| metric | value |
| --- | --- |
| sampled experts | 18 |
| worst-expert W2 rel MSE | 1.181 (L24 E128, cos 0.6052) |
| median W2 rel MSE | 0.977 |
| mean W2 rel MSE | 0.9997 |
| median W2 cosine | 0.5784 |
| median W3 rel MSE (in-memory probe) | 0.2637 |
| median W4 rel MSE (in-memory probe) | 0.04575 |
| Engram W4 rel MSE (mean/max) | 0.01873 / 0.03887 |

## Mixed-bit proposal

- Error budget: relative MSE <= **0.05**.
- Over budget at W2 in sample: **18/18** (~100.0% of experts).
- Minimal escalation that meets budget (sample fractions): **W4: 77.8%**; unresolved by any probed grid: **4** (budget fully met by escalation: **False**).
- Routed-expert W2 footprint: **129.7 GiB** (15360 experts).
- Mixed-bit escalation byte delta: **+129.73 GiB** (mean +2.00 bits/weight, bit-tight store).
- Note: byte delta assumes a bit-tight store (k bits/weight); today's pack_codes writes W3 at 2 codes/byte (4 bits/weight) and W4 at 2 codes/byte (4 bits/weight), so a W3 escalation costs the same on-disk bytes as W4 with the current packer.

## Verdict

~100% of sampled experts exceed the 0.05 budget at W2 and 4 are not rescued by any probed grid (W3/W4); **naive round-to-nearest W2 looks risky** here — calibration (GPTQ/AWQ-style), not just wider bits, is likely the real mitigation, and a runtime/rental check is needed. This is a functional weight-error proxy only: it does not capture the INT8 activation-quant path, routing, or end-to-end task accuracy, so the final go/no-go still needs a runtime or rental run.
