# 310P prefill measurement

Three scripts for measuring prefill on the 4x Ascend 310P3 box. They assume a
vLLM server config like `~/start_server.sh` (Qwen3.8-27B-w8a8-310p, TP4).

| script | needs | answers |
| --- | --- | --- |
| `prefill_probe.py` | a running server | end-to-end prefill tok/s |
| `prefill_budget.py` | exclusive chips | where a step's time goes |
| `prefill_session.sh` | exclusive chips | both, unattended, then restores production |

Start here:

```bash
setsid nohup bash ~/prefill_session.sh > ~/logs/prefill_session.log 2>&1 &
tail -f ~/prefill_session_results.txt
```

## Things that will mislead you

**`Avg prompt throughput` in the vLLM log is not a device-side number.** It is
`(tokens finished in the 10 s window) / 10 s`. Two 5839-token requests landing
in one window read as "1167 tok/s" while each actually took 7.16 s. Use
`prefill_probe.py`, which times the request.

**Prefix caching will serve a cached prefill and flatter the result.** Every
prompt `prefill_probe.py` builds carries a unique prefix. Do the same for any
prompt you add, or you will measure the cache.

**A CANN inner error poisons the device context, and ops after it return
without executing.** This does not raise -- it shows up as throughput above the
hardware peak (14163 TOP/s against ~70 TOPS/chip). `prefill_budget.py` runs each
section in its own process and flags above-peak rates, but if you write your own
benchmark, check the numbers against peak before believing them.

**Importing `vllm_ascend` inside a plain benchmark dies in `SetPrecisionMode`**,
because it re-runs the CANN env bootstrap after torch_npu has initialised.
Inline what you need instead (`ACL_FORMAT_FRACTAL_NZ = 29`).

**`npu_quant_matmul` needs the runtime's weight layout** --
`npu_format_cast(w, 29).transpose(0, 1)` -- or it fails with
`aclnnQuantMatmulV5 error 161002`.

**Budget 15-20 minutes for `prefill_budget.py`, not five.** CANN compiles the
uncached fp32 5D shapes in the WY/UT sections on first call; that alone has
taken ~10 minutes. Results stream to `~/prefill_budget_results.txt` as each
section lands, so an interrupted run still leaves what it had.

**Never SIGKILL anything holding an NPU.** It wedges the 310P chips until a
reboot. `prefill_session.sh` traps SIGTERM, stops the server gracefully and
restores production before exiting.

## The measured budget

Per chip, per 8192-token step, from `prefill_budget.py` on 2026-09-21. The step
itself is ~10.1 s (8192 tok at the 813 tok/s the probe measures end to end), so
these account for about 45% of it:

| item | s/step | rate | vs peak |
| --- | ---: | --- | --- |
| UT transform (blocked) | 1.479 | | |
| MLP gate+up int8 | 0.828 | 56.4 TOP/s | 81% |
| WY attn build (fp32) | 0.601 | | |
| dynamic_quant [T x hidden] | 0.421 | 76.6 GB/s | ~38% |
| MLP down int8 | 0.389 | 60.1 TOP/s | 86% |
| GDN out_proj fp16 | 0.373 | 16.6 TFLOP/s | 47% |
| GDN in_proj_qkv int8 | 0.188 | 54.8 TOP/s | 78% |
| GDN in_proj_z int8 | 0.129 | 47.8 TOP/s | 68% |
| dynamic_quant [T x inter/TP] | 0.102 | 67.4 GB/s | ~33% |

Peaks are per chip: ~70 TOPS INT8, ~35 TFLOPS FP16 (Atlas 300I Duo, 2 chips per
card). What the table says:

**The GDN WY prefix is the biggest thing in prefill, not a rounding error.**
The UT transform plus the attn build is 2.08 s/step, about a fifth of the step
and more than the whole MLP. Counting its FLOPs badly understates it -- it runs
in fp32, and fp32 matmul is a slow path on 310P. (The tell was that CANN needed
~10 minutes to compile these shapes on first call.) An earlier revision of this
file called the WY prefix ~0.25% of step FLOPs and told you not to bother with
it; that was arithmetic, not measurement, and it was wrong.

**The INT8 GEMMs have no headroom.** 48-60 TOP/s against a ~70 TOPS peak is
68-86% of the hardware. 1.53 s/step of the budget is essentially irreducible.

**`dynamic_quant` costs 0.523 s/step to do no arithmetic.** 256 standalone calls
per step -- four per layer -- converting activations to INT8 before each matmul,
at ~38% of memory bandwidth. `vllm_ascend/compilation/passes/norm_quant_fusion_pass.py`
already fuses this into `npu_add_rms_norm_quant`, but only under torch.compile,
and the box serves with `--enforce-eager`, so none of it fires.

**`out_proj` in fp16 costs 0.373 s/step at 47% of FP16 peak.** INT8 would be
roughly 0.11 s. It is fp16 on purpose -- quantizing it corrupted prose -- so
this needs finer-grained quantization, not a flag.

## What is already ruled out

Measured on this box, so do not re-investigate without new evidence:

- **Per-step fixed overhead and O(T^2) attention.** Prefill is linear in tokens:
  5839 tok took 7.16 s, 11599 tok took 14.30 s against 14.23 s predicted from
  the first. Raising `--max-num-batched-tokens` buys nothing.
- **Slow ND weight layout.** The 310P hardware profile is `FORCE_NZ`.
- **Big GEMMs running unquantized.** 22.8 B of 24.3 B non-embedding params run
  INT8. Only GDN `out_proj` is fp16.
- **INT8 Cube utilization.** 68-86% of peak, measured. Nothing to win.

## Served prefill, measured

Same server config throughout (TP4, MTP k=5, GDN W8A8), one knob at a time,
`prefill_probe.py`, 2026-09-21:

| config | 5839 tok | 11599 tok |
| --- | ---: | ---: |
| before GDN W8A8 and the blocked inverse | ~714 | ~707 |
| row-wise WY substitution | 806 | 784 |
| blocked UT inverse (default) | 823 | 823 |
| + grouped WY gram (default) | **852** | **845** |

So the blocked inverse is worth 2-5% and the grouped gram another 3.5%, and
both agree with the per-op budget above. Rollback knobs, both exact:

- `VLLM_ASCEND_GDN_UT_BLOCKED=0` -- the row-wise WY substitution the blocked
  inverse replaced (1.810 s/step against 1.479).
- `VLLM_ASCEND_GDN_WY_GROUPED_GRAM=0` -- the per-V-head gram. The 6D broadcast
  matmul this needs is now confirmed working on Ascend.
