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

## What is already ruled out

Measured on this box, so do not re-investigate without new evidence:

- **Per-step fixed overhead and O(T^2) attention.** Prefill is linear in tokens:
  5839 tok took 7.16 s, 11599 tok took 14.30 s against 14.23 s predicted from
  the first. Raising `--max-num-batched-tokens` buys nothing.
- **Slow ND weight layout.** The 310P hardware profile is `FORCE_NZ`.
- **Big GEMMs running unquantized.** 22.8 B of 24.3 B non-embedding params run
  INT8. Only GDN `out_proj` is fp16, and deliberately -- quantizing it corrupted
  prose.
- **The GDN WY prefix.** ~0.25% of step FLOPs. Worth low single digits at best.

## Opt-in knobs these scripts A/B

Both are exact and both are off by default:

- `VLLM_ASCEND_GDN_UT_BLOCKED=0` -- back to the row-wise WY substitution the
  blocked inverse replaced. For attribution, and as a rollback.
- `VLLM_ASCEND_GDN_WY_GROUPED_GRAM=1` -- build the WY gram matrix once per K
  head rather than once per V head. Exact to 1 fp16 ULP on CPU, but it needs
  torch_npu to broadcast a 6D matmul, which is unverified on Ascend.
