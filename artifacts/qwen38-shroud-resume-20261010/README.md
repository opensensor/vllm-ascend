# Qwen resume after the thermal shroud replacement

The user authorized resuming the six-chip server after replacing the thermal
shroud on October 10, 2026. This directory records the restart and the bounded
idle checks completed before the user requested immediate startup.

## Device access recovery

After the reboot, `/dev/davinci0` through `/dev/davinci5` were owned by
`root:root`, and the manager, SVM, and HDC nodes were root-only. Unprivileged
`npu-smi info` failed with DCMI initialization error `-8005`. Root readings
confirmed six healthy chips at 43–55°C and no NPU processes.

Restoring group ownership to the existing `HwHiAiUser` group and mode `0660`
on the manager, SVM, and HDC nodes restored unprivileged monitoring. A narrow
udev rule at `/etc/udev/rules.d/99-qwen-ascend-access.rules` preserves group
access after reboot. No world-accessible device permissions were added.

## Idle checks

The checked-in `benchmark_idle_ddr_310.py` ran without a model using six fresh
workers for each phase. Device initialization, 128-MiB allocation per worker,
HCCL initialization, eager all-reduce, and captured graph replay passed on all
six chips. The retire phase was interrupted and the close phase was skipped
when the user prioritized restoring service. Neither is qualified by this run.

Each five-second idle window contains one complete six-chip usage sweep; these
are brief snapshots, not sustained traffic measurements. Peak sampled
temperature was 62°C. Allocation, communicator, eager, and graph snapshots
reported 0% DDR bandwidth on all six chips. The device-only sweep reported 47%
on its first chip and 0% on the remaining chips.

Before device initialization, the unloaded host also reported 47% DDR bandwidth
with no NPU processes. That reading alone cannot establish transfers caused by
Qwen or HCCL. Telemetry timing and driver activity remain unresolved.

The first device probe failed because its invocation replaced the toolkit's
Python path, preventing import of the Ascend compiler module. The corrected invocation preserved
the sourced toolkit path and passed. No dependency upgrade was needed.

## Restored configuration

- API: `http://192.168.53.187:8001/v1`; model ID: `qwen38-flash-next`.
- Runtime: `/srv/ai/src/qwen-performance-paced-tp6-20261010`.
- Native INT4 target, FP16 PLE baseline, grouped W8 draft, MTP2, TP6/EP6.
- Six sequence slots, 262,144 maximum tokens per sequence, image processing
  enabled with the encoder in data mode. Six full-length simultaneous contexts
  are not stress-qualified by this restart.
- Graph sizes `[3, 18]`; existing prefill pacing and decoder fairness retained.
- Thermal hold at 94°C, resume only when all six chips reach 85°C or below,
  separate 96°C emergency cutoff.
- New logs: `/srv/ai/src/qwen-shroud-resume-20261010/`; earlier receipts were
  preserved.

The restored service passed three text checks and fresh/cached image
understanding checks. All six resident workers were present with clean graphs;
the observed smoke temperatures stayed at or below 62°C. The cache planner
reported 1,620,838 tokens (6.18 full-length slots). This brief run does not
establish sustained thermal behavior.

The v3 residual candidate and the final combined count/GDN/histogram candidate
were not activated during this restoration. Their whole-model qualification
and the historical 50% decode improvement target remain open.

## Repository checks

The scoped manual hooks passed for this receipt. The required `bash format.sh
ci` was also run; it still fails on existing repository-wide formatting and
forbidden-import findings. Unrelated formatter changes were restored.
