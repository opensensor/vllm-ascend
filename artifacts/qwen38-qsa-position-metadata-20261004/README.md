# QSA position metadata AI CPU quick gate

On one idle Ascend 310P, the isolated AI CPU metadata candidate matched the
Torch reference exactly for all eight query-count/padding cases and changed
inputs. The corrected gate measured 0.042-0.044 ms at 3-256 queries and
0.069-0.070 ms at 2,048 queries, against 0.101-0.108 ms and 0.093-0.095 ms
for the equivalent Torch operations. These are five-trial medians of 20-call
batches with a device synchronization after each batch, not model latency.

- `metadata-gate-fair-20261004.jsonl`: corrected tuple-to-tuple comparison.
- `metadata-gate-quick-20261004.jsonl`: first parity gate; its Torch baseline
  included an extra `torch.stack`, so its timing is superseded.

At the 2,048-query prefill size, the 0.022-0.026 ms per-call gain is too small
to materially change the approximately 120 s cold 40K prompt. The serving
path was untouched. See `tools/qwen4exp/QSA_AICPU_METADATA_EXPERIMENT.md` for
the setup, methodology, and interpretation. Further NPU testing is deferred.
