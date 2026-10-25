# M3.1 independent llama.cpp reference preparation

There is **no native-backend inference result yet**, and the reasoning loop is
not established as fixed. The user took over the NPUs during the fresh Ascend
trial. Its [interrupted result](../minimax-m31-history-reference-20261025/fp8-fresh-kilo-interrupted.json)
must not be counted as a completed failure. All six workers were restored to
baseline, with weights resident and the control lock released.

## Source and conversion

The reference uses unmodified [Unsloth llama.cpp revision
822be707](https://github.com/unslothai/llama.cpp/tree/822be707b23345b291f3042defa7508aa813cc4b).
It was built on the Threadripper host for its CPU instruction set, with CUDA
disabled. See [build identity](build.json).

The converter runs against the original mixed NVFP4/MXFP8 release with
`--outtype auto --split-max-size 20G`. This preserves NVFP4 expert codes/scales,
writes dense MXFP8 weights as BF16, retains router/indexer weights in FP32, and
forms Gemma normalization weights in FP32. It does not requantize experts into
signed INT4. Output is written directly to the host's NVMe through a temporary
SSHFS mount. Conversion and transfer completed with writer exit code zero. All
14 shards are readable with complete tensor bounds: 1,359 unique tensors,
including 171 NVFP4 expert banks, 766 FP32 tensors and 422 BF16 tensors. Index
projections are present in all 60 layers. See [complete accounting](conversion.json).

The local download revision metadata matches the current Hugging Face revision
`54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e` for all 48 weight shards, the config,
and chat template. That release was last modified October 23, 2026. This is a
revision-metadata check, not a fresh hash of all checkpoint bytes
([receipt](source-revision.json)).

## Completed checks

- Samples from 14 expert banks in completed shard one preserve all 6,048 sampled
  packed bytes and 28 inverse-global scale values. Sampling covers experts zero
  and 127, first/middle/last rows, and first/last superblocks. This is not a full
  model checksum ([receipt](weight-repacking-sample.json)).
- Native `llama-tokenize`, using vocabulary metadata extracted from completed
  shard one, reproduces all 5,549 captured prompt IDs exactly. Source-tokenizer
  decode/encode also round-trips exactly. No BOS insertion or escape
  interpretation is enabled ([receipt](tokenizer-gate.json)).

## Prepared experiments, not executed

The frozen controller and launch script prepare a CPU-only server on localhost
port 8002, one slot, 32K context, memory-mapped weights and explicit Flash
Attention. In this revision, disabling Flash Attention makes M3 fall back to
dense attention; that would invalidate a sparse-model comparison. KV storage is
FP16. The CPU backend has different activation/cache arithmetic from the hosted
BF16/QAT implementation, so it is an independent implementation rather than a
cloud-equivalence oracle.

The controller first checks complete tensor bounds, 171 NVFP4 banks with scale
vectors, and index projections in all 60 layers. It then compares native chat
rendering with captured prompt IDs and runs the identical-token, greedy held
history through `/completion`, bypassing chat templating. A chat-rendering
mismatch prevents the subsequent paired fresh Kilo trial until investigated.

The prepared fresh Kilo trial retains the original agent prompt, temperature 1,
top-p 0.95 and seed 71. Every request uses the same captured initial system/user
messages and tool definitions as the NPU trial; actual generated assistant/tool
history remains unchanged. Both original incoming and normalized outgoing
requests are retained privately. Identical seed integers across backends do not
imply identical random-number streams. Repository changes since the NPU trial
must also be considered before interpreting a later comparison.

A separate prepared NPU experiment replaces only the agent prompt in that
canonical prefix with the concise-overview guidance already in the user's Kilo
configuration. It retains the FP8/residual candidate to isolate that prompt
change. This experiment has **not run** and must not start while the user owns
the NPUs.

The CPU reference server has also **not started**: its large host-memory and CPU
load could compete with the user's NPU work. There is no automatic launch job.
Conversion has finished and its temporary SSHFS process has exited; the
comparison controller is a manual entry point. Private prompts, token arrays,
model bytes and tool-result contents are excluded from this artifact.

Prepared Python and shell scripts pass syntax checks. Scoped manual hooks pass;
whole-tree formatting still fails on seven existing unrelated checks in an
isolated checkout. None of these task files was modified or named by those
failures. See [validation](validation.json).
