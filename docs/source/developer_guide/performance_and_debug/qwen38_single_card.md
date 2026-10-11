# Qwen3.8 Flash-Next on one Ascend 310P card

One dual-chip card uses tensor parallelism of two, with expert parallelism.
The model has 48 layers: 36 recurrent GDN layers and 12 attention layers,
512 routed experts and ten selected experts per token. This recipe keeps the
original W4 G128 checkpoint with FP16 scales; changing the runtime does not
re-quantize its expert weights. Activation quantization is INT8 per group,
implemented by the native INT4 backend's activation limbs.

Read the [runtime runbook](qwen38_310p_runtime_runbook.md) for source and operator
package coherence. A successful import or health response is insufficient:
the first inference request, cold prefill, prefix reuse and decode must pass.

## Josh: clone and run the exact qualified profile

The [single-card release](https://github.com/opensensor/vllm-ascend/releases/tag/qwen38-single-card-20261011)
contains the frozen runtime, matching compiled operators, native residual
kernel and validation receipts. Clone the repository for the source and tests;
use the release bundle for the qualified runtime. Building whatever operator
packages happen to be installed does not reproduce this configuration.

Use Josh's existing matching Python environment: Python 3.12.13 on x86_64,
CANN 9.1.0 with ATB, torch 2.13.0+cpu, torch_npu 2.13.0rc1 and
transformers 5.17.0. Keep the working OpenSensor vLLM and registered Ascend
plugin installation in that environment; the bundle overrides their source
and operator paths. The complete observed package inventory is included in
the release. Package availability and installation on a clean OS have not
been qualified; this recipe was validated against the existing environment.
The host had 247 GiB RAM and the unchanged checkpoint occupies about 169 GiB
on disk, including its large host-resident PLE table. This is not a 64 GB RAM
recipe.

```bash
git clone https://github.com/opensensor/vllm-ascend.git
cd vllm-ascend
git checkout qwen38-single-card-20261011
mkdir -p single-card-release
cd single-card-release
curl -fLO https://github.com/opensensor/vllm-ascend/releases/download/qwen38-single-card-20261011/qwen38-single-card-runtime-20261011.tar.gz
curl -fLO https://github.com/opensensor/vllm-ascend/releases/download/qwen38-single-card-20261011/SHA256SUMS
sha256sum -c SHA256SUMS
tar -xzf qwen38-single-card-runtime-20261011.tar.gz
cd single-card-runtime-bundle
python3 verify.py .
```

Reuse the existing checkpoint if Josh already downloaded it. Otherwise download
[the published W4 G128 checkpoint](https://huggingface.co/matteiuspi/Qwen3.8-Flash-Next-W4A16-G128-300i)
at the qualified revision. It has the same `config.json` SHA-256 as the test host.
Do not substitute the W8 checkpoint or a generic AWQ/GPTQ export.

```bash
hf download matteiuspi/Qwen3.8-Flash-Next-W4A16-G128-300i \
  --revision d32978ccdde1729caec0034571b03a14779698e9 \
  --local-dir /path/to/Qwen3.8-Flash-Next-W4A16-G128-300i

qwen_python=/path/to/existing/matched/venv/bin/python
qwen_model=/path/to/Qwen3.8-Flash-Next-W4A16-G128-300i
bash serve.sh "$qwen_model" "$qwen_python" --show
bash serve.sh "$qwen_model" "$qwen_python" > qwen-single-card.log 2>&1 &
echo $! > qwen-single-card.pid
tail -f qwen-single-card.log
```

The log is in the current bundle directory. Leave the service running; pressing
Ctrl-C in `tail` stops only the log viewer. Once automatic memory profiling,
weight loading and graph capture complete, run the native gate from a second
terminal in the same bundle directory:

```bash
curl -f http://127.0.0.1:8002/health
bash native.sh "$qwen_python"
curl -s http://127.0.0.1:8002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-single-card","messages":[{"role":"user","content":"Reply with only the integer: 17 * 23"}],"max_tokens":128,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
```

`native.sh` performs the worker numerical gates, installs the residual fusion
and histogram dispatch, and recaptures target and draft graphs. Run it after
each service restart; the server command alone omits those two optimizations.
Check for `391` in the arithmetic response and retain `native-qualification.json`.
The default API is `http://127.0.0.1:8002/v1`, model `qwen38-single-card`.
For a remote coding client, forward that port over SSH; resident development
controls remain on loopback.

The exact profile is TP2/EP on one physical dual-chip card, 131,072 total
tokens, 512-token chunks, one sequence, MTP2, seven FP32 resident state slots,
sequential QSA gathers and decode graph `[3]`. Images and video are disabled.
If chip IDs differ, append `--devices 2,3` only after checking `npu-smi info`.
Automatic memory profiling stays enabled. A lower-memory host may need a
smaller context; do not bypass a failed admission check.

## Environment and launch

The qualification host has 247 GiB of RAM, torch 2.13.0+cpu,
torch_npu 2.13.0rc1, transformers 5.17.0 and the OpenSensor vLLM revision
3ab5dda29. CANN is 9.1.0 and Python is 3.12.13. Its two selected chips report approximately
47 GB each. Check
`npu-smi info` to identify both chips on the same physical card before selecting
`0,1` or `2,3`. Leave chips assigned to another service out of the device list.

Install a coherent Ascend plugin and its matching custom operator packages.
Source the CANN and ATB environment first. Preserve their library paths when
adding the qualified operator packages; the host API and device kernel for
each operator must come from the same package. The launch helper does not
install or discover those packages.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export VLLM_ASCEND_ENABLE_310P=1
export VLLM_ASCEND_KV_CACHE_FRACTION=0.80
python -m tools.qwen4exp.single_card_profile \
  --model /path/to/Qwen3.8-Flash-Next-W4A16-G128-300i \
  --devices 0,1 --port 8002 --context 131072 \
  --batch-tokens 512 --sequences 1 --speculative-tokens 2 \
  --prefix-slots 7 --updated --show
```

Inspect the generated command, then remove `--show` to serve. It binds to
loopback. The helper uses real weights and automatic memory profiling. It
does not transplant a fixed TP4 or TP6 cache budget. `--updated` selects the
FP16 SwiGLU and fused route finalization paths, and sets
`additional_config.qwen_prefix_mamba_min_slots` to the `--prefix-slots` value
(default 16). The large-context recipe uses seven slots. Omitting `--updated`
retains the historical profile and the default 64-slot resident pool. The
131,072-token recipe completed real near-limit inference on one physical
card. Keep automatic profiling enabled on another host: capacity depends on
its memory and matching operator packages.

The original checkpoint's configured maximum is 262,144 tokens. A generated
command with a larger context or batch is a candidate, not a measured capacity
result. Profile memory and validate a near-limit request before advertising a
larger window. The native grouped route ceiling is 25,600 rows; top-k ten
therefore limits a single prefill chunk to 2,560 tokens. Older operator packages
have smaller ceilings. Larger chunks also consume more activation workspace.

MTP2 uses three verification rows per request. With one sequence the helper
captures `[3]`; two sequences use `[3,6]`. Both target and drafter must replay
valid graphs. Adding more capture sizes can exhaust 310P HCCL capture events.

The updated single-card helper uses sequential QSA gathers by default.
`--parallel-gather` selects the two-stream schedule. The 128K parallel-gather
trial admitted memory and passed short requests, then stalled after 6,144 tokens
of its near-limit prompt. Worker stacks and CANN pool failures are retained.
The sequential schedule completed the near-limit cold prompt and replay.
This is not proof that stream overlap alone caused the earlier stall. CANN logging-pool messages also occur in
the progressing sequential run, so those messages alone do not diagnose a
device memory failure.

Images are disabled in this text-performance profile. `--images` enables one
image per prompt and requires a separate memory and image-quality test.
`--quantize-dense-projections` explicitly changes LM-head and PLE projection
arithmetic to W8A8; it is a separate candidate, not part of the W4 runtime A/B.

## Recurrent-state memory

The newer runtime originally failed automatic admission on one card because
its 64-slot primary GDN pool reserved 3,741,843,456 bytes per rank. The 16-slot
pool reserves 935,460,864 bytes, saving 2,806,382,592 bytes (2.61 GiB) per rank.
Eight slots save a further 467,730,432 bytes (446 MiB) per rank. Seven slots
reserve 409,264,128 bytes (390 MiB), the minimum for one request with MTP2.
One sequence with MTP2 needs seven null/live slots, so it still fits. State
precision remains FP32. The slot count always covers the null state and
both live speculative windows; reducing the residency floor never merges
their states. Evicted checkpoint bytes remain available in the host tier.

Worker allocation and optional bounded-scheduler retention use the same
`qwen_prefix_mamba_min_slots` configuration. The default remains 64 for existing
deployments. A low floor can increase checkpoint transfers; it does not change
attention KV ownership or prefix hashes. The separate device archive can use
post-capture headroom while preserving the existing workspace reserve.

The startup `310P physical cache profile` line reports weights, activation
peak, non-Torch memory and physical cache budget before layout translation.
The subsequent compact-cache planner number can be larger than the physical
allocation: it represents virtual Mamba pages, not additional device RAM.

## Verify inference and prefix reuse

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-single-card","messages":[{"role":"user","content":"Reply with only the integer: 17 * 23"}],"max_tokens":128,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'

python -m tools.qwen38_decode_study.smoke \
  --base-url http://127.0.0.1:8002 --output smoke.json
python -m tools.qwen4exp.single_card_prefill \
  --base-url http://127.0.0.1:8002 \
  --model-dir /path/to/Qwen3.8-Flash-Next-W4A16-G128-300i \
  --prompt-tokens 8192 --label fresh-single-card-comparison \
  --output prefill.jsonl
python -m tools.qwen38_decode_study.benchmark \
  --base-url http://127.0.0.1:8002 --lengths 512 --repeats 3 \
  --output decode.jsonl
```

Use a fresh label for each cold/warm pair; use the same label for the two arms
of an A/B comparison after each server starts with an empty cache. The client
submits exact token IDs, checks recall, and reports actual cached and uncached
tokens. Missing cache details are unknown, not zero. Its prompt rate divides
uncached tokens by time to first token and therefore includes first-token work.
Decode throughput uses completed output-token counts, not SSE chunk counts.

The deterministic smoke includes arithmetic, short answers and a parsed tool
call. It is not an accuracy evaluation. In the historical W4 baseline, six of
seven checks passed; reversing `ASCEND` returned `DNESCA` instead of `DNECSA`.
Preserve failures when comparing candidates rather than changing expectations.

## Qualification evidence

The checked-in [qualification receipt](../../../../artifacts/qwen38-single-card-20261011/qualification.json)
contains measured timings, source hashes, memory accounting and known failures.
Host evidence is under
`/srv/ai/storage/artifacts/qwen-single-card-20261011/`. The initial baseline used
only chips 0 and 1, 24,576 context, a 512-token prefill batch, MTP2 and graph `[3]`.

| Fresh measurement | Historical runtime | Updated short profile | Selected 128K profile |
| --- | --- | --- | --- |
| Text context / prefill chunk | 24,576 / 512 | 32,768 / 1,024 | 131,072 / 512 |
| Cold 8,192-token prompt | 31.323 s; 261.54 tokens/s | 25.261 s; 324.29 tokens/s | 28.591 s; 286.52 tokens/s |
| First repeated prompt | 7,680 cached; 2.751 s | 8,064 cached; 0.781 s | 8,064 cached; 0.906 s |
| Three 512-token decode runs | 20.08, 18.12, 17.40 tokens/s | 26.19, 21.73, 22.24 tokens/s | 23.31, 20.46, 20.94 tokens/s |
| Median decode | 18.12 tokens/s | 22.24 tokens/s | 20.94 tokens/s |

The updated short profile used the 16-slot pool and parallel gathers. The
selected profile used seven slots and sequential gathers. These are short-request rates;
long-context processing and generation need separate measurements. The original
W4 checkpoint and dense weight precision were unchanged. The updated runtime
also corrects the recurrent decode boundary to preserve FP32 state, so the
comparison does not isolate one fusion or promise identical text across runs.

The first-replay scheduler retains the final checkpoint usable by an identical
prompt. Without it, the first repeat of an 8,192-token prompt recomputed the
prefix and only the third request hit. With it, the first repeat reported
8,064 cached tokens. Parent alignment and speculation rules still apply.

The 16-slot pool passed 96-step interleaved spill/restore, copy-on-write and
recycling tests with two and four windows on both chips. The eight- and seven-slot pools
passed the same tests with two and three windows on both chips. These tests
prove state transport on synthetic tensors, not whole-model accuracy.

The native residual kernel passed 70 byte-exact cases per worker before the
resident controller recaptured target and draft graphs. It preserves the
injection projection's original shape and rounding. Cold/warm recall, completed
512-token streams, a reasoning-enabled arithmetic problem and 13 generated
`is_prime` test cases then passed. The text/tool smoke remains 6/7, including
the historical reverse-word failure; it is not a broad accuracy evaluation.

| Feature | Qualification scope |
| --- | --- |
| TP2 and expert filtering | Real-weight startup and inference on one physical card |
| MTP2 and decode graphs | Captured and recaptured graphs, completed streams |
| First-replay prefix caching | Actual HTTP usage: 8,064 cached tokens on the first repeat |
| Reduced resident state pool | Seven-slot real 128K serving and synthetic spill/restore tests on both chips |
| Images and video | Disabled; vision tower weights skipped by zero modality limits |
| Flashcomm1 | Disabled on the qualified 310P environment |
| Two concurrent sessions | Generated graph geometry tested offline; hardware test pending |
| 131,072-token context | 131,008-token recall + replay and 130,560-token prompt + 512 generated tokens passed |
| 128K context with batch size 16 | Not qualified; exceeds this profile's sequence limit and memory evidence |

The automatic 512-token profile with 16 slots reported 93,485 cache tokens;
the 1,024-token profile reported 72,138. This is allocation capacity, not a
validated recall window. Smaller chunks reduce activation workspace. Images
were already disabled in both profiles, so toggling them cannot explain the
32K launch limit. Eight slots are predicted to allow about 124,204 cache tokens
from the measured physical byte ledger, without increasing the KV fraction or
bypassing the profiler. The eight-slot real profile admitted 130,465 cache tokens, narrowly short of
a 131,072-token window. Seven slots then admitted 134,614 cache tokens. The
validated serving limit is 131,072 total prompt and output tokens. No KV-budget
increase, profiling bypass or expert re-quantization was needed.

The sequential 128K run completed a cold 131,008-token prompt with correct
passcode recall in 528.366 seconds to first token (247.95 prompt tokens/s). Its
first repeat reported 130,944 cached tokens and 4.932 seconds to first token.
A separate 130,560-token prompt completed all 512 requested output tokens,
reaching exactly 131,072 total tokens at 19.73 generation tokens/s. That
shortened, changed prompt reported zero cached tokens and required another
cold prefill. Identical replay is qualified; broader changed-prompt cache
reuse needs further work. The probe's cache-hit assertion failed while its
capacity and complete-output checks passed; both outcomes are retained.

If an experiment fails, retain its log and source receipt. Select the complete
qualified baseline runtime and matching operators to recover; do not mix
individual model, drafter or attention files from different snapshots.

## Frozen runtime handoff

The qualification host also retains `single-card-runtime-bundle/` under the
artifact directory. It includes the complete matched Ascend Python source,
upstream vLLM source, compiled plugin binding, runtime reference dependencies, the four matching operator
packages and the versioned native residual assets. Symlinks are dereferenced
and every included file has a SHA-256 record. It does not include model weights,
a Python environment or CANN/ATB. The binding requires the recorded Python,
PyTorch and host architecture; this is not a generic wheel.

`serve.sh MODEL_DIRECTORY MATCHED_PYTHON --show` prints the text-only profile.
After serving starts, `native.sh MATCHED_PYTHON` validates the native kernel on
each worker and recaptures the resident graphs. Keep development controls on
loopback. The relocated launcher, model/drafter imports and file hashes passed offline
checks. Its initial import exposed a loader dependency on `tests` reference
modules; those are now included. The full relocated source and operator
snapshot subsequently completed the real near-limit serving gates.

The source commit contains the profile, cache fixes, tests and documentation;
it is not a substitute for the matched compiled operator snapshot. Do not
combine an older model/drafter with a newer loader or native operator package.
