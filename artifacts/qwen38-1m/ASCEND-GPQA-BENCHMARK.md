# Ascend GPQA Diamond Replay

This runbook freezes the quality comparison between the RTX 6000 PRO
`llama.cpp` UD-IQ4_XS baseline and the four-chip Ascend 310P3
`W8A8_DYNAMIC` release candidate.

## Reference baseline

The reference run is:

Use the immutable `gpqa-diamond-deterministic-iq4xs/20260923_103113` result
directory from the RTX workstation as the reference.

Its frozen settings were:

| Setting | Value |
| --- | --- |
| Dataset | AISBench `gpqa_gen_0_shot_cot_chat_prompt` |
| Dataset version | `b1ed2c` |
| Questions | 198 |
| AISBench revision | `19018e9c` |
| Temperature | `0.0` |
| Seed | `1024` |
| Top-p / top-k / min-p | `1.0` / `20` / `0.0` |
| Repetition / presence penalty | `1.0` / `0.0` |
| Maximum output | 8,192 tokens |
| Client concurrency | 3 |
| Warmups | 1 |
| Result | 138/198, 69.70% |

The reference generated 981,338 output tokens. Fifty-four responses reached
the 8,192-token ceiling without an extractable answer. Preserve that ceiling
for the first Ascend run: raising it would answer a useful question, but not
the same question.

## Ascend server gate

Do not start the scored run until the server configuration is frozen and the
following evidence has been saved:

- vLLM and vLLM Ascend Git revisions, with a clean-tree indicator or patch;
- complete `vllm serve` command;
- driver, firmware, CANN, Python, PyTorch, `torch-npu`, Transformers, vLLM,
  and vLLM Ascend versions;
- ECC state, per-chip memory, configured context limit, and allocated KV-token
  capacity;
- whether MTP, prefix caching, chunked prefill, and decode graphs are enabled;
- one cold prompt and one warmed continuation with request-timing records.

The September 25 milestone configuration is not yet the frozen release build.
It used TP=4, one MTP draft token, prefix caching, chunked prefill, full-decode
graphs, `max_model_len=131072`, and `max_num_seqs=1`. Fix or explicitly accept
the cold-prefill issue before starting GPQA, then restart once so the entire
run uses one code and configuration state.

## Run the exact replay

Run the client on the RTX workstation. It already contains the pinned
AISBench environment and dataset, while the OpenAI-compatible endpoint remains
on the Ascend host.

```bash
artifacts/qwen38-1m/run-ascend-gpqa-replay.sh
```

Set the Ascend server address. Other paths and names can be overridden without
editing the script:

```bash
SERVER_HOST=<ASCEND_SERVER> \
SERVER_PORT=8001 \
MODEL_NAME=qwen38-flash-next-w8a8 \
AIS_BENCH_BIN=/path/to/pinned/ais_bench \
WORK_ROOT=/path/to/benchmark-results/qwen38-quality \
RUN_NAME=gpqa-diamond-deterministic-ascend-w8a8-final \
  artifacts/qwen38-1m/run-ascend-gpqa-replay.sh
```

### Five-question smoke run

Before committing to the complete 198-question replay, run the first five
dataset prompts with all other generation and evaluation settings unchanged:

```bash
SERVER_HOST=<ASCEND_SERVER> \
SERVER_PORT=8001 \
NUM_PROMPTS=5 \
RUN_NAME=gpqa-diamond-first5-ascend-w8a8 \
  artifacts/qwen38-1m/run-ascend-gpqa-replay.sh
```

AISBench implements `--num-prompts 5` as the dataset range `[:5]`, so this is
a deterministic first-five subset rather than a random sample. Treat its score
as a pipeline and qualitative smoke check, not as a statistically meaningful
model-quality estimate.

With the current server limited to one active sequence, the reference token
volume implies roughly 17–18.5 hours of decode at 14.8–16.0 tokens/s. Client
concurrency remains three to reproduce the reference request schedule, but the
server may queue two requests. Record actual wall time rather than presenting
this estimate as measured benchmark throughput.

## Required result checks

Before publishing a score:

1. Confirm all 198 prediction rows exist and every API request succeeded.
2. Record raw accuracy and the framework-reported extraction rate.
3. Independently count extractable answer letters from item-level grader
   output; do not equate request success with answer extraction.
4. Report the number of outputs at 8,192 tokens, cap-without-answer count, and
   accuracy among responses with extracted choices.
5. Compare correct-to-incorrect and incorrect-to-correct flips against the RTX
   result by item ID.
6. Report Biology, Chemistry, and Physics raw and answered-only accuracy.
7. Archive the generated AISBench config, prediction JSONL, detailed result
   JSON, summaries, server command, and server log together.

Generate the paired summary after the Ascend timestamp is known:

```bash
python artifacts/qwen38-1m/compare-gpqa-runs.py \
  <RTX_BASELINE_TIMESTAMP_DIR> \
  <ASCEND_TIMESTAMP_DIR> \
  --output /tmp/qwen38-gpqa-paired-comparison.md
```

For the Biology/Chemistry/Physics table, download the GPQA metadata CSV after
accepting its dataset terms and add `--metadata-csv /path/to/gpqa_diamond.csv`.
The comparison script reports only aggregate domains and item IDs; it does not
republish benchmark questions.

## Follow-up runs

Only after the exact replay is complete:

- repeat the 54 RTX cap-without-answer items with a 16,384-token ceiling or an
  answer-first instruction to measure completion behavior;
- run target-only and MTP-enabled configurations on the same item subset to
  detect speculative-decoding quality regressions;
- measure performance separately with a versioned latency/throughput harness.

Quality and speed are separate claims. GPQA answers the quality question; the
request-timing and load tests answer TTFT, prefill, decode, concurrency, and
long-context performance questions.
