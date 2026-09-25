# Hugging Face Release Plan: Qwen3.8 Flash Next W8A8 for Ascend 300I

## Decision summary

The derivative checkpoint inherits **Qwen Community License 1.0** from
`Qwen/Qwen3.8-Flash-Next`. Do not replace it with GPL. Code written for the
vLLM Ascend integration can remain Apache-2.0 in its code repository, but that
does not relicense the model weights.

The intended Hugging Face repository is:

`matteiuspi/Qwen3.8-Flash-Next-W8A8-DYNAMIC-300i`

The required implementation repositories are already public:

| Component | Public reference |
| --- | --- |
| vLLM Ascend | [`opensensor/vllm-ascend@e3091d230`](https://github.com/opensensor/vllm-ascend/commit/e3091d23015e3fa4a75471fc9d57f45e8b6c7a6b), branch `main` |
| vLLM | [`opensensor/vllm@3756b28a3`](https://github.com/opensensor/vllm/commit/3756b28a3c63d7d5c5503912cdc5e9dfbbe3c7f6), branch `main` |

The private repository has been created, but the account's ability to hold the
full 240.06 GB checkpoint still needs to be confirmed. The two viable
publication paths are:

1. Upgrade the account and stage privately while validation is completed.
2. Complete all mandatory gates below locally, then create the repository as
   public and upload once.

The recommended path is the first one if there is any uncertainty about the
source license, metadata, generated output, or reproducibility. If avoiding PRO
is the priority, do not create the public repository until every mandatory gate
has passed.

The repository was created privately on September 23, 2026, and seeded with the
four publishable metadata files in Hub commit
`184904b9c4d31aebfd2ad871a2119fffcdd97565`. The selected local credential has
write access. Do not store access tokens in this directory or commit them to
Git.

The MTP adapter and 310P v1 runner route were added to public vLLM Ascend
`main` in [`839b6657f`](https://github.com/opensensor/vllm-ascend/commit/839b6657f2e08b60fcd2d3fa124341003465104f)
on September 25, 2026. A subsequent experimental device build loaded the full
checkpoint with one-token MTP and full-decode graphs, advertised a 131,072-token
limit, allocated 142,237 KV tokens, and produced representative batch-one
decode measurements of 14.8–16.0 tokens/s. The associated Ascend performance
snapshot and cold-prefill fix are not yet frozen on public `main`, so these
remain provisional findings rather than the final release baseline.

The exact AISBench GPQA Diamond replay is now frozen in
`../ASCEND-GPQA-BENCHMARK.md` and `../run-ascend-gpqa-replay.sh`. It reproduces
the RTX 6000 PRO settings before any diagnostic rerun changes the 8,192-token
output ceiling.

## Local release overlay

This directory is an overlay for the checkpoint directory; it intentionally
does not duplicate 240 GB of weights.

| File | Destination | Purpose |
| --- | --- | --- |
| `README.md` | repository root | Draft Hugging Face model card |
| `LICENSE` | repository root | Verbatim inherited Qwen license |
| `PROVENANCE.json` | repository root | Machine-readable provenance and smoke-test record |
| `calibration_manifest.json` | repository root | Public-safe conversion and calibration report |
| `RELEASE_PLAN.md` | local-only by default | Curation and publication checklist |

Before upload, copy the first three files into a temporary release view of the
checkpoint. Do not overwrite the only checkpoint copy. Exclude
`RELEASE_PLAN.md` unless we decide that its operational notes are useful to
users.

## Mandatory gates before public upload

### 1. Provenance and licensing

- [x] Identify the official base repository.
- [x] Pin the exact base revision.
- [x] Copy the base model license verbatim.
- [x] Declare the model as a quantized derivative.
- [ ] Independently review Qwen Community License 1.0 before publication.
- [ ] Decide whether the publisher name should be an individual account or a
      project organization.

### 2. Metadata hygiene

- [x] Produce a sanitized `calibration_manifest.json` with local absolute paths
      replaced by stable repository identifiers.
- [ ] Preserve the original manifest outside the upload view for auditability.
- [ ] Scan every non-Safetensors file for credentials, private hostnames, IP
      addresses, usernames, and local filesystem paths.
- [ ] Validate all JSON and YAML files after sanitization.
- [ ] Confirm that every filename in the Safetensors index exists and that no
      unindexed weight shard is present.
- [ ] Verify the 254 shard SHA-256 values against
      `../checkpoint-manifest.json`.

Recommended path replacements:

| Existing field | Public value |
| --- | --- |
| `checkpoint_dir` | `matteiuspi/Qwen3.8-Flash-Next-W8A8-DYNAMIC-300i` |
| `quality.before.model_path` | `Qwen/Qwen3.8-Flash-Next@de4b8e4d...` |
| `quality.after.model_path` | `matteiuspi/Qwen3.8-Flash-Next-W8A8-DYNAMIC-300i` |

### 3. Runtime reproducibility

- [x] Publish the required Qwen4Exp vLLM and vLLM Ascend changes.
- [x] Replace local-only commit references with public commit links.
- [x] Merge the vLLM Ascend release candidate into public `main`.
- [x] Merge the paired vLLM Qwen4Exp changes into public `main`.
- [ ] Tag the final validated commit once tonight's baseline is frozen.
- [ ] Complete the runtime inventory. Python, PyTorch, torch-npu, Transformers,
      vLLM, and `npu-smi` versions are captured; driver/CANN detail and the final
      vLLM Ascend revision remain pending.
- [ ] Re-run a clean-room load from a staged release directory.
- [ ] Verify at least one deterministic text prompt and record the complete
      request and response.
- [ ] Decide whether ECC-disabled operation is merely the test configuration or
      an actual requirement; do not present it as required without evidence.

#### MTP integration and device gate

- [x] Implement the FP16 MTP draft expert bank and map its checkpoint tensor
      shapes without constructing the target INT8 expert bank.
- [x] Route Qwen4Exp MTP to the 310P v1 speculative runner, widen its draft
      hidden-state buffer for all `hc_count` streams, and stage PLE history
      across speculative rollback.
- [x] Pass focused host-side MTP and runner tests (17 tests on September 25).
- [x] Load the full checkpoint with MTP on four 310P devices and complete
      generation requests.
- [ ] Complete an item-level deterministic comparison against the no-MTP target.
- [x] Exercise MTP generation with full-decode graph capture and replay.
- [ ] Measure acceptance length, TTFT, inter-token latency, output throughput,
      and peak memory with and without MTP using the same prompts.

### 4. Quality and performance

- [ ] Run a held-out perplexity comparison against the pinned BF16 base.
- [ ] Run the frozen 198-item GPQA Diamond replay and archive its prediction
      ledger, detailed grader output, server command, and server log.
- [ ] Benchmark TTFT, inter-token latency, output throughput, and end-to-end
      latency with a versioned harness.
- [x] Exercise prompts beyond the original 2,304-token smoke context; captured
      development requests reached 43,603 prompt tokens.
- [ ] Validate one complete 100K/131K prompt rather than inferring support only
      from the configured limit and allocated KV-token capacity.
- [ ] Clearly separate measured results from projections for 262K and 1M
      context.
- [ ] Record warm-start and cold-start memory per chip with ECC state noted.

### 5. Model-card accuracy

- [ ] Decide whether to advertise `text-generation` only or multimodal support.
      The current Ascend path has only validated text generation.
- [x] State the MTP status precisely: real-weight generation and decode graphs
      are functional in a provisional build; quality parity, controlled
      performance, and the final public runtime revision remain pending.
- [ ] Add limitations, intended use, out-of-scope use, and risk information.
- [ ] Add original Qwen citation details from the official model card.
- [ ] Replace the draft warning once the release is reproducible.
- [x] Run a model-card metadata validation before upload.

### 6. Upload rehearsal

- [ ] Build a temporary release view without duplicating the weight payload,
      using reflinks or hard links on the same filesystem where safe.
- [ ] Compare its file inventory and total bytes with the approved manifest.
- [x] Create the repository privately and confirm its visibility.
- [ ] Use current `hf upload`, which resumes interrupted folder uploads.
- [ ] Set `HF_XET_HIGH_PERFORMANCE=1` for the large upload.
- [ ] After upload, compare Hub file names and sizes with the release manifest.
- [ ] Download at least the configuration and one shard and verify checksums.
- [ ] Test the published model card and gated/private access behavior, as
      applicable.

## Suggested publication sequence

1. Resolve the code-release dependency and metadata hygiene gates.
2. Freeze an immutable candidate manifest and checksums.
3. Run functional, quality, and performance validation from that candidate.
4. Review the model card and license one final time.
5. Create the Hub repository with the chosen visibility.
6. Upload from the Ascend server using the resumable HF CLI path.
7. Verify every remote file before announcing the repository.

At the measured server uplink of roughly 4.6 MB/s, the initial 240 GB transfer
is expected to take approximately 14 to 16 hours, excluding retries and Hub-side
processing.
