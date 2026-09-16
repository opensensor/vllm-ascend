# Environment Freeze: Qwen3.8-Flash-Next (Qwen4Exp) 1M Context on Ascend 310P

**Companion to**: `qwen38_flash_next_1m_310p_prd.md` (PRD),
`qwen38-flash-next-1m-310p-plan.md` (plan, task T0.2),
`qwen38_flash_next_1m_runtime_requirements.md` (runtime requirements).
**Tool**: `tools/qwen38_1m/env_freeze.py`
**Tests**: `tests/ut/qwen38_1m/test_env_freeze.py`

## Why we freeze the environment

A 1M-context run on four 310P chips is only meaningful if it is reproducible. Every
run artifact (plan R1, R13) embeds the exact software stack it ran against so that a
later regression, OOM, or accuracy drift can be attributed to a specific revision
rather than to an unknown, drifted container. The freeze document is the
machine-readable record of that stack, produced by `tools/qwen38_1m/env_freeze.py`
and consumed by the run-artifact writer (plan T4.x).

## What is frozen, and why

The recorder pins the revision of each component that can change model behaviour,
memory footprint, or numerics:

| Component | Why it is pinned |
| --- | --- |
| `vllm` | Scheduler, KV-cache manager, and model-execution semantics. |
| `vllm_ascend` | The NPU plugin: platform patches, 310P model runner, allocator behaviour. |
| `torch_npu` | Device kernels and the ACL runtime binding; numerics and memory. |
| `cann` | The Ascend toolkit under `torch_npu`; driver/runtime compatibility. |
| `transformers` | Model config / tokenizer plumbing; affects tokenization and shapes. |
| `tokenizers` | The fast tokenizer implementation; token boundaries must be stable at 1M. |
| `modelslim` | The quantization toolchain that produced the W8A8 routed-expert weights. |
| `checkpoint_hash` | The exact native-Ascend checkpoint the run loaded (see runtime requirements). |

Each entry records a `version`, an optional resolved `path`, and a `source` (how the
value was obtained), so a reader can judge how authoritative each value is.

## Pinned import-path assertion

The pinned container places the working trees at `/vllm-workspace/vllm` and
`/vllm-workspace/vllm-ascend`. A stray `pip`-installed copy of either package on
`sys.path` silently changes behaviour while still "importing fine". Before a run
produces any artifact, `import_check()` asserts that the server's `vllm` and
`vllm_ascend` imports resolve **under** those pinned prefixes, and raises
`ImportCheckError` loudly otherwise. The expected prefixes are parameters
(`expected_vllm_prefix`, `expected_vllm_ascend_prefix`) so the check is testable
off-target and reusable if the container layout changes. The default path collector
uses `importlib.util.find_spec` and therefore resolves the origins **without**
importing the packages, avoiding the heavy (and NPU-touching) import side effects.

## Design (matches the T0.3 probe conventions)

The recorder follows the same shape as `tools/qwen38_1m/hw_probe.py`:

- A `SCHEMA_VERSION`-tagged dataclass report (`EnvFreezeReport`) with
  `to_dict` / `to_json` / `from_dict` round-trip and a `human_summary()`.
- **Injectable collectors**: `freeze()` takes a `revision_collector` and an
  optional `checkpoint_hash_collector`. The default collectors read package
  metadata (`importlib.metadata`) and container environment variables
  (`ASCEND_TOOLKIT_VERSION`, `ASCEND_TOOLKIT_HOME`, `QWEN38_CHECKPOINT_HASH`), and
  are exercised only on the target; the unit tests inject fakes.
- **No hard torch-npu import at load**: the module imports nothing NPU-specific at
  import time; the default `torch_npu` collector reads metadata lazily.
- `freeze(require_all=True)` raises `ValueError` if any of the required components
  (`REQUIRED_COMPONENTS`) is absent, so an incomplete freeze cannot silently be
  embedded in an artifact.

## Output

`env_freeze.py` emits a JSON document, for example:

```json
{
  "schema_version": 1,
  "timestamp": 1757894400.0,
  "components": {
    "vllm": {"name": "vllm", "version": "0.11.0", "path": "/vllm-workspace/vllm/__init__.py", "source": "importlib.metadata"},
    "vllm_ascend": {"name": "vllm_ascend", "version": "0.11.0.dev", "path": "/vllm-workspace/vllm-ascend/vllm_ascend/__init__.py", "source": "importlib.metadata"},
    "torch_npu": {"name": "torch_npu", "version": "2.5.1", "path": null, "source": "importlib.metadata"},
    "cann": {"name": "cann", "version": "8.0.RC3", "path": "/usr/local/Ascend", "source": "env"},
    "transformers": {"name": "transformers", "version": "4.51.0", "path": null, "source": "importlib.metadata"},
    "tokenizers": {"name": "tokenizers", "version": "0.21.0", "path": null, "source": "importlib.metadata"},
    "modelslim": {"name": "modelslim", "version": "master-abc123", "path": null, "source": "importlib.metadata"}
  },
  "checkpoint_hash": "sha256:..."
}
```

## Usage

```bash
# On the pinned container/target (asserts pinned import paths, then records):
python3 tools/qwen38_1m/env_freeze.py --json-out run/env_freeze.json

# Off-target authoring (skip the /vllm-workspace assertion):
python3 tools/qwen38_1m/env_freeze.py --skip-import-check --quiet
```
