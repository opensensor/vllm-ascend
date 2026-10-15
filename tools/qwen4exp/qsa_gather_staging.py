# SPDX-License-Identifier: Apache-2.0
"""Private eager QSA NZ gather staging experiment; no selection/math changes."""

import hashlib
import importlib
import json
import time
from contextlib import nullcontext
from pathlib import Path
from types import FunctionType

import torch

NAMESPACE = "qwen_qsa_gather_staging_v1"
MAX_ROWS = 64
MAX_CONFIGS = 256
HEAD_DIM = 256
GROUP_WIDTH = 512
PADDED_TOKENS = 2064
CACHE_PAGES = 40
COUNTS = (0, 1, 15, 16, 17, 255, 256, 511, 512)
TABLE_WIDTHS = (2048, 4096)
VARIANTS = (4, 8)
QUALIFIED_HEADER_SHA256 = "dbceeef186d29656390a644061f781ecb682ad4ae5a9d83f0c4969db169671a1"
QUALIFIED_TILING_SHA256 = "5193b22a2ca386b1fff93f807472090a2f8e202b1a11d32477bf0dd0ac9d3681"
GATHER_MODULE = "vllm_ascend.models.qwen4_exp.ops.qsa_gather_nz_310"
ATTENTION_MODULE = "vllm_ascend.models.qwen4_exp.ops.qsa_batched_attention_310"
ATTENTION_FILE_SHA256 = "eae684c0ed7efe1d9c4c8b388cc8160d0efd0b1616bb37eed42304ead63b9661"
REQUIRED_SOURCES = frozenset(
    (
        "native_qsa_gather_staging.cpp",
        "native_owned_bridge.cpp",
        "shared_launch_arguments.h",
        "qsa_gather_value_nz_v310.h",
        "qsa_gather_value_nz_v310_tiling_data.h",
        "qsa_gather_staging_4.h",
        "qsa_gather_staging_8.h",
    )
)


def _on_npu(device):
    return device.type == "npu"


def _is_capturing():
    return torch.npu.is_current_stream_capturing()


def _npu_module():
    import torch_npu

    return torch_npu


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_build(build):
    root = Path(build).resolve()
    provenance = json.loads((root / "provenance.json").read_text())
    if (
        provenance.get("namespace") != NAMESPACE
        or provenance.get("bridge_source") != "native_owned_bridge.cpp"
        or provenance.get("flush_host_queue") is not True
        or provenance.get("eager_only") is not True
        or provenance.get("native_task_type") != "KERNEL_TYPE_AICORE"
    ):
        raise ValueError("QSA staging requires its owned, queue-draining eager bridge")
    if not REQUIRED_SOURCES.issubset(provenance["sources"]):
        raise ValueError("QSA staging requires frozen native and original/generated headers")
    if provenance["sources"]["qsa_gather_value_nz_v310.h"] != QUALIFIED_HEADER_SHA256:
        raise ValueError("QSA staging original header differs from qualified runtime")
    if provenance["sources"]["qsa_gather_value_nz_v310_tiling_data.h"] != QUALIFIED_TILING_SHA256:
        raise ValueError("QSA staging tiling descriptor differs from qualified runtime")
    expected = {f"native_qsa_gather_staging_{variant}" for variant in VARIANTS}
    if set(provenance["binaries"]) != expected:
        raise ValueError("QSA staging requires both independent native variants")
    for name, checksum in provenance["sources"].items():
        if Path(name).name != name or _digest(root / name) != checksum:
            raise ValueError("QSA staging frozen source digest differs")
    original_header = (root / "qsa_gather_value_nz_v310.h").read_text()
    for variant in VARIANTS:
        expected_header = original_header.replace(
            "constexpr int64_t TOKEN_BLOCKS_PER_TILE = 4;", f"constexpr int64_t TOKEN_BLOCKS_PER_TILE = {variant};"
        )
        if (root / f"qsa_gather_staging_{variant}.h").read_text() != expected_header:
            raise ValueError("QSA staging candidate changes more than its staging width")
        entry = provenance["binaries"][f"native_qsa_gather_staging_{variant}"]
        if entry.get("entry") != f"qwen_qsa_gather_staging_{variant}_v1":
            raise ValueError("QSA staging native entry differs")
    for entry in (provenance["bridge"], *provenance["binaries"].values()):
        path = Path(entry["path"])
        if not path.is_absolute() or path.resolve().parent != root or _digest(path) != entry["sha256"]:
            raise ValueError("QSA staging artifact path or digest differs")
    return provenance


def geometry(cache, selection, table, head_dim, transpose_output, output=None):
    """Metadata-only admission; selection validity remains the original contract."""
    values = (
        cache,
        selection.group_indices,
        selection.group_counts,
        selection.tail_starts,
        selection.tail_counts,
        table,
    )
    if not all(isinstance(value, torch.Tensor) for value in values):
        return None
    if (
        not _on_npu(cache.device)
        or cache.dtype != torch.float16
        or cache.ndim != 4
        or cache.shape[0] < 1
        or cache.shape[1] not in (16, 32)
        or cache.shape[2:] != (64, 16)
        or head_dim != HEAD_DIM
        or type(transpose_output) is not bool
    ):
        return None
    groups = selection.group_indices
    if groups.ndim != 2 or not 1 <= groups.shape[0] <= MAX_ROWS or groups.shape[1] != GROUP_WIDTH:
        return None
    rows, heads = groups.shape[0], cache.shape[1] // 16
    if table.ndim != 2 or table.shape[0] != 1 or not 1 <= table.shape[1] <= 4096:
        return None
    if any(value.shape != (rows,) for value in values[2:5]):
        return None
    if any(value.device != cache.device or not value.is_contiguous() for value in values):
        return None
    if any(value.dtype not in (torch.int32, torch.int64) for value in values[1:]):
        return None
    npu = _npu_module()
    shape = (rows, heads, HEAD_DIM, PADDED_TOKENS) if transpose_output else (rows, heads, PADDED_TOKENS, HEAD_DIM)
    if npu.get_npu_format(cache) != 29:
        return None
    if output is not None and (
        not isinstance(output, torch.Tensor)
        or output.shape != shape
        or output.dtype != torch.float16
        or output.device != cache.device
        or not output.is_contiguous()
        or npu.get_npu_format(output) != 29
    ):
        return None
    return (rows, heads, table.shape[1], transpose_output, cache.device)


def descriptor(key):
    rows, heads, table_width, transpose, _ = key
    tasks = rows * heads
    blocks = min(tasks, 8)
    return [
        rows,
        heads,
        16,
        heads * 16,
        64,
        GROUP_WIDTH,
        129,
        table_width,
        tasks,
        (tasks + blocks - 1) // blocks,
        int(transpose),
    ], blocks


def _descriptor_ready():
    stream = torch.npu.current_stream()
    return stream, stream.record_event()


def _wait_descriptor(ready):
    producer, event = ready
    consumer = torch.npu.current_stream()
    if consumer != producer:
        consumer.wait_event(event)


class GatherStaging:
    WRAPPER_VERSION = 2

    def __init__(self, kernels, launch):
        self.kernels, self.launch = kernels, launch
        self._configs = {}
        self._config_events = {}
        self._config_blocks = {}
        self.failed = False
        self.failure = self.failure_traceback = None

    def __call__(self, cache, selection, table, *, head_dim, transpose_output, output=None, variant=8):
        if self.failed:
            raise RuntimeError("QSA staging failed; retain owner and tensor frames")
        if type(variant) is not int or variant not in VARIANTS:
            raise ValueError("QSA staging variant must be 4 or 8")
        if _is_capturing():
            raise RuntimeError("QSA staging is eager-only, including cached descriptors")
        key = geometry(cache, selection, table, head_dim, transpose_output, output)
        if key is None:
            raise ValueError("unsupported QSA staging layout")
        return self.launch_admitted(key, cache, selection, table, output=output, variant=variant)

    def launch_admitted(self, key, cache, selection, table, *, output=None, variant=8):
        """Internal eager launch after this call's geometry/capture admission.

        Both the public wrapper and source-guarded model binding admit exactly
        once immediately before invoking this method. Never use this entry to
        bypass admission for externally supplied tensors or graph capture.
        """
        if self.failed:
            raise RuntimeError("QSA staging failed; retain owner and tensor frames")
        if type(variant) is not int or variant not in VARIANTS:
            raise ValueError("QSA staging variant must be 4 or 8")
        if key not in self._configs and len(self._configs) >= MAX_CONFIGS:
            raise RuntimeError("QSA staging descriptor capacity reached")
        try:
            config = self._configs.get(key)
            if config is None:
                fields, blocks = descriptor(key)
                config = torch.tensor(fields, dtype=torch.int64, device=cache.device)
                self._configs[key] = config
                self._config_blocks[key] = blocks
                self._config_events[key] = _descriptor_ready()
            blocks = self._config_blocks[key]
            _wait_descriptor(self._config_events[key])
            if output is None:
                rows, heads = key[:2]
                shape = (rows, heads, HEAD_DIM, PADDED_TOKENS) if key[3] else (rows, heads, PADDED_TOKENS, HEAD_DIM)
                output = _npu_module().empty_with_format(
                    size=shape, dtype=cache.dtype, device=cache.device, acl_format=29
                )
            metadata = [
                value.to(dtype=torch.int32).contiguous()
                for value in (
                    selection.group_indices,
                    selection.group_counts,
                    selection.tail_starts,
                    selection.tail_counts,
                    table,
                )
            ]
            arguments = [cache, *metadata, output, config]
            # Same stream as the caller; original attention owns all K/V events.
            self.launch(self.kernels[variant], arguments, blocks)
            return output
        except BaseException as error:
            self.failed = True
            self.failure, self.failure_traceback = error, error.__traceback__
            raise


def upgrade_owned_wrapper(state):
    """Cold paused/drained upgrade: preserve native owners and every descriptor.

    The controller must drain the worker before calling this explicit helper.
    Old Python wrappers remain strongly owned for rollback/in-flight inspection;
    no class mutation, native constructor, allocation, teardown or unload occurs.
    """
    if _is_capturing():
        raise RuntimeError("wrapper upgrade is eager-only")
    resources = state.get("resources", {})
    old = resources.get("gather")
    if (
        state.get("failed")
        or not state.get("loaded")
        or resources.get("eager_only") is not True
        or old is None
        or old.failed
    ):
        raise RuntimeError("wrapper upgrade requires a healthy loaded eager owner")
    version = getattr(old, "WRAPPER_VERSION", 1)
    if type(version) is not int or version not in (1, GatherStaging.WRAPPER_VERSION):
        raise RuntimeError("unknown wrapper version; do not reinterpret its ownership layout")
    if version == GatherStaging.WRAPPER_VERSION:
        return old
    if set(old._configs) != set(old._config_events):
        raise RuntimeError("wrapper descriptor/event ownership is incomplete")
    upgraded = GatherStaging(old.kernels, old.launch)
    upgraded._configs.update(old._configs)
    upgraded._config_events.update(old._config_events)
    upgraded._config_blocks.update({key: descriptor(key)[1] for key in old._configs})
    state.setdefault("previous_gather_wrappers", []).append(old)
    resources["gather"] = upgraded
    return upgraded


def load_resources(build, *, owner):
    if owner:
        raise ValueError("QSA staging owner must begin empty")
    provenance = verify_build(build)
    torch.ops.load_library(provenance["bridge"]["path"])
    owner.update(eager_only=True, kernels={}, provenance=provenance)
    for variant in VARIANTS:
        entry = provenance["binaries"][f"native_qsa_gather_staging_{variant}"]
        owner["kernels"][variant] = getattr(torch.classes, NAMESPACE).Kernel(entry["path"], entry["entry"])
    owner["gather"] = GatherStaging(owner["kernels"], getattr(torch.ops, NAMESPACE).launch)
    return owner


def raw_bytes(tensor):
    """Cold full-storage snapshot, without an NZ-to-ND conversion.

    Private gate tensors own their full storage. Unsupported backend storage
    copies fail the gate explicitly; logical equality is never substituted.
    """
    if tensor.storage_offset() != 0:
        raise ValueError("physical byte snapshots require standalone storage")
    storage = tensor.untyped_storage().cpu()
    return torch.empty(0, dtype=torch.uint8).set_(storage, 0, (storage.nbytes(),), (1,)).clone()


def _selection_type():
    return importlib.import_module("vllm_ascend.models.qwen4_exp.ops.qsa_indexer").QSAGroupSelection


def private_inputs(rows, heads, table_width, step, device):
    generator = torch.Generator().manual_seed(781 + step)
    cache = torch.randn(CACHE_PAGES, heads * 16, 64, 16, dtype=torch.float16, generator=generator)
    # Every logical page maps into a small independent private cache.
    table = torch.randint(CACHE_PAGES, (1, table_width), dtype=torch.int32, generator=generator)
    table[0, :CACHE_PAGES] = torch.randperm(CACHE_PAGES, generator=generator).int()
    groups = torch.randint(512, (rows, GROUP_WIDTH), generator=generator, dtype=torch.int32)
    groups[:, ::7] = 0  # repeated groups, nonmonotonic page ordering
    counts = torch.tensor([COUNTS[(step + row) % len(COUNTS)] for row in range(rows)], dtype=torch.int32)
    tails = torch.tensor([(step + row) % 4 for row in range(rows)], dtype=torch.int32)
    starts = torch.tensor([63 + ((step + row) % 4) * 64 for row in range(rows)], dtype=torch.int32)
    metadata_dtype = torch.int64 if step % 2 else torch.int32
    selection = _selection_type()(
        *(value.to(device=device, dtype=metadata_dtype) for value in (groups, counts, starts, tails))
    )
    return _npu_module().npu_format_cast(cache.to(device), 29), selection, table.to(device)


def _selection_bytes(selection):
    return [
        raw_bytes(getattr(selection, name)) for name in ("group_indices", "group_counts", "tail_starts", "tail_counts")
    ]


def attention_function(gather):
    module = importlib.import_module(ATTENTION_MODULE)
    if _digest(module.__file__) != ATTENTION_FILE_SHA256:
        raise RuntimeError("whole-attention gate requires the qualified untouched attention file")
    original = module.qsa_batched_prefill_310
    namespace = dict(original.__globals__)
    for transpose, name in ((True, "qsa_gather_key_transposed_nz_310"), (False, "qsa_gather_value_nz_310")):

        def wrapper(cache, selection, table, *, head_dim, output=None, _transpose=transpose):
            return gather(cache, selection, table, head_dim=head_dim, transpose_output=_transpose, output=output)

        namespace[name] = wrapper
    clone = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    clone.__kwdefaults__ = original.__kwdefaults__
    return original, clone


def validate_attention(resource, device):
    records = []
    for heads in (1, 2):
        cache, selection, table = private_inputs(95, heads, 4096, 2, device)
        # The first request exceeds the original64-query reusable-buffer tile.
        table = table.expand(2, -1).contiguous()
        query = torch.randn(95, heads * 2, HEAD_DIM, dtype=torch.float16, device=device)
        loc = torch.tensor([0, 65, 95], dtype=torch.int32, device=device)
        original, _ = attention_function(resource)
        before = [raw_bytes(cache), raw_bytes(table), raw_bytes(query), *_selection_bytes(selection)]
        for parallel in (False, True):
            streams = (torch.npu.Stream(), torch.npu.Stream()) if parallel else None
            kwargs = dict(scale=1 / 16, query_tile=64, query_lens=[65, 30], gather_streams=streams)
            expected = original(query, cache, cache, selection, table, loc, **kwargs)
            torch.npu.synchronize()
            expected_bytes = raw_bytes(expected)
            for variant in VARIANTS:

                def gather(*args, _variant=variant, **kw):
                    return resource(*args, variant=_variant, **kw)

                _, candidate = attention_function(gather)
                actual = candidate(query, cache, cache, selection, table, loc, **kwargs)
                torch.npu.synchronize()
                records.append(
                    dict(
                        heads=heads,
                        parallel=parallel,
                        variant=variant,
                        exact=torch.equal(raw_bytes(actual), expected_bytes),
                        inputs_preserved=all(
                            torch.equal(a, b)
                            for a, b in zip(
                                before,
                                [raw_bytes(cache), raw_bytes(table), raw_bytes(query), *_selection_bytes(selection)],
                                strict=True,
                            )
                        ),
                    )
                )
                del actual
            del expected, expected_bytes, streams
        del cache, selection, table, query, loc, before
    return records


def validate(resource, device, *, rows=(1, 3, 64), repeats=3):
    baseline = importlib.import_module(GATHER_MODULE)._qsa_gather_nz_310
    baseline = getattr(baseline, "_qfn_qsa_gather_staging_base", baseline)
    records = []
    for count in rows:
        for heads in (1, 2):
            for table_width in TABLE_WIDTHS:
                for transpose in (False, True):
                    outputs = {}
                    for step in range(len(COUNTS)):
                        cache, selection, table = private_inputs(count, heads, table_width, step, device)
                        before = [raw_bytes(cache), raw_bytes(table), *_selection_bytes(selection)]
                        expected = baseline(
                            cache,
                            selection,
                            table,
                            head_dim=HEAD_DIM,
                            transpose_output=transpose,
                            output=outputs.get("baseline"),
                        )
                        outputs["baseline"] = expected
                        torch.npu.synchronize()
                        expected_bytes = raw_bytes(expected)
                        for variant in VARIANTS:
                            actual = resource(
                                cache,
                                selection,
                                table,
                                head_dim=HEAD_DIM,
                                transpose_output=transpose,
                                output=outputs.get(variant),
                                variant=variant,
                            )
                            reused = variant not in outputs or actual is outputs[variant]
                            outputs[variant] = actual
                            torch.npu.synchronize()
                            exact = torch.equal(raw_bytes(actual), expected_bytes)
                            after = [raw_bytes(cache), raw_bytes(table), *_selection_bytes(selection)]
                            preserved = all(torch.equal(a, b) for a, b in zip(before, after, strict=True))
                            records.append(
                                dict(
                                    rows=count,
                                    heads=heads,
                                    table_width=table_width,
                                    transpose=transpose,
                                    step=step,
                                    variant=variant,
                                    exact=exact,
                                    inputs_preserved=preserved,
                                    output_reused=reused,
                                )
                            )
                            del after
                        del cache, selection, table, before, expected_bytes
                    del outputs, actual, expected
    attention = validate_attention(resource, device) if 64 in rows else []
    return dict(
        passed=all(r["exact"] and r["inputs_preserved"] and r["output_reused"] for r in records)
        and all(r["exact"] and r["inputs_preserved"] for r in attention),
        cases=records,
        attention=attention,
        scope="private physical NZ bytes and unchanged full attention; no serving qualification",
    )


def timing(resource, device, *, rows, repeats, record_function=None, diagnostic_routes=False):
    baseline = importlib.import_module(GATHER_MODULE)._qsa_gather_nz_310
    baseline = getattr(baseline, "_qfn_qsa_gather_staging_base", baseline)
    records = []
    routes = ("installed", 4, 8)
    model = None
    if diagnostic_routes:
        from tools.qwen4exp.resident_candidates.qfn_qsa_gather_staging import build_forward

        model = build_forward(baseline, resource)
        routes += ("admitted4", "admitted8", "model8")
    for count in rows:
        for dense in (False, True):
            cache, selection, table = private_inputs(count, 1, 4096, 0, device)
            selection.group_counts.fill_(512 if dense else 16)
            for transpose in (False, True):
                outputs = {}
                for repeat in range(repeats + 1):
                    order = routes if repeat % 2 == 0 else tuple(reversed(routes))
                    for variant in order:
                        kwargs = dict(head_dim=HEAD_DIM, transpose_output=transpose, output=outputs.get(variant))
                        key = None
                        if variant in ("admitted4", "admitted8"):
                            if _is_capturing():
                                raise RuntimeError("private admitted timing is eager-only")
                            key = geometry(cache, selection, table, HEAD_DIM, transpose, outputs.get(variant))
                            if key is None:
                                raise RuntimeError("private timing operands failed explicit admission")
                        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                        torch.npu.synchronize()
                        wall = time.perf_counter()
                        start.record()
                        layout = "K" if transpose else "V"
                        density = "dense" if dense else "sparse"
                        label = f"qsa_gather/rows{count}/{density}/{layout}/{variant}/repeat{repeat}"
                        with record_function(label) if record_function is not None else nullcontext():
                            if variant == "installed":
                                output = baseline(cache, selection, table, **kwargs)
                            elif variant == "model8":
                                output = model(cache, selection, table, **kwargs)
                            elif variant in ("admitted4", "admitted8"):
                                output = resource.launch_admitted(
                                    key, cache, selection, table, output=kwargs["output"], variant=int(variant[-1])
                                )
                            else:
                                output = resource(cache, selection, table, variant=variant, **kwargs)
                        outputs[variant] = output
                        end.record()
                        end.synchronize()
                        elapsed = (time.perf_counter() - wall) * 1000
                        if repeat:
                            records.append(
                                dict(
                                    rows=count,
                                    dense=dense,
                                    transpose=transpose,
                                    variant=variant,
                                    repeat=repeat,
                                    wall_ms=elapsed,
                                    device_ms=start.elapsed_time(end),
                                )
                            )
                del outputs, output
            del cache, selection, table
    return dict(
        passed=True,
        records=records,
        scope=(
            "alternating wall/device gather timing; admitted routes are a hoisted-admission ceiling, "
            "model8 is real K-only binding"
        ),
    )
