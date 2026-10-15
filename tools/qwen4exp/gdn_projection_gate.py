# SPDX-License-Identifier: Apache-2.0
"""Private real-weight QKV/Z concatenation experiment; never a model binding."""

import hashlib
import importlib
import inspect
import statistics
import textwrap
import time
from functools import partial

import torch

MODEL_MODULE = "vllm_ascend.models.qwen4_exp.model"
CLASS_SHA256 = "2443170bfc98d501034519d2b897e4d06b567d1d0f06f86b52c493b6aaad769c"
LINEAR_SHA256 = "0efbbe722fd2656dfafc39042e76b0aa23cc7797f238c5a7cdc91b4ebc878f6e"
OPERAND_SHA256 = "52a70f8f8efe7721e2a4af83caa4df1e6b4fe8510864dcc9df79e46d1433337f"
HIDDEN = 2560
QKV_WIDTH = 2560
Z_WIDTH = 1536
ROWS = (3, 6, 64)
INPUT_DTYPES = (torch.float32, torch.float16)
INPUT_STEPS = 4
MEMORY_REQUIRED_BYTES = (128 + 256) << 20


def source_hash(value):
    return hashlib.sha256(textwrap.dedent(inspect.getsource(value)).encode()).hexdigest()


def resident_module(worker, name, *, model_module=None):
    module = importlib.import_module(MODEL_MODULE) if model_module is None else model_module
    for value, digest in (
        (module._GDNAttention, CLASS_SHA256),
        (module._linear, LINEAR_SHA256),
        (module._linear_operand_dtype, OPERAND_SHA256),
    ):
        if source_hash(value) != digest:
            raise RuntimeError("GDN gate requires exact pinned class and linear source")
    matches = [
        (path, layer)
        for path, layer in worker.model_runner.get_model().named_modules()
        if type(layer) is module._GDNAttention
    ]
    if not matches:
        raise RuntimeError("no exact pinned GDN module in resident model")
    if name is None:
        name, layer = matches[0]
    else:
        selected = [layer for path, layer in matches if path == name]
        if not selected:
            raise ValueError("requested module is not an exact pinned GDN layer")
        layer = selected[0]
    qkv, z = layer.in_proj_qkv, layer.in_proj_z
    if (
        qkv.shape != (QKV_WIDTH, HIDDEN)
        or z.shape != (Z_WIDTH, HIDDEN)
        or qkv.dtype != torch.float16
        or z.dtype != torch.float16
        or qkv.device != z.device
        or qkv.device.type != "npu"
        or layer.params_dtype != torch.float16
        or layer.compute_dtype != torch.float32
        or layer.tp_size != 4
    ):
        raise RuntimeError("GDN gate requires qualified TP4 FP16 projection geometry")
    return name, layer, module._linear


def memory_preflight(torch_module=torch):
    free, total = torch_module.npu.mem_get_info()
    return dict(
        passed=free >= MEMORY_REQUIRED_BYTES,
        free_bytes=free,
        total_bytes=total,
        required_bytes=MEMORY_REQUIRED_BYTES,
        scope="one private GDN module plus allocator reserve",
    )


def bits(value):
    return value.detach().cpu().contiguous().view(torch.int16).clone()


def outputs_exact(actual, expected):
    return len(actual) == len(expected) and all(
        left.shape == right.shape and left.dtype == right.dtype and torch.equal(bits(left), bits(right))
        for left, right in zip(actual, expected, strict=True)
    )


def input_case(rows, step, *, width=HIDDEN, dtype=torch.float16):
    # CPU-only deterministic construction: no device RNG/global seed changes.
    offsets = torch.arange(rows * width, dtype=torch.float32).reshape(rows, width)
    scales = (0.03125, 1.0, 4.0, 0.00006103515625)
    result = (((offsets * (step + 3)).remainder(257) - 128) * (scales[step] / 128)).to(dtype)
    result[:, 0] = -0.0
    result[:, 1] = 0.0
    result[:, 2] = 1.0009765625
    result[:, 3] = -1.0009765625
    if dtype == torch.float32:
        # Production block_input is FP32; retain non-half-exact inputs so the
        # two original operand casts and one concatenated cast are exercised.
        result[:, 2] = 1.0003
        result[:, 3] = -1.0007
        result[:, 4] = 0.3333333432674408
    return result


def _weight_versions(values):
    # Resident weights can be inference tensors, which have no version counter.
    return tuple(None if torch.is_inference(value) else value._version for value in values)


def prepare(layer, linear, owner, *, format_weight):
    """Publish tensors before launches and retain the frozen original parameters."""
    if owner:
        raise RuntimeError("GDN private owner cannot be replaced")
    owner.update(
        layer=layer,
        linear=linear,
        qkv=layer.in_proj_qkv,
        z=layer.in_proj_z,
        graphs=[],
        tensors=[],
        outputs=[],
        pools=[],
        failed=True,
    )
    owner["weight_versions"] = _weight_versions((owner["qkv"], owner["z"]))
    owner["weight_pointers"] = (owner["qkv"].data_ptr(), owner["z"].data_ptr())
    owner["weight_bits"] = (bits(owner["qkv"]), bits(owner["z"]))
    with torch.inference_mode():
        owner["packed_nd"] = torch.cat((owner["qkv"], owner["z"]), dim=0)
        owner["packed"] = format_weight(owner["packed_nd"])
    owner["failed"] = False


def baseline(owner, inputs):
    linear, dtype = owner["linear"], owner["layer"].compute_dtype
    return linear(inputs, owner["qkv"], dtype), linear(inputs, owner["z"], dtype)


def concatenated(owner, inputs):
    result = owner["linear"](inputs, owner["packed"], owner["layer"].compute_dtype)
    width = owner["qkv"].shape[0]
    # Materialize both splits: unlike the old outputs, raw views have stride 4096.
    return result[:, :width].contiguous(), result[:, width:].contiguous()


def weights_preserved(owner):
    values = (owner["qkv"], owner["z"])
    return (
        _weight_versions(values) == owner["weight_versions"]
        and tuple(value.data_ptr() for value in values) == owner["weight_pointers"]
        and all(torch.equal(bits(value), before) for value, before in zip(values, owner["weight_bits"], strict=True))
    )


def eager_case(owner, inputs, source):
    inputs.copy_(source)
    expected = baseline(owner, inputs)
    owner["outputs"].append(expected)
    repeated = baseline(owner, inputs)
    owner["outputs"].append(repeated)
    actual = concatenated(owner, inputs)
    owner["outputs"].append(actual)
    return dict(
        baseline_stable=outputs_exact(repeated, expected),
        exact=outputs_exact(actual, expected),
        inputs_preserved=torch.equal(bits(inputs), bits(source)),
        contiguous=all(value.is_contiguous() for value in actual),
    )


def _timing(owner, functions, synchronize, *, repeats, replays):
    records = {name: [] for name in functions}
    names = tuple(functions)
    for repeat in range(repeats):
        for name in names[:: 1 if repeat % 2 == 0 else -1]:
            synchronize()
            start = time.perf_counter()
            for _ in range(replays):
                result = functions[name]()
            synchronize()
            records[name].append((time.perf_counter() - start) * 1000 / replays)
            # Keep only the final timed eager outputs; failed frames retain earlier ones.
            if result is not None:
                owner["outputs"].append(result)
    return {name: dict(wall_ms=samples, median_wall_ms=statistics.median(samples)) for name, samples in records.items()}


def _capture_pair(owner, inputs, stream, npu):
    graphs, outputs = {}, {}
    for name, function in (("baseline", baseline), ("concatenated", concatenated)):
        graph = npu.NPUGraph()
        owner["graphs"].append(graph)
        pool = npu.graph_pool_handle()
        owner["pools"].append(pool)
        with npu.graph(graph, pool=pool, stream=stream):
            output = function(owner, inputs)
        owner["outputs"].append(output)
        graphs[name], outputs[name] = graph, output
    return graphs, outputs


def validate(owner, *, rows=ROWS, repeats=3, replays=5, torch_module=torch):
    """Retain failed graphs; reset only owned graphs after every numerical gate passes.

    The caller owns ``owner`` before invoking this function and must pause/drain
    the resident server. Both graph pools and input/output tensors stay alive
    through final synchronization and graph reset. No live graph/observer is used.
    """
    if owner.get("validated") or owner.get("failed"):
        raise RuntimeError("GDN gate is one-shot; failed or validated owner cannot be reused")
    owner["failed"] = True
    npu = torch_module.npu
    device = owner["qkv"].device
    stream = npu.Stream(device=device)
    owner["stream"] = stream
    stream.wait_stream(npu.current_stream(device))
    checks, timings = [], []
    owner["checks"], owner["timings"] = checks, timings
    with torch_module.inference_mode(), npu.stream(stream):
        # Validate all eager cases before recording any graph.
        cases = [(count, dtype) for dtype in INPUT_DTYPES for count in rows]
        for count, dtype in cases:
            inputs = torch_module.empty((count, owner["qkv"].shape[1]), device=device, dtype=dtype)
            owner["tensors"].append(inputs)
            for step in range(INPUT_STEPS):
                source = input_case(count, step, width=inputs.shape[1], dtype=inputs.dtype)
                check = dict(
                    rows=count, input_dtype=str(dtype), step=step, mode="eager", **eager_case(owner, inputs, source)
                )
                checks.append(check)
                if not all(check[key] for key in ("baseline_stable", "exact", "inputs_preserved", "contiguous")):
                    raise RuntimeError("GDN eager numerical gate failed; no capture attempted")
        for (count, dtype), inputs in zip(cases, owner["tensors"], strict=True):
            stream.synchronize()
            graphs, outputs = _capture_pair(owner, inputs, stream, npu)
            for step in range(INPUT_STEPS):
                source = input_case(count, step, width=inputs.shape[1], dtype=inputs.dtype)
                inputs.copy_(source)
                expected = baseline(owner, inputs)
                owner["outputs"].append(expected)
                for graph in graphs.values():
                    graph.replay()
                stream.synchronize()
                check = dict(
                    rows=count,
                    input_dtype=str(dtype),
                    step=step,
                    mode="graph",
                    baseline_stable=outputs_exact(outputs["baseline"], expected),
                    exact=outputs_exact(outputs["concatenated"], expected),
                    inputs_preserved=torch.equal(bits(inputs), bits(source)),
                    contiguous=all(value.is_contiguous() for value in outputs["concatenated"]),
                )
                checks.append(check)
                if not all(check[key] for key in ("baseline_stable", "exact", "inputs_preserved", "contiguous")):
                    raise RuntimeError("GDN captured replay numerical gate failed; retain private graphs")
            functions = {name: graph.replay for name, graph in graphs.items()}
            timings.append(
                dict(
                    rows=count,
                    input_dtype=str(dtype),
                    mode="graph",
                    measurements=_timing(owner, functions, stream.synchronize, repeats=repeats, replays=replays),
                )
            )
            functions = {
                "baseline": partial(baseline, owner, inputs),
                "concatenated": partial(concatenated, owner, inputs),
            }
            timings.append(
                dict(
                    rows=count,
                    input_dtype=str(dtype),
                    mode="eager",
                    measurements=_timing(owner, functions, stream.synchronize, repeats=repeats, replays=replays),
                )
            )
        stream.synchronize()
        if not weights_preserved(owner):
            raise RuntimeError("GDN original weight preservation gate failed")
    npu.synchronize(device)
    for graph in owner["graphs"]:
        graph.reset()
    owner["failed"], owner["validated"] = False, True
    return dict(
        passed=True,
        scope="one private real-weight GDN QKV/Z group; not model parity or serving speed",
        checks=checks,
        timings=timings,
        weights_preserved=True,
        reset_graphs=len(owner["graphs"]),
        packed_weight_bytes=owner["packed"].numel() * owner["packed"].element_size(),
    )
