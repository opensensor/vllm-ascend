# SPDX-License-Identifier: Apache-2.0
"""Owner-filtered dense and native expert loading for the layer actors."""

import ctypes
import json
from contextlib import ExitStack
from pathlib import Path

from .checkpoint import read_header
from .device_backend import CUDA_COMPACT_PLANES, CudaOracle
from .native_expert_probe import contained, digest_file
from .resident import NativeResidentPlan

MAX_CHUNK = 32
TOP_K = 8
WIDTH = 4096
GPU_LAYERS = tuple(range(18)) + tuple(range(33, 45))
NPU_LAYERS = tuple(range(18, 33))


def private_tensor(handle, descriptor, payload_offset, torch):
    """Read file bytes into anonymous writable storage, never mmap into CANN."""
    start, end = descriptor["data_offsets"]
    handle.seek(payload_offset + start)
    raw = bytearray(handle.read(end - start))
    if len(raw) != end - start:
        raise ValueError("truncated private tensor payload")
    dtype = {"F16": torch.float16, "F32": torch.float32, "I8": torch.int8, "U8": torch.uint8}[descriptor["dtype"]]
    return torch.frombuffer(raw, dtype=dtype).reshape(descriptor["shape"]).clone()


def private_native_banks(plan, torch):
    if not plan.authenticated:
        raise RuntimeError("native payload must be authenticated before private staging")
    for layer, experts in plan.by_layer.items():
        parts = []
        with ExitStack() as stack:
            files = {}
            for expert in experts:
                values = {}
                for projection in ("gate", "up", "down"):
                    for kind in ("codes", "scale"):
                        short = f"{projection}_proj_{kind}"
                        name = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{short}"
                        filename = plan.index[name]
                        if filename not in files:
                            path = contained(plan.root, filename)
                            header, receipt = read_header(path)
                            files[filename] = (stack.enter_context(path.open("rb")), header, receipt["payload_offset"])
                        handle, header, offset = files[filename]
                        values[short] = private_tensor(handle, header[name], offset, torch)
                parts.append(
                    (
                        torch.cat((values["gate_proj_codes"], values["up_proj_codes"]), 0),
                        torch.cat((values["gate_proj_scale"], values["up_proj_scale"]), 0),
                        values["down_proj_codes"],
                        values["down_proj_scale"],
                    )
                )
        tensors = tuple(torch.stack(rows).contiguous() for rows in zip(*parts, strict=True))
        plan.check_unchanged()
        yield layer, {"tensors": tensors}
        del tensors, parts, values


def shard_dense_tensor(name, value, config, rank):
    """Select complete head/intermediate shards while still in CPU staging."""
    if type(rank) is not int or rank not in (0, 1):
        raise ValueError("middle stage requires exactly two tensor parallel ranks")
    column_projections = (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.b_proj.weight",
        "self_attn.f_b_proj.weight",
        "self_attn.g_b_proj.weight",
        "self_attn.q_b_proj.weight",
        "self_attn.kv_b_proj.weight",
        "self_attn.q_conv1d.weight",
        "self_attn.k_conv1d.weight",
        "self_attn.v_conv1d.weight",
        "self_attn.A_log",
        "self_attn.dt_bias",
        "mlp.shared_experts.gate_proj.weight",
        "mlp.shared_experts.up_proj.weight",
    )
    row_projections = ("self_attn.o_proj.weight", "mlp.shared_experts.down_proj.weight")
    dimension = 0 if name in column_projections else 1 if name in row_projections else None
    if dimension is None:
        return value
    if value.shape[dimension] % 2:
        raise ValueError("weight cannot be split into equal complete TP2 shards")
    return value.chunk(2, dimension)[rank].contiguous()


def load_dense(root, layers, device, torch, *, include_globals=False, tp_rank=None):
    from safetensors import safe_open

    root = Path(root)
    config = json.loads((root / "config.json").read_bytes())["text_config"]
    index = json.loads((root / "model.safetensors.index.json").read_bytes())["weight_map"]
    prefixes = {f"model.language_model.layers.{index}.": index for index in layers}
    globals_map = {
        "model.language_model.embed_tokens.weight": "embedding",
        "model.language_model.norm.weight": "norm",
        "lm_head.weight": "head",
    }
    manifest = json.loads((root / "native-layout.json").read_bytes())
    authenticated = {record["file"]: record["sha256"] for record in manifest["dense_shards"]}
    result, global_weights = {index: {} for index in layers}, {}
    # Dense shards contain only ~17 GiB. Keep maps open, never retain unowned
    # tensors or convert the complete model before choosing ownership.
    with ExitStack() as stack:
        handles = {}
        for name, filename in index.items():
            if ".experts." in name:
                continue
            owner = next(((prefix, layer) for prefix, layer in prefixes.items() if name.startswith(prefix)), None)
            if owner is None and not (include_globals and name in globals_map):
                continue
            if filename not in handles:
                path = contained(root, filename)
                if filename not in authenticated or digest_file(path) != authenticated[filename]:
                    raise ValueError("owner-selected dense shard failed payload authentication")
                if device == "cuda":
                    handles[filename] = stack.enter_context(safe_open(str(path), framework="pt", device="cpu"))
                else:
                    header, receipt = read_header(path)
                    handles[filename] = (stack.enter_context(path.open("rb")), header, receipt["payload_offset"])
            # The 310P loader qualification requires private CPU staging;
            # uploading an mmap alias can corrupt file-cache pages.
            if device == "cuda":
                value = handles[filename].get_tensor(name).clone()
            else:
                handle, header, offset = handles[filename]
                value = private_tensor(handle, header[name], offset, torch)
            if tp_rank is not None and owner is not None:
                value = shard_dense_tensor(name[len(owner[0]) :], value, config, tp_rank)
            value = value.to(device=device, dtype=torch.float16).contiguous()
            if owner is None:
                global_weights[globals_map[name]] = value
            else:
                prefix, layer = owner
                result[layer][name[len(prefix) :]] = value
    return result, global_weights


class PlanBanks:
    def __init__(self, plan):
        self.plan = plan

    def items(self):
        yield from self.plan.items()


class CudaExperts:
    def __init__(self, root, checkpoint_sha256, specification, torch, *, keys=None):
        self.torch = torch
        plan = NativeResidentPlan(
            root,
            tuple((layer, expert) for layer in GPU_LAYERS if layer >= 3 for expert in range(288))
            if keys is None
            else keys,
            checkpoint_sha256,
        )
        plan.authenticate()
        spec = dict(specification, weight_layout=CUDA_COMPACT_PLANES, partial_reduce=True, tp_rank_inputs=False)
        self.oracle = CudaOracle(PlanBanks(plan), spec, MAX_CHUNK * TOP_K)
        self.configure_stream(specification.get("torch_stream_experts", False))

    def configure_stream(self, enabled):
        """Bind explicitly stream-aware kernels without changing legacy APIs."""
        self.use_stream = enabled
        if not enabled:
            return
        library = self.oracle.library
        names = (
            "sidecard_oracle_forward_packed_stream",
            "sidecard_oracle_forward_compact_stream",
            "sidecard_oracle_reduce_local_stream",
        )
        if not all(hasattr(library, name) for name in names):
            raise ValueError("CUDA expert library lacks the qualified Torch-stream ABI")
        self.stream_packed, self.stream_compact, self.stream_reduce = [getattr(library, name) for name in names]
        for function, integers in ((self.stream_packed, 4), (self.stream_compact, 5), (self.stream_reduce, 3)):
            pointers = 4 if function is self.stream_reduce else 13
            function.argtypes = [ctypes.c_void_p] * pointers + [ctypes.c_int] * integers + [ctypes.c_void_p]
            function.restype = ctypes.c_int
        self.oracle.cuda.library.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.oracle.cuda.library.cudaMemcpyAsync.restype = ctypes.c_int

    def __call__(self, layer, x, ids, weights):
        t, oracle = self.torch, self.oracle
        rows = x.shape[0] * TOP_K
        if not 1 <= x.shape[0] <= MAX_CHUNK:
            raise ValueError("CUDA expert chunk exceeds admitted scratch")
        expanded = x[:, None, :].expand(-1, TOP_K, -1).reshape(rows, WIDTH).contiguous()
        ids = ids.to(t.int32).contiguous().flatten()
        row_map = t.arange(rows, device="cuda", dtype=t.int32).reshape(-1, TOP_K)
        weights = weights.float().contiguous()
        runtime, check = oracle.cuda.library, oracle.cuda.check
        use_stream = getattr(self, "use_stream", False)
        stream = ctypes.c_void_p(t.cuda.current_stream().cuda_stream) if use_stream else None
        if not use_stream:
            t.cuda.synchronize()
        copy = runtime.cudaMemcpyAsync if use_stream else runtime.cudaMemcpy
        for name, value in (("input", expanded), ("ids", ids), ("row_map", row_map), ("weights", weights)):
            check(
                copy(
                    oracle.buffers[name],
                    ctypes.c_void_p(value.data_ptr()),
                    value.numel() * value.element_size(),
                    3,
                    *([stream] if use_stream else []),
                )
            )
        pointers = oracle.bank_pointers[layer]
        names = ("input_q", "input_scales", "gate_up", "hidden", "hidden_q", "hidden_scales", "output")
        arguments = (
            oracle.buffers["input"],
            oracle.buffers["ids"],
            *pointers,
            *(oracle.buffers[name] for name in names),
            rows,
            WIDTH,
            2048,
            4,
        )
        if use_stream:
            bits = oracle.weight_bits[pointers[0].value]
            launch = self.stream_packed if bits == 4 else self.stream_compact
            check(launch(*arguments, *([] if bits == 4 else [bits]), stream))
        else:
            check(oracle.launch(*arguments))
        check(
            (self.stream_reduce if use_stream else oracle.reduce)(
                oracle.buffers["output"],
                oracle.buffers["row_map"],
                oracle.buffers["weights"],
                oracle.buffers["partial_output"],
                x.shape[0],
                WIDTH,
                rows,
                *([stream] if use_stream else []),
            )
        )
        output = t.empty((x.shape[0], WIDTH), dtype=t.float32, device="cuda")
        check(
            copy(
                ctypes.c_void_p(output.data_ptr()),
                oracle.buffers["partial_output"],
                output.numel() * output.element_size(),
                3,
                *([stream] if use_stream else []),
            )
        )
        if not use_stream:
            check(runtime.cudaDeviceSynchronize())
        return output

    def close(self):
        self.oracle.close()


class NpuExperts:
    def __init__(
        self,
        root,
        checkpoint_sha256,
        rank,
        torch,
        *,
        gate_override,
        reduce_results=True,
        stable_reduction=True,
        layers=NPU_LAYERS,
    ):
        from tools.glm_perf.fused_moe_profile import frozen_helper

        self.torch, self.rank = torch, rank
        self.reduce_results = reduce_results
        if (
            not layers
            or len(set(layers)) != len(layers)
            or any(type(layer) is not int or layer not in NPU_LAYERS for layer in layers)
        ):
            raise ValueError("cold banks require unique supported middle layers")
        plan = NativeResidentPlan(
            root,
            tuple((layer, expert) for layer in layers for expert in range(rank * 144, (rank + 1) * 144)),
            checkpoint_sha256,
        )
        plan.authenticate()
        bundle = contained(Path(root), plan.manifest["kernel_bundle"])
        if digest_file(bundle / "provenance.json") != plan.manifest["kernel_provenance_sha256"]:
            raise ValueError("native layer kernel provenance differs")
        helper, options = frozen_helper(bundle)
        override_path = Path(gate_override)
        override = json.loads(override_path.read_bytes())
        if (
            override["swiglu_limit"] != 10.0
            or override["original_provenance_sha256"] != plan.manifest["kernel_provenance_sha256"]
            or set(override["native_gate_files"]) != {"glm_fused_gate_up.bin", "glm_fused_gate_up_w4.bin"}
        ):
            raise ValueError("routed activation override differs from GLM's frozen kernel contract")
        for name, checksum in override["native_gate_files"].items():
            if digest_file(override_path.parent / name) != checksum:
                raise ValueError("clipped routed kernel failed authentication")

        def factory(path, entry):
            filename = Path(path).name
            if filename in override["native_gate_files"]:
                path = str(override_path.parent / filename)
            return getattr(torch.classes, options["namespace"]).Kernel(path, entry)

        self.stable_rows = helper.FUSED_REDUCTION_TOKENS + 1 if stable_reduction else 1
        self.native = helper.NativeFusedMoE(
            bundle,
            namespace=options["namespace"],
            activation_bits=4,
            kernel_factory=factory,
            prepared_weight_layout=True,
            **({"fp16_route_workspace": True} if options.get("fp16_route_workspace") else {}),
        )
        self.banks = {}
        for layer, bank in private_native_banks(plan, torch):
            self.banks[layer] = tuple(value.to(f"npu:{rank}") for value in bank["tensors"])
            print(json.dumps(dict(kind="npu_layer_loaded", rank=rank, layer=layer)), flush=True)
        plan.authenticate()

    def __call__(self, layer, x, ids, weights):
        t = self.torch
        original_rows = x.shape[0]
        if original_rows < self.stable_rows:
            # The small decode path adds expert outputs atomically. Select the
            # bundle's fixed-slot reducer with zero-weight padding instead;
            # padding contributes no routes and changes no activation scales.
            padding = self.stable_rows - original_rows
            x = t.cat((x, t.zeros((padding, WIDTH), dtype=x.dtype, device=x.device)))
            ids = t.cat((ids, t.full((padding, TOP_K), self.rank * 144, dtype=ids.dtype, device=ids.device)))
            weights = t.cat((weights, t.zeros((padding, TOP_K), dtype=weights.dtype, device=weights.device)))
        owned = (ids >= self.rank * 144) & (ids < (self.rank + 1) * 144)
        local_ids = t.where(owned, ids, self.rank * 144).to(t.int32)
        local_weights = t.where(owned, weights, 0).float()
        output = self.native(x, *self.banks[layer], local_weights, local_ids, self.rank * 144)
        if self.reduce_results:
            t.distributed.all_reduce(output)
        return output[:original_rows]

    def validate_repeat(self, output, *, combine_results=True):
        t = self.torch
        cases = []
        for layer in self.banks:
            for offset in (0, 144):
                t.manual_seed(20261014 + layer)
                hidden = t.randn(1, WIDTH).half().to(f"npu:{self.rank}")
                ids = (t.arange(TOP_K, dtype=t.int64) * 17 + offset).reshape(1, TOP_K).to(hidden.device)
                weights = t.full((1, TOP_K), 2.5 / TOP_K, device=hidden.device)
                golden = None
                equal = True
                for _ in range(4):
                    actual = self(layer, hidden, ids, weights)
                    if not self.reduce_results and combine_results:
                        t.distributed.all_reduce(actual)
                    actual = actual.cpu()
                    equal = equal and bool(t.isfinite(actual).all())
                    if golden is None:
                        golden = actual
                    else:
                        equal = equal and t.equal(golden, actual)
                cases.append(dict(layer=layer, all_routes_on_rank=offset // 144, bitwise_repeat_equal=equal))
        report = dict(
            passed=all(case["bitwise_repeat_equal"] for case in cases),
            rank=self.rank,
            zero_weight_padding_rows=self.stable_rows,
            cases=cases,
        )
        output.write_text(json.dumps(report, indent=2) + "\n")
        if not report["passed"]:
            raise RuntimeError("native complete-bank repeated expert output is nondeterministic")
