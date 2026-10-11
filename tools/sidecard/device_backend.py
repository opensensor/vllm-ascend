# SPDX-License-Identifier: Apache-2.0
"""Bounded real-weight device backends for split-expert correctness gates.

These reject the full GLM seed: at most three layers, sixteen experts per
worker and 256 routes. Full-model memory admission and optimized CUDA MoE are
separate work. Device contexts initialize only during worker preparation.
"""

import ctypes
import json
import struct
import time
from pathlib import Path

from .hardware_probe import DEVICE_TO_HOST, HOST_TO_DEVICE, CudaCopies
from .native_expert_probe import contained, digest_file, select_expert

MAX_GATE_EXPERTS = 16
MAX_GATE_LAYERS = 3
MAX_GATE_ROUTES = 256


def owned_keys(placement, owner):
    return tuple(
        (layer, expert)
        for layer in range(placement.first_moe_layer, placement.first_moe_layer + len(placement.layers))
        for expert, assigned in enumerate(placement.layer(layer))
        if assigned == owner
    )


def prepare_cpu_banks(root, keys, expected_manifest):
    """Authenticate selected native files and copy independent host banks."""
    if digest_file(root / "native-layout.json") != expected_manifest:
        raise ValueError("native worker checkpoint differs from pinned identity")
    plans = [select_expert(root, layer, expert) for layer, expert in keys]
    files = {}
    for plan in plans:
        for filename, digest in plan["shard_sha256"].items():
            if filename in files and files[filename] != digest:
                raise ValueError("native worker shard receipts disagree")
            files[filename] = digest
    snapshots = {}
    for filename, digest in files.items():
        path = contained(root, filename)
        stat = path.stat()
        snapshots[filename] = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if digest_file(path) != digest:
            raise ValueError("native worker shard checksum mismatch")

    # Host Torch is used only for CPU safetensors manipulation here. No NPU
    # device is selected until the Ascend worker adapter prepares its banks.
    import torch
    from safetensors import safe_open

    torch.set_num_threads(1)
    banks = {}
    for plan in plans:
        values = {}
        for name, filename in plan["tensor_files"].items():
            with safe_open(str(contained(root, filename)), framework="pt", device="cpu") as handle:
                values[name.rsplit(".", 1)[1]] = handle.get_tensor(name).clone()
        layer = plan["layer"]
        if layer not in banks:
            banks[layer] = {"experts": [], "parts": [], "plan": plan}
        banks[layer]["experts"].append(plan["expert"])
        banks[layer]["parts"].append(
            (
                torch.cat((values["gate_proj_codes"], values["up_proj_codes"]), 0),
                torch.cat((values["gate_proj_scale"], values["up_proj_scale"]), 0),
                values["down_proj_codes"],
                values["down_proj_scale"],
            )
        )
    for bank in banks.values():
        bank["experts"] = tuple(bank["experts"])
        bank["tensors"] = tuple(torch.stack(parts).contiguous() for parts in zip(*bank.pop("parts"), strict=True))
    for filename, snapshot in snapshots.items():
        stat = contained(root, filename).stat()
        if snapshot != (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns):
            raise ValueError("native worker shard changed while loading")
    if digest_file(root / "native-layout.json") != expected_manifest:
        raise ValueError("native manifest changed while loading")
    return banks


def compact_routes(request, experts):
    """Explicit arbitrary global -> local expert and original -> compact token maps."""
    expert_map = {expert: index for index, expert in enumerate(experts)}
    token_map = {token: index for index, token in enumerate(request.token_ids)}
    try:
        return (
            tuple(token_map[route.token] for route in request.routes),
            tuple(expert_map[route.expert] for route in request.routes),
        )
    except KeyError as error:
        raise ValueError("native gate received a foreign expert or token") from error


CUDA_ROW_MAJOR = "row_major_v1"
CUDA_WORD_PLANES = "k32_words_n_v1"
CUDA_NIBBLE_PLANES = "k32_nibbles_n_v1"
TENSOR_TILE_ROWS = 16
MIN_GROUPED_ROUTES = 32
MAX_TENSOR_TILE_PADDING = 4


def expert_tiles(experts):
    """Bounded contiguous tiles; never mix experts or assume IDs are sorted."""
    tiles = []
    start = 0
    while start < len(experts):
        stop = start + 1
        while stop < min(start + TENSOR_TILE_ROWS, len(experts)) and experts[stop] == experts[start]:
            stop += 1
        tiles.append((start, stop - start))
        start = stop
    return tuple(tiles)


def encode_cuda_weights(values, layout):
    if layout == CUDA_ROW_MAJOR:
        return values.tobytes()
    if (
        layout not in (CUDA_WORD_PLANES, CUDA_NIBBLE_PLANES)
        or values.ndim != 3
        or values.dtype.name != "int8"
        or min(values.shape) < 1
        or values.shape[-1] % 32
        or values.shape[-2] % 32
    ):
        raise ValueError("invalid CUDA weight layout or signed block32 geometry")
    experts, channels, width = values.shape
    planes = values.reshape(experts, channels, width // 32, 8, 4).transpose(0, 2, 3, 1, 4)
    if layout == CUDA_WORD_PLANES:
        return planes.tobytes()
    if values.min() < -8 or values.max() > 7:
        raise ValueError("packed CUDA weights require signed four-bit codes")
    # Keep W2/W3 codes exact too: storage compression, not requantization.
    codes = (planes.astype("int16") + 8).astype("uint16")
    packed = codes[..., 0] | (codes[..., 1] << 4) | (codes[..., 2] << 8) | (codes[..., 3] << 12)
    return packed.astype("<u2").tobytes()


class CudaOracle:
    def __init__(self, banks, specification, routes):
        from tools.glm_perf.fused_weight_layout import unpack_cube

        library_path = Path(specification["library"])
        if digest_file(library_path) != specification["library_sha256"]:
            raise ValueError("CUDA oracle library differs from pinned identity")
        layout = specification.get("weight_layout", CUDA_ROW_MAJOR)
        if layout not in (CUDA_ROW_MAJOR, CUDA_WORD_PLANES, CUDA_NIBBLE_PLANES):
            raise ValueError("unsupported CUDA weight layout")
        experimental = specification.get("experimental_projection", False)
        if type(experimental) is not bool or (experimental and layout != CUDA_NIBBLE_PLANES):
            raise ValueError("experimental CUDA projections require explicit packed selection")
        self.cuda = CudaCopies(Path(specification["cuda_runtime"]), 1, 1)
        self.buffers, self.bank_pointers = {}, {}
        self.allocation_budget = specification.get("cuda_memory_budget_bytes")
        self.allocated_bytes = 0
        self.max_routes = routes
        self.max_tokens = min(routes, 640)
        self.reduce = None
        self.library = ctypes.CDLL(str(library_path))
        self.launch = getattr(
            self.library,
            {
                CUDA_ROW_MAJOR: "sidecard_oracle_forward",
                CUDA_WORD_PLANES: "sidecard_oracle_forward_transposed",
                CUDA_NIBBLE_PLANES: "sidecard_oracle_forward_packed",
            }[layout],
        )
        self.launch.argtypes = [ctypes.c_void_p] * 13 + [ctypes.c_int] * 4
        self.launch.restype = ctypes.c_int
        self.grouped_launch = getattr(self.library, "sidecard_oracle_forward_grouped", None) if experimental else None
        if self.grouped_launch is not None:
            self.grouped_launch.argtypes = [ctypes.c_void_p] * 14 + [ctypes.c_int] * 5
            self.grouped_launch.restype = ctypes.c_int
        self.split_launch = getattr(self.library, "sidecard_oracle_forward_splitk", None) if experimental else None
        if experimental and (self.grouped_launch is None or self.split_launch is None):
            self.cuda.close()
            raise ValueError("experimental CUDA library lacks grouped or split-K projections")
        if self.split_launch is not None:
            self.split_launch.argtypes = [ctypes.c_void_p] * 14 + [ctypes.c_int] * 4
            self.split_launch.restype = ctypes.c_int
        pack = getattr(self.library, "sidecard_oracle_pack_cube", None) if layout == CUDA_NIBBLE_PLANES else None
        staging_size = 0
        prepared_at = time.perf_counter()
        if pack is not None:
            pack.argtypes, pack.restype = [ctypes.c_void_p] * 2 + [ctypes.c_int] * 4, ctypes.c_int
        try:
            if specification.get("partial_reduce", False):
                self.reduce = self.library.sidecard_oracle_reduce_local
                self.reduce.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3
                self.reduce.restype = ctypes.c_int
            for layer, bank in banks.items():
                if "encoded" in bank:
                    tensors = bank["encoded"]
                else:
                    gate, gs, down, ds = bank["tensors"]
                    tensors = (
                        (gate, gs, down, ds)
                        if pack is not None
                        else (unpack_cube(gate, 4096), gs, unpack_cube(down, 2048), ds)
                    )
                pointers = []
                for index, tensor in enumerate(tensors):
                    if pack is not None and "encoded" not in bank and index in (0, 2):
                        experts, channels, packed_width = tensor.shape
                        width = 4096 if index == 0 else 2048
                        bits = packed_width * 8 // width
                        data = tensor.numpy().tobytes()
                        if len(data) > staging_size:
                            if staging_size:
                                self.cuda.check(self.cuda.library.cudaFree(self.buffers["cube_staging"]))
                                self.buffers["cube_staging"].value = None
                                self.allocated_bytes -= staging_size
                            self.allocate("cube_staging", len(data))
                            staging_size = len(data)
                        self.upload(self.buffers["cube_staging"], data)
                        pointer = self.allocate(f"layer-{layer}-{index}", experts * channels * width // 2)
                        self.cuda.check(pack(self.buffers["cube_staging"], pointer, experts, channels, width, bits))
                        self.cuda.check(self.cuda.library.cudaDeviceSynchronize())
                        pointers.append(pointer)
                        continue
                    data = (
                        tensor
                        if "encoded" in bank
                        else encode_cuda_weights(tensor.numpy(), layout)
                        if index in (0, 2)
                        else tensor.numpy().tobytes()
                    )
                    pointer = self.allocate(f"layer-{layer}-{index}", len(data))
                    self.upload(pointer, data)
                    pointers.append(pointer)
                self.bank_pointers[layer] = tuple(pointers)
                print(
                    json.dumps(
                        {
                            "kind": "cuda_layer_loaded",
                            "layer": layer,
                            "cuda_allocated_bytes": self.allocated_bytes,
                            "native_cube_prepare": pack is not None,
                            "elapsed_seconds": time.perf_counter() - prepared_at,
                        }
                    ),
                    flush=True,
                )
            if staging_size:
                self.cuda.check(self.cuda.library.cudaFree(self.buffers["cube_staging"]))
                self.buffers["cube_staging"].value = None
                self.allocated_bytes -= staging_size
            sizes = {
                "input": routes * 4096 * 2,
                "ids": routes * 4,
                "input_q": routes * 4096,
                "input_scales": routes * 128 * 4,
                "gate_up": routes * 4096 * 2,
                "hidden": routes * 2048 * 2,
                "hidden_q": routes * 2048,
                "hidden_scales": routes * 64 * 4,
                "output": routes * 4096 * 2,
            }
            if self.grouped_launch is not None:
                sizes["expert_tiles"] = routes * 8
            if self.split_launch is not None:
                sizes["split_groups"] = min(routes, MIN_GROUPED_ROUTES - 1) * 4096 * 128 * 4
            if self.reduce is not None:
                sizes.update(
                    row_map=self.max_tokens * 8 * 4,
                    weights=self.max_tokens * 8 * 4,
                    partial_output=self.max_tokens * 4096 * 4,
                )
            for name, size in sizes.items():
                self.allocate(name, size)
            self.cuda.check(self.cuda.library.cudaDeviceSynchronize())
        except BaseException:
            self.close()
            raise

    def allocate(self, name, size):
        if self.allocation_budget is not None and self.allocated_bytes + size > self.allocation_budget:
            raise ValueError("CUDA resident allocations exceed the admitted byte budget")
        pointer = ctypes.c_void_p()
        self.cuda.check(self.cuda.library.cudaMalloc(ctypes.byref(pointer), size))
        self.buffers[name] = pointer
        self.allocated_bytes += size
        return pointer

    def upload(self, pointer, body):
        storage = ctypes.create_string_buffer(body, len(body))
        self.cuda.check(self.cuda.library.cudaMemcpy(pointer, storage, len(body), HOST_TO_DEVICE))
        self.cuda.check(self.cuda.library.cudaDeviceSynchronize())

    def compute(self, layer, hidden, experts, bits):
        rows = len(hidden)
        body = b"".join(struct.pack("<4096e", *row) for row in hidden)
        output = self.compute_bytes(layer, body, experts, bits)
        return tuple(struct.unpack_from("<4096e", output, row * 4096 * 2) for row in range(rows))

    def compute_bytes(self, layer, body, experts, bits):
        self.project_bytes(layer, body, experts, bits)
        return self.read_output("output", len(experts) * 4096 * 2)

    def project_bytes(self, layer, body, experts, bits):
        rows = len(experts)
        if not 1 <= rows <= self.max_routes or len(body) != rows * 4096 * 2 or bits not in (4, 8):
            raise ValueError("CUDA raw input differs from admitted route geometry")
        self.upload(self.buffers["input"], body)
        self.upload(self.buffers["ids"], struct.pack(f"<{rows}i", *experts))
        gate, gs, down, ds = self.bank_pointers[layer]
        names = ("input_q", "input_scales", "gate_up", "hidden", "hidden_q", "hidden_scales", "output")
        arguments = (
            self.buffers["input"],
            self.buffers["ids"],
            gate,
            gs,
            down,
            ds,
            *(self.buffers[name] for name in names),
        )
        if self.split_launch is not None and rows < MIN_GROUPED_ROUTES:
            self.cuda.check(self.split_launch(*arguments, self.buffers["split_groups"], rows, 4096, 2048, bits))
            return
        if self.grouped_launch is not None and rows >= MIN_GROUPED_ROUTES:
            tiles = expert_tiles(experts)
            if len(tiles) * TENSOR_TILE_ROWS <= rows * MAX_TENSOR_TILE_PADDING:
                self.upload(self.buffers["expert_tiles"], b"".join(struct.pack("<2i", *tile) for tile in tiles))
                self.cuda.check(
                    self.grouped_launch(*arguments, self.buffers["expert_tiles"], rows, 4096, 2048, bits, len(tiles))
                )
                return
        self.cuda.check(
            self.launch(
                *arguments,
                rows,
                4096,
                2048,
                bits,
            )
        )

    def read_output(self, name, size):
        self.cuda.check(self.cuda.library.cudaDeviceSynchronize())
        output = ctypes.create_string_buffer(size)
        self.cuda.check(self.cuda.library.cudaMemcpy(output, self.buffers[name], len(output), DEVICE_TO_HOST))
        self.cuda.check(self.cuda.library.cudaDeviceSynchronize())
        return output.raw

    def compute_partial_bytes(self, layer, body, experts, bits, row_map, weights, token_rows):
        if (
            self.reduce is None
            or not 1 <= token_rows <= self.max_tokens
            or len(row_map) != token_rows * 8 * 4
            or len(weights) != token_rows * 8 * 4
        ):
            raise ValueError("CUDA partial combination differs from admitted geometry")
        self.upload(self.buffers["row_map"], row_map)
        self.upload(self.buffers["weights"], weights)
        self.project_bytes(layer, body, experts, bits)
        self.cuda.check(
            self.reduce(
                self.buffers["output"],
                self.buffers["row_map"],
                self.buffers["weights"],
                self.buffers["partial_output"],
                token_rows,
                4096,
                len(experts),
            )
        )
        return self.read_output("partial_output", token_rows * 4096 * 4)

    def close(self):
        self.cuda.check(self.cuda.library.cudaDeviceSynchronize())
        for pointer in self.buffers.values():
            if pointer.value:
                self.cuda.check(self.cuda.library.cudaFree(pointer))
                pointer.value = None
        self.cuda.close()


class NativeGateBackend:
    def __init__(self, kind, specification, owner, placement, identity, limits):
        fields = (
            {"checkpoint_root", "device_index"}
            if kind == "ascend_native_gate"
            else {"checkpoint_root", "library", "library_sha256", "cuda_runtime"}
        )
        if set(specification) != fields or limits.hidden_width != 4096 or limits.max_routes > MAX_GATE_ROUTES:
            raise ValueError("invalid or unbounded native gate specification")
        if kind == "ascend_native_gate" and (
            type(specification["device_index"]) is not int
            or not 0 <= specification["device_index"] < 4
            or owner != f"npu:{specification['device_index']}"
        ):
            raise ValueError("native gate device and owner differ")
        if kind == "cuda_oracle_gate" and owner != "cuda:0":
            raise ValueError("CUDA oracle requires owner cuda:0")
        keys = owned_keys(placement, owner)
        if len({layer for layer, _ in keys}) > MAX_GATE_LAYERS or any(
            sum(key_layer == layer for key_layer, _ in keys) > MAX_GATE_EXPERTS for layer, _ in keys
        ):
            raise ValueError("native gate bank exceeds the bounded correctness scope")
        self.kind, self.specification, self.owner = kind, dict(specification), owner
        self.placement, self.identity, self.limits, self.keys = placement, identity, limits, keys
        self.banks, self.native, self.cuda = {}, None, None
        self.metrics = {"owner": owner, "requests": 0, "routes": 0, "compute_total_ms": 0.0, "compute_max_ms": 0.0}

    def prepare(self):
        self.close()
        root = Path(self.specification["checkpoint_root"])
        self.banks = prepare_cpu_banks(root, self.keys, self.identity.checkpoint_sha256)
        if self.kind == "cuda_oracle_gate":
            self.cuda = CudaOracle(self.banks, self.specification, self.limits.max_routes)
        else:
            import torch
            import torch_npu

            from tools.glm_perf.fused_moe_profile import frozen_helper

            torch.npu.set_device(self.specification["device_index"])
            torch_npu.npu.set_compile_mode(jit_compile=False)
            manifest = json.loads((root / "native-layout.json").read_bytes())
            bundle = contained(root, manifest["kernel_bundle"])
            if digest_file(bundle / "provenance.json") != manifest["kernel_provenance_sha256"]:
                raise ValueError("native gate kernel provenance differs from checkpoint")
            helper, options = frozen_helper(bundle)
            self.native = helper.NativeFusedMoE(
                bundle,
                namespace=options["namespace"],
                activation_bits=manifest["activation_bits"],
                prepared_weight_layout=True,
                **({"fp16_route_workspace": True} if options.get("fp16_route_workspace") else {}),
            )
            if self.native.weight_scale_dtype != torch.float16:
                raise ValueError("native gate kernel requires matching FP16 scale storage")
            for bank in self.banks.values():
                bank["device_tensors"] = tuple(value.to(self.native.device).contiguous() for value in bank["tensors"])
            torch.npu.synchronize()
        return self.keys

    def compute(self, request):
        started = time.perf_counter()
        result = self._compute(request)
        elapsed = (time.perf_counter() - started) * 1000
        self.metrics["requests"] += 1
        self.metrics["routes"] += len(request.routes)
        self.metrics["compute_total_ms"] += elapsed
        self.metrics["compute_max_ms"] = max(self.metrics["compute_max_ms"], elapsed)
        return result

    def _compute(self, request):
        if request.layer_id not in self.banks or (self.native is None and self.cuda is None):
            raise RuntimeError("native gate is not prepared for this layer")
        bank = self.banks[request.layer_id]
        tokens, experts = compact_routes(request, bank["experts"])
        if self.cuda is not None:
            hidden = tuple(request.hidden[index] for index in tokens)
            return self.cuda.compute(request.layer_id, hidden, experts, bank["plan"]["activation_bits"])
        import torch

        hidden = torch.tensor(request.hidden, dtype=torch.float16).to(self.native.device)
        gather = torch.tensor(tokens, dtype=torch.int32).to(self.native.device)
        routed = hidden.index_select(0, gather).contiguous()
        ids = torch.tensor(experts, dtype=torch.int32).reshape(-1, 1).to(self.native.device)
        weights = torch.ones(len(tokens), 1, dtype=torch.float32, device=self.native.device)
        output = self.native(routed, *bank["device_tensors"], weights, ids, 0)
        torch.npu.synchronize()
        return tuple(tuple(row) for row in output.cpu().tolist())

    def close(self):
        if self.native is not None:
            import torch

            torch.npu.synchronize()
        if self.cuda is not None:
            self.cuda.close()
            self.cuda = None
        self.banks.clear()
        self.native = None
