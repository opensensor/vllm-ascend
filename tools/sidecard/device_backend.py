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


class CudaOracle:
    def __init__(self, banks, specification, routes):
        from tools.glm_perf.fused_weight_layout import unpack_cube

        library_path = Path(specification["library"])
        if digest_file(library_path) != specification["library_sha256"]:
            raise ValueError("CUDA oracle library differs from pinned identity")
        self.cuda = CudaCopies(Path(specification["cuda_runtime"]), 1, 1)
        self.buffers, self.bank_pointers = {}, {}
        self.allocation_budget = specification.get("cuda_memory_budget_bytes")
        self.allocated_bytes = 0
        self.max_routes = routes
        self.library = ctypes.CDLL(str(library_path))
        self.launch = self.library.sidecard_oracle_forward
        self.launch.argtypes = [ctypes.c_void_p] * 13 + [ctypes.c_int] * 4
        self.launch.restype = ctypes.c_int
        try:
            for layer, bank in banks.items():
                gate, gs, down, ds = bank["tensors"]
                tensors = (unpack_cube(gate, 4096), gs, unpack_cube(down, 2048), ds)
                pointers = []
                for index, tensor in enumerate(tensors):
                    data = tensor.numpy().tobytes()
                    pointer = self.allocate(f"layer-{layer}-{index}", len(data))
                    self.upload(pointer, data)
                    pointers.append(pointer)
                self.bank_pointers[layer] = tuple(pointers)
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
        rows = len(experts)
        if not 1 <= rows <= self.max_routes or len(body) != rows * 4096 * 2 or bits not in (4, 8):
            raise ValueError("CUDA raw input differs from admitted route geometry")
        self.upload(self.buffers["input"], body)
        self.upload(self.buffers["ids"], struct.pack(f"<{rows}i", *experts))
        gate, gs, down, ds = self.bank_pointers[layer]
        names = ("input_q", "input_scales", "gate_up", "hidden", "hidden_q", "hidden_scales", "output")
        self.cuda.check(
            self.launch(
                self.buffers["input"],
                self.buffers["ids"],
                gate,
                gs,
                down,
                ds,
                *(self.buffers[name] for name in names),
                rows,
                4096,
                2048,
                bits,
            )
        )
        self.cuda.check(self.cuda.library.cudaDeviceSynchronize())
        output = ctypes.create_string_buffer(rows * 4096 * 2)
        self.cuda.check(self.cuda.library.cudaMemcpy(output, self.buffers["output"], len(output), DEVICE_TO_HOST))
        self.cuda.check(self.cuda.library.cudaDeviceSynchronize())
        return output.raw

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
