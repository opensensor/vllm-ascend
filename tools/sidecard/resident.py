# SPDX-License-Identifier: Apache-2.0
"""Stream authenticated native expert banks into an admitted CUDA partition.

The first service uses the qualified Ascend backbone and expert parallel
layout. CUDA replaces rank zero's routed experts; it does not own attention.
Only one layer of packed CPU weights is staged at a time.
"""

import ctypes
import json
import time
from contextlib import ExitStack
from pathlib import Path

from .checkpoint import read_header
from .device_backend import CUDA_NIBBLE_PLANES, CUDA_ROW_MAJOR, CUDA_WORD_PLANES, CudaOracle, compact_routes, owned_keys
from .native_expert_probe import NATIVE_LAYOUT, SCALE_LAYOUT, contained, digest_file
from .placement import validate_glm_config
from .reference import ExpertReply
from .tensor_wire import TensorPacket

WIDTH = 4096
INTERMEDIATE = 2048
MAIN_LAYERS = range(3, 45)
MAX_RESIDENT_EXPERTS = 72
MAX_PACKED_RESIDENT_EXPERTS = 144
MAX_RESIDENT_ROUTES = 5120
MAX_RESIDENT_TOKENS = 640
CUDA_RESERVE_BYTES = 4 * 1024**3
SCRATCH_ALLOWANCE_BYTES = 256 * 1024**2
PACKED_STAGING_ALLOWANCE_BYTES = MAX_PACKED_RESIDENT_EXPERTS * 2 * INTERMEDIATE * WIDTH // 2
EXPERT_UNPACKED_BYTES = 3 * WIDTH * INTERMEDIATE
EXPERT_SCALE_BYTES = 3 * (WIDTH // 32) * (INTERMEDIATE // 32) * 2
RESIDENT_PREPARE_THREADS = 8


def snapshot(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


class NativeResidentPlan:
    """Read metadata once, validate every selected tensor, then hash each shard."""

    def __init__(self, root, keys, checkpoint_sha256):
        self.root = Path(root).resolve()
        self.keys = tuple(keys)
        if (
            not self.keys
            or len(set(self.keys)) != len(self.keys)
            or any(
                layer not in MAIN_LAYERS or type(expert) is not int or not 0 <= expert < 288
                for layer, expert in self.keys
            )
        ):
            raise ValueError("invalid resident expert keys")
        manifest_path = self.root / "native-layout.json"
        if digest_file(manifest_path) != checkpoint_sha256:
            raise ValueError("resident manifest differs from pinned checkpoint")
        config = json.loads((self.root / "config.json").read_bytes())
        validate_glm_config(config)
        self.manifest = json.loads(manifest_path.read_bytes())
        if (
            self.manifest.get("complete") is not True
            or self.manifest.get("schema_version") != 1
            or self.manifest.get("layout") != NATIVE_LAYOUT
            or self.manifest.get("weight_scale_layout") != SCALE_LAYOUT
            or config.get("ascend_glm_expert_layout") != NATIVE_LAYOUT
            or config.get("ascend_glm_expert_scale_layout") != SCALE_LAYOUT
            or self.manifest.get("activation_bits") not in (4, 8)
        ):
            raise ValueError("resident loader requires native Cube bytes and FP16 scales")
        self.index = json.loads((self.root / "model.safetensors.index.json").read_bytes())["weight_map"]
        self.metadata_digests = {
            name: digest_file(self.root / name)
            for name in ("native-layout.json", "config.json", "model.safetensors.index.json")
        }
        code_records = {record["file"]: record for record in self.manifest["native_shards"]}
        scale_records = {record["file"]: record for record in self.manifest["scale_shards"]}
        self.files, self.headers, self.by_layer = {}, {}, {}
        for layer, expert in self.keys:
            bits = self.manifest["layer_bits"].get(str(layer))
            if bits not in (2, 3, 4):
                raise ValueError("unsupported resident weight precision")
            self.by_layer.setdefault(layer, []).append(expert)
            for projection in ("gate", "up", "down"):
                rows, columns = (WIDTH, INTERMEDIATE) if projection == "down" else (INTERMEDIATE, WIDTH)
                for kind in ("codes", "scale"):
                    name = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}_proj_{kind}"
                    filename = self.index[name]
                    records = code_records if kind == "codes" else scale_records
                    if filename not in records:
                        raise ValueError("resident tensor lacks an authenticated native receipt")
                    record = records[filename]
                    digest = record["sha256"]
                    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                        raise ValueError("invalid resident shard digest")
                    if filename in self.files and self.files[filename] != digest:
                        raise ValueError("resident shard receipts disagree")
                    self.files[filename] = digest
                    if filename not in self.headers:
                        self.headers[filename] = read_header(contained(self.root, filename))[0]
                    header = self.headers[filename]
                    metadata = header.get("__metadata__", {})
                    if metadata.get("layout") != NATIVE_LAYOUT or metadata.get("weight_scale_layout") != SCALE_LAYOUT:
                        raise ValueError("resident shard has incompatible layout metadata")
                    descriptor = header[name]
                    shape = (rows, columns * bits // 8) if kind == "codes" else (rows // 32, columns // 32)
                    dtypes = ("I8", "U8") if kind == "codes" else ("F16",)
                    if tuple(descriptor["shape"]) != shape or descriptor["dtype"] not in dtypes:
                        raise ValueError("resident tensor has incompatible geometry or dtype")
        self.by_layer = {layer: tuple(experts) for layer, experts in sorted(self.by_layer.items())}
        self.snapshots = {name: snapshot(contained(self.root, name)) for name in self.files}
        self.required_cuda_bytes = len(self.keys) * (EXPERT_UNPACKED_BYTES + EXPERT_SCALE_BYTES)
        self.authenticated = False

    def authenticate(self):
        for filename, digest in self.files.items():
            if digest_file(contained(self.root, filename)) != digest:
                raise ValueError("resident shard checksum mismatch")
        self.check_unchanged()
        self.authenticated = True

    def check_unchanged(self):
        if any(snapshot(contained(self.root, name)) != state for name, state in self.snapshots.items()):
            raise ValueError("resident shard changed during loading")
        if any(digest_file(self.root / name) != digest for name, digest in self.metadata_digests.items()):
            raise ValueError("resident checkpoint metadata changed during loading")

    def items(self):
        if not self.authenticated:
            raise RuntimeError("resident payload must be authenticated before loading")
        # CPU-only imports in the isolated CUDA process. No torch_npu context.
        import torch
        from safetensors import safe_open

        torch.set_num_threads(RESIDENT_PREPARE_THREADS)
        for layer, experts in self.by_layer.items():
            parts = []
            # Large scale-shard headers contain thousands of tensors. Parse
            # each file once per layer rather than once per expert tensor.
            with ExitStack() as stack:
                handles = {}
                for expert in experts:
                    values = {}
                    for projection in ("gate", "up", "down"):
                        for kind in ("codes", "scale"):
                            short = f"{projection}_proj_{kind}"
                            name = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{short}"
                            filename = self.index[name]
                            if filename not in handles:
                                handles[filename] = stack.enter_context(
                                    safe_open(str(contained(self.root, filename)), framework="pt", device="cpu")
                                )
                            values[short] = handles[filename].get_tensor(name).clone()
                    parts.append(
                        (
                            torch.cat((values["gate_proj_codes"], values["up_proj_codes"]), 0),
                            torch.cat((values["gate_proj_scale"], values["up_proj_scale"]), 0),
                            values["down_proj_codes"],
                            values["down_proj_scale"],
                        )
                    )
            tensors = tuple(torch.stack(rows).contiguous() for rows in zip(*parts, strict=True))
            del parts, values
            self.check_unchanged()
            yield layer, {"tensors": tensors}
            del tensors


class CudaResidentBackend:
    def __init__(self, specification, owner, placement, identity, limits):
        fields = {"checkpoint_root", "library", "library_sha256", "cuda_runtime", "cuda_memory_budget_bytes"}
        if (
            not fields
            <= set(specification)
            <= fields | {"weight_layout", "cache_root", "cache_manifest_sha256", "reply_fp16", "partial_reduce"}
            or specification.get("weight_layout", CUDA_ROW_MAJOR)
            not in (CUDA_ROW_MAJOR, CUDA_WORD_PLANES, CUDA_NIBBLE_PLANES)
            or ("cache_root" in specification) != ("cache_manifest_sha256" in specification)
            or type(specification.get("reply_fp16", False)) is not bool
            or type(specification.get("partial_reduce", False)) is not bool
            or (specification.get("partial_reduce", False) and specification.get("reply_fp16", False))
            or owner != "cuda:0"
            or limits.hidden_width != WIDTH
            or limits.max_routes > MAX_RESIDENT_ROUTES
            or limits.max_tokens > MAX_RESIDENT_TOKENS
            or type(specification["cuda_memory_budget_bytes"]) is not int
            or specification["cuda_memory_budget_bytes"] <= 0
        ):
            raise ValueError("invalid resident CUDA specification")
        keys = owned_keys(placement, owner)
        max_experts = (
            MAX_PACKED_RESIDENT_EXPERTS
            if specification.get("weight_layout") == CUDA_NIBBLE_PLANES
            else MAX_RESIDENT_EXPERTS
        )
        if not keys or any(
            sum(key_layer == layer for key_layer, _ in keys) > max_experts for layer in {layer for layer, _ in keys}
        ):
            raise ValueError("resident CUDA partition exceeds the admitted experts per main layer")
        self.keys, self.specification, self.identity, self.limits = keys, dict(specification), identity, limits
        self.cuda, self.plan = None, None
        self.metrics = {"owner": owner, "requests": 0, "routes": 0, "compute_total_ms": 0.0, "compute_max_ms": 0.0}

    def prepare(self):
        self.close()
        started = time.perf_counter()
        self.plan = NativeResidentPlan(
            self.specification["checkpoint_root"], self.keys, self.identity.checkpoint_sha256
        )
        required = self.plan.required_cuda_bytes + SCRATCH_ALLOWANCE_BYTES
        if self.specification.get("weight_layout") == CUDA_NIBBLE_PLANES:
            required -= len(self.keys) * EXPERT_UNPACKED_BYTES // 2
            required += PACKED_STAGING_ALLOWANCE_BYTES
        budget = self.specification["cuda_memory_budget_bytes"]
        if required > budget:
            raise ValueError("resident CUDA plan exceeds configured byte budget")
        # Admission before loading payloads or allocating expert weights.
        runtime = ctypes.CDLL(self.specification["cuda_runtime"])
        runtime.cudaSetDevice.argtypes, runtime.cudaSetDevice.restype = [ctypes.c_int], ctypes.c_int
        runtime.cudaMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t)] * 2
        runtime.cudaMemGetInfo.restype = ctypes.c_int
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        if runtime.cudaSetDevice(0) or runtime.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)):
            raise RuntimeError("CUDA memory admission query failed")
        if required + CUDA_RESERVE_BYTES > free.value:
            raise ValueError("resident CUDA plan leaves less than the 4 GiB device reserve")
        print(
            json.dumps({"kind": "resident_admission", "required_bytes": required, "free_bytes": free.value}), flush=True
        )
        if "cache_root" in self.specification:
            from .cuda_bank_cache import CachedCudaBanks

            banks = CachedCudaBanks(
                self.specification["cache_root"],
                self.specification["cache_manifest_sha256"],
                self.identity.checkpoint_sha256,
                self.keys,
                self.specification.get("weight_layout", CUDA_ROW_MAJOR),
            )
            banks.authenticate()
        else:
            self.plan.authenticate()
            banks = self.plan
        self.cuda = CudaOracle(banks, self.specification, self.limits.max_routes)
        banks.check_unchanged()
        self.plan.check_unchanged()
        self.metrics["prepare_total_ms"] = (time.perf_counter() - started) * 1000
        print(
            json.dumps(
                {
                    "kind": "resident_ready",
                    "layers": len(self.plan.by_layer),
                    "experts": len(self.keys),
                    "cuda_allocated_bytes": self.cuda.allocated_bytes,
                }
            ),
            flush=True,
        )
        return self.keys

    def compute(self, request):
        if self.cuda is None or request.layer_id not in self.plan.by_layer:
            raise RuntimeError("resident CUDA backend is not prepared for this layer")
        start = time.perf_counter()
        tokens, experts = compact_routes(request, self.plan.by_layer[request.layer_id])
        output = self.cuda.compute(
            request.layer_id,
            tuple(request.hidden[token] for token in tokens),
            experts,
            self.plan.manifest["activation_bits"],
        )
        elapsed = (time.perf_counter() - start) * 1000
        self.metrics["requests"] += 1
        self.metrics["routes"] += len(tokens)
        self.metrics["compute_total_ms"] += elapsed
        self.metrics["compute_max_ms"] = max(self.metrics["compute_max_ms"], elapsed)
        return output

    def close(self):
        if self.cuda is not None:
            self.cuda.close()
            self.cuda = None

    def compute_tensor(self, packet):
        import numpy as np

        request = packet.control
        if self.cuda is None or request.layer_id not in self.plan.by_layer:
            raise RuntimeError("resident CUDA backend is not prepared for this layer")
        started = time.perf_counter()
        tokens, experts = compact_routes(request, self.plan.by_layer[request.layer_id])
        inputs = np.frombuffer(packet.tensor, dtype="<f2").reshape(len(request.token_ids), WIDTH)
        # Group equal experts for weight-cache reuse, then restore exact wire order.
        order = np.argsort(np.asarray(experts), kind="stable")
        body = inputs[np.asarray(tokens, dtype=np.int64)[order]].tobytes()
        sorted_experts = tuple(experts[index] for index in order)
        if packet.partial_reduce:
            output_tokens = tuple(sorted(request.token_ids))
            token_map = {token: index for index, token in enumerate(output_tokens)}
            inverse = np.empty_like(order)
            inverse[order] = np.arange(len(order))
            row_map = np.full((len(output_tokens), 8), -1, dtype="<i4")
            weights = np.zeros((len(output_tokens), 8), dtype="<f4")
            raw_weights = np.frombuffer(packet.route_weights, dtype="<f4")
            for index, route in enumerate(request.routes):
                row_map[token_map[route.token], route.slot] = inverse[index]
                weights[token_map[route.token], route.slot] = raw_weights[index]
            encoded = self.cuda.compute_partial_bytes(
                request.layer_id,
                body,
                sorted_experts,
                self.plan.manifest["activation_bits"],
                row_map.tobytes(),
                weights.tobytes(),
                len(output_tokens),
            )
        else:
            output = self.cuda.compute_bytes(
                request.layer_id, body, sorted_experts, self.plan.manifest["activation_bits"]
            )
            sorted_rows = np.frombuffer(output, dtype="<f2").reshape(len(tokens), WIDTH)
            original_rows = np.empty_like(sorted_rows)
            original_rows[order] = sorted_rows
        reply_fp16 = getattr(self, "specification", {}).get("reply_fp16", False)
        if not packet.partial_reduce:
            encoded = original_rows.tobytes() if reply_fp16 else original_rows.astype("<f4").tobytes()
        elapsed = (time.perf_counter() - started) * 1000
        self.metrics["requests"] += 1
        self.metrics["routes"] += len(tokens)
        self.metrics["compute_total_ms"] += elapsed
        self.metrics["compute_max_ms"] = max(self.metrics["compute_max_ms"], elapsed)
        return TensorPacket(
            ExpertReply(request.execution_id, request.layer_id, request.owner, request.routes, ()),
            encoded,
            reply_fp16=reply_fp16,
            partial_reduce=packet.partial_reduce,
        )
