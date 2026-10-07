# SPDX-License-Identifier: Apache-2.0
"""GLM W4A8 experiment: unchanged signed weights, new activation quantization.

The two INT4 limbs reconstruct INT8 exactly. Quantizing FP16 activations and
moving weight scaling outside the integer dot product are numerical changes;
neither CPU reference parity nor successful graph capture qualifies model quality.
"""

from dataclasses import dataclass

import torch

BLOCK = 32
N_TILE = 16
K_TILE = 256
FRACTAL_K = 64
INT8_MAX = 127
RADIX = 16
BIAS = 8
MAX_GROUPED_ROUTES = 65536


@dataclass(frozen=True)
class GroupedProjectionGeometry:
    rows: int
    experts: int
    n: int
    k: int

    def __post_init__(self):
        values = self.rows, self.experts, self.n, self.k
        if any(type(value) is not int for value in values):
            raise ValueError("projection dimensions must be integers")
        if not 1 <= self.rows <= MAX_GROUPED_ROUTES or not 1 <= self.experts <= 288:
            raise ValueError("unsupported grouped row or expert count")
        if self.n <= 0 or self.n % 128 or not 256 <= self.k <= 4096 or self.k % 256:
            raise ValueError("grouped projections require complete GLM NZ tiles")


def pack_nz_codes(signed, bits=4):
    """Independent CPU reference for GLM's field-major 16x256 NZ packing."""
    if signed.device.type != "cpu" or signed.dtype != torch.int8 or signed.ndim != 3 or bits not in (2, 3, 4):
        raise ValueError("expected CPU signed INT8 [experts,N,K], bits 2, 3 or 4")
    experts, n, k = signed.shape
    if (
        not signed.numel()
        or n % N_TILE
        or k % K_TILE
        or signed.min() < -(1 << (bits - 1))
        or signed.max() >= (1 << (bits - 1))
    ):
        raise ValueError("codes must fit their signed bit width and complete NZ tiles")
    fields = (signed.int() & ((1 << bits) - 1)).reshape(experts, n // N_TILE, N_TILE, k // K_TILE, K_TILE)
    fields = fields.permute(0, 1, 3, 4, 2).contiguous()
    count = 8 if bits == 3 else 8 // bits
    planes = fields.reshape(experts, n // N_TILE, k // K_TILE, count, -1)
    words = torch.zeros_like(planes[..., 0, :])
    for field in range(count):
        words |= planes[..., field, :] << (bits * field)
    packed = torch.stack([(words >> (8 * byte)) & 255 for byte in range(3 if bits == 3 else 1)], -2)
    return packed.to(torch.uint8).view(torch.int8).reshape(experts, n, k * bits // 8).contiguous()


def activation_limbs(inputs):
    if inputs.ndim != 2 or inputs.dtype != torch.float16 or inputs.shape[1] % BLOCK:
        raise ValueError("activation requires FP16 [rows,K], K divisible by 32")
    groups = inputs.float().reshape(inputs.shape[0], -1, BLOCK)
    maximum = groups.abs().amax(-1)
    scales = torch.where(maximum > 0, maximum / INT8_MAX, torch.ones_like(maximum))
    quant = (groups / scales.unsqueeze(-1)).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int32)
    high = torch.div(quant, RADIX, rounding_mode="floor")
    low = quant - RADIX * high - BIAS
    return low, high, scales, quant


def signed_nibbles(packed):
    if packed.dtype not in (torch.int8, torch.uint8):
        raise ValueError("packed nibbles require byte storage")
    unsigned = packed.to(torch.int32) & 255
    fields = torch.stack((unsigned & 15, (unsigned >> 4) & 15), -1)
    return torch.where(fields >= BIAS, fields - RADIX, fields).to(torch.int8)


def unpack_canonical_codes(packed, k):
    """Decode checkpoint row-order bytes independently of the serving loader."""
    if packed.ndim != 2 or packed.dtype not in (torch.int8, torch.uint8) or k <= 0:
        raise ValueError("canonical weights require byte [N,packedK] storage")
    bits = packed_weight_bits(k, packed.shape[1])
    unsigned = packed.int() & 255
    if bits == 3:
        planes = unsigned.reshape(packed.shape[0], -1, 3)
        words = planes[..., 0] | (planes[..., 1] << 8) | (planes[..., 2] << 16)
        fields = torch.stack([(words >> (bits * field)) & 7 for field in range(8)], -1)
    else:
        fields = torch.stack([(unsigned >> (bits * field)) & ((1 << bits) - 1) for field in range(8 // bits)], -1)
    signed = torch.where(fields >= 1 << (bits - 1), fields - (1 << bits), fields)
    return signed.to(torch.int8).reshape(packed.shape[0], k)


def unpack_nz_w4(packed, k):
    """Invert GLM's field-major 16x256 layout without changing signed codes."""
    if packed.ndim != 3 or k % K_TILE or packed.shape[1] % N_TILE or packed.shape[2] * 2 != k:
        raise ValueError("expected W4 [experts,N,K/2] with 16x256 NZ tiles")
    experts, n, _ = packed.shape
    fields = signed_nibbles(packed).reshape(experts, n // N_TILE, k // K_TILE, -1, 2)
    logical = fields.transpose(-1, -2).reshape(experts, n // N_TILE, k // K_TILE, K_TILE, N_TILE)
    return logical.permute(0, 1, 4, 2, 3).reshape(experts, n, k).contiguous()


def block_reference(inputs, signed, scales):
    """Reference for the native arithmetic, separately from W4A16 baseline."""
    if signed.ndim != 2 or signed.dtype != torch.int8 or signed.shape[1] != inputs.shape[1]:
        raise ValueError("expected signed expert [N,K] matching activation K")
    n, k = signed.shape
    if n % BLOCK or k % BLOCK or scales.shape != (n // BLOCK, k // BLOCK):
        raise ValueError("GLM native reference requires a 32x32 block scale grid")
    low, high, xs, _ = activation_limbs(inputs)
    result = torch.zeros((inputs.shape[0], n), dtype=torch.float32, device=inputs.device)
    rounded_scales = scales.half().float().repeat_interleave(BLOCK, 0)
    for group in range(k // BLOCK):
        weight = signed[:, group * BLOCK : (group + 1) * BLOCK].to(torch.int32)
        # Integer products fit INT32; FP32 also represents each group result
        # exactly, and supports the same host/device reference implementation.
        integer = low[:, group].float() @ weight.float().T
        integer += RADIX * (high[:, group].float() @ weight.float().T)
        integer += BIAS * weight.sum(-1).float()
        result += integer * (xs[:, group, None] * rounded_scales[None, :, group])
    return result.half()


def fp16_weight_reference(inputs, signed, scales):
    weights = signed.half() * scales.half().repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
    return (inputs.float() @ weights.float().T).half()


def pipeline_gather_offsets(output_columns, bits=4):
    """Prepare immutable byte offsets on CPU, before any graph capture."""
    if output_columns not in (16, 32, 64, 128) or bits not in (2, 3, 4):
        raise ValueError("tile pipeline output tile must be 16, 32, 64 or 128")
    physical = torch.arange(output_columns)[None, :, None]
    channel = torch.where(physical < output_columns // 2, 2 * physical, 2 * (physical - output_columns // 2) + 1)
    group = torch.arange(K_TILE // BLOCK)[:, None, None]
    word = torch.arange(FRACTAL_K // 4)[None, None, :]
    offsets = channel // N_TILE * (N_TILE * K_TILE * bits // 8)
    inner = 4 * word if bits == 3 else group % bits * BLOCK + 4 * word
    plane = group * bits // 8 if bits == 3 else torch.zeros_like(group)
    offsets = offsets + inner * N_TILE + channel % N_TILE // 2 * 2 + plane * (N_TILE * K_TILE // 8)
    return torch.where(word < BLOCK // 4, offsets, output_columns * K_TILE * bits // 8).flatten().contiguous()


def packed_weight_bits(k, packed_k):
    bits, remainder = divmod(packed_k * 8, k)
    if remainder or bits not in (2, 3, 4):
        raise ValueError("expected complete W2/W3/W4 packed input dimension")
    return bits


class NativeW4Projection:
    def __init__(
        self,
        pack_binary,
        matrix_binary,
        geometries,
        *,
        namespace="glm_reconstruction_v1",
        tile_pipeline=False,
        output_columns=16,
        all_bits=False,
    ):
        from .reconstruction_native import ProjectionGeometry

        factory = getattr(torch.classes, namespace).Kernel
        self.pack_kernel = factory(str(pack_binary), "glm_w4a8_pack_v1")
        self.matrix_kernel = factory(str(matrix_binary), "glm_w4a8_matmul_v1")
        self.launch = getattr(torch.ops, namespace).launch
        self.configs = {}
        self.pack_configs = {}
        if all_bits and not tile_pipeline:
            raise ValueError("all bit widths require the tile pipeline")
        self.all_bits = all_bits
        weight_bits = (2, 3, 4) if all_bits else (4,)
        metadata = {}
        for bits in weight_bits:
            metadata[bits] = torch.empty(0, dtype=torch.int64)
            if tile_pipeline:
                offsets = pipeline_gather_offsets(output_columns, bits).reshape(-1, 2)
                metadata[bits] = offsets[:, 0] | (offsets[:, 1] << 32)
        self.metadata = metadata
        for geometry in geometries:
            if not isinstance(geometry, (ProjectionGeometry, GroupedProjectionGeometry)):
                raise ValueError("native W4 requires explicit projection geometries")
            if all_bits:
                geometry = GroupedProjectionGeometry(geometry.rows, geometry.experts, geometry.n, geometry.k)
            for bits in weight_bits:
                values = (geometry.rows, geometry.experts, geometry.n, geometry.k)
                if all_bits:
                    values += (bits, 0, 0, 0)  # Metadata DMA starts at aligned byte 64.
                header = torch.tensor(values, dtype=torch.int64)
                key = (geometry, bits) if all_bits else geometry
                self.configs[key] = torch.cat((header, metadata[bits])).npu()
                self.prepared_device = self.configs[key].device
            key = (geometry.rows, geometry.k)
            if key not in self.pack_configs:
                self.pack_configs[key] = torch.tensor((geometry.rows * geometry.k // BLOCK,), dtype=torch.int64).npu()

    def geometry(self, inputs, codes, scales, ends):
        from .reconstruction_native import ProjectionGeometry

        if inputs.ndim != 2 or codes.ndim != 3 or scales.ndim != 3 or ends.ndim != 1:
            raise ValueError("native W4 requires grouped projection tensors")
        all_bits = getattr(self, "all_bits", False)
        geometry_type = GroupedProjectionGeometry if all_bits else ProjectionGeometry
        geometry = geometry_type(inputs.shape[0], codes.shape[0], codes.shape[1], inputs.shape[1])
        bits = packed_weight_bits(geometry.k, codes.shape[2])
        key = (geometry, bits) if getattr(self, "all_bits", False) else geometry
        if (bits != 4 and not all_bits) or (not all_bits and key not in self.configs):
            raise ValueError("W4 geometry is not prepared")
        if scales.shape != (geometry.experts, geometry.n // BLOCK, geometry.k // BLOCK):
            raise ValueError("native W4 scale shape mismatch")
        if ends.shape != (geometry.experts,):
            raise ValueError("native W4 boundary shape mismatch")
        if (inputs.dtype, scales.dtype, ends.dtype) != (
            torch.float16,
            torch.float32,
            torch.int64,
        ) or codes.dtype not in (torch.int8, torch.uint8):
            raise ValueError("native W4 requires FP16/NZ byte/FP32/INT64 tensors")
        if (
            inputs.device.type != "npu"
            or not inputs.is_contiguous()
            or any(
                tensor.device != inputs.device or not tensor.is_contiguous() for tensor in (inputs, codes, scales, ends)
            )
            or (self.prepared_device if all_bits else self.configs[key].device) != inputs.device
        ):
            raise ValueError("native W4 tensors must be contiguous on the prepared NPU")
        return geometry

    def supports(self, inputs, codes, scales, ends):
        try:
            self.geometry(inputs, codes, scales, ends)
        except ValueError:
            return False
        return True

    def pack(self, inputs):
        if (
            inputs.ndim != 2
            or inputs.dtype != torch.float16
            or inputs.device.type != "npu"
            or not inputs.is_contiguous()
        ):
            raise ValueError("packing requires contiguous NPU FP16 [rows,K]")
        rows, k = inputs.shape
        if self.all_bits and (rows, k) not in self.pack_configs and inputs.device == self.prepared_device:
            self.pack_configs[(rows, k)] = torch.tensor((rows * k // BLOCK,), dtype=torch.int64).to(inputs.device)
        if (rows, k) not in self.pack_configs or self.pack_configs[(rows, k)].device != inputs.device:
            raise ValueError("activation geometry was not prepared on this NPU")
        groups = k // BLOCK
        # Each 32-value group occupies a padded 64-value Cube fractal. Only
        # activation limbs are padded; the resident weight bank is unchanged.
        low = torch.empty((rows, groups, FRACTAL_K // 2), dtype=torch.int8, device=inputs.device)
        high = torch.empty_like(low)
        xs = torch.empty((rows, groups, BIAS), dtype=torch.float32, device=inputs.device)
        self.launch(self.pack_kernel, [inputs, low, high, xs, self.pack_configs[(rows, k)]], 8)
        return low, high, xs

    def project(self, inputs, codes, scales, ends, limbs):
        geometry = self.geometry(inputs, codes, scales, ends)
        key = (geometry, packed_weight_bits(geometry.k, codes.shape[2])) if self.all_bits else geometry
        if self.all_bits and key not in self.configs:
            # Prefill uses host shape metadata to prepare a new immutable header
            # once. Capture shapes are prepared earlier; no route tensor read.
            bits = key[1]
            header = torch.tensor(
                (geometry.rows, geometry.experts, geometry.n, geometry.k, bits, 0, 0, 0), dtype=torch.int64
            )
            self.configs[key] = torch.cat((header, self.metadata[bits])).to(inputs.device)
        expected = ((geometry.rows, geometry.k // BLOCK, FRACTAL_K // 2),) * 2 + (
            (geometry.rows, geometry.k // BLOCK, BIAS),
        )
        if len(limbs) != 3 or any(
            tensor.shape != shape
            or tensor.dtype != dtype
            or tensor.device != inputs.device
            or not tensor.is_contiguous()
            for tensor, shape, dtype in zip(limbs, expected, (torch.int8, torch.int8, torch.float32))
        ):
            raise ValueError("activation limbs must match the prepared shape, dtype and NPU")
        output = torch.empty((geometry.rows, geometry.n), dtype=torch.float16, device=inputs.device)
        self.launch(
            self.matrix_kernel,
            [*limbs, codes, scales, ends, output, self.configs[key]],
            8,
        )
        return output

    def __call__(self, inputs, codes, scales, ends, zero_outputs=True):
        if type(zero_outputs) is not bool:
            raise ValueError("zero_outputs must be boolean")
        self.geometry(inputs, codes, scales, ends)
        # The native kernel always writes peer zeros. This is valid for both
        # caller contracts and makes component output hashes reproducible.
        return self.project(inputs, codes, scales, ends, self.pack(inputs))
