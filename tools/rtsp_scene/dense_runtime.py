"""Instance-local decode kernels; original modules handle other batch shapes."""

from pathlib import Path

import torch
from torch import nn

from tools.rtsp_scene.dense_bench import checked_build, pack_weight
from tools.rtsp_scene.dense_shapes import decode_rows, kernel_namespace

MAX_DECODE_ROWS = 8
TP_SIZE = 2
FFN_HIDDEN = 6144
MODEL_HIDDEN = 4096
SCALE_STRIDE = 8
KERNEL_CORES = 8


def eligible(x, inner: int, max_rows: int = MAX_DECODE_ROWS) -> bool:
    return (
        x.ndim == 2
        and x.dtype == torch.float16
        and x.shape[1] == inner
        and 0 < x.shape[0] <= max_rows
        and x.is_contiguous()
    )


class KernelBackend:
    def __init__(self, build: Path):
        # Imported only by initialized NPU workers, never by CPU test collection.
        import torch_npu

        self.manifest = checked_build(build)
        self.max_decode_rows = decode_rows(self.manifest)
        self.format_cast = torch_npu.npu_format_cast
        self.get_format = torch_npu.get_npu_format
        self.dynamic_quant = torch_npu.npu_dynamic_quant
        self.input_quant = None
        self.add_layer_norm = torch_npu.npu_add_layer_norm
        torch.ops.load_library(str(build / "dense-bridge.so"))
        namespace = kernel_namespace(self.manifest)
        kernel_class = getattr(torch.classes, namespace).Kernel
        self.native_launch = getattr(torch.ops, namespace).launch
        self.matrices = {
            inner: kernel_class(str(build / f"dense-k{inner}.bin"), "rtsp_dense_i8")
            for inner in (2048, MODEL_HIDDEN, FFN_HIDDEN)
        }
        self.activation = kernel_class(str(build / "nz-swiglu.bin"), "rtsp_nz_swiglu")

    def empty(self, rows: int, columns: int, dtype, device, nz: bool = False):
        return torch.ops.npu.empty_with_format([rows, columns], dtype=dtype, device=device, acl_format=29 if nz else 2)

    def launch(self, kernel, arguments):
        self.native_launch(kernel, arguments, KERNEL_CORES)


class DenseProjection(nn.Module):
    def __init__(self, layer, backend, *, output_nz: bool = False, input_nz: bool = False, scale_stride: int = 1):
        super().__init__()
        self.backend = backend
        self.max_decode_rows = backend.max_decode_rows
        self.inner, self.columns = layer.weight.shape
        if layer.weight.dtype != torch.int8 or layer.bias is not None:
            raise ValueError("Only bias-free INT8 checkpoint projections are supported")
        canonical = backend.format_cast(layer.weight.detach().t(), 2).contiguous()
        packed = pack_weight(canonical, backend.manifest["columns"])
        self.register_buffer("packed_weight", packed)
        self.register_buffer("weight_scale", layer.weight_scale.detach())
        self.output_nz = output_nz
        self.input_quant = getattr(backend, "input_quant", None)
        self.input_nz = input_nz or self.input_quant is not None
        self.scale_stride = SCALE_STRIDE if self.input_quant is not None else scale_stride
        self.configs = {
            rows: torch.tensor(
                [rows, self.columns, int(output_nz), int(self.input_nz), self.scale_stride],
                dtype=torch.int64,
                device=packed.device,
            )
            for rows in range(1, self.max_decode_rows + 1)
        }

    def quantized(self, qx, scales):
        rows = qx.shape[0]
        output = self.backend.empty(rows, self.columns, torch.float16, qx.device, self.output_nz)
        self.backend.launch(
            self.backend.matrices[self.inner],
            [qx, self.packed_weight, self.weight_scale, scales, output, self.configs[rows]],
        )
        return output

    def forward(self, x):
        qx, scales = self.input_quant(x) if self.input_quant is not None else self.backend.dynamic_quant(x)
        return self.quantized(qx, scales.reshape(-1))


class DecodeLinearMethod:
    """Delegate prefill/unsupported inputs and preserve the original linear forward."""

    def __init__(self, original, projection):
        self.original = original
        self.projection = projection

    def __getattr__(self, name):
        return getattr(self.original, name)

    def apply(self, layer, x, bias=None, *args, **kwargs):
        if bias is None and eligible(x, self.projection.inner, self.projection.max_decode_rows):
            return self.projection(x)
        return self.original.apply(layer, x, bias, *args, **kwargs)


class NZMLP(nn.Module):
    def __init__(self, original, backend, reduce):
        super().__init__()
        down = original.down_proj
        if (
            down.tp_size != TP_SIZE
            or not down.input_is_parallel
            or down.bias is not None
            or tuple(original.gate_up_proj.weight.shape) != (MODEL_HIDDEN, 2 * FFN_HIDDEN)
            or tuple(down.weight.shape) != (FFN_HIDDEN, MODEL_HIDDEN)
        ):
            raise ValueError("NZ FFN requires the validated Qwen3-VL TP2 shard geometry")
        self.original = original
        self.backend = backend
        self.reduce = reduce
        self.reduce_results = down.reduce_results
        self.gate = DenseProjection(original.gate_up_proj, backend, output_nz=True)
        self.down = DenseProjection(down, backend, input_nz=True, scale_stride=SCALE_STRIDE)
        self.activation_configs = {
            rows: torch.tensor([rows, 0], dtype=torch.int64, device=down.weight.device)
            for rows in range(1, backend.max_decode_rows + 1)
        }

    def forward(self, x):
        if not eligible(x, MODEL_HIDDEN, self.backend.max_decode_rows):
            return self.original(x)
        rows = x.shape[0]
        gate = self.gate(x)
        q = self.backend.empty(rows, FFN_HIDDEN, torch.int8, x.device, True)
        scales = torch.empty(rows, SCALE_STRIDE, dtype=torch.float32, device=x.device)
        # Debug output is disabled; reuse a valid pointer rather than allocate it.
        self.backend.launch(self.backend.activation, [gate, q, scales, self.activation_configs[rows], gate])
        output = self.down.quantized(q, scales)
        return self.reduce(output) if self.reduce_results else output
