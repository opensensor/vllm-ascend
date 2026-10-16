"""Probe direct CANN NZ output without changing weights or the serving runtime."""

import argparse
import ctypes
import json
from pathlib import Path

from tools.rtsp_scene.dense_shapes import nz_shape


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Device libraries are imported only by this isolated hardware worker.
    import torch
    import torch_npu

    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.manual_seed(42)
    acl = ctypes.CDLL("libopapi.so")
    opapi = ctypes.CDLL("libopapi.so")
    pointer = ctypes.c_void_p
    integer = ctypes.c_int64
    acl.aclCreateTensor.restype = pointer
    acl.aclCreateTensor.argtypes = [
        pointer,
        ctypes.c_uint64,
        ctypes.c_int,
        pointer,
        integer,
        ctypes.c_int,
        pointer,
        ctypes.c_uint64,
        pointer,
    ]
    acl.aclDestroyTensor.argtypes = [pointer]
    workspace = opapi.aclnnQuantMatmulWeightNzGetWorkspaceSize
    workspace.argtypes = [pointer] * 9 + [ctypes.c_bool, ctypes.c_bool, integer, pointer, pointer, pointer]
    execute = opapi.aclnnQuantMatmulWeightNz
    execute.argtypes = [pointer, ctypes.c_uint64, pointer, pointer]

    def descriptor(tensor, dimensions=None, strides=None, storage=None, format_id=None):
        dimensions = tuple(tensor.shape) if dimensions is None else dimensions
        strides = tuple(tensor.stride()) if strides is None else strides
        format_id = torch_npu.get_npu_format(tensor) if format_id is None else format_id
        storage = dimensions if storage is None else storage
        dtype = {torch.int8: 2, torch.float16: 1, torch.float32: 0}[tensor.dtype]
        dims = (integer * len(dimensions))(*dimensions)
        stride = (integer * len(strides))(*strides)
        physical = (integer * len(storage))(*storage)
        result = acl.aclCreateTensor(
            dims, len(dims), dtype, stride, 0, format_id, physical, len(physical), tensor.data_ptr()
        )
        if not result:
            raise RuntimeError("Could not create ACL tensor descriptor")
        return result

    records = []
    for rows, output_format in ((1, 2), (4, 2), (8, 2), (1, 29), (4, 29), (8, 29)):
        k, n = 4096, 12288
        x = torch.randn(rows, k, dtype=torch.float16, device="npu") * 0.1
        w = torch.randint(-127, 128, (n, k), dtype=torch.int8, device="npu")
        w = torch_npu.npu_format_cast(w, 29)
        scale = torch.rand(n, dtype=torch.float32, device="npu") * 0.01
        qx, xs = torch_npu.npu_dynamic_quant(x)
        expected = torch_npu.npu_quant_matmul(qx, w.t(), scale, pertoken_scale=xs, output_dtype=torch.float16)
        y = torch.ops.npu.empty_with_format([rows, n], dtype=torch.float16, device=x.device, acl_format=output_format)
        torch.npu.synchronize()
        descriptors = [
            descriptor(qx),
            descriptor(w, storage=nz_shape(n, k, True)),
            descriptor(xs),
            descriptor(scale),
            descriptor(y, storage=nz_shape(rows, n) if output_format == 29 else (rows, n)),
        ]
        size, executor = ctypes.c_uint64(), pointer()
        status = workspace(
            *descriptors[:4],
            None,
            None,
            None,
            None,
            None,
            False,
            True,
            0,
            descriptors[4],
            ctypes.byref(size),
            ctypes.byref(executor),
        )
        record = {
            "rows": rows,
            "workspace_status": status,
            "workspace_bytes": size.value,
            "output_format": torch_npu.get_npu_format(y),
        }
        if status == 0:
            scratch = torch.empty(size.value, dtype=torch.uint8, device=x.device)
            record["launch_status"] = execute(
                scratch.data_ptr(), size.value, executor, torch.npu.current_stream().npu_stream
            )
            torch.npu.synchronize()
            actual = y.cpu()
            reference = expected.cpu()
            record["bitwise_equal"] = torch.equal(actual, reference)
            record["finite"] = bool(torch.isfinite(actual).all())
            record["max_abs_error"] = (actual - reference).abs().max().item() if record["finite"] else None
            record["actual_first"] = actual[0, :12].tolist() if record["finite"] else None
            record["reference_first"] = reference[0, :12].tolist()
        for desc in descriptors:
            acl.aclDestroyTensor(desc)
        records.append(record)
        args.output.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
