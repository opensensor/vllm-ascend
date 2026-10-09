# SPDX-License-Identifier: Apache-2.0
"""Build append-only reconstruction probes; never connect to a serving engine."""

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sysconfig
from pathlib import Path

import torch
from torch.utils.cpp_extension import include_paths, library_paths

HERE = Path(__file__).resolve().parent
W3_DEFINES = (
    "GLM_W2_GROUPED_L1_WIDE",
    "GLM_W2_GROUPED_L1_WIDE_PIPELINED",
    "GLM_W2_GROUPED_L1_W3",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(
    build_dir,
    cann_root,
    source_root,
    include_w3=False,
    version=1,
    output_columns=16,
    tile_pipeline=False,
    all_bits=False,
    fused_moe=False,
    prepared_weight_layout=False,
    pair_scale_groups=False,
    fp16_swiglu=False,
    pair_hidden_quant=False,
    wide_cube_k=0,
    pair_prefill_scale_groups=False,
    weight_decode_lut=False,
    strided_product_copy=False,
    repeat_product_cast=False,
    w3_float_fragments=False,
    specialize_w3=False,
    prefill_weight_cache=False,
    fp16_route_workspace=False,
    share_gate_up_input=False,
    cache_gate_up_activations=False,
    vector_scale_products=False,
    gather_product_matrix=False,
    prefill_rows_32=False,
    quad_hidden_quant=False,
    direct_hidden_gather=False,
    route_packed_input=False,
    route_packed_down=False,
    route_compact_down_scales=False,
    raw_hidden_scales=False,
    raw_input_scales=False,
    nz_prefill_accumulator=False,
    prefill_product_cast=False,
    native_route_columns=False,
    prerounded_weight_scales=False,
    compact_w4_scratch=False,
    fp16_weight_scales=False,
    active_cube_rows=False,
    direct_w4_l1=False,
    prepared_offset_tables=False,
    product_pipe_events=False,
    bulk_route_store=False,
    fused_scale_accumulation=False,
    nz_prefill_min_rows=0,
    cache_expert_ends=False,
    prefill_reduce_meta_cache=False,
    direct_compact_down_scales=False,
):
    """Freeze helper sources and compile a unique, append-only native version."""
    if type(version) is not int or version < 1 or output_columns not in (16, 32, 64, 128):
        raise ValueError("version must be positive and output tile 16, 32, 64 or 128")
    if any(
        type(flag) is not bool
        for flag in (
            prefill_weight_cache,
            fp16_route_workspace,
            share_gate_up_input,
            cache_gate_up_activations,
            vector_scale_products,
            gather_product_matrix,
            prefill_rows_32,
            quad_hidden_quant,
            direct_hidden_gather,
            route_packed_input,
            route_packed_down,
            route_compact_down_scales,
            raw_hidden_scales,
            raw_input_scales,
            nz_prefill_accumulator,
            prefill_product_cast,
            native_route_columns,
            prerounded_weight_scales,
            compact_w4_scratch,
            fp16_weight_scales,
            active_cube_rows,
            direct_w4_l1,
            prepared_offset_tables,
            product_pipe_events,
            bulk_route_store,
            fused_scale_accumulation,
            cache_expert_ends,
            prefill_reduce_meta_cache,
            direct_compact_down_scales,
        )
    ):
        raise ValueError("prefill experiment flags must be boolean")
    if output_columns == 128 and not tile_pipeline:
        raise ValueError("128 columns requires the tile pipeline")
    if all_bits and not tile_pipeline:
        raise ValueError("all bit widths require the tile pipeline")
    if fused_moe and (not all_bits or output_columns != 128):
        raise ValueError("fused MoE requires all bits and a 128-column tile")
    if prepared_weight_layout and not fused_moe:
        raise ValueError("prepared weight layout requires fused MoE")
    if pair_scale_groups and not fused_moe:
        raise ValueError("paired scale groups require fused MoE")
    if fp16_swiglu and not fused_moe:
        raise ValueError("FP16 SwiGLU requires fused MoE")
    if pair_hidden_quant and not fused_moe:
        raise ValueError("paired hidden quantization requires fused MoE")
    if direct_hidden_gather and not fused_moe:
        raise ValueError("direct hidden gather requires fused MoE")
    if quad_hidden_quant and not fused_moe:
        raise ValueError("four-row hidden quantization requires fused MoE")
    if quad_hidden_quant and pair_hidden_quant:
        raise ValueError("two-row and four-row hidden quantization are alternative schedules")
    if strided_product_copy and not fused_moe:
        raise ValueError("strided product copy requires fused MoE")
    if repeat_product_cast and not fused_moe:
        raise ValueError("repeated product cast requires fused MoE")
    if specialize_w3 and not fused_moe:
        raise ValueError("W3 specialization requires fused MoE")
    if compact_w4_scratch and not (fused_moe and prepared_weight_layout):
        raise ValueError("compact W4 scratch requires prepared fused MoE")
    if compact_w4_scratch and weight_decode_lut:
        raise ValueError("compact W4 scratch cannot retain weight decode lookup tables")
    if prefill_weight_cache and not (fused_moe and prepared_weight_layout):
        raise ValueError("prefill weight cache requires prepared fused MoE")
    if prefill_weight_cache and weight_decode_lut:
        raise ValueError("prefill weight cache does not support decode lookup tables")
    if fp16_route_workspace and not fused_moe:
        raise ValueError("FP16 route workspace requires fused MoE")
    if native_route_columns and not (fused_moe and fp16_route_workspace):
        raise ValueError("native route columns require paired fused MoE and FP16 workspace reduction")
    if prerounded_weight_scales and not (fused_moe and prepared_weight_layout):
        raise ValueError("prerounded weight scales require the prepared fused MoE layout")
    if fp16_weight_scales and not (fused_moe and prepared_weight_layout):
        raise ValueError("FP16 weight scales require prepared fused MoE")
    if fp16_weight_scales and prerounded_weight_scales:
        raise ValueError("choose FP16 storage or prerounded FP32 weight scales")
    if share_gate_up_input and not fused_moe:
        raise ValueError("shared gate/up input requires fused MoE")
    if cache_gate_up_activations and not (fused_moe and share_gate_up_input):
        raise ValueError("gate/up activation cache requires shared-input fused MoE")
    if vector_scale_products and not (fused_moe and prepared_weight_layout):
        raise ValueError("vector scale products require prepared fused MoE")
    if gather_product_matrix and not (fused_moe and prepared_weight_layout):
        raise ValueError("matrix product gather requires prepared fused MoE")
    if gather_product_matrix and (weight_decode_lut or strided_product_copy or repeat_product_cast):
        raise ValueError("matrix product gather needs its scratch and a single readback schedule")
    if prefill_rows_32 and not (fused_moe and prepared_weight_layout and vector_scale_products):
        raise ValueError("wide prefill rows require prepared vector-scale fused MoE")
    if prefill_rows_32 and (
        weight_decode_lut
        or gather_product_matrix
        or fp16_swiglu
        or strided_product_copy
        or repeat_product_cast
        or wide_cube_k
    ):
        raise ValueError("wide prefill rows need lifetime-compatible scratch and the default K64 readback schedule")
    if w3_float_fragments and not fused_moe:
        raise ValueError("W3 fragment reconstruction requires fused MoE")
    if w3_float_fragments and weight_decode_lut:
        raise ValueError("W3 fragments and lookup tables are alternative reconstruction paths")
    if strided_product_copy and repeat_product_cast:
        raise ValueError("product copy and repeated cast are alternative readback schedules")
    if type(wide_cube_k) is not int or wide_cube_k not in (0, 128, 256):
        raise ValueError("wide Cube K must be 0 (disabled), 128 or 256")
    if (wide_cube_k or pair_prefill_scale_groups or weight_decode_lut) and not (
        fused_moe and prepared_weight_layout and pair_scale_groups
    ):
        raise ValueError("wide Cube and prefill scale pairing require prepared paired fused MoE")
    if route_packed_input and not (
        prefill_rows_32
        and prepared_weight_layout
        and pair_scale_groups
        and pair_prefill_scale_groups
        and share_gate_up_input
        and cache_gate_up_activations
    ):
        raise ValueError("routed input packing requires paired wide rows and shared cached gate/up activations")
    if route_packed_down and not route_packed_input:
        raise ValueError("packed down input requires routed input packing")
    if route_compact_down_scales and not route_packed_down:
        raise ValueError("compact down scales require packed down input")
    if raw_hidden_scales and not (route_compact_down_scales and quad_hidden_quant):
        raise ValueError("raw hidden scales require compact down scales and four-row quantization")
    if raw_input_scales and not route_packed_input:
        raise ValueError("raw input scales require routed input packing")
    if nz_prefill_accumulator and not (prefill_rows_32 and pair_prefill_scale_groups):
        raise ValueError("NZ prefill accumulator requires wide paired prefill rows")
    # M32 reserves one row for the A8 bias product; its route ABI batches 31.
    # Zero preserves the original >16-row selection for existing bundles.
    if type(nz_prefill_min_rows) is not int or nz_prefill_min_rows not in (0, *range(17, 32)):
        raise ValueError("NZ prefill minimum rows must be 0 (legacy) or 17 through 31")
    if nz_prefill_min_rows and not nz_prefill_accumulator:
        raise ValueError("NZ prefill minimum rows require the NZ accumulator")
    if cache_expert_ends and not (
        fused_moe and prepared_weight_layout and vector_scale_products and not weight_decode_lut
    ):
        raise ValueError("expert boundary cache requires prepared vector-scale fused MoE without lookup tables")
    if prefill_reduce_meta_cache and not (fused_moe and fp16_route_workspace):
        raise ValueError("cached reducer metadata requires fused MoE with FP16 route workspace")
    if direct_compact_down_scales and not (raw_hidden_scales and prefill_rows_32 and vector_scale_products):
        raise ValueError("direct compact scales require raw M32 vector-scale down projection")
    if prefill_product_cast and not (prefill_rows_32 and pair_prefill_scale_groups):
        raise ValueError("contiguous prefill cast requires wide paired prefill rows")
    if prefill_product_cast and nz_prefill_accumulator:
        raise ValueError("contiguous prefill cast and NZ accumulation are alternative readback schedules")
    if active_cube_rows and not (prefill_rows_32 and pair_prefill_scale_groups):
        raise ValueError("active Cube rows require paired M32 prefill")
    if active_cube_rows and (nz_prefill_accumulator or prefill_product_cast):
        raise ValueError("active Cube rows require the qualified strided readback")
    if direct_w4_l1 and not (prepared_weight_layout and prefill_weight_cache):
        raise ValueError("direct W4 L1 requires prepared projection cache allocation")
    if prepared_offset_tables and not (fused_moe and prepared_weight_layout):
        raise ValueError("prepared offsets require prepared fused MoE")
    namespace = f"glm_reconstruction_v{version}"
    build_dir = build_dir.resolve()
    if product_pipe_events and not fused_moe:
        raise ValueError("product pipe events require fused MoE")
    if bulk_route_store and not (fused_moe and fp16_route_workspace and native_route_columns):
        raise ValueError("bulk route store requires fused native-column FP16 workspace")
    if fused_scale_accumulation and not (fused_moe and vector_scale_products):
        raise ValueError("fused scale accumulation requires vector-scale fused MoE")
    if fused_scale_accumulation and nz_prefill_accumulator:
        raise ValueError("fused scale accumulation requires the row accumulator")
    build_dir.mkdir(parents=True, exist_ok=False)
    helper_package = namespace + "_helpers"
    helper_root = build_dir / helper_package
    helper_root.mkdir()
    (helper_root / "__init__.py").write_text("# SPDX-License-Identifier: Apache-2.0\n")
    helpers = ("glm_int4.py", "reconstruction_native.py", "reconstruction_probe.py")
    if fused_moe:
        helpers += ("glm_fused_moe.py", "fused_moe_probe.py", "fused_weight_layout.py", "fused_offset_tables.py")
    for name in helpers:
        shutil.copy2(HERE / name, helper_root / name)
    shutil.copy2(HERE / "resident_candidates/expert_reconstruction.py", helper_root / "expert_reconstruction.py")
    cann_root = cann_root.resolve(strict=True)
    source_root = source_root.resolve(strict=True)
    compiler = build_dir / "compile-reconstruction"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O2",
            str(HERE / "compile_reconstruction.cpp"),
            f"-I{cann_root / 'include'}",
            f"-L{cann_root / 'lib64'}",
            "-lacl_rtc",
            "-lascendcl",
            f"-Wl,-rpath,{cann_root / 'lib64'}",
            "-o",
            str(compiler),
        ],
        check=True,
    )
    target = cann_root / "tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0"
    options = [
        "--npu-arch=dav-2002",
        f"--sysroot={cann_root / 'tools/hcc/sysroot'}",
        f"-isystem{target}",
        f"-isystem{target / 'aarch64-target-linux-gnu'}",
    ]
    matrix_source = "glm_w4a8_pipeline.cpp" if tile_pipeline else "glm_w4a8_matmul.cpp"
    if all_bits:
        matrix_source = "glm_lowbit_a8_pipeline.cpp"
    sources = ["glm_w4a8_pack.cpp", matrix_source]
    provenance = {
        "_build": {
            "namespace": namespace,
            "version": version,
            "output_columns": output_columns,
            "helper_package": helper_package,
            "tile_pipeline": tile_pipeline,
            "all_bits": all_bits,
            "prefill_native": all_bits,
            "fused_moe": fused_moe,
            "prepared_weight_layout": prepared_weight_layout,
            "pair_scale_groups": pair_scale_groups,
            "fp16_swiglu": fp16_swiglu,
            "pair_hidden_quant": pair_hidden_quant,
            "strided_product_copy": strided_product_copy,
            "repeat_product_cast": repeat_product_cast,
            "w3_float_fragments": w3_float_fragments,
            "specialize_w3": specialize_w3,
            "compact_w4_scratch": compact_w4_scratch,
            "prefill_weight_cache": prefill_weight_cache,
            "fp16_route_workspace": fp16_route_workspace,
            "native_route_columns": native_route_columns,
            "prerounded_weight_scales": prerounded_weight_scales,
            "fp16_weight_scales": fp16_weight_scales,
            "active_cube_rows": active_cube_rows,
            "direct_w4_l1": direct_w4_l1,
            "prepared_offset_tables": prepared_offset_tables,
            "product_pipe_events": product_pipe_events,
            "bulk_route_store": bulk_route_store,
            "fused_scale_accumulation": fused_scale_accumulation,
            "share_gate_up_input": share_gate_up_input,
            "cache_gate_up_activations": cache_gate_up_activations,
            "vector_scale_products": vector_scale_products,
            "gather_product_matrix": gather_product_matrix,
            "prefill_rows_32": prefill_rows_32,
            "quad_hidden_quant": quad_hidden_quant,
            "direct_hidden_gather": direct_hidden_gather,
            "route_packed_input": route_packed_input,
            "route_packed_down": route_packed_down,
            "route_compact_down_scales": route_compact_down_scales,
            "raw_hidden_scales": raw_hidden_scales,
            "raw_input_scales": raw_input_scales,
            "nz_prefill_accumulator": nz_prefill_accumulator,
            "nz_prefill_min_rows": nz_prefill_min_rows,
            "cache_expert_ends": cache_expert_ends,
            "prefill_reduce_meta_cache": prefill_reduce_meta_cache,
            "direct_compact_down_scales": direct_compact_down_scales,
            "prefill_product_cast": prefill_product_cast,
            "weight_decode_lut": weight_decode_lut,
            "wide_cube_k": wide_cube_k,
            "pair_prefill_scale_groups": pair_prefill_scale_groups,
        }
    }
    provenance["_helpers"] = {p.name: sha256(p) for p in helper_root.glob("*.py")}
    for name in sources:
        source = HERE / name
        output = build_dir / ("glm_w4a8_matmul.bin" if name == matrix_source else source.stem + ".bin")
        subprocess.run(
            [
                str(compiler),
                str(source),
                str(output),
                *options,
                f"-DGLM_INT4_OUTPUT_COLUMNS={output_columns}",
                *(["-DGLM_INT4_METADATA_GATHER"] if tile_pipeline else []),
            ],
            check=True,
        )
        provenance[name] = {"source_sha256": sha256(source), "binary_sha256": sha256(output)}
    if fused_moe:
        source = HERE / "glm_fused_moe.cpp"
        stages = [(stage, None) for stage in ("gate_up", "down")]
        if specialize_w3:
            stages.extend((stage, 3) for stage in ("gate_up", "down"))
        if compact_w4_scratch:
            stages.extend((stage, 4) for stage in ("gate_up", "down"))
        for stage, precision in stages:
            suffix = f"_w{precision}" if precision else ""
            output = build_dir / f"glm_fused_{stage}{suffix}.bin"
            subprocess.run(
                [
                    str(compiler),
                    str(source),
                    str(output),
                    *options,
                    f"-I{HERE}",
                    *(["-DGLM_PREPARED_WEIGHT_LAYOUT"] if prepared_weight_layout else []),
                    *(["-DGLM_PREROUNDED_WEIGHT_SCALES"] if prerounded_weight_scales else []),
                    *(["-DGLM_FP16_WEIGHT_SCALES"] if fp16_weight_scales else []),
                    *(["-DGLM_PAIR_SCALE_GROUPS"] if pair_scale_groups else []),
                    *(["-DGLM_FP16_SWIGLU"] if fp16_swiglu else []),
                    *(["-DGLM_PAIR_HIDDEN_QUANT"] if pair_hidden_quant else []),
                    *(
                        ["-DGLM_QUAD_HIDDEN_QUANT", "-DGLM_HIDDEN_QUANT_BATCH=16"]
                        if stage == "gate_up" and quad_hidden_quant
                        else []
                    ),
                    *([f"-DGLM_NATIVE_WIDE_CUBE_K={wide_cube_k}"] if wide_cube_k else []),
                    *(["-DGLM_STRIDED_PRODUCT_COPY"] if strided_product_copy else []),
                    *(["-DGLM_REPEAT_PRODUCT_CAST"] if repeat_product_cast else []),
                    *(
                        ["-DGLM_W3_FLOAT_FRAGMENTS"]
                        if w3_float_fragments and precision != 4 and (not specialize_w3 or precision == 3)
                        else []
                    ),
                    *([f"-DGLM_STATIC_WEIGHT_BITS={precision}"] if precision else []),
                    *(["-DGLM_COMPACT_W4_SCRATCH"] if precision == 4 else []),
                    *(["-DGLM_PREFILL_WEIGHT_CACHE"] if prefill_weight_cache else []),
                    *(["-DGLM_VECTOR_SCALE_PRODUCTS"] if vector_scale_products else []),
                    *(["-DGLM_PREFILL_ROWS_32"] if prefill_rows_32 else []),
                    *(["-DGLM_ACTIVE_CUBE_ROWS"] if active_cube_rows else []),
                    *(["-DGLM_DIRECT_W4_L1"] if direct_w4_l1 else []),
                    *(["-DGLM_PREPARED_OFFSET_TABLES"] if prepared_offset_tables else []),
                    *(["-DGLM_PRODUCT_PIPE_EVENTS"] if product_pipe_events else []),
                    *(["-DGLM_FUSED_SCALE_ACCUMULATION"] if fused_scale_accumulation else []),
                    *(["-DGLM_BULK_ROUTE_STORE"] if stage == "down" and bulk_route_store else []),
                    *(["-DGLM_GATHER_PRODUCT_MATRIX"] if gather_product_matrix else []),
                    *(["-DGLM_FP16_ROUTE_WORKSPACE"] if fp16_route_workspace else []),
                    *(["-DGLM_NATIVE_ROUTE_COLUMNS"] if stage == "down" and native_route_columns else []),
                    *(["-DGLM_WEIGHT_DECODE_LUT"] if weight_decode_lut else []),
                    *(["-DGLM_PAIR_PREFILL_SCALE_GROUPS"] if pair_prefill_scale_groups else []),
                    *(["-DGLM_FUSED_GATE_UP"] if stage == "gate_up" else []),
                    *(["-DGLM_ROUTE_PACKED_INPUT"] if stage == "gate_up" and route_packed_input else []),
                    *(["-DGLM_ROUTE_PACKED_DOWN"] if route_packed_down else []),
                    *(["-DGLM_COMPACT_DOWN_SCALES"] if route_compact_down_scales else []),
                    *(["-DGLM_DIRECT_COMPACT_DOWN_SCALES"] if stage == "down" and direct_compact_down_scales else []),
                    *(["-DGLM_RAW_HIDDEN_SCALES"] if raw_hidden_scales else []),
                    *(["-DGLM_NZ_PREFILL_ACCUMULATOR"] if nz_prefill_accumulator else []),
                    *([f"-DGLM_NZ_PREFILL_MIN_ROWS={nz_prefill_min_rows}"] if nz_prefill_min_rows else []),
                    *(["-DGLM_CACHE_EXPERT_ENDS"] if cache_expert_ends else []),
                    *(["-DGLM_PREFILL_PRODUCT_CAST"] if prefill_product_cast else []),
                    *(["-DGLM_DIRECT_HIDDEN_GATHER"] if stage == "gate_up" and direct_hidden_gather else []),
                    *(["-DGLM_SHARE_GATE_UP_INPUT"] if stage == "gate_up" and share_gate_up_input else []),
                    *(["-DGLM_CACHE_GATE_UP_ACTIVATIONS"] if stage == "gate_up" and cache_gate_up_activations else []),
                ],
                check=True,
            )
            provenance[output.name] = {"source_sha256": sha256(source), "binary_sha256": sha256(output)}
        provenance["glm_fused_scratch.h"] = {"source_sha256": sha256(HERE / "glm_fused_scratch.h")}
        provenance["glm_fused_compact_scales.h"] = {"source_sha256": sha256(HERE / "glm_fused_compact_scales.h")}
        provenance["glm_fused_route_cache.h"] = {"source_sha256": sha256(HERE / "glm_fused_route_cache.h")}
        provenance["glm_route_input_layout.h"] = {"source_sha256": sha256(HERE / "glm_route_input_layout.h")}
        if route_packed_input:
            source = HERE / "glm_fused_route_input.cpp"
            output = build_dir / "glm_fused_route_input.bin"
            subprocess.run(
                [
                    str(compiler),
                    str(source),
                    str(output),
                    *options,
                    f"-I{HERE}",
                    *(["-DGLM_RAW_INPUT_SCALES"] if raw_input_scales else []),
                ],
                check=True,
            )
            provenance[output.name] = {"source_sha256": sha256(source), "binary_sha256": sha256(output)}
        source = HERE / "glm_fused_pack.cpp"
        output = build_dir / "glm_fused_pack.bin"
        subprocess.run(
            [
                str(compiler),
                str(source),
                str(output),
                *options,
                f"-I{HERE}",
                *(["-DGLM_RAW_INPUT_SCALES"] if raw_input_scales else []),
            ],
            check=True,
        )
        provenance[output.name] = {"source_sha256": sha256(source), "binary_sha256": sha256(output)}
        provenance["glm_fused_quantize.h"] = {"source_sha256": sha256(HERE / "glm_fused_quantize.h")}
        source = HERE / "glm_fused_reduce.cpp"
        output = build_dir / "glm_fused_reduce.bin"
        subprocess.run(
            [
                str(compiler),
                str(source),
                str(output),
                *options,
                *(["-DGLM_FP16_ROUTE_WORKSPACE"] if fp16_route_workspace else []),
                f"-I{HERE}",
                *(["-DGLM_NATIVE_ROUTE_COLUMNS"] if native_route_columns else []),
                *(["-DGLM_PREFILL_REDUCE_META_CACHE"] if prefill_reduce_meta_cache else []),
            ],
            check=True,
        )
        provenance[output.name] = {"source_sha256": sha256(source), "binary_sha256": sha256(output)}
        provenance["glm_fused_reduce_schedule.h"] = {"source_sha256": sha256(HERE / "glm_fused_reduce_schedule.h")}
    if include_w3:
        csrc = source_root / "csrc" if (source_root / "csrc").is_dir() else source_root
        kernel = csrc / "gmm/w2_blocked_dequant_matmul_v310/op_kernel"
        common = csrc / "moe/common"
        catlass = source_root / "third_party/catlass/include"
        headers = ("w2_blocked_dequant_matmul_v310.h", "row_reuse_geometry.h", "compat_310p.h")
        snapshot = build_dir / "decoder"
        snapshot.mkdir()
        for name in headers:
            data = (kernel / name).read_bytes()
            (snapshot / name).write_bytes(data)
            provenance[name] = {"source_sha256": hashlib.sha256(data).hexdigest()}
        extra = [f"-I{snapshot}", f"-I{common}", f"-I{catlass}", *[f"-D{name}" for name in W3_DEFINES]]
        source = HERE / "reconstruction_kernel.cpp"
        output = build_dir / "reconstruction_kernel.bin"
        subprocess.run([str(compiler), str(source), str(output), *options, *extra], check=True)
        provenance[source.name] = {"source_sha256": sha256(source), "binary_sha256": sha256(output)}
    spec = importlib.util.find_spec("torch_npu")
    if spec is None or spec.origin is None:
        raise RuntimeError("build the bridge with the serving torch-npu environment")
    npu_root = Path(spec.origin).parent
    includes = [*include_paths(), str(npu_root / "include"), str(cann_root / "include"), sysconfig.get_path("include")]
    libraries = [*library_paths(), str(npu_root / "lib"), str(cann_root / "lib64")]
    bridge = build_dir / f"glm_reconstruction_bridge_v{version}.so"
    subprocess.run(
        [
            "c++",
            "-shared",
            "-fPIC",
            "-std=c++20",
            "-O2",
            f"-DGLM_RECONSTRUCTION_NAMESPACE={namespace}",
            f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
            *[f"-I{path}" for path in includes],
            str(HERE / "reconstruction_bridge.cpp"),
            *[f"-L{path}" for path in libraries],
            *[f"-Wl,-rpath,{path}" for path in libraries],
            "-ltorch_npu",
            "-lascendcl",
            "-lc10",
            "-ltorch_cpu",
            "-ltorch",
            "-ltorch_python",
            "-o",
            str(bridge),
        ],
        check=True,
    )
    provenance["reconstruction_bridge.cpp"] = {
        "source_sha256": sha256(HERE / "reconstruction_bridge.cpp"),
        "binary_sha256": sha256(bridge),
    }
    (build_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return build_dir


def main():
    """Expose opt-in schedules and freeze their full configuration in a unique build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--cann-root", type=Path, default=Path("/usr/local/Ascend/ascend-toolkit/latest"))
    parser.add_argument("--source-root", type=Path, default=HERE.parents[1])
    parser.add_argument("--include-w3", action="store_true")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--output-columns", type=int, choices=(16, 32, 64, 128), default=16)
    parser.add_argument("--tile-pipeline", action="store_true")
    parser.add_argument("--all-bits", action="store_true")
    parser.add_argument("--fused-moe", action="store_true")
    parser.add_argument("--prepared-weight-layout", action="store_true")
    parser.add_argument("--pair-scale-groups", action="store_true")
    parser.add_argument("--fp16-swiglu", action="store_true", help="quality-gated FP16 SwiGLU experiment")
    parser.add_argument("--pair-hidden-quant", action="store_true", help="quality-gated paired hidden quantization")
    parser.add_argument(
        "--wide-cube-k",
        type=int,
        choices=(0, 128, 256),
        default=0,
        help="pack independent block32 dots into a wider native Cube (0 disables)",
    )
    parser.add_argument(
        "--pair-prefill-scale-groups",
        action="store_true",
        help="use M32 for paired prefill block32 dots; preserve FP32 accumulation order",
    )
    parser.add_argument("--weight-decode-lut", action="store_true", help="exact W2/W3 reconstruction with byte lookup")
    parser.add_argument("--strided-product-copy", action="store_true", help="contiguous sparse Cube product casts")
    parser.add_argument("--repeat-product-cast", action="store_true", help="cast sparse Cube rows across N16 strips")
    parser.add_argument("--w3-float-fragments", action="store_true")
    parser.add_argument("--specialize-w3", action="store_true")
    parser.add_argument("--prefill-weight-cache", action="store_true")
    parser.add_argument("--fp16-route-workspace", action="store_true")
    parser.add_argument(
        "--prerounded-weight-scales", action="store_true", help="require permanent FP16-rounded FP32 scale banks"
    )
    parser.add_argument(
        "--fp16-weight-scales", action="store_true", help="FP16 expert scale storage with FP32 scale computation"
    )
    parser.add_argument(
        "--native-route-columns", action="store_true", help="permute columns once after stable route reduction"
    )
    parser.add_argument(
        "--share-gate-up-input", action="store_true", help="reuse activation scales and routes from gate in up"
    )
    parser.add_argument(
        "--cache-gate-up-activations",
        action="store_true",
        help="reuse gate's packed L1 activations in up within each batch",
    )
    parser.add_argument(
        "--vector-scale-products", action="store_true", help="batch prefill scale products without scalar row reads"
    )
    parser.add_argument(
        "--gather-product-matrix", action="store_true", help="gather bulk Cube rows before one contiguous cast"
    )
    parser.add_argument(
        "--prefill-rows-32",
        action="store_true",
        help="use up to 31 expert rows with shared scratch; K64 products, unchanged 16-token route ABI",
    )
    parser.add_argument(
        "--quad-hidden-quant",
        action="store_true",
        help="quantize four gate/up output rows per pass; keep block32 scales independent",
    )
    parser.add_argument(
        "--direct-hidden-gather",
        action="store_true",
        help="write permuted hidden rows directly into quantizer scratch",
    )
    parser.add_argument(
        "--route-packed-input",
        action="store_true",
        help="pack routed A4 rows and compact scales once before all gate/up output tiles",
    )
    parser.add_argument(
        "--route-packed-down",
        action="store_true",
        help="write bulk A4 hidden codes directly in the paired layout consumed by down",
    )
    parser.add_argument(
        "--route-compact-down-scales",
        action="store_true",
        help="write four FP32 hidden scales per tile row and bulk-load them in down",
    )
    parser.add_argument(
        "--raw-hidden-scales",
        action="store_true",
        help="copy scalar FP32 scales from four-row quantizer directly to the down bank",
    )
    parser.add_argument(
        "--raw-input-scales",
        action="store_true",
        help="retain scalar FP32 input scales and copy routed rows without broadcast gathers",
    )
    parser.add_argument(
        "--nz-prefill-accumulator",
        action="store_true",
        help="accumulate dense A4 prefill in native Cube layout with unique FP32 scale factors",
    )
    parser.add_argument(
        "--nz-prefill-min-rows",
        type=int,
        default=0,
        help="NZ accumulator active-row threshold: 0 preserves legacy 17; 31 selects full route batches",
    )
    parser.add_argument(
        "--cache-expert-ends", action="store_true", help="cache bulk expert boundaries in existing UB scratch"
    )
    parser.add_argument(
        "--prefill-reduce-meta-cache",
        action="store_true",
        help="reuse route positions and weights across a balanced token's output tiles",
    )
    parser.add_argument(
        "--direct-compact-down-scales",
        action="store_true",
        help="retain dense down activation scales in their compact producer layout",
    )
    parser.add_argument(
        "--prefill-product-cast",
        action="store_true",
        help="cast paired dense prefill Cube output contiguously before row copies",
    )
    parser.add_argument(
        "--fused-scale-accumulation",
        action="store_true",
        help="fuse A4 prefill product scaling and FP32 accumulation; changes FP32 rounding",
    )
    parser.add_argument(
        "--compact-w4-scratch",
        action="store_true",
        help="add paired W4 entries with reconstruction-only scratch removed",
    )
    parser.add_argument(
        "--active-cube-rows", action="store_true", help="compute only populated M16 row blocks in paired A4 prefill"
    )
    parser.add_argument(
        "--direct-w4-l1", action="store_true", help="load prepared W4 weights into L1 without UB staging"
    )
    parser.add_argument(
        "--prepared-offset-tables",
        action="store_true",
        help="DMA immutable gather tables prepared with each projection config",
    )
    parser.add_argument("--product-pipe-events", action="store_true", help="use explicit M/V readback dependencies")
    parser.add_argument("--bulk-route-store", action="store_true", help="store FP16 route rows with one strided DMA")
    args = parser.parse_args()
    print(
        build(
            args.build_dir,
            args.cann_root,
            args.source_root,
            args.include_w3,
            args.version,
            args.output_columns,
            args.tile_pipeline,
            args.all_bits,
            args.fused_moe,
            args.prepared_weight_layout,
            args.pair_scale_groups,
            args.fp16_swiglu,
            args.pair_hidden_quant,
            args.wide_cube_k,
            args.pair_prefill_scale_groups,
            args.weight_decode_lut,
            args.strided_product_copy,
            args.repeat_product_cast,
            args.w3_float_fragments,
            args.specialize_w3,
            args.prefill_weight_cache,
            args.fp16_route_workspace,
            args.share_gate_up_input,
            args.cache_gate_up_activations,
            args.vector_scale_products,
            args.gather_product_matrix,
            args.prefill_rows_32,
            args.quad_hidden_quant,
            args.direct_hidden_gather,
            args.route_packed_input,
            args.route_packed_down,
            args.route_compact_down_scales,
            args.raw_hidden_scales,
            args.raw_input_scales,
            args.nz_prefill_accumulator,
            args.prefill_product_cast,
            args.native_route_columns,
            args.prerounded_weight_scales,
            args.compact_w4_scratch,
            args.fp16_weight_scales,
            args.active_cube_rows,
            args.direct_w4_l1,
            args.prepared_offset_tables,
            args.product_pipe_events,
            args.bulk_route_store,
            args.fused_scale_accumulation,
            nz_prefill_min_rows=args.nz_prefill_min_rows,
            cache_expert_ends=args.cache_expert_ends,
            prefill_reduce_meta_cache=args.prefill_reduce_meta_cache,
            direct_compact_down_scales=args.direct_compact_down_scales,
        )
    )


if __name__ == "__main__":
    main()
