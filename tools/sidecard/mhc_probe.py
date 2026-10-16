# SPDX-License-Identifier: Apache-2.0
"""Offline-built mHC candidate gates on owner-local NPU; never launches a service.

Requires an exclusive hardware lease, assigned ASCEND_RT_VISIBLE_DEVICES=2,3,
and a T11-built ops_library in --config. The config uses existing checkpoint,
ops_library and checkpoint_sha256 keys; optional mhc_layers defaults to 18/19.
This command records native steady-state timings only, with CPU fingerprints
outside all timing regions. It does not qualify model accuracy or promote rows.
"""

import argparse
import hashlib
import json
import os
import statistics
import time
from functools import partial
from pathlib import Path

from .execution_options import load_native, validate_options
from .layer_weights import load_dense
from .npu_mhc import NUMERICAL_CONTRACT, post_reference, pre_reference, prepare_weights, resolve_ops

ROWS = (1, 4, 32)
CASES = ((0.0, False), (0.125, False), (1.0, True), (100000.0, True), (-100000.0, True))


def summarize(report, requested):
    """Qualify each requested operation/row independently using fixed bounds."""
    tested_rows = sorted({row["rows"] for row in report["cases"]})
    for operation in ("pre", "post"):
        numerical = [
            rows
            for rows in tested_rows
            if all(
                row["pre"]["passed"] if operation == "pre" else row["post_passed"]
                for row in report["cases"]
                if row["rows"] == rows
            )
        ]
        report[f"numerically_eligible_{operation}_rows"] = numerical
        report[f"probe_eligible_{operation}_rows"] = [
            rows
            for rows in numerical
            if any(row["rows"] == rows for row in report["measurements"])
            and all(
                row[f"candidate_{operation}"]["median_host_microseconds"]
                < row[f"baseline_{operation}"]["median_host_microseconds"]
                for row in report["measurements"]
                if row["rows"] == rows
            )
        ]
    report["requested_rows"] = requested
    report["passed"] = any(requested.values()) and all(
        set(rows) <= set(report[f"numerically_eligible_{operation}_rows"]) for operation, rows in requested.items()
    )


def byte_equal(actual, expected, torch):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return False
    bits = torch.int16 if actual.dtype == torch.float16 else torch.int32
    return torch.equal(actual.contiguous().view(bits).cpu(), expected.contiguous().view(bits).cpu())


def compare(actual, expected, torch):
    """Finite outputs and fixed contracts; this gate cannot tune its bounds."""
    fields = {}
    for name, got, wanted in zip(("post", "comb", "hidden", "raw_logits"), actual, expected, strict=True):
        finite = bool(got.isfinite().all()) and bool(wanted.isfinite().all())
        same_metadata = got.shape == wanted.shape and got.dtype == wanted.dtype
        close = False
        if finite and same_metadata:
            if name == "raw_logits":
                close = bool(torch.allclose(got, wanted, **NUMERICAL_CONTRACT[name]))
            else:
                close = byte_equal(got, wanted, torch)
        fields[name] = dict(
            passed=finite and same_metadata and close,
            finite=finite,
            same_metadata=same_metadata,
            max_abs_error=float((got.float() - wanted.float()).abs().max()) if same_metadata else None,
        )
    return dict(passed=all(value["passed"] for value in fields.values()), fields=fields)


def timing(torch, operation, warmups=5, repeats=25):
    for _ in range(warmups):
        operation()
    torch.npu.synchronize()
    measurements = []
    for _ in range(repeats):
        torch.npu.synchronize()
        begin = time.perf_counter()
        operation()
        torch.npu.synchronize()
        measurements.append((time.perf_counter() - begin) * 1e6)
    return dict(median_host_microseconds=statistics.median(measurements), repeats=repeats, warmups=warmups)


def complete_pre(operation, inputs, sinkhorn, iterations, hc_eps):
    post, logits, hidden = operation(*inputs)
    return post, sinkhorn(logits.contiguous(), iterations, hc_eps).bfloat16().float(), hidden


def run(config, output, rank):
    if type(rank) is not int or rank not in (0, 1) or os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "2,3":
        raise ValueError("mHC probe requires leased physical chips 2,3 with local rank 0/1")
    spec = json.loads(config.read_bytes())
    execution = validate_options(spec)
    layers = spec.get("mhc_layers", [18, 19])
    if (
        not isinstance(layers, list)
        or not layers
        or any(type(layer) is not int or not 18 <= layer < 33 for layer in layers)
        or len(set(layers)) != len(layers)
    ):
        raise ValueError("mhc_layers must be unique middle layers 18 through 32")
    if output.exists():
        raise ValueError("mHC report already exists; retain prior evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    library = Path(spec.get("npu_ops_library", spec.get("ops_library", "")))
    if not library.is_absolute() or not library.is_file():
        raise ValueError("mHC probe requires an existing absolute base ops library")
    requested = (
        {"pre": sorted(execution.mhc_pre_rows), "post": sorted(execution.mhc_post_rows)}
        if "execution_candidates" in spec
        else {"pre": list(ROWS), "post": list(ROWS)}
    )
    if not any(requested.values()):
        raise ValueError("mHC component gate requires at least one selected pre/post row policy")
    test_rows = sorted(set(ROWS) | set(requested["pre"]) | set(requested["post"]))
    import torch
    import torch_npu

    report = dict(
        passed=False,
        rank=rank,
        physical_chip=rank + 2,
        checkpoint_sha256=spec["checkpoint_sha256"],
        library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        numerical_contract=NUMERICAL_CONTRACT,
        full_model_validated=False,
        qualified_rows={"pre": [], "post": []},
        cases=[],
        measurements=[],
    )
    try:
        torch.set_num_threads(2)
        torch.npu.set_device(rank)
        torch_npu.npu.set_compile_mode(jit_compile=False)
        torch.ops.load_library(str(library))
        _, native_manifest = load_native(spec, torch)
        if native_manifest is not None:
            report["native_build_sha256"] = execution.native_build.manifest_sha256
        native_pre, native_post = resolve_ops()
        sinkhorn = torch.ops._C_ascend.mhc_sinkhorn_310
        model = json.loads((Path(spec["checkpoint"]) / "config.json").read_bytes())["text_config"]
        norm_eps, hc_eps, iterations = model["rms_norm_eps"], model["hc_eps"], model["hc_sinkhorn_iters"]
        with torch.inference_mode():
            for layer in layers:
                loaded, _ = load_dense(spec["checkpoint"], (layer,), f"npu:{rank}", torch, tp_rank=rank)
                for kind, norm_name in (("attn", "input_layernorm.weight"), ("ffn", "post_attention_layernorm.weight")):
                    prepared = prepare_weights(loaded[layer], kind, norm_name)
                    for rows in test_rows:
                        for case, (scale, changed) in enumerate(CASES):
                            torch.manual_seed(937 + 17 * layer + case)
                            residual = (
                                (torch.randn(rows, 4, 4096, device=f"npu:{rank}") * scale).to(torch.bfloat16).float()
                            )
                            fn, coef, base, norm = prepared
                            if changed:
                                # Opposite signed coefficients and nonuniform logits expose orientation/rounding bugs.
                                coef = -coef * 0.75
                                base = base + torch.linspace(-2, 2, 24, device=base.device)
                            inputs = (residual, fn, coef, base, norm, norm_eps, hc_eps)
                            expected_post, expected_logits, expected_hidden = pre_reference(*inputs, torch)
                            post, logits, hidden = native_pre(*inputs)
                            expected_comb = (
                                sinkhorn(expected_logits.contiguous(), iterations, hc_eps).bfloat16().float()
                            )
                            comb = sinkhorn(logits.contiguous(), iterations, hc_eps).bfloat16().float()
                            result = compare(
                                (post, comb, hidden, logits),
                                (expected_post, expected_comb, expected_hidden, expected_logits),
                                torch,
                            )
                            # Arbitrary matrices and signed post coefficients separately test post, so a pre failure
                            # cannot contaminate the post candidate's arithmetic qualification.
                            matrix = torch.randn(rows, 4, 4, device=residual.device) * 0.75
                            gates = torch.randn(rows, 4, 1, device=residual.device)
                            x = torch.randn(rows, 4096, device=residual.device).half()
                            reference = post_reference(x, residual, gates, matrix, torch)
                            actual = native_post(x, residual, gates, matrix)
                            post_passed = bool(actual.isfinite().all()) and byte_equal(actual, reference, torch)
                            report["cases"].append(
                                dict(
                                    layer=layer,
                                    kind=kind,
                                    rows=rows,
                                    scale=scale,
                                    changed_coefficients=changed,
                                    pre=result,
                                    post_passed=post_passed,
                                    post_max_abs_error=float((actual - reference).abs().max()),
                                )
                            )
                        # The unchanged, finite real-weight fixture times all pre outputs, including Sinkhorn/comb.
                        # Use an ordinary-scale fixture rather than the final range-stress case.
                        timing_residual = (
                            (torch.randn(rows, 4, 4096, device=residual.device) * 0.125).bfloat16().float()
                        )
                        timing_inputs = (timing_residual, *prepared, norm_eps, hc_eps)
                        baseline_pre = partial(
                            complete_pre, pre_reference, (*timing_inputs, torch), sinkhorn, iterations, hc_eps
                        )
                        candidate_pre = partial(complete_pre, native_pre, timing_inputs, sinkhorn, iterations, hc_eps)
                        report["measurements"].append(
                            dict(
                                layer=layer,
                                kind=kind,
                                rows=rows,
                                residual_scale=0.125,
                                baseline_pre=timing(torch, baseline_pre),
                                candidate_pre=timing(torch, candidate_pre),
                                baseline_post=timing(
                                    torch, partial(post_reference, x, timing_residual, gates, matrix, torch)
                                ),
                                candidate_post=timing(torch, partial(native_post, x, timing_residual, gates, matrix)),
                            )
                        )
                del loaded
                torch.npu.synchronize()
                torch.npu.empty_cache()
        # Component row eligibility is distinct from full-model/deployment qualification.
        summarize(report, requested)
    except Exception as error:
        report["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(passed=report["passed"], cases=len(report["cases"]), full_model_validated=False)), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    if not run(args.config, args.output, args.rank)["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
