#!/usr/bin/env python3
"""Bounded eager splitfuse qualification. No model/runtime import or service changes.

--device cpu checks fixtures/reference only; npu mode is an explicit hardware job.
--source-root is the staged overlay directory containing vllm_ascend/.
--utils supplies the installed vllm_ascend/utils.py for exact mask layout conversion.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch


def load_function(path, name, namespace, class_name=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def reference(query, keys, values, qlens, lengths, window, causal):
    rows = []
    start = 0
    for request, (count, length) in enumerate(zip(qlens, lengths)):
        for j in range(count):
            position = length - count + j
            indices = [
                i
                for i in range(length)
                if (not causal or i <= position) and (window is None or abs(i - position) < window)
            ]
            q = query[start + j].float()
            groups = query.shape[1] // keys[request].shape[1]
            k = keys[request][indices].float().repeat_interleave(groups, dim=1).transpose(0, 1)
            v = values[request][indices].float().repeat_interleave(groups, dim=1).transpose(0, 1)
            scores = (q[:, None, :] @ k.transpose(1, 2)) / math.sqrt(query.shape[-1])
            rows.append((torch.softmax(scores, dim=-1) @ v).squeeze(1))
        start += count
    return torch.stack(rows)


def build_cache(tensors, table, block_size=64):
    blocks = int(table.max()) + 1
    kv_heads, head_dim = tensors[0].shape[1:]
    cache = torch.zeros(blocks, block_size, kv_heads, head_dim, dtype=torch.float16)
    for req, seq in enumerate(tensors):
        for i, item in enumerate(seq):
            cache[int(table[req, i // block_size]), i % block_size] = item
    # Actual 310P dense cache logical layout: [pages, heads*dim/16, block, 16].
    return cache.reshape(blocks, block_size, kv_heads * head_dim // 16, 16).permute(0, 2, 1, 3).contiguous()


def make_case(qlens, lengths, heads, kv_heads, head_dim):
    generator = torch.Generator().manual_seed(127 + sum(lengths))
    query = torch.randn(sum(qlens), heads, head_dim, generator=generator).half() * 0.25
    keys = [torch.randn(n, kv_heads, head_dim, generator=generator).half() * 0.25 for n in lengths]
    values = [torch.randn(n, kv_heads, head_dim, generator=generator).half() * 0.25 for n in lengths]
    width = (max(lengths) + 63) // 64
    table = torch.arange(len(lengths) * width, dtype=torch.int32).flip(0).reshape(len(lengths), width)
    return query, keys, values, table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--utils", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()
    if args.heads < 1 or args.kv_heads < 1 or args.heads % args.kv_heads or args.head_dim % 16:
        parser.error("Require positive GQA head counts and head dimension divisible by 16")
    root = args.source_root / "vllm_ascend"
    paths = [
        root / "_310p/attention/noncausal_mask.py",
        root / "_310p/attention/attention_v1.py",
        root / "_310p/attention/attention_mask.py",
        args.utils,
    ]
    report = dict(
        completed=False,
        passed=False,
        device=args.device,
        geometry={"heads": args.heads, "kv_heads": args.kv_heads, "head_dim": args.head_dim},
        torch=torch.__version__,
        hardware_executed=False,
        hardware_requested=args.device.startswith("npu"),
        cases=[],
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    )
    started = time.time()
    try:
        spec = importlib.util.spec_from_file_location("isolated_noncausal_mask", paths[0])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if args.device.startswith("npu"):
            import torch_npu
        device = torch.device(args.device)
        native = None
        if device.type == "npu":
            torch.npu.set_device(device)
            torch.npu.set_compile_mode(jit_compile=False)
            report["torch_npu"] = torch_npu.__version__
            report["device_name"] = torch.npu.get_device_name(device)
            namespace = {"torch": torch}
            nz = load_function(args.utils, "nd_to_nz_spec", namespace)
            namespace = {
                "torch": torch,
                "torch_npu": torch_npu,
                "noncausal_draft_mask": module.noncausal_draft_mask,
                "nd_to_nz_spec": nz,
                "ACL_FORMAT_FRACTAL_NZ": 29,
            }
            mask_method = load_function(paths[2], "get_noncausal_splitfuse_mask", namespace, "AttentionMaskBuilder310")
            namespace = {
                "torch": torch,
                "torch_npu": torch_npu,
                "get_query_lens_cpu": lambda metadata: metadata.query_lens_cpu,
                "AttentionMaskBuilder310": SimpleNamespace(),
            }
            native = load_function(
                paths[1], "forward_noncausal_chunked_prefill_310", namespace, "AscendAttentionBackendImpl310"
            )
        for qlens, lengths in [([3], [3]), ([3], [9]), ([2, 3], [66, 70]), ([7], [19]), ([3, 7], [66, 74])]:
            for window in [None, 3]:
                case = dict(query_lengths=qlens, kv_lengths=lengths, window=window, runs=[])
                report["cases"].append(case)
                query, keys, values, table = make_case(qlens, lengths, args.heads, args.kv_heads, args.head_dim)
                outputs = {}
                for causal in [False, True]:
                    for perturb in [False, True]:
                        k = [v.clone() for v in keys]
                        v = [v.clone() for v in values]
                        if perturb:
                            future = lengths[0] - qlens[0] + 1
                            k[0][future] += 0.5
                            v[0][future] += 4.0
                        expected = reference(query, k, v, qlens, lengths, window, causal)
                        mask = module.noncausal_draft_mask(
                            qlens, lengths, device=torch.device("cpu"), sliding_window=window, max_key_length=256
                        )
                        if causal:
                            row = 0
                            for count, length in zip(qlens, lengths):
                                for j in range(count):
                                    mask[row + j, length - count + j + 1 :] = -float("inf")
                                row += count
                        if native is None:
                            # Exercise the mask math independently of the scalar allowed-key reference.
                            actual_rows = []
                            row = 0
                            for req, count in enumerate(qlens):
                                padded_k = torch.zeros(max(lengths), args.kv_heads, args.head_dim)
                                padded_v = torch.zeros_like(padded_k)
                                padded_k[: lengths[req]] = k[req].float()
                                padded_v[: lengths[req]] = v[req].float()
                                groups = args.heads // args.kv_heads
                                padded_k = padded_k.repeat_interleave(groups, dim=1)
                                padded_v = padded_v.repeat_interleave(groups, dim=1)
                                scores = torch.einsum(
                                    "thd,khd->thk", query[row : row + count].float(), padded_k
                                ) / math.sqrt(args.head_dim)
                                actual_rows.append(
                                    torch.einsum(
                                        "thk,khd->thd",
                                        torch.softmax(scores + mask[row : row + count, None, :], dim=-1),
                                        padded_v,
                                    )
                                )
                                row += count
                            actual = torch.cat(actual_rows)
                        else:
                            key_cache = torch_npu.npu_format_cast(build_cache(k, table).to(device), 29)
                            value_cache = torch_npu.npu_format_cast(build_cache(v, table).to(device), 29)
                            impl = SimpleNamespace(
                                sinks=None,
                                alibi_slopes=None,
                                key_cache=key_cache,
                                value_cache=value_cache,
                                sliding_window=window,
                                num_heads=args.heads,
                                head_size=args.head_dim,
                                num_kv_heads=args.kv_heads,
                                scale=1 / math.sqrt(args.head_dim),
                            )
                            metadata = SimpleNamespace(
                                query_lens_cpu=torch.tensor(qlens, dtype=torch.int32),
                                num_actual_tokens=sum(qlens),
                                seq_lens=torch.tensor(lengths, dtype=torch.int32, device=device),
                                block_tables=table.to(device),
                            )
                            if causal:
                                namespace["AttentionMaskBuilder310"].get_noncausal_splitfuse_mask = (
                                    lambda *a, mask=mask: torch_npu.npu_format_cast(
                                        nz(mask.to(device)).contiguous(), 29
                                    )
                                )
                            else:
                                namespace["AttentionMaskBuilder310"].get_noncausal_splitfuse_mask = lambda q, k, d, w: (
                                    mask_method(SimpleNamespace(max_seqlen=256), q, k, d, w)
                                )
                            output = torch.full(query.shape, -77, dtype=torch.float16, device=device)
                            native(impl, query.to(device), metadata, output)
                            torch.npu.synchronize(device)
                            report["hardware_executed"] = True
                            actual = output.cpu().float()
                        run = dict(
                            causal=causal,
                            perturb_future=perturb,
                            max_abs_error=float((actual - expected).abs().max()),
                            output_first_row=actual[0, 0, :8].tolist(),
                        )
                        case["runs"].append(run)
                        try:
                            torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
                        except Exception:
                            run.update(actual=actual.tolist(), expected=expected.tolist())
                            raise
                        outputs[causal, perturb] = actual
                noncausal_delta = float((outputs[False, True][0] - outputs[False, False][0]).abs().max())
                causal_delta = float((outputs[True, True][0] - outputs[True, False][0]).abs().max())
                case.update(noncausal_future_delta=noncausal_delta, causal_future_delta=causal_delta)
                assert noncausal_delta > 0.01, "First query did not observe the future draft key/value"
                assert causal_delta < 0.002, "Causal control observed a future draft key/value"
                if len(qlens) > 1:
                    torch.testing.assert_close(
                        outputs[False, True][qlens[0] :], outputs[False, False][qlens[0] :], atol=0.002, rtol=0
                    )
                case["passed"] = True
        report.update(completed=True, passed=True)
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        report["elapsed_s"] = time.time() - started
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: report[k] for k in ["completed", "passed", "device", "hardware_executed", "elapsed_s"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
