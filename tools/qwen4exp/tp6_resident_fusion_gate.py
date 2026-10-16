# SPDX-License-Identifier: Apache-2.0
"""Private-tensor W4 gate/pack and HC reuse comparisons; no live bindings."""

import ast
import hashlib
import inspect
from pathlib import Path

import torch

from tools.qwen4exp import tp6_finalize_gate
from tools.qwen4exp.tp6_finalize_gate import MIN_FREE_BYTES, NAMESPACE, measurements, worker_snapshot


def exact_pair(actual, expected):
    if isinstance(actual, tuple):
        return len(actual) == len(expected) and all(exact_pair(a, b) for a, b in zip(actual, expected))
    a, b = actual.cpu(), expected.cpu()
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))
    )


def function_metadata(function):
    result = dict(name=function.__qualname__, file=function.__code__.co_filename)
    try:
        result["source"] = inspect.getsource(function)
    except (OSError, TypeError):
        result["source"] = None
    return result


def run_gate(worker, config, resource_source, owner):
    # Imports are intentionally worker-local, after a no-clear engine drain.
    import torch_npu

    from vllm_ascend.models.qwen4_exp.model import _GatedResidual
    from vllm_ascend.models.qwen4_exp.w4_moe import W4SparseMoE, pack_activation_device, swiglu_pack_activation_device

    if getattr(worker.model_runner, "execute_model_state", None) is not None:
        raise RuntimeError("worker has pending execution")
    if torch.npu.mem_get_info()[0] < MIN_FREE_BYTES:
        raise RuntimeError("insufficient private-tensor headroom")
    owner["before"] = worker_snapshot(worker)
    owner["namespace"] = namespace = {}
    exec(compile(resource_source, "<tp6-private-fusion-resources>", "exec"), namespace)
    modules = [module for wrapper in worker._resident_wrappers() for module in wrapper.runnable.modules()]
    device = worker.model_runner.device
    generator = torch.Generator(device="cpu").manual_seed(71016 + worker.rank)
    owner["rows"] = results = []
    with torch.inference_mode():
        if config["kind"] == "hc":
            modules = [module for module in modules if isinstance(module, _GatedResidual) and module.use_combine]
            if not modules:
                raise ValueError("no retained HC modules")
            owner["live_mix"] = _GatedResidual.mix
            candidate = namespace["make_mix"](_GatedResidual.mix)
            owner["candidate"] = candidate
            # Diagnostic extension only; no live class attribute is changed.
            namespace["DECODE_ROWS"] = (3, 6, 12, 18, 128, 256, 512, 1024)
            for index in (0, len(modules) // 2, len(modules) - 1):
                module = modules[index]
                for rows in namespace["DECODE_ROWS"]:
                    for pattern in ("random", "changed", "zero"):
                        value = torch.randn((rows, 10240), dtype=torch.float16, generator=generator).to(device)
                        block = torch.randn((rows, 2560), dtype=torch.float16, generator=generator).to(device)
                        if pattern == "changed":
                            value *= 0.01
                        elif pattern == "zero":
                            value.zero_()

                        def retained(module=module, value=value, block=block):
                            mixed, residual = module.mix(value)
                            return mixed, module.combine(block, residual)

                        def alternative(module=module, value=value, block=block):
                            mixed, residual = candidate(module, value)
                            return mixed, module.combine(block, residual)

                        if not namespace["eligible"](module, value):
                            raise ValueError("HC candidate would silently fall back")
                        exact = exact_pair(alternative(), retained())
                        record = dict(kind="hc", module=index, rows=rows, pattern=pattern, exact=exact)
                        if exact and pattern == "random" and index == 0:
                            record["timing"] = measurements(dict(retained=retained, candidate=alternative))
                        results.append(record)
            policy = dict(
                mix=_GatedResidual.mix.__qualname__,
                combine=_GatedResidual.combine.__qualname__,
                mix_source=function_metadata(_GatedResidual.mix),
                modules=len(modules),
            )
        elif config["kind"] == "chunk":
            from vllm_ascend.models.qwen4_exp.w4_moe import route_topk

            for key in ("binary", "bridge"):
                if hashlib.sha256(Path(config[key]).read_bytes()).hexdigest() != config[key + "_sha256"]:
                    raise ValueError(f"changed {key}")
            modules = [module for module in modules if isinstance(module, W4SparseMoE)]
            owner["kernel"] = getattr(torch.classes, NAMESPACE).Kernel(config["binary"], "qwen_tp6_gate_pack_v1")
            owner["resource"] = resource = namespace["GatePack"](owner["kernel"], getattr(torch.ops, NAMESPACE).launch)
            owner["state"] = state = dict(mode="enabled", eligible_calls=0)
            owner["chunk"] = chunk = namespace["make_chunk"](W4SparseMoE._forward_grouped_chunk, resource, state)
            for index, module in enumerate(modules):
                for rows in (65, 128, 256):
                    value = torch.randn((rows, 2560), dtype=torch.float16, generator=generator).to(device)
                    weights, ids = route_topk(
                        torch.nn.functional.linear(value, module.gate),
                        module.top_k,
                        renormalize=module.renormalize,
                        routed_scaling_factor=module.routed_scaling_factor,
                    )
                    if not namespace["admitted"](module, value, weights, ids, resource):
                        raise ValueError("complete chunk would silently fall back")

                    def retained(module=module, value=value, weights=weights, ids=ids):
                        return module._forward_grouped_chunk(value, weights, ids)

                    def alternative(module=module, value=value, weights=weights, ids=ids):
                        return chunk(module, value, weights, ids)

                    exact = exact_pair(alternative(), retained())
                    record = dict(kind="chunk", module=index, rows=rows, exact=exact)
                    if exact and index in (0, len(modules) // 2, len(modules) - 1):
                        record["timing"] = measurements(dict(retained=retained, candidate=alternative))
                    results.append(record)
            policy = dict(modules=len(modules), finalizer=modules[0].grouped_finalize, state=state)
        elif config["kind"] == "gate_pack":
            for key in ("binary", "bridge"):
                if hashlib.sha256(Path(config[key]).read_bytes()).hexdigest() != config[key + "_sha256"]:
                    raise ValueError(f"changed {key}")
            modules = [module for module in modules if isinstance(module, W4SparseMoE)]
            if not modules or any(module.grouped_activation != "cann_builtin_fp16" for module in modules):
                raise ValueError("expected the retained CANN FP16 activation policy")
            owner["kernel"] = getattr(torch.classes, NAMESPACE).Kernel(config["binary"], "qwen_tp6_gate_pack_v1")
            owner["resource"] = resource = namespace["GatePack"](owner["kernel"], getattr(torch.ops, NAMESPACE).launch)
            for index in (0, len(modules) // 2, len(modules) - 1):
                module = modules[index]
                gate, down = module.projections["gate_up_proj"], module.projections["down_proj"]
                cases = ((False, 30), (False, 130), (False, 650), (False, 2560), (True, 30), (True, 120))
                for routed, rows in cases:
                    for active in (0, rows // 6, rows):
                        source_rows = rows // 10 if routed else rows
                        value = torch.randn((source_rows, 2560), dtype=torch.float16, generator=generator).to(device)
                        prepared = pack_activation_device(value)
                        if routed:
                            indices = torch.arange(rows, dtype=torch.int32) % module.num_local_experts
                            indices[active:] = -1
                            ends = indices.to(device)
                        else:
                            ends = torch.tensor(
                                [(i + 1) * active // module.num_local_experts for i in range(module.num_local_experts)],
                                dtype=torch.int64,
                                device=device,
                            )

                        def retained(gate=gate, prepared=prepared, ends=ends):
                            projected = gate.native_linear(prepared, ends)
                            return pack_activation_device(torch_npu.npu_swiglu(projected, dim=-1))

                        def fp32_reference(gate=gate, prepared=prepared, ends=ends):
                            return swiglu_pack_activation_device(gate.native_linear(prepared, ends))

                        def alternative(gate=gate, prepared=prepared, ends=ends):
                            return resource(gate, prepared, ends)

                        expected, reference, actual = retained(), fp32_reference(), alternative()
                        exact = exact_pair(actual, expected)
                        record = dict(
                            kind="gate_pack",
                            module=index,
                            rows=rows,
                            routed=routed,
                            active=active,
                            exact=exact,
                            fp32_exact=exact_pair(actual, reference),
                            downstream_exact=exact_pair(
                                down.native_linear(actual, ends), down.native_linear(expected, ends)
                            ),
                        )
                        if exact and index == 0 and rows in (650, 2560):
                            record["timing"] = measurements(dict(retained=retained, candidate=alternative))
                        results.append(record)
            policy = dict(activation=modules[0].grouped_activation, modules=len(modules))
        else:
            raise ValueError("unknown diagnostic kind")
        torch.npu.synchronize()
    owner["rows"] = results
    owner["after"] = worker_snapshot(worker)
    return dict(
        rank=worker.rank,
        passed=owner["before"] == owner["after"],
        before=owner["before"],
        after=owner["after"],
        rows=results,
        policy=policy,
        exact=all(row["exact"] for row in results),
        installed=False,
    )


def engine_source(config):
    if config["kind"] == "hc":
        resources = Path(__file__).with_name("tp6_operand_reuse.py").read_text()
    elif config["kind"] in ("gate_pack", "chunk"):
        source = Path(__file__).with_name("tp6_fusions.py").read_text()
        tree = ast.parse(source)
        resources = "\n\n".join(
            ast.get_source_segment(source, node)
            for node in tree.body
            if not (
                isinstance(node, ast.ImportFrom)
                and node.module in ("tools.qwen4exp.build_tp6_fusions", "tools.qwen4exp.tp6_fusion_sources")
            )
            and not (isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in ("Finalize", "load_resources"))
        )
        if config["kind"] == "chunk":
            resources += "\n" + Path(__file__).with_name("tp6_eager_gate_pack.py").read_text()
    else:
        raise ValueError("unknown diagnostic kind")
    source = Path(__file__).read_text()
    extension = "\n\n".join(
        ast.get_source_segment(source, node)
        for node in ast.parse(source).body
        if not (isinstance(node, ast.ImportFrom) and node.module.startswith("tools.qwen4exp"))
        and not (isinstance(node, ast.FunctionDef) and node.name == "engine_source")
    )
    source = Path(tp6_finalize_gate.__file__).read_text() + "\n" + extension
    return (
        f"CONFIG = {config!r}\nRESOURCE_SOURCE = {resources!r}\nGATE_SOURCE = {source!r}\n"
        + tp6_finalize_gate.ENGINE_SOURCE
    )
