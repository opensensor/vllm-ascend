"""Isolate repeatability of native GDN kernels on frozen private inputs."""

import argparse
import json
from pathlib import Path

from tools.qwen4exp.apply_ple_serving import apply
from tools.sidecard.qfn_resident_server import QfnResidentClient

SOURCE = """
def operation(wrapper):
    import torch
    import torch_npu
    from types import FunctionType
    from tools.qwen4exp.wy_graph import WYGraph
    from vllm_ascend._310p.ops.fla.chunk_gated_delta_rule import chunk_gated_delta_rule_310

    worker = getattr(wrapper, "worker", wrapper)
    result = dict(rank=worker.rank, passed=True)
    try:
        assert worker.model_runner.execute_model_state is None
        owner = worker._qfn_wy_graph
        assert not owner.enabled and not owner.failed
        candidate = WYGraph(owner.record)
        candidate.verify = False
        values = tuple(value.clone() for value in owner.record.inputs)

        def function(callback):
            fn = FunctionType(
                chunk_gated_delta_rule_310.__code__,
                dict(chunk_gated_delta_rule_310.__globals__, _compute_kernel_inputs_from_torch_wy=callback),
                chunk_gated_delta_rule_310.__name__,
                chunk_gated_delta_rule_310.__defaults__,
                chunk_gated_delta_rule_310.__closure__,
            )
            fn.__kwdefaults__ = chunk_gated_delta_rule_310.__kwdefaults__
            return fn

        def info(tensor):
            return dict(
                shape=list(tensor.shape),
                stride=list(tensor.stride()),
                format=torch_npu.get_npu_format(tensor),
                offset=tensor.storage_offset(),
            )

        base_inputs = owner.original(*values, 64)
        replay_inputs = candidate(*values, 64)
        result["inputs"] = [
            dict(reference=info(a), candidate=info(b), exact=torch.equal(a, b))
            for a, b in zip(base_inputs, replay_inputs)
        ]
        state = torch.zeros(1, 12, 128, 128, device=values[0].device)

        def compare(a, b):
            return dict(
                exact=torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16)),
                nan_a=torch.isnan(a).sum().item(),
                nan_b=torch.isnan(b).sum().item(),
                max_abs=(a.float() - b.float()).abs().max().item(),
                mismatches=(a != b).sum().item(),
            )

        q, k, w, u, g = base_inputs

        def fwd_h():
            return torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h(
                k,
                w,
                u,
                g=g,
                gk=None,
                initial_state=state,
                output_final_state=True,
                chunk_size=64,
                save_new_value=True,
                cu_seqlens=None,
                chunk_indices=None,
                use_exp2=False,
                transpose_state_layout=False,
            )

        h, vn, fs = fwd_h()
        from tools.qwen4exp.gdn_output_native import GDNOutputNative, build_forward
        import hashlib, inspect, textwrap

        native = GDNOutputNative(worker._qfn_gdn_output_probe["kernel"], torch.ops._C_ascend.chunk_fwd_o_vllm, q.device)
        worker._qfn_gdn_output_adapter = native
        cases = []
        for tokens, cu in (
            (64, None),
            (128, None),
            (512, None),
            (2048, None),
            (192, [0, 64, 192]),
            (1024, [0, 512, 1024]),
        ):
            repeats = (tokens + 511) // 512
            qt, kt, vt = [t.repeat(1, 1, repeats, 1)[:, :, :tokens].contiguous() for t in (q, k, vn)]
            ht = h.repeat(1, 1, repeats, 1, 1)[:, :, : tokens // 64].contiguous()
            gt = g.repeat(1, 1, repeats)[:, :, :tokens].contiguous()
            chunks = (
                None
                if cu is None
                else [x for i, (a, b) in enumerate(zip(cu, cu[1:])) for j in range((b - a) // 64) for x in (i, j)]
            )
            actual = native(qt, kt, vt, ht, 128**-0.5, g=gt, cu_seqlens=cu, chunk_indices=chunks).cpu()
            again = native(qt, kt, vt, ht, 128**-0.5, g=gt, cu_seqlens=cu, chunk_indices=chunks).cpu()
            qc, kc, vc, hc, gc = [t.cpu().float() for t in (qt, kt, vt, ht, gt)]
            expected = torch.empty_like(vc)
            for head in range(12):
                for chunk in range(tokens // 64):
                    sl = slice(chunk * 64, (chunk + 1) * 64)
                    qi, ki, vi = qc[0, head // 3, sl], kc[0, head // 3, sl], vc[0, head, sl]
                    gi = gc[0, head, sl]
                    mask = torch.ones(64, 64, dtype=torch.bool).tril()
                    weights = (
                        ((qi @ ki.T) * (gi[:, None] - gi[None, :]).clamp_max(0).exp())
                        .masked_fill(~mask, 0)
                        .half()
                        .float()
                    )
                    expected[0, head, sl] = ((weights @ vi) + gi.exp()[:, None] * (qi @ hc[0, head, chunk])) * 128**-0.5
            cases.append(
                dict(
                    tokens=tokens,
                    cu=cu,
                    repeat=compare(actual, again),
                    reference=compare(expected.half(), actual),
                    allclose=torch.allclose(expected.half(), actual, atol=0.002, rtol=0.02),
                )
            )
        original = chunk_gated_delta_rule_310
        source_hash = hashlib.sha256(textwrap.dedent(inspect.getsource(original)).encode()).hexdigest()
        corrected = build_forward(original, native, source_hash)

        def chunk_fn(callback):
            fn = FunctionType(
                corrected.__code__,
                dict(corrected.__globals__, _compute_kernel_inputs_from_torch_wy=callback),
                corrected.__name__,
                corrected.__defaults__,
                corrected.__closure__,
            )
            fn.__kwdefaults__ = corrected.__kwdefaults__
            return fn

        full = []
        for varlen in (False, True):
            cu = torch.tensor([0, 512], device=q.device, dtype=torch.int32) if varlen else None
            eager = chunk_fn(owner.original)(
                *values, initial_state=state.clone(), output_final_state=True, cu_seqlens=cu
            )
            replay = chunk_fn(candidate)(*values, initial_state=state.clone(), output_final_state=True, cu_seqlens=cu)
            full.append(dict(varlen=varlen, comparison=[compare(a.cpu(), b.cpu()) for a, b in zip(eager, replay)]))
        result.update(cases=cases, full=full, source_sha256=source_hash, status=native.status())
        result["passed"] = (
            all(row["repeat"]["exact"] and row["allclose"] for row in cases)
            and all(all(c["exact"] for c in row["comparison"]) for row in full)
            and native.fallbacks == 0
        )
        native.original_function, native.replacement = original, corrected
        native.receipt = result
    except Exception as error:
        result["error"] = type(error).__name__ + ": " + str(error)
        result["passed"] = False
    import json
    from pathlib import Path

    Path(OUTPUT, f"rank{worker.rank}-gate.json").write_text(json.dumps(result, indent=2))
    return result


class Candidate:
    def validate(self, engine):
        assert engine._pause_complete() and not engine.batch_queue
        assert not engine.scheduler.get_num_unfinished_requests()
        if not hasattr(self, "rows"):
            self.rows = engine.model_executor.collective_rpc(operation)
        return dict(passed=len(self.rows) == 4 and all(r["passed"] for r in self.rows), workers=self.rows)

    def apply(self, engine):
        return self.validate(engine)


def prepare(engine):
    return Candidate()
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    client = QfnResidentClient("http://127.0.0.1:8001", args.run_root / "control.token", timeout=900)
    result = apply(client, f"OUTPUT={str(args.output)!r}\nBINARY={str(args.binary)!r}\n" + SOURCE, args.output)
    print(json.dumps(result["applied"], indent=2))


if __name__ == "__main__":
    main()
