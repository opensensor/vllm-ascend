# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual ACLNN bindings, including changed-input graph replay."""

import ast
import json
from pathlib import Path

import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op


def main():
    root = Path(__file__).resolve().parent
    enable_custom_op()
    torch.ops.load_library(str(root / "glm_decode_flags.so"))
    swiglu = torch.ops._C_ascend.npu_w2_swiglu_310
    combine = torch.ops._C_ascend.npu_w2_route_combine_310
    records = []
    for device in range(4):
        torch.npu.set_device(device)
        torch_npu.npu.set_compile_mode(jit_compile=False)
        for tokens in (1, 2, 4, 8, 9, 63, 640):
            torch.manual_seed(3102026 + tokens)
            rows = tokens * 8
            data = (torch.randn(rows, 4096) * 3).half().npu()
            gate, up = data.chunk(2, dim=-1)
            expected = (torch.nn.functional.silu(gate.float()) * up.float()).half()
            torch.testing.assert_close(swiglu(data).cpu(), expected.cpu(), rtol=1e-3, atol=2e-6)
            routed = torch.randn(rows, 4096, dtype=torch.float16)
            live = rows // 4
            routed[live:] = torch.nan
            inverse = torch.randperm(rows)
            weights = torch.rand(tokens, 8)
            weights[:, -1] = 0
            weights /= weights.sum(1, keepdim=True)
            active = ((inverse < live).reshape(tokens, 8) & (weights != 0)).unsqueeze(-1)
            products = torch.where(active, routed[inverse].reshape(tokens, 8, 4096).double(), 0)
            products *= weights.double().unsqueeze(-1)
            actual = combine(routed.npu(), inverse.npu(), weights.npu(), torch.tensor([live]).npu()).cpu()
            bound = 16 * torch.finfo(torch.float32).eps * products.abs().sum(1) + 1e-7
            assert torch.isfinite(actual).all() and torch.all((actual.double() - products.sum(1)).abs() <= bound)
            records.append({"device": device, "tokens": tokens, "operators_passed": True})
        for tokens in (2, 8):
            data = torch.ones(tokens * 8, 4096, dtype=torch.float16, device="npu")
            routed = torch.ones(tokens * 8, 4096, dtype=torch.float16, device="npu")
            inverse = torch.arange(tokens * 8, device="npu")
            weights = torch.full((tokens, 8), 0.125, device="npu")
            ends = torch.tensor([tokens * 8], device="npu")
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                for _ in range(3):
                    swiglu(data)
                    combine(routed, inverse, weights, ends)
            torch.npu.current_stream().wait_stream(stream)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                activation = swiglu(data)
                reduced = combine(routed, inverse, weights, ends)
            for value, live in ((-3.0, tokens * 8), (0.0, 0), (2.0, tokens * 4)):
                data.fill_(value)
                routed.fill_(value)
                ends.fill_(live)
                graph.replay()
                gate, up = data.chunk(2, dim=-1)
                torch.testing.assert_close(
                    activation.cpu(),
                    (torch.nn.functional.silu(gate.float()) * up.float()).half().cpu(),
                    rtol=1e-3,
                    atol=2e-6,
                )
                expected = torch.full((tokens, 8), value)
                expected.reshape(-1)[live:] = 0
                torch.testing.assert_close(
                    reduced.cpu(), expected.sum(1)[:, None].mul(0.125).expand(tokens, 4096), rtol=0, atol=0
                )
            records.append({"device": device, "graph_tokens": tokens, "changed_inputs_passed": True})
            del graph
        # Run the exact slot helper body, isolated from model/plugin imports.
        source = Path("vllm_ascend/spec_decode/multi_kv_cache_group_proposer.py").read_text()
        definition = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == "compute_packed_draft_slots"
        )
        namespace = {"torch": torch, "PADDING_SLOT_ID": -1}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), "<actual-slot-helper>", "exec"), namespace)
        table = torch.tensor([[3, -1], [10, 11]], dtype=torch.int32, device="npu")
        boundaries = torch.tensor([0, 3, 5], dtype=torch.int32, device="npu")
        positions = torch.tensor([0, -1, 640, 640, 1280, 0], device="npu")
        indices = torch.arange(14, dtype=torch.int32, device="npu")
        actual = namespace["compute_packed_draft_slots"](table, boundaries, positions, 640, indices)
        assert actual.cpu().tolist() == [1920, -1, -1, 7040, -1, -1]
        records.append({"device": device, "slot_boundaries_passed": True})
        torch.npu.synchronize()
        print(json.dumps({"device": device, "passed": True}), flush=True)
    (root / "operator-gates.json").write_text(json.dumps({"passed": True, "cases": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
