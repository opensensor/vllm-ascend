# SPDX-License-Identifier: Apache-2.0
"""Qualify CANN start/stop around native graph replays in a disposable process."""

import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401


def main():
    root = Path(__file__).resolve().parent
    namespace = {}
    exec(compile((root / "acl-profile.py").read_text(), "<acl-profile>", "exec"), namespace)
    profile = namespace["AclProfile"]()
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.ops.load_library("/home/matteius/experiments/glm-sinkhorn-resident-20261005/glm_sinkhorn_bridge_v1.so")
    kernel = torch.classes.glm_sinkhorn_v1.Kernel(
        "/home/matteius/experiments/glm-score-batch-20261005/score-baseline.bin", "glm_kda_score_probe_v1"
    )
    q = torch.randn(16, 64, 128, dtype=torch.float16, device="npu:0") * 0.088
    k = torch.randn_like(q) * 0.088
    g = torch.zeros_like(q)
    outputs = [torch.empty(16, 64, 64, device="npu:0") for _ in range(2)]

    def run():
        torch.ops.glm_sinkhorn_v1.launch(kernel, [q, k, g] + outputs, 8)

    for _ in range(5):
        run()
    torch.npu.synchronize()
    expected = [x.cpu() for x in outputs]
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    torch.npu.synchronize()
    for cycle in range(2):
        profile.start(root / f"isolated-cycle{cycle}", 0)
        try:
            for _ in range(20):
                graph.replay()
            torch.npu.synchronize()
        finally:
            profile.stop()
        for _ in range(10):
            graph.replay()
        torch.npu.synchronize()
        assert all(torch.equal(a.view(torch.int32), b.cpu().view(torch.int32)) for a, b in zip(expected, outputs))
        print(f"cycle {cycle} profile stop and post-stop replays passed", flush=True)
    (root / "isolated-qualification.json").write_text(
        json.dumps(
            {
                "passed": True,
                "cycles": 2,
                "profiled_graph_replays": 40,
                "post_stop_graph_replays": 20,
                "exact_outputs": True,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
