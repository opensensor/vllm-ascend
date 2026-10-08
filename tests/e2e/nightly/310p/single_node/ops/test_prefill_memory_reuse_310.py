# SPDX-License-Identifier: Apache-2.0
"""Real-weight paired gates; neither candidate is loaded into a serving engine.

Run directly with --baseline-build-dir, --candidate-build-dir and --checkpoint.
The checkpoint must be the canonical selective W3 model, not the native bundle.
"""

import argparse
import json
from pathlib import Path

import pytest
import torch

from tools.glm_perf.fused_moe_profile import frozen_helper
from tools.glm_perf.glm_int4 import unpack_canonical_codes


class BuildOptions:
    def pytest_addoption(self, parser):
        parser.addoption("--baseline-build-dir")
        parser.addoption("--candidate-build-dir")
        parser.addoption("--checkpoint")
        parser.addoption("--decode-build-dir")


@pytest.fixture(scope="module")
def bundles(pytestconfig):
    paths = [
        pytestconfig.getoption(name, default=None)
        for name in ("--baseline-build-dir", "--candidate-build-dir", "--checkpoint")
    ]
    if any(path is None for path in paths):
        pytest.skip("requires two independent compiled bundles and real checkpoint")
    import torch_npu

    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    baseline, candidate = [frozen_helper(Path(path).resolve(strict=True)) for path in paths[:2]]
    flags = (
        "fp16_route_workspace",
        "prefill_weight_cache",
        "share_gate_up_input",
        "cache_gate_up_activations",
        "vector_scale_products",
        "gather_product_matrix",
        "prefill_rows_32",
        "quad_hidden_quant",
        "direct_hidden_gather",
        "route_packed_input",
        "route_packed_down",
        "route_compact_down_scales",
        "raw_hidden_scales",
        "raw_input_scales",
        "nz_prefill_accumulator",
        "prefill_product_cast",
    )
    assert any(candidate[1].get(flag, False) != baseline[1].get(flag, False) for flag in flags)
    assert baseline[1]["namespace"] != candidate[1]["namespace"]
    assert baseline[1]["prepared_weight_layout"] and candidate[1]["prepared_weight_layout"]
    return list(zip(paths[:2], (baseline, candidate))), Path(paths[2])


@pytest.fixture(scope="module", params=[10, 11, 33], ids=["real-W3", "real-W4", "real-W2"])
def real_weights(request, bundles):
    from safetensors import safe_open

    _, checkpoint = bundles
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.language_model.layers.{request.param}.mlp.experts.0."
    projections = []
    bits = []
    for name in ("gate", "up", "down"):
        keys = [prefix + name + "_proj" + suffix for suffix in ("_codes", "_scale")]
        values = []
        for key in keys:
            with safe_open(str(checkpoint / index[key]), framework="pt", device="cpu") as handle:
                values.append(handle.get_tensor(key))
        codes, scales = values
        k = scales.shape[1] * 32
        projections.append((unpack_canonical_codes(codes, k), scales.float()))
        bits.append(codes.shape[1] * 8 // k)
    expected = {10: 3, 11: 4, 33: 2}[request.param]
    assert bits == [expected] * 3
    gate = torch.cat((projections[0][0], projections[1][0]), dim=0)[None]
    scales = torch.cat((projections[0][1], projections[1][1]), dim=0)[None]
    return expected, gate, scales, projections[2][0][None], projections[2][1][None]


@pytest.mark.parametrize("activation_bits", [4, 8])
@pytest.mark.parametrize("tokens", [2, 15, 16, 17, 30, 31, 32, 33, 62, 63, 640])
def test_real_multibatch_exact_replay(bundles, real_weights, activation_bits, tokens, pytestconfig):
    builds, _ = bundles
    bits, gate, gs, down, ds = real_weights
    generator = torch.Generator().manual_seed(7321 + bits + tokens)
    x = torch.randn(tokens, gate.shape[-1], generator=generator).half().npu()
    # Duplicates exercise stable expert ordering. Mixed zero/peer slots and
    # all-peer replay must never consume unwritten half workspace rows.
    ids = torch.zeros(tokens, 2, dtype=torch.int64).npu()
    ids[::3, 1] = 1
    weights = torch.rand(tokens, 2, generator=generator).npu()
    weights[::4, 0] = 0
    gc, dc, ggs, gds = None, None, gs.npu(), ds.npu()
    graphs, outputs, natives = [], [], []
    for directory, (helper, options) in builds:
        native = helper.NativeFusedMoE(
            Path(directory),
            namespace=options["namespace"],
            activation_bits=activation_bits,
            prepared_weight_layout=True,
            **({"weight_decode_lut": True} if options.get("weight_decode_lut") else {}),
            **({"fp16_route_workspace": True} if options.get("fp16_route_workspace") else {}),
        )
        decode_path = pytestconfig.getoption("--decode-build-dir", default=None)
        if decode_path and directory == builds[-1][0]:
            from tools.glm_perf.resident_candidates.prefill_decode import PrefillDecodeNative

            decode_helper, decode_options = frozen_helper(Path(decode_path).resolve(strict=True))
            decode = decode_helper.NativeFusedMoE(
                Path(decode_path),
                namespace=decode_options["namespace"],
                activation_bits=activation_bits,
                prepared_weight_layout=True,
                **({"fp16_route_workspace": True} if decode_options.get("fp16_route_workspace") else {}),
            )
            native = PrefillDecodeNative(native, decode)
        if gc is None:
            gc, dc = native.pack_weight_codes(gate, bits).npu(), native.pack_weight_codes(down, bits).npu()
        # Warm descriptors before capture. This test does not claim cold capture.
        native(x, gc, ggs, dc, gds, weights, ids)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = native(x, gc, ggs, dc, gds, weights, ids)
        graphs.append(graph)
        outputs.append(output)
        natives.append(native)

    def compare():
        for graph in graphs:
            graph.replay()
        torch.npu.synchronize()
        assert torch.equal(outputs[0].cpu(), outputs[1].cpu())

    compare()
    x.mul_(1.25)
    weights.mul_(0.75)
    ids.zero_()  # All rows now select one hot expert, including partial tails.
    gc.copy_(natives[0].pack_weight_codes(torch.roll(gate, 1, -1), bits).npu())
    dc.copy_(natives[0].pack_weight_codes(torch.roll(down, 2, -1), bits).npu())
    ggs.mul_(1.03125)
    gds.mul_(0.96875)
    compare()
    ids.fill_(1)
    compare()
    assert torch.equal(outputs[1].cpu(), torch.zeros(tokens, down.shape[1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-build-dir", required=True)
    parser.add_argument("--candidate-build-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    args, extra = parser.parse_known_args()
    raise SystemExit(
        pytest.main(
            [
                __file__,
                "--noconftest",
                "-q",
                "--baseline-build-dir",
                args.baseline_build_dir,
                "--candidate-build-dir",
                args.candidate_build_dir,
                "--checkpoint",
                args.checkpoint,
                *extra,
            ],
            plugins=[BuildOptions()],
        )
    )
