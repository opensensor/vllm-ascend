# SPDX-License-Identifier: Apache-2.0
"""Arithmetic and changed-input graph gates for the append-only post mixer.

Run this file with ``--mhc-post-build-dir PATH`` after build_mhc_post.py.
The direct entry point registers the option without modifying shared conftest.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch
import torch_npu


class BuildOptions:
    def pytest_addoption(self, parser):
        parser.addoption("--mhc-post-build-dir")


@pytest.fixture(scope="module")
def native_post(pytestconfig):
    directory = pytestconfig.getoption("--mhc-post-build-dir", default=None)
    if directory is None:
        pytest.skip("requires a separately compiled append-only mHC build")
    root = Path(directory).resolve(strict=True)
    provenance = json.loads((root / "mhc-provenance.json").read_text())
    for name, record in provenance["binaries"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == record["sha256"]
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.ops.load_library(str(root / f"glm_reconstruction_bridge_v{provenance['version']}.so"))
    helper = root / "mhc_post_native.py"
    assert hashlib.sha256(helper.read_bytes()).hexdigest() == provenance["helper_sha256"]
    spec = importlib.util.spec_from_file_location("qualified_mhc_post_test", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NativeMhcPost(root, provenance["namespace"], finish_only=provenance.get("finish_only", False))


def reference(x, residual, post, comb):
    return (torch.einsum("...ij,...ih->...jh", comb, residual) + post * x.float().unsqueeze(1)).half().float()


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("rows", [1, 2, 8, 640])
def test_post_arithmetic_and_changed_input_graph(native_post, rows, dtype):
    generator = torch.Generator().manual_seed(7305)
    x = torch.randn(rows, 4096, generator=generator, dtype=dtype).npu()
    residual = torch.randn(rows, 4, 4096, generator=generator).half().float().npu()
    post = torch.randn(rows, 4, 1, generator=generator).sigmoid().half().float().npu()
    comb = torch.randn(rows, 4, 4, generator=generator).softmax(-1).half().float().npu()
    tolerance = dict(rtol=0, atol=0) if native_post.finish_only else dict(rtol=1e-3, atol=2e-6)
    torch.testing.assert_close(
        native_post(x, residual, post, comb).cpu(), reference(x, residual, post, comb).cpu(), **tolerance
    )
    native_post.configs.clear()  # First-use metadata must also be capture-safe.
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = native_post(x, residual, post, comb)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(result.cpu(), reference(x, residual, post, comb).cpu(), **tolerance)
    x.mul_(1.125)
    residual.copy_((residual * 0.875).half().float())
    post.copy_((post * 0.75).half().float())
    comb.copy_(comb.flip(-1).contiguous())
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(result.cpu(), reference(x, residual, post, comb).cpu(), **tolerance)


def test_fp32_expert_input_is_not_rounded_before_mixing(native_post):
    x = torch.full((1, 4096), 1 + 2**-11 + 2**-22, dtype=torch.float32, device="npu")
    residual = torch.zeros((1, 4, 4096), dtype=torch.float32, device="npu")
    residual[:, 0, :] = -0.5
    post = torch.ones((1, 4, 1), dtype=torch.float32, device="npu")
    comb = torch.zeros((1, 4, 4), dtype=torch.float32, device="npu")
    comb[:, 0, :] = 1
    expected = reference(x, residual, post, comb).cpu()
    assert torch.equal(native_post(x, residual, post, comb).cpu(), expected)
    assert not torch.equal(native_post(x.half(), residual, post, comb).cpu(), expected)


def test_backend_overflow_and_nan_rounding(native_post):
    x = torch.zeros((1, 4096), dtype=torch.float32, device="npu")
    x[0, :5] = torch.tensor([65504, 1e6, -1e6, float("inf"), float("nan")], device="npu")
    residual = torch.zeros((1, 4, 4096), dtype=torch.float32, device="npu")
    post = torch.ones((1, 4, 1), dtype=torch.float32, device="npu")
    comb = torch.zeros((1, 4, 4), dtype=torch.float32, device="npu")
    torch.testing.assert_close(
        native_post(x, residual, post, comb).cpu(),
        reference(x, residual, post, comb).cpu(),
        rtol=0,
        atol=0,
        equal_nan=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mhc-post-build-dir", required=True)
    args, remaining = parser.parse_known_args()
    raise SystemExit(
        pytest.main([__file__, "--mhc-post-build-dir", args.mhc_post_build_dir, *remaining], plugins=[BuildOptions()])
    )
