# SPDX-License-Identifier: Apache-2.0
"""First-use descriptor creation and replay for the frozen BF16 bundle.

Direct invocation registers --bf16-cast-build-dir without a shared conftest.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import pytest
import torch
import torch_npu

from tools.glm_perf.resident_native import NativeManifest


class BuildOptions:
    def pytest_addoption(self, parser):
        parser.addoption("--bf16-cast-build-dir")


@pytest.fixture(scope="module")
def native_factory(pytestconfig):
    directory = pytestconfig.getoption("--bf16-cast-build-dir", default=None)
    if directory is None:
        pytest.skip("requires a qualified frozen BF16 bundle")
    root = Path(directory).resolve(strict=True)
    manifest = NativeManifest(json.loads((root / "manifest.json").read_text()))
    manifest.verify_files()
    options = json.loads((root / "options.json").read_text())
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.ops.load_library(str(root / options["bridge"]))
    spec = importlib.util.spec_from_file_location("qualified_cold_bf16_test", root / "bf16_cast.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return lambda: helper.NativeBF16Cast(root, options["namespace"])


MODES = (
    (0, torch.float32, torch.bfloat16),
    (1, torch.bfloat16, torch.float32),
    (2, torch.float16, torch.bfloat16),
    (3, torch.bfloat16, torch.float16),
    (4, torch.float32, torch.float32),
    (5, torch.float32, torch.float16),
)


@pytest.mark.parametrize("count", [1, 333, 8192])
@pytest.mark.parametrize("mode,source_dtype,result_dtype", MODES)
def test_first_conversion_inside_capture_and_changed_input_replay(
    native_factory, count, mode, source_dtype, result_dtype
):
    native = native_factory()
    generator = torch.Generator().manual_seed(7805)
    source = torch.randn(count, generator=generator).to(source_dtype).npu()
    assert not native.configs
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = native._convert(source, result_dtype, mode)
    storage_dtype = torch.int32 if result_dtype == torch.float32 else torch.int16
    for _ in range(2):
        graph.replay()
        torch.npu.synchronize()
        expected = (source.bfloat16() if mode in (0, 2, 4, 5) else source).to(result_dtype)
        assert torch.equal(actual.cpu().view(storage_dtype), expected.cpu().view(storage_dtype))
        source.copy_(torch.randn(count, generator=generator).to(source_dtype).npu())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16-cast-build-dir", required=True)
    args, remaining = parser.parse_known_args()
    raise SystemExit(
        pytest.main([__file__, "--bf16-cast-build-dir", args.bf16_cast_build_dir, *remaining], plugins=[BuildOptions()])
    )
