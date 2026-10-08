# SPDX-License-Identifier: Apache-2.0
"""Complete QSA alias/nonalias correctness and changed-input graph gates.

Run directly with --qsa-build-dir pointing to an immutable paired QSA bundle.
The environment must expose the same custom OPP stack as the serving engine.
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

import pytest
import torch


class BuildOptions:
    def pytest_addoption(self, parser):
        parser.addoption("--qsa-build-dir")


def test_qsa_parent_shared_production_and_changed_graph_replay(pytestconfig):
    directory = pytestconfig.getoption("--qsa-build-dir", default=None)
    if directory is None:
        pytest.skip("requires independent compiled parent/shared QSA bundle and 310P hardware")
    import torch_npu

    from vllm_ascend.utils import enable_custom_op

    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    root = Path(directory).resolve(strict=True)
    provenance = json.loads((root / "provenance.json").read_text())
    sys.path.insert(0, str(root))
    native = importlib.import_module(provenance["helper_package"] + ".qsa_shared_native")
    probe = importlib.import_module(provenance["helper_package"] + ".qsa_shared_probe")
    namespace = provenance["namespace"]
    torch.ops.load_library(str(root / (namespace + ".so")))
    kernel = getattr(torch.classes, namespace).Kernel
    launch = getattr(torch.ops, namespace).launch
    operations = [
        native.NativeQsaShared(
            kernel(str(root / (name + ".bin")), "qsa_sparse_attention_v310"), launch, torch.device("npu:0")
        )
        for name in ("parent", "shared")
    ]
    result = probe.qualify(*operations, production=torch.ops._C_ascend.npu_qsa_sparse_attention_310)
    assert result["passed"] and len(result["cases"]) == 36


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qsa-build-dir", required=True)
    args = parser.parse_args()
    raise SystemExit(
        pytest.main([__file__, "--noconftest", "-q", "--qsa-build-dir", args.qsa_build_dir], plugins=[BuildOptions()])
    )
