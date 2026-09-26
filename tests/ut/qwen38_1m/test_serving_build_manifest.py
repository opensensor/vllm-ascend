# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_qsa_gather_value_nz_is_built_and_bound():
    build = (REPO_ROOT / "csrc/build_aclnn.sh").read_text()
    defaults_310p = build.split('if [[ "$SOC_VERSION" =~ ^ascend310 ]]', 1)[1].split('elif [[ "$SOC_VERSION"', 1)[0]
    assert '"qsa_gather_value_nz_v310"' in defaults_310p
    binding = (REPO_ROOT / "csrc/torch_binding.cpp").read_text()
    assert "qsa_gather_value_nz_310_torch_adpt.h" in binding
    assert (
        'ops.impl("qsa_gather_value_nz_310", torch::kPrivateUse1, &vllm_ascend::qsa_gather_value_nz_310)'
        in " ".join(binding.split())
    )
    op_path = REPO_ROOT / "csrc/attention/qsa_gather_value_nz_v310"
    assert (op_path / "CMakeLists.txt").is_file()
    assert (op_path / "op_host/CMakeLists.txt").is_file()
    assert "EXEC_NPU_CMD(aclnnQsaGatherValueNzV310," in (op_path / "qsa_gather_value_nz_310_torch_adpt.h").read_text()


def test_request_timing_logger_has_installed_entrypoint():
    setup = (REPO_ROOT / "setup.py").read_text()
    assert '"vllm.stat_logger_plugins"' in setup
    assert "vllm_ascend.observability.request_timing:AscendRequestTimingLogger" in setup
