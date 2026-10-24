# SPDX-License-Identifier: Apache-2.0
"""Global plugin setup must not require another model's test dependencies."""

import subprocess
import sys


def test_dtype_package_import_without_test_tree():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "sys.modules['tests'] = None\n"
            "import vllm_ascend.models.deepseek_v41 as package\n"
            "assert package.DeepseekV41DtypePolicy is not None\n"
            "assert package.w2_group_qdq_linear is not None\n"
            "assert not hasattr(package, 'unknown_export')\n",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
