# SPDX-License-Identifier: Apache-2.0
"""Experimental serving bootstrap; also registers operators in spawned workers."""

import torch

from vllm_ascend.utils import enable_custom_op

# Register before importing the CLI, and again when multiprocessing imports
# this file as __mp_main__. No device is initialized by loading the library.
enable_custom_op()
torch.ops.load_library(
    "/srv/ai/src/glm-l1-wide-build-20261004/build-strata-bindings-owned-20261004/glm_moe_candidates.so"
)

if __name__ == "__main__":
    from vllm.entrypoints.cli.main import main

    main()
