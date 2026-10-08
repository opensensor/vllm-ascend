import sys
import torch
from vllm_ascend.utils import enable_custom_op
enable_custom_op()
torch.ops.load_library('/srv/ai/src/glm-l1-wide-build-20261004/build-strata-bindings-owned-20261004/glm_moe_candidates.so')
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
