# SPDX-License-Identifier: Apache-2.0
"""Keep QFN DSpark admission isolated from the native MTP serving profile."""

from vllm.model_executor.models.config import Qwen4ExpForConditionalGenerationConfig

from vllm_ascend._310p.qwen4exp_dspark_config import verify_qfn_config
from vllm_ascend.utils import is_310p

_ORIGINAL_VERIFY = Qwen4ExpForConditionalGenerationConfig.verify_and_update_config


def verify_and_update_config(config):
    return verify_qfn_config(_ORIGINAL_VERIFY, config, is_310p=is_310p())


Qwen4ExpForConditionalGenerationConfig.verify_and_update_config = staticmethod(verify_and_update_config)
