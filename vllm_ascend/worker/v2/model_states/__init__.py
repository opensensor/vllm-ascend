# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_states/__init__.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache

from vllm_ascend.device.device_config import is_310p


def init_asecnd_model_state(
    vllm_config: VllmConfig,
    model: nn.Module,
    encoder_cache: EncoderCache | None,
    device: torch.device,
):
    # Keep model-provided state overrides ahead of the platform defaults.
    if hasattr(model, "get_model_state_cls"):
        try:
            cls = model.get_model_state_cls()
        except NotImplementedError:
            # A model may declare the stable ``get_model_state_cls`` hook while
            # deferring the concrete class to the platform dispatch below (e.g.
            # Qwen4Exp PLE, whose 310P state is registered here in T1.3).
            cls = None
        if cls is not None:
            return cls(vllm_config, model, encoder_cache, device)

    # 310P Qwen4Exp PLE n-gram model state (T1.3). The n-gram context buffers
    # only exist on the 310P Triton-free hybrid path; ``ple_layer_ids`` on the
    # text config is the authoritative PLE marker (see model_state module).
    if is_310p() and vllm_config.model_config.is_hybrid:
        text_config = getattr(vllm_config.model_config, "hf_text_config", None)
        if getattr(text_config, "ple_layer_ids", None):
            from vllm_ascend._310p.worker.v2.model_state import (
                Ascend310PQwen4ExpModelState,
            )

            return Ascend310PQwen4ExpModelState(vllm_config, model, encoder_cache, device)

    # 310P uses Triton-free states under ``vllm_ascend._310p.worker.v2.model_state``.
    if vllm_config.model_config.is_hybrid:
        if is_310p():
            from vllm_ascend._310p.worker.v2.model_state import Ascend310PMambaHybridModelState

            return Ascend310PMambaHybridModelState(vllm_config, model, encoder_cache, device)

        from vllm_ascend.worker.v2.model_states.mamba_hybrid import (
            AscendMambaHybridModelState,
        )

        return AscendMambaHybridModelState(
            vllm_config,
            model,
            encoder_cache,
            device,
        )

    if is_310p():
        from vllm_ascend._310p.worker.v2.model_state import (
            Ascend310PModelState,
        )

        return Ascend310PModelState(
            vllm_config,
            model,
            encoder_cache,
            device,
        )

    from vllm_ascend.worker.v2.model_states.default import AscendModelState

    return AscendModelState(vllm_config, model, encoder_cache, device)
