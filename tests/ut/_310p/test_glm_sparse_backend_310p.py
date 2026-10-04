# SPDX-License-Identifier: Apache-2.0
"""The GLM kpool MLA selector must never fall through to generic attention."""

from types import SimpleNamespace

import pytest
import vllm.config as vllm_config_module

import vllm_ascend.envs as ascend_envs
import vllm_ascend.platform as platform
from vllm_ascend.device.hardware_profile import AttentionBackendFamily


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("kpool", [False, True])
def test_310p_sparse_mla_routes_only_enabled_kpool(monkeypatch, enabled, kpool):
    text_config = SimpleNamespace(index_topk=2048)
    if kpool:
        text_config.index_kpool = 4
    monkeypatch.setattr(
        vllm_config_module,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=SimpleNamespace(hf_text_config=text_config)),
    )
    monkeypatch.setattr(
        platform,
        "get_current_hardware_profile",
        lambda: SimpleNamespace(attention_backend_family=AttentionBackendFamily.COMPATIBILITY),
    )
    monkeypatch.setattr(platform, "_validate_fa3_backend", lambda *args: False)
    monkeypatch.setattr(ascend_envs, "VLLM_ASCEND_310P_ENABLE_MLA", enabled)
    selector = SimpleNamespace(use_mla=True, use_sparse=True, use_pcp=False)
    if enabled and kpool:
        assert platform.NPUPlatform.get_attn_backend_cls(None, selector) == (
            "vllm_ascend._310p.attention.mla_v1_310.AscendMLABackend310"
        )
    else:
        with pytest.raises(NotImplementedError, match="Sparse MLA on 310P"):
            platform.NPUPlatform.get_attn_backend_cls(None, selector)
