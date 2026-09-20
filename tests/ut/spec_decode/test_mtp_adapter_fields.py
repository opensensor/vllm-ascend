#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Regression tests for two fork-drift gaps that broke MTP speculative decode.

Both surfaced as AttributeError at the first draft step on Ascend 310P with
Qwen3.5/3.8 MTP: ``prepare_inputs_padded`` read
``AscendCommonAttentionMetadata._num_computed_tokens_cpu``, and the proposer
read ``uses_xdrope_dim`` on a model whose rope is partial rather than
extended/decoupled. Neither was ever assigned.
"""

from __future__ import annotations

from dataclasses import fields

from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer

# `slice_for_reqs` and the spec-decode proposers read these underscore-prefixed
# CPU views that the vLLM fork parent no longer declares.
_FORK_DRIFT_CPU_FIELDS = ("_seq_lens_cpu", "_num_computed_tokens_cpu", "dcp_local_seq_lens_cpu")


def test_common_attention_metadata_declares_fork_drift_cpu_fields():
    declared = {f.name for f in fields(AscendCommonAttentionMetadata)}
    missing = [name for name in _FORK_DRIFT_CPU_FIELDS if name not in declared]
    assert not missing, f"AscendCommonAttentionMetadata is missing {missing}"


def test_fork_drift_cpu_fields_default_to_none():
    defaults = {f.name: f.default for f in fields(AscendCommonAttentionMetadata)}
    for name in _FORK_DRIFT_CPU_FIELDS:
        assert defaults[name] is None, f"{name} must default to None so callers can omit it"


def test_proposer_defaults_xdrope_dims_to_zero():
    """Non-xdrope models must read 0, not raise AttributeError."""
    assert AscendSpecDecodeBaseProposer.uses_xdrope_dim == 0
    assert AscendSpecDecodeBaseProposer.draft_uses_xdrope_dim == 0
