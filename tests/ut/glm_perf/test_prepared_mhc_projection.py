# SPDX-License-Identifier: Apache-2.0
"""Inject before the parent builds its dispatch closures; retain fallback calls."""

import torch

from tools.glm_perf.resident_candidates.mhc_projection import PreparedProjection, extend_replacements


class Projection:
    configs = {640: None}

    def __call__(self, residual, fn):
        return torch.ones(residual.shape[0], 24)


class Original:
    def __init__(self):
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
        return "original"

    def epilogue(self, residual, mixes, *args):
        return mixes


def test_projection_proxy_is_injected_before_parent_captures_operation():
    original, projection = Original(), Projection()
    captured = []
    prefix = "vllm_ascend._310p.worker_310p:NPUWorker310."

    def parent(resources):
        captured.append(resources["mhc_pre_vector_v1005_corrected_reference"])
        return {prefix + "resident_capture": lambda self: {}, prefix + "resident_status": lambda self: {}}

    resources = dict(mhc_pre_vector_v1005_corrected_reference=original, mhc_projection_v1011=projection)
    changes = extend_replacements(parent, resources)
    assert isinstance(captured[0], PreparedProjection)
    assert captured[0].original is original
    assert resources["mhc_pre_vector_v1005_corrected_reference"] is original
    assert set(changes) == {prefix + "resident_capture", prefix + "resident_status"}


def test_unqualified_weights_and_decode_keep_the_original_dispatch():
    original, projection = Original(), Projection()
    proxy = PreparedProjection(original, projection)
    fn = torch.zeros(24, 1)
    residual = torch.zeros(640, 4, 1)
    assert proxy(residual, fn, None, None, None, None) == "original"
    proxy.accepted.add(fn.data_ptr())
    projected = proxy(residual, fn, None, None, None, None)
    assert projected.shape == (640, 24) and proxy.calls == {640: 1}
    assert proxy(residual[:8], fn, None, None, None, None) == "original"
    assert original.calls == 2
