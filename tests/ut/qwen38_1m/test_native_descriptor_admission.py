# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from tools.qwen4exp.native_descriptor_admission import descriptors_available


def test_existing_descriptor_does_not_consult_capture_or_mutate():
    descriptor = object()
    resource = SimpleNamespace(_configs={"existing": descriptor})

    def capture():
        raise AssertionError("existing descriptor does not need capture state")

    assert descriptors_available(resource, ["existing"], capacity=1, is_capturing=capture)
    assert resource._configs == {"existing": descriptor}


@pytest.mark.parametrize("capturing", [False, True])
def test_257th_shape_declines_but_every_existing_address_survives(capturing):
    resource = SimpleNamespace(_configs={index: object() for index in range(256)})
    before = resource._configs.copy()
    assert not descriptors_available(resource, [256], capacity=256, is_capturing=lambda: capturing)
    assert descriptors_available(resource, range(256), capacity=256, is_capturing=lambda: capturing)
    assert resource._configs == before


def test_multiple_missing_tail_descriptors_need_combined_capacity():
    resource = SimpleNamespace(_configs={index: object() for index in range(255)})
    before = resource._configs.copy()
    assert not descriptors_available(resource, [255, 256], capacity=256, is_capturing=lambda: False)
    assert descriptors_available(resource, [255, 255], capacity=256, is_capturing=lambda: False)
    assert resource._configs == before


def test_unknown_capture_descriptor_declines_and_empty_plan_is_safe():
    resource = SimpleNamespace(_configs={})
    assert not descriptors_available(resource, ["new"], capacity=256, is_capturing=lambda: True)
    assert descriptors_available(resource, [], capacity=256, is_capturing=lambda: True)
    assert not descriptors_available(object(), [], capacity=256, is_capturing=lambda: False)
