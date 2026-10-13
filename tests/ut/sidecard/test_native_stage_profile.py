# SPDX-License-Identifier: Apache-2.0
"""Reject unsupported native profiling scopes before device initialization."""

from pathlib import Path

import pytest

from tools.sidecard.native_stage_profile import run, validate_layers


@pytest.mark.parametrize("layers", [(), (18, 18), (0,), (33,), (18, 99), (18.0,), ("18",)])
def test_native_profile_rejects_unsupported_or_duplicate_layers(layers):
    with pytest.raises(ValueError, match="supported cold layers"):
        validate_layers(layers)


def test_native_profile_rejects_unassigned_card_visibility_before_device_import(monkeypatch):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1")
    with pytest.raises(ValueError, match="other Ascend card"):
        run(Path("missing-config.json"), Path("missing-output.json"), (18,))
