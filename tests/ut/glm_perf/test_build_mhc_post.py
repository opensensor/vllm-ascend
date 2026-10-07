# SPDX-License-Identifier: Apache-2.0
"""Append-only build contracts protect binaries already held by live graphs."""

import pytest

from tools.glm_perf.build_mhc_post import build_mhc


@pytest.mark.parametrize("version", [0, -1, True])
def test_invalid_namespace_version_creates_no_assets(tmp_path, version):
    output = tmp_path / "build"
    with pytest.raises(ValueError, match="version must be positive"):
        build_mhc(output, tmp_path, tmp_path, version)
    assert not output.exists()


def test_existing_build_is_never_overwritten(tmp_path):
    output = tmp_path / "build"
    output.mkdir()
    binary = output / "mhc_post_fp32.bin"
    binary.write_bytes(b"resident kernel")
    with pytest.raises(FileExistsError):
        build_mhc(output, tmp_path, tmp_path, 76)
    assert binary.read_bytes() == b"resident kernel"
    assert list(output.iterdir()) == [binary]
