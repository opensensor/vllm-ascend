# SPDX-License-Identifier: Apache-2.0
import shlex
import subprocess

import pytest

from tools.qwen4exp import single_card_lm_head_launcher as launcher


def test_launcher_preserves_frozen_environment_and_quotes_current_profile(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    profile_dir = tmp_path / "repo with 'quotes' and $dollars"
    profile_dir.mkdir()
    (profile_dir / "single_card_profile.py").write_text("# test profile\n")
    monkeypatch.setattr(launcher, "__file__", str(profile_dir / "single_card_lm_head_launcher.py"))
    checked = []
    monkeypatch.setattr(launcher, "prepare", lambda root, check: checked.append((root, check)) or {})
    original = "#!/bin/bash\nexport OMP_NUM_THREADS=1\n" + launcher.SEAM + '--updated "$@"\n'
    (bundle / "serve.sh").write_text(original)
    receipt = launcher.create_launcher(bundle)
    assert checked == [(bundle, True)]
    assert (bundle / "serve.sh").read_text() == original
    script = bundle / "serve-w8-head.sh"
    command = shlex.split(script.read_text().splitlines()[-1])
    assert command[:4] == [
        "exec",
        "$launcher_python",
        str(profile_dir / "single_card_profile.py"),
        "--quantize-lm-head",
    ]
    assert receipt["profile_sha256"]
    subprocess.run(["bash", "-n", str(script)], check=True)
    with pytest.raises(FileExistsError):
        launcher.create_launcher(bundle)


def test_unverified_bundle_cannot_produce_launcher(tmp_path, monkeypatch):
    def reject(*args, **kwargs):
        raise ValueError("unverified")

    monkeypatch.setattr(launcher, "prepare", reject)
    with pytest.raises(ValueError, match="unverified"):
        launcher.create_launcher(tmp_path)
    assert not (tmp_path / "serve-w8-head.sh").exists()
