# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from tools.qwen4exp import single_card_kv_fix as fix


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    source = "def update():\n    if True:\n        if True:\n" + fix.OLD + "\n"
    monkeypatch.setattr(fix, "ORIGINAL_SHA256", fix.digest(source.encode()))
    path = tmp_path / fix.TARGET
    path.parent.mkdir(parents=True)
    path.write_text(source)
    (tmp_path / "asset").write_text("preserved")
    (tmp_path / "files.sha256.json").write_text(
        json.dumps({fix.TARGET: fix.ORIGINAL_SHA256, "asset": fix.digest(b"preserved")})
    )
    monkeypatch.setattr(fix, "MANIFEST_SHA256", fix.digest((tmp_path / "files.sha256.json").read_bytes()))
    return tmp_path


def test_exact_patch_is_idempotent_and_preserves_manifest(bundle):
    manifest = (bundle / "files.sha256.json").read_bytes()
    with pytest.raises(ValueError, match="not applied"):
        fix.prepare(bundle, check=True)
    receipt = fix.prepare(bundle)
    assert fix.prepare(bundle) == receipt == fix.prepare(bundle, check=True)
    assert (bundle / "files.sha256.json").read_bytes() == manifest
    assert fix.NEW in (bundle / fix.TARGET).read_text()


@pytest.mark.parametrize("target", [fix.TARGET, "asset"])
def test_unrelated_drift_fails_before_write(bundle, target):
    path = bundle / target
    path.write_text(path.read_text() + "\n# drift\n")
    before = (bundle / fix.TARGET).read_bytes()
    with pytest.raises(ValueError):
        fix.prepare(bundle)
    assert (bundle / fix.TARGET).read_bytes() == before
    assert not (bundle / "draft-kv-fix.json").exists()


def test_receipt_does_not_hide_later_drift(bundle):
    fix.prepare(bundle)
    path = bundle / fix.TARGET
    path.write_text(path.read_text() + "\n# extra\n")
    with pytest.raises(ValueError, match="unrelated"):
        fix.prepare(bundle, check=True)


def test_manifest_escape_is_rejected(bundle, monkeypatch):
    path = bundle / "files.sha256.json"
    records = json.loads(path.read_text())
    records["../escape"] = fix.digest(b"x")
    path.write_text(json.dumps(records))
    monkeypatch.setattr(fix, "MANIFEST_SHA256", fix.digest(path.read_bytes()))
    with pytest.raises(ValueError, match="escapes"):
        fix.prepare(bundle)


def test_manifest_cannot_be_rewritten_to_hide_missing_assets(bundle):
    path = bundle / "files.sha256.json"
    path.write_text(json.dumps({fix.TARGET: fix.ORIGINAL_SHA256}))
    with pytest.raises(ValueError, match="manifest digest"):
        fix.prepare(bundle)
    assert fix.OLD in (bundle / fix.TARGET).read_text()
