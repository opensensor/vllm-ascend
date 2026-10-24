# SPDX-License-Identifier: Apache-2.0
"""Historical builds use pinned, complete inputs and reject altered sources."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from tools.qwen4exp.custom_op_sources import RELEASE_PATH, prepare_sources, verify_sources


@pytest.fixture
def source_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    source = repo / "csrc/host.cpp"
    source.parent.mkdir()
    source.write_bytes(b"original host")
    subprocess.run(["git", "-C", str(repo), "add", "csrc"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qsm",
            "test: fixture",
        ],
        check=True,
    )
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    release = repo / RELEASE_PATH
    release.mkdir(parents=True)
    (release / "tiling.cpp").write_bytes(b"historical tiling")
    files = {
        "host.cpp": {"git_path": "csrc/host.cpp", "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
        "op_host/tiling.cpp": {
            "snapshot": "tiling.cpp",
            "sha256": hashlib.sha256(b"historical tiling").hexdigest(),
        },
    }
    for record in files.values():
        record["executable"] = False
    (release / "files.json").write_text(json.dumps(files))
    manifest = {"base_revision": revision, "packages": {"embedded": {"manifest": "files.json", "file_count": 2}}}
    (release / "manifest.json").write_text(json.dumps(manifest))
    return repo


def test_pinned_tree_ignores_checkout_edits(source_repo, tmp_path):
    (source_repo / "csrc/host.cpp").write_bytes(b"unrelated working changes")
    work = tmp_path / "work"
    assert prepare_sources(source_repo, "embedded", work) == 2
    assert (work / "csrc/host.cpp").read_bytes() == b"original host"
    assert (work / "csrc/op_host/tiling.cpp").read_bytes() == b"historical tiling"


def test_existing_directory_is_preserved(source_repo, tmp_path):
    work = tmp_path / "existing"
    work.mkdir()
    (work / "keep").write_text("keep")
    with pytest.raises(ValueError, match="new work directory"):
        prepare_sources(source_repo, "embedded", work)
    assert (work / "keep").read_text() == "keep"


def test_altered_snapshot_rejected(source_repo, tmp_path):
    (source_repo / RELEASE_PATH / "tiling.cpp").write_bytes(b"modified tiling")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        prepare_sources(source_repo, "embedded", tmp_path / "work")


@pytest.mark.parametrize("linked", [False, True])
def test_modified_or_linked_prepared_file_rejected(source_repo, tmp_path, linked):
    work = tmp_path / "work"
    prepare_sources(source_repo, "embedded", work)
    target = work / "csrc/host.cpp"
    if linked:
        target.unlink()
        target.symlink_to(source_repo / "csrc/host.cpp")
    else:
        target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="linked|escapes|SHA-256 mismatch"):
        verify_sources(source_repo, "embedded", work)


@pytest.mark.parametrize("path", ["../escape.cpp", "/tmp/escape.cpp"])
def test_manifest_escape_rejected(source_repo, tmp_path, path):
    manifest = source_repo / RELEASE_PATH / "files.json"
    files = json.loads(manifest.read_text())
    files[path] = files.pop("host.cpp")
    manifest.write_text(json.dumps(files))
    with pytest.raises(ValueError, match="escapes"):
        prepare_sources(source_repo, "embedded", tmp_path / "work")


@pytest.mark.parametrize("package", ["coherent", "retained", "gdn", "embedded"])
def test_complete_release_source_tree(package, tmp_path):
    repo = Path(__file__).resolve().parents[3]
    work = tmp_path / package
    count = prepare_sources(repo, package, work)
    assert count > 0
    assert verify_sources(repo, package, work) == count
    assert (work / "csrc/third_party/makeself/makeself.sh").is_file()
    assert (work / "csrc/third_party/catlass/include/catlass/catlass.hpp").is_file()
