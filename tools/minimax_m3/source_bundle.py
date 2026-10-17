# SPDX-License-Identifier: Apache-2.0
"""Freeze committed local serving sources and CATLASS; never install or launch."""

import argparse
import json
import subprocess
from pathlib import Path

from tools.minimax_m3.checkpoint import read_manifest
from tools.minimax_m3.export_w4 import file_digest

EXCLUDED_ROOTS = frozenset({"artifacts"})
CATLASS_PATH = "csrc/third_party/catlass"


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def archive_tree(repo, revision, destination, prefix):
    if destination.exists():
        raise ValueError("Archive destination already exists")
    commit = git(repo, "rev-parse", "--verify", revision + "^{commit}").decode().strip()
    roots = [
        name.decode()
        for name in git(repo, "ls-tree", "--name-only", "-z", commit).split(b"\0")
        if name and name.decode() not in EXCLUDED_ROOTS
    ]
    with destination.open("xb") as stream:
        subprocess.run(
            ["git", "-C", str(repo), "archive", "--format=tar.gz", "--prefix=" + prefix, commit, "--", *roots],
            stdout=stream,
            check=True,
        )
    gitlinks = [
        line.decode().split("\t", 1)[1]
        for line in git(repo, "ls-tree", "-r", "-z", commit).split(b"\0")
        if line.startswith(b"160000 ")
    ]
    return dict(
        gitlinks_not_embedded=gitlinks,
        commit=commit,
        archive=destination.name,
        bytes=destination.stat().st_size,
        sha256=file_digest(destination),
        excluded_uncommitted_changes=bool(git(repo, "diff", "--name-only", "HEAD").strip()),
        checkout_differs_from_archived_commit=git(repo, "rev-parse", "HEAD").decode().strip() != commit,
    )


def prepare(ascend, vllm, catlass, model, destination):
    if destination.exists():
        raise ValueError("Use a fresh bundle directory")
    manifest = read_manifest(model)
    ascend_commit = git(ascend, "rev-parse", "HEAD").decode().strip()
    entry = git(ascend, "ls-tree", ascend_commit, CATLASS_PATH).decode().split()
    if len(entry) != 4 or entry[:2] != ["160000", "commit"]:
        raise ValueError("Expected a pinned CATLASS gitlink")
    catlass_commit = entry[2]
    git(catlass, "cat-file", "-e", catlass_commit + "^{commit}")
    destination.mkdir(parents=True)
    archives = {
        "ascend": archive_tree(ascend, ascend_commit, destination / "vllm-ascend.tar.gz", "vllm-ascend/"),
        "vllm": archive_tree(vllm, "HEAD", destination / "vllm.tar.gz", "vllm/"),
        "catlass": archive_tree(
            catlass, catlass_commit, destination / "catlass.tar.gz", "vllm-ascend/" + CATLASS_PATH + "/"
        ),
    }
    record = dict(
        complete=True,
        source_only=True,
        hardware_validated=False,
        archives=archives,
        model=dict(
            path=str(model.resolve()),
            tensor_count=manifest["tensor_count"],
            manifest_sha256=file_digest(model / "minimax_m3_w4.json"),
            config_sha256=file_digest(model / "config.json"),
            weights_included=False,
        ),
        requirements=[
            "matching CANN and torch_npu runtime",
            "compiled 310P extension and coherent OPP",
            "separate checkpoint transfer and destination payload verification",
        ],
        installation_performed=False,
    )
    (destination / "bundle.json").write_text(json.dumps(record, indent=2) + "\n")
    checksums = "".join(f"{a['sha256']}  {a['archive']}\n" for a in archives.values())
    (destination / "SHA256SUMS").write_text(checksums)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ascend-repo", type=Path, required=True)
    parser.add_argument("--vllm-repo", type=Path, required=True)
    parser.add_argument("--catlass-repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.ascend_repo, args.vllm_repo, args.catlass_repo, args.model, args.output), indent=2))


if __name__ == "__main__":
    main()
