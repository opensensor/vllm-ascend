# SPDX-License-Identifier: Apache-2.0
"""Restore the four October 11 operator source trees from versioned inputs."""

import argparse
import hashlib
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

RELEASE_PATH = Path("csrc/release_sources/qwen38_single_card_20261011")


def child_path(root: Path, name: str) -> Path:
    relative = Path(name)
    path = root / relative
    if relative.is_absolute() or ".." in relative.parts or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Source path escapes root: {name}")
    return path


def package_manifest(repo: Path, package: str):
    release = repo / RELEASE_PATH
    manifest = json.loads((release / "manifest.json").read_text())
    if package not in manifest["packages"]:
        raise ValueError(f"Unknown operator package: {package}")
    config = manifest["packages"][package]
    files = json.loads(child_path(release, config["manifest"]).read_text())
    if not files or len(files) != config["file_count"]:
        raise ValueError("Incomplete source manifest")
    for name, record in files.items():
        child_path(Path("/source"), name)
        if ("git_path" in record) == ("snapshot" in record):
            raise ValueError(f"Expected exactly one source for {name}")
        if "snapshot" in record:
            child_path(release, record["snapshot"])
        elif not record["git_path"].startswith("csrc/"):
            raise ValueError(f"Expected csrc Git path: {name}")
    return manifest, config, files


def verify_sources(repo: Path, package: str, work: Path) -> int:
    _, _, files = package_manifest(repo, package)
    for name, record in files.items():
        path = child_path(work / "csrc", name)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or linked source: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Source SHA-256 mismatch: {name}")
    return len(files)


def prepare_sources(repo: Path, package: str, work: Path) -> int:
    manifest, _, files = package_manifest(repo, package)
    if work.exists() or work.is_symlink():
        raise ValueError(f"Use a new work directory: {work}")
    # Read the pinned Git tree, not a potentially edited checkout. No archive is
    # downloaded or installed; this temporary stream is only a Git export.
    with tempfile.TemporaryFile() as exported:
        subprocess.run(
            ["git", "-C", str(repo), "archive", manifest["base_revision"], "csrc"],
            stdout=exported,
            check=True,
        )
        exported.seek(0)
        with tarfile.open(fileobj=exported) as sources:
            work.mkdir(parents=True, exist_ok=False)
            for name, record in files.items():
                if "snapshot" in record:
                    data = child_path(repo / RELEASE_PATH, record["snapshot"]).read_bytes()
                else:
                    member = sources.getmember(record["git_path"])
                    if not member.isfile():
                        raise ValueError(f"Not a regular Git source: {name}")
                    with sources.extractfile(member) as stream:
                        data = stream.read()
                if hashlib.sha256(data).hexdigest() != record["sha256"]:
                    raise ValueError(f"Source SHA-256 mismatch: {name}")
                target = child_path(work / "csrc", name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                target.chmod(0o755 if record["executable"] else 0o644)
    (work / "source-receipt.json").write_text(
        json.dumps({"package": package, "base_revision": manifest["base_revision"], "files": len(files)}, indent=2)
        + "\n"
    )
    return verify_sources(repo, package, work)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", choices=("coherent", "retained", "gdn", "embedded"))
    parser.add_argument("work", type=Path)
    parser.add_argument("--verify", action="store_true", help="Check an existing prepared tree without rebuilding")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    try:
        action = verify_sources if args.verify else prepare_sources
        count = action(repo, args.package, args.work.absolute())
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    print(f"Verified {count} source files for {args.package}")


if __name__ == "__main__":
    main()
