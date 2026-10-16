# SPDX-License-Identifier: Apache-2.0
"""Apply/check the October 16 draft-addressing fix in a stopped October 11 bundle.

The original manifest stays unchanged. This verifier accepts exactly one
reviewed source change and checks every other frozen file against that manifest.
Run preflight on the pristine bundle first; restart after patching and discard
all earlier prefix caches. This does not modify a running worker's Python.
"""

import argparse
import hashlib
import json
from pathlib import Path

TARGET = "runtime/vllm_ascend/spec_decode/llm_base_proposer.py"
ORIGINAL_SHA256 = "39c96f22e16c624695165241dfe0d5827d25cc9acbc9a2c18da4aaa96d2a71f9"
MANIFEST_SHA256 = "bb8bacdbc16f4fd90a523e54eb746141ed1865fa016c8837b7c1abb35a6ad2ec"
OLD = """            # NOTE: In vllm, `block_size = attn_metadata_builder.kv_cache_spec.block_size`.
            # However, in vllm-ascend, the above value can be multiple of `kernel_block_size`,
            # which is not correct for computing `slot_mapping` below.
            if self.has_gdn:
                block_size = self.kernel_block_size
            else:
                block_size = self.block_size"""
NEW = """            # initialize_attn_backend selects the block size used by the
            # actual draft block table. The backend's first supported size
            # (kernel_block_size) may differ: 310P QSA advertises 128 first,
            # but its 256-wide heads use 64-token pages. Using 128 here can
            # write draft KV beyond its allocation and corrupt other caches.
            block_size = self.block_size"""


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare(root, *, check=False):
    root = root.resolve()
    manifest_bytes = (root / "files.sha256.json").read_bytes()
    if digest(manifest_bytes) != MANIFEST_SHA256:
        raise ValueError("requires the original October 11 single-card manifest digest")
    records = json.loads(manifest_bytes)
    if not isinstance(records, dict) or records.get(TARGET) != ORIGINAL_SHA256:
        raise ValueError("requires the original October 11 single-card manifest")
    source = (root / TARGET).read_text()
    if source.count(OLD) == 1 and digest(source.encode()) == ORIGINAL_SHA256:
        original, patched = source, source.replace(OLD, NEW)
        applied = False
    elif source.count(NEW) == 1:
        original, patched = source.replace(NEW, OLD), source
        if digest(original.encode()) != ORIGINAL_SHA256:
            raise ValueError("patched proposer contains unrelated changes")
        applied = True
    else:
        raise ValueError("proposer is neither the exact original nor the exact reviewed fix")
    # Validate all frozen files before any write; never trust a receipt as proof.
    for name, expected in records.items():
        if not isinstance(name, str) or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("invalid original manifest entry")
        relative = Path(name)
        path = root / relative
        if relative.is_absolute() or ".." in relative.parts or not path.resolve().is_relative_to(root):
            raise ValueError(f"manifest path escapes bundle: {name}")
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing or linked file: {name}")
        actual = digest(original.encode()) if name == TARGET else digest(path.read_bytes())
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch: {name}")
    if check and not applied:
        raise ValueError("correctness fix is not applied")
    compile(patched, str(root / TARGET), "exec")
    receipt = dict(
        fix_commit="4ee1d73d5",
        target=TARGET,
        original_sha256=ORIGINAL_SHA256,
        patched_sha256=digest(patched.encode()),
        original_manifest_sha256=digest(manifest_bytes),
        verified_files=len(records),
    )
    if not check:
        if not applied:
            path = root / TARGET
            temporary = path.with_suffix(".kv-fix.tmp")
            with temporary.open("x") as output:
                output.write(patched)
            temporary.chmod(path.stat().st_mode)
            temporary.replace(path)
        (root / "draft-kv-fix.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.bundle, check=args.check), indent=2))


if __name__ == "__main__":
    main()
