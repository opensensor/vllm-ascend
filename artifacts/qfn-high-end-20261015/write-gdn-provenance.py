"""Record the exact isolated source tree and binary used by the native gate."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    parser.add_argument("compiler", type=Path)
    args = parser.parse_args()
    sources = {}
    for path in sorted(args.build.rglob("*")):
        if path.is_file() and path.suffix in (".h", ".hpp", ".cpp", ".bin"):
            sources[str(path.relative_to(args.build))] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = dict(
        entry="qwen_gdn_output_aligned_v1",
        architecture="dav-2002",
        compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
        files=[dict(path=path, sha256=digest) for path, digest in sources.items()],
        note="Build provenance only; qualification is recorded separately. No OPP packages replaced.",
    )
    (args.build / "provenance.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
