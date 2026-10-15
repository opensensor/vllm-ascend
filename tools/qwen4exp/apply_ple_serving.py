# SPDX-License-Identifier: Apache-2.0
"""Install native PLE once in an already qualified, drained resident server."""

import argparse
import hashlib
import json
import uuid
from pathlib import Path


def main():
    # The qualified runtime overlay supplies the resident transaction client.
    from tools.glm_perf.resident_control import Control
    from tools.sidecard.qfn_resident_server import QfnResidentClient

    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("run-root", "table", "library", "selector-build", "output"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("use a new output directory")
    manifest = json.loads((args.selector_build / "provenance.json").read_text())
    binary = Path(manifest["binary"]["path"]).resolve(strict=True)
    if binary.parent != args.selector_build.resolve() or manifest["entry"] != "qwen_ple_request_v1":
        raise ValueError("selector build differs from expected contract")
    if hashlib.sha256(binary.read_bytes()).hexdigest() != manifest["binary"]["sha256"]:
        raise ValueError("selector binary differs from provenance")
    library = args.library.resolve(strict=True)
    config = dict(
        table=str(args.table.resolve(strict=True)),
        library=str(library),
        library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        binary=str(binary),
        binary_sha256=manifest["binary"]["sha256"],
    )
    source = Path(__file__).with_name("ple_serving.py").read_text()
    source += "\ndef replacements():\n    return make_replacements(" + repr(config) + ")\n"
    client = QfnResidentClient(args.url, args.run_root / "control.token", timeout=1800)
    if any("native_ple" in worker for worker in client.rpc("resident_status")):
        raise RuntimeError("native PLE already installed; retain its owners until process shutdown")
    args.output.mkdir(mode=0o700, parents=True)
    (args.output / "candidate.py").write_text(source)
    control = Control(generation=uuid.uuid4().hex, candidate="native_host_ple", source=source, recapture=True)
    result = client.switch(control)
    (args.output / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Native PLE installed; verification remains enabled until inference qualification.")


if __name__ == "__main__":
    main()
