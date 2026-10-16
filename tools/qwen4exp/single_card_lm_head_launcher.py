# SPDX-License-Identifier: Apache-2.0
"""Create a launcher for the corrected frozen bundle with only its head in W8.

Keep the original launcher and its environment. Use the current profile tool
by absolute filename so the frozen tools package cannot hide the new option.
"""

import argparse
import hashlib
import json
import shlex
from pathlib import Path

from tools.qwen4exp.single_card_kv_fix import prepare

SEAM = 'exec "$launcher_python" -m tools.qwen4exp.single_card_profile '


def create_launcher(bundle):
    bundle = bundle.resolve()
    receipt = prepare(bundle, check=True)
    script = (bundle / "serve.sh").read_text()
    if script.count(SEAM) != 1:
        raise ValueError("frozen launcher entry point changed")
    profile = Path(__file__).with_name("single_card_profile.py").resolve()
    script = script.replace(SEAM, f'exec "$launcher_python" {shlex.quote(str(profile))} --quantize-lm-head ')
    path = bundle / "serve-w8-head.sh"
    with path.open("x") as output:
        output.write(script)
    receipt.update(
        launcher=str(path),
        launcher_sha256=hashlib.sha256(script.encode()).hexdigest(),
        profile=str(profile),
        profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
        precision="W4 experts, W8 dynamic target/draft output heads, unchanged FP16 PLE projection",
    )
    (bundle / "w8-head-launcher.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    print(json.dumps(create_launcher(args.bundle), indent=2))


if __name__ == "__main__":
    main()
