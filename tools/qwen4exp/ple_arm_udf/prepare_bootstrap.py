# SPDX-License-Identifier: Apache-2.0
"""Prepare a private CANN overlay for the missing model staging root.

The standard common-package loader's install_path=1 is the model staging root.
Reuse the installed, signed UDF runtime archive without changing a byte; the
device still verifies its CMS signature. A plain README-only archive was
rejected by this firmware. This changes no installed toolkit file. Deployment
duplicates the vendor runtime under the model staging root, so it is a private
diagnostic workaround, not a serving installation step.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PACKAGE_NAME = "ple-model-dir-bootstrap.tar.gz"
VENDOR_ARCHIVE = "compat/cann-udf-compat.tar.gz"


def prepare(runtime_root: Path, output: Path) -> Path:
    runtime_root = runtime_root.resolve(strict=True)
    config = (runtime_root / "conf/ascend_package_load.ini").read_text()
    if PACKAGE_NAME in config:
        raise ValueError("bootstrap package already in source configuration")
    vendor = runtime_root / VENDOR_ARCHIVE
    # Recognize the container, not its signature: signature validation is device
    # work. Reject the failed unsigned-marker experiment before creating files.
    with vendor.open("rb") as stream:
        if stream.read(4) != bytes.fromhex("aa55aa55"):
            raise ValueError("expected installed vendor CMS archive container")
    output.mkdir(parents=True, exist_ok=False)
    for child in runtime_root.iterdir():
        if child.name != "conf":
            (output / child.name).symlink_to(child, target_is_directory=child.is_dir())
    conf = output / "conf"
    conf.mkdir()
    for child in (runtime_root / "conf").iterdir():
        if child.name != "ascend_package_load.ini":
            (conf / child.name).symlink_to(child, target_is_directory=child.is_dir())
    package = conf / PACKAGE_NAME
    shutil.copyfile(vendor, package)
    source_hash = hashlib.sha256(vendor.read_bytes()).hexdigest()
    if hashlib.sha256(package.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError("vendor package changed during preparation")
    (conf / "ascend_package_load.ini").write_text(
        config.rstrip() + f"\nname:{PACKAGE_NAME}\ninstall_path:1\noptional:true\n"
        "package_path:conf\nload_as_per_soc:false\n"
    )
    (output / "ple-bootstrap-manifest.json").write_text(
        json.dumps(
            {
                "vendor_archive": str(vendor),
                "sha256": source_hash,
                "install_path": 1,
                "device_signature_verified": False,
                "device_deployment_verified": False,
                "device_submit": False,
            },
            indent=2,
        )
        + "\n"
    )
    return package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.runtime_root, args.output))


if __name__ == "__main__":
    main()
