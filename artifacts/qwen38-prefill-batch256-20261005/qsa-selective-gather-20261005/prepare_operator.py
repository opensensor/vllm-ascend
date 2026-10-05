# SPDX-License-Identifier: Apache-2.0
"""Create an isolated named-operator snapshot without editing the serving source."""

from pathlib import Path


def rename(text: str) -> str:
    for old, new in (
        ("qsa_gather_value_nz_v310", "qsa_gather_value_nz_zero_v310"),
        ("qsa_gather_value_nz_310", "qsa_gather_value_nz_zero_310"),
        ("QsaGatherValueNzV310", "QsaGatherValueNzZeroV310"),
        ("QSA_GATHER_VALUE_NZ_V310", "QSA_GATHER_VALUE_NZ_ZERO_V310"),
        ("ASCEND_OPS_QSA_GATHER_VALUE_NZ_V310", "ASCEND_OPS_QSA_GATHER_VALUE_NZ_ZERO_V310"),
        ("NsQsaGatherValueNz", "NsQsaGatherValueNzZero"),
    ):
        text = text.replace(old, new)
    return text


def main() -> None:
    experiment = Path(__file__).resolve().parent
    repo = experiment.parents[2]
    original = repo / "csrc/attention/qsa_gather_value_nz_v310"
    variant = experiment.parent / "qsa-zero-invalid-20261005/qsa_gather_value_nz_v310.h"
    destination = experiment / "csrc/attention/qsa_gather_value_nz_zero_v310"
    for source in sorted(original.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(original)
        output = destination / rename(str(relative))
        output.parent.mkdir(parents=True, exist_ok=True)
        content = (
            variant.read_text() if relative == Path("op_kernel/qsa_gather_value_nz_v310.h") else source.read_text()
        )
        output.write_text(rename(content))
    print(destination)


if __name__ == "__main__":
    main()
