# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate isolated 40-route gate/up patches without editing serving source."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path

KERNEL_DIR = Path("csrc/gmm/qwen_w4_a8_int4_matmul_v310/op_kernel")
MATMUL = KERNEL_DIR / "qwen_w4_a8_int4_matmul_v310.cpp"
SCHEDULE = KERNEL_DIR / "native_int4_schedule.h"
POLICY = KERNEL_DIR / "gate_up40_policy.h"
PROFILES = ("tile40", "stream40")
POLICY_SOURCE = """// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#ifndef QWEN_GATE_UP40_POLICY_H
#define QWEN_GATE_UP40_POLICY_H
#include <cstdint>
namespace native_int4 {
constexpr uint32_t EXTENDED_GATE_UP_ROUTE_LIMIT = 40;
// Isolate MTP3 C1's 31..40 routed rows. Preserve grouped/prefill, down,
// the selected <=30-row path, and every >=41-row concurrency schedule.
template <typename TilingData>
__aicore__ inline bool UseExtendedGateUpSchedule(const TilingData& td) {
  return td.metadataLanes == 8 && td.routed != 0 && td.numRows > 30 &&
         td.numRows <= EXTENDED_GATE_UP_ROUTE_LIMIT && td.nDim == 1280 && td.kDim == 2560;
}
}  // namespace native_int4
#endif
"""


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"candidate source changed; expected one occurrence of {old!r}")
    return source.replace(old, new, 1)


def candidate_files(root: Path, profile: str):
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile}")
    source = (root / MATMUL).read_text()
    source = replace_once(
        source,
        '#include "native_int4_schedule.h"',
        '#include "native_int4_schedule.h"\n#include "gate_up40_policy.h"',
    )
    source = replace_once(
        source,
        "if (modelC1GateUp || modelDecodeDown)",
        "if (modelC1GateUp || native_int4::UseExtendedGateUpSchedule(*td) || modelDecodeDown)",
    )
    source = replace_once(
        source,
        "// cores per tile. Larger gate/up concurrency and every prefill shape",
        "// cores per tile. The experimental 31..40 routed gate/up rows also\n"
        "      // use this width. Larger concurrency and every prefill shape",
    )
    files = {MATMUL: source, POLICY: POLICY_SOURCE}
    if profile == "stream40":
        # Only the narrowly selected gate/up specialization receives the
        # extended stream budget. The down/reduction specialization keeps 30.
        schedule = (root / SCHEDULE).read_text()
        # Newer bases also expose metadata caching. Preserve its default and
        # append the route budget after it, rather than binding 40 to a bool.
        cached_template = "bool ENABLE_DIRECT_LOOKUP = false, bool CACHE_GROUP_METADATA = false>"
        template = cached_template if cached_template in schedule else "bool ENABLE_DIRECT_LOOKUP = false>"
        arguments = "16, N, false, true, false" + (", false" if template == cached_template else "")
        schedule = replace_once(schedule, template, template[:-1] + ", uint32_t PIPELINED_ROUTE_LIMIT = 30>")
        schedule = replace_once(
            schedule,
            "MAX_PIPELINED_ROUTES = 30;",
            "MAX_PIPELINED_ROUTES = PIPELINED_ROUTE_LIMIT;",
        )
        schedule = replace_once(
            schedule,
            "// The streamed schedule is qualified only for the 30-route c1 shape.",
            "// The default streamed budget stays at 30; only the experimental\n"
            "    // gate/up specialization receives an explicit 40-row budget.",
        )
        source = replace_once(
            source,
            "if (td->numRows <= DECODE_ROUTE_LIMIT) {\n    if (td->numRows <= MODEL_C1_ROUTE_LIMIT) {",
            "if (td->numRows <= DECODE_ROUTE_LIMIT) {\n"
            "    if constexpr (N == MODEL_DECODE_COLUMNS) {\n"
            "      if (native_int4::UseExtendedGateUpSchedule(*td)) {\n"
            f"        native_int4::Schedule<{arguments}, native_int4::EXTENDED_GATE_UP_ROUTE_LIMIT> op;\n"
            "        op.Init(low, high, xs, sums, codes, scale, offset, weight_sum, ends, nullptr, y, td);\n"
            "        op.Process();\n"
            "        return;\n"
            "      }\n"
            "    }\n"
            "    if (td->numRows <= MODEL_C1_ROUTE_LIMIT) {",
        )
        files[MATMUL] = source
        files[SCHEDULE] = schedule
    return files


def patch_text(root: Path, files):
    chunks = []
    for path, content in files.items():
        original = (root / path).read_text() if (root / path).exists() else ""
        chunks.extend(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                content.splitlines(keepends=True),
                fromfile=f"a/{path}" if original else "/dev/null",
                tofile=f"b/{path}",
            )
        )
    return "".join(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="complete source snapshot used as the base")
    parser.add_argument("--output", type=Path, required=True, help="new artifact directory")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output directory")
    prepared = {profile: candidate_files(args.root, profile) for profile in PROFILES}
    args.output.mkdir(parents=True)
    report = {
        "scope": "unapplied_offline_gate_up_kernel_candidates",
        "root": str(args.root.resolve()),
        "npu_used": False,
        "abi_changed": False,
        "down_fusion_capacity_changed": False,
        "base_sha256": {
            str(path): hashlib.sha256((args.root / path).read_bytes()).hexdigest() for path in (MATMUL, SCHEDULE)
        },
        "profiles": {},
    }
    for profile, files in prepared.items():
        patch = patch_text(args.root, files)
        (args.output / f"{profile}.patch").write_text(patch)
        report["profiles"][profile] = {
            "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
            "candidate_sha256": {
                str(path): hashlib.sha256(content.encode()).hexdigest() for path, content in files.items()
            },
        }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
