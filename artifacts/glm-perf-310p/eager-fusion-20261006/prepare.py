# SPDX-License-Identifier: Apache-2.0
"""Export existing fusion kernels with versioned resident bridge entry points."""

import hashlib
import json
from pathlib import Path


def main():
    study = Path(__file__).resolve().parent
    root = study.parents[2]
    kernels = {
        "swiglu": (
            "w2_swiglu_v310",
            "swiglu_geometry.h",
            "SwigluTiling",
            "Swiglu",
            "GM_ADDR gate_up, GM_ADDR y, GM_ADDR config",
            "gate_up, y",
        ),
        "combine": (
            "w2_route_combine_v310",
            "route_combine_geometry.h",
            "RouteCombineTiling",
            "RouteCombine",
            "GM_ADDR routed, GM_ADDR inverse, GM_ADDR weights, GM_ADDR ends, GM_ADDR y, GM_ADDR config",
            "routed, inverse, weights, ends, y",
        ),
        "mhc_post": (
            "glm_mhc_post_v310",
            "mhc_post_geometry.h",
            "MhcPostTiling",
            "MhcPost",
            "GM_ADDR x, GM_ADDR residual, GM_ADDR post, GM_ADDR comb, GM_ADDR y, GM_ADDR config",
            "x, residual, post, comb, y",
        ),
    }
    hashes = {}
    for label, (directory, geometry, tiling, cls, arguments, call) in kernels.items():
        source_root = root / "csrc/gmm" / directory
        source = (source_root / "op_kernel" / (directory + ".cpp")).read_text()
        geometry_text = (source_root / geometry).read_text()
        hashes[label] = {
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "geometry_sha256": hashlib.sha256(geometry_text.encode()).hexdigest(),
        }
        source = source.replace(f'#include "{geometry}"', geometry_text)
        source = source.split('extern "C" __global__')[0]
        # Config tensor preserves the existing int64 tiling layout. All inputs
        # and configuration tensors are retained by the validated launch bridge.
        fields = "raw->tokens, raw->hidden, raw->top_k, raw->experts" if label == "combine" else "raw->rows, raw->width"
        source += f"""extern "C" __global__ __aicore__ void glm_resident_{label}_v1({arguments}) {{
  AscendC::InitSocState();
  auto raw = reinterpret_cast<__gm__ {tiling}*>(config);
  const {tiling} td{{{fields}}};
  {cls} op;
  op.Run({call}, td);
}}
"""
        (study / f"{label}.cpp").write_text(source)
    (study / "source-hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")


if __name__ == "__main__":
    main()
