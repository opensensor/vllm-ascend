#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build a recovered vendor in a new directory. Does not install or touch an NPU.
set -euo pipefail

if (( $# < 3 || $# > 6 )); then
    echo "Usage: bash tools/qwen4exp/build_custom_ops.sh {coherent|retained|gdn|embedded} SDK_ROOT NEW_WORK_DIR [host|full] [JOBS] [THIRD_PARTY_CACHE]" >&2
    exit 2
fi
package=$1
sdk=$(realpath -e -- "$2")
work=$(realpath -m -- "$3")
mode=${4:-host}
jobs=${5:-4}
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
handoff="$repo/csrc/release_sources/qwen38_single_card_20261011"
case "$work/" in "$handoff/"*) echo "Build outside the source handoff" >&2; exit 2 ;; esac
case "$package" in coherent|retained|gdn|embedded) ;; *) echo "Unknown package" >&2; exit 2 ;; esac
case "$mode" in host|full) ;; *) echo "Mode must be host or full" >&2; exit 2 ;; esac
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { echo "JOBS must be positive" >&2; exit 2; }
[[ -x "$sdk/tools/opbuild/op_build" ]] || { echo "Missing SDK op_build" >&2; exit 2; }
python3 -c 'import regex' || { echo "Activate a Python build environment with regex and the CANN build dependencies" >&2; exit 2; }
[[ ! -e "$work" ]] || { echo "Use a new work directory: $work" >&2; exit 2; }
mapfile -t config < <(python3 - "$handoff/manifest.json" "$package" <<'PY'
import json
import sys
config = json.load(open(sys.argv[1]))['packages'][sys.argv[2]]
print(config['vendor'])
print(','.join(config['operators']))
PY
)
(( ${#config[@]} == 2 )) || { echo "Cannot read package configuration" >&2; exit 2; }
if (( $# == 6 )); then
    third_party=$(realpath -e -- "$6")
else
    third_party="$work/csrc/third_party"
fi

# Source the target SDK environment in the caller's shell before invoking this.
# Pin both CMake discovery and the generator to the requested SDK.
export ASCEND_HOME_PATH="$sdk"
export ASCEND_OPP_PATH="$sdk/opp"
export LD_LIBRARY_PATH="$sdk/lib64:${LD_LIBRARY_PATH:-}"
export PATH="$sdk/bin:$sdk/compiler/ccec_compiler/bin:$PATH"
export MAX_JOBS="$jobs" CMAKE_BUILD_PARALLEL_LEVEL="$jobs"
python3 "$repo/tools/qwen4exp/custom_op_sources.py" "$package" "$work"
# The historical GDN command supplied these headers through CPATH.
export CPATH="$work/csrc/third_party/catlass/include:${CPATH:-}"
cd -- "$work/csrc"
args=(--pkg "--ops=${config[1]}" --soc=ascend310p "--vendor_name=${config[0]}"
      --package-path "$sdk" "--cann_3rd_lib_path=$third_party" "-j$jobs")
if [[ "$mode" == host ]]; then args+=(--jit); fi
printf '%q ' bash build.sh "${args[@]}" | tee "$work/command.txt"
printf '\n' | tee -a "$work/command.txt"
bash build.sh "${args[@]}" 2>&1 | tee "$work/build.log"
test -s build/libcust_opmaster_rt2.0.so
test -s build/libcust_opapi.so
sha256sum build/libcust_opmaster_rt2.0.so build/libcust_opapi.so > "$work/libraries.sha256"
readelf --version-info build/libcust_opapi.so > "$work/opapi-version-info.txt"
readelf --version-info build/libcust_opmaster_rt2.0.so > "$work/ophost-version-info.txt"
nm -D --defined-only build/libcust_opapi.so > "$work/opapi-symbols.txt"
echo "Build artifacts: $work/csrc/build (not installed; inference not validated)"
