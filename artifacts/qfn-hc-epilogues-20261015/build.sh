#!/usr/bin/env bash
set -eo pipefail
build=$1
compiler=$2
cann=/usr/local/Ascend/ascend-toolkit/latest
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
"$compiler" "$build/native_hc_epilogues.cpp" "$build/native_hc_epilogues.bin" \
  --npu-arch=dav-2002 -std=c++17 "-I$build" \
  "--sysroot=$cann/tools/hcc/sysroot" \
  "-isystem$cann/tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0" \
  "-isystem$cann/tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0/aarch64-target-linux-gnu"
