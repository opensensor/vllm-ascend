#!/usr/bin/env bash
set -euo pipefail
build=$1
source_root=$2
catlass_root=$3
compiler=$4
cann=/usr/local/Ascend/ascend-toolkit/latest
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
mkdir -p "$build"
cp "$(dirname "$0")/native-gdn-output.cpp" "$build/"
cp -a "$source_root/csrc/moe/chunk_fwd_o_vllm/op_kernel/arch20" "$build/"
cp -a "$source_root/csrc/moe/common/kernel_utils" "$build/"
cp -a "$catlass_root/include" "$build/catlass-include"
"$compiler" "$build/native-gdn-output.cpp" "$build/native-gdn-output.bin" \
  --npu-arch=dav-2002 -std=c++17 \
  "-I$build" "-I$build/catlass-include" \
  "--sysroot=$cann/tools/hcc/sysroot" \
  "-isystem$cann/tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0" \
  "-isystem$cann/tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0/aarch64-target-linux-gnu"

python3 "$(dirname "$0")/write-gdn-provenance.py" "$build" "$compiler"
