#!/usr/bin/env bash
set -eo pipefail
root=/srv/ai/storage/artifacts/qfn-recovery-20261016
cann=/usr/local/Ascend/ascend-toolkit/latest
compiler=/srv/ai/storage/artifacts/qfn-hc-combine-compile-20261014-v1/build/compile-hc-combine
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
[[ ! -e "$root/native_activation_epilogue.bin" ]]
"$compiler" "$root/native_activation_epilogue.cpp" "$root/native_activation_epilogue.bin" \
  --npu-arch=dav-2002 -std=c++17 "-I$root" \
  "--sysroot=$cann/tools/hcc/sysroot" \
  "-isystem$cann/tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0" \
  "-isystem$cann/tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0/aarch64-target-linux-gnu"
