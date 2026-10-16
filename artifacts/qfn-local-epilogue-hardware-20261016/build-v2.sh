#!/bin/bash
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
root=/srv/ai/storage/artifacts/qfn-local-epilogue-hardware-20261016/build-v2
cann=/usr/local/Ascend/ascend-toolkit/latest
compiler=/srv/ai/storage/artifacts/qfn-hc-combine-compile-20261014-v1/build/compile-hc-combine
for name in native_local_down_projection native_local_route_epilogue; do
  [[ ! -e "$root/$name.bin" ]]
  "$compiler" "$root/$name.cpp" "$root/$name.bin" --npu-arch=dav-2002 -std=c++17 "-I$root" \
    "--sysroot=$cann/tools/hcc/sysroot" \
    "-isystem$cann/tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0" \
    "-isystem$cann/tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0/aarch64-target-linux-gnu"
done
sha256sum "$root/"*.bin
