#!/usr/bin/env bash
set -eo pipefail
root=/srv/ai/storage/artifacts/qfn-local-pack-20261016
cann=/usr/local/Ascend/ascend-toolkit/latest
compiler=/srv/ai/storage/artifacts/qfn-hc-combine-compile-20261014-v1/build/compile-hc-combine
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
frozen=/srv/ai/storage/artifacts/qfn-route-finalize-compile-20261014-v1/build-queue-v3
[[ $(sha256sum "$frozen/native_int4_schedule.h" | cut -d " " -f 1) == cebf66752cfe9a9606a584dab07a25dbf8a4f8a8d4057ad0a372dfb65c7a7f70 ]]
[[ ! -e "$root/schedule-m16.bin" ]]
"$compiler" "$root/schedule-m16.cpp" "$root/schedule-m16.bin" \
  --npu-arch=dav-2002 -std=c++17 "-I$frozen" \
  "--sysroot=$cann/tools/hcc/sysroot" \
  "-isystem$cann/tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0" \
  "-isystem$cann/tools/hcc/aarch64-target-linux-gnu/include/c++/7.3.0/aarch64-target-linux-gnu"
