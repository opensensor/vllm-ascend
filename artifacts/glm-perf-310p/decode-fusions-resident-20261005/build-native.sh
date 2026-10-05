#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# CPU compilation only. Does not initialize an NPU or alter a running server.
set -euo pipefail
glm_study_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
glm_cann_root=/usr/local/Ascend/cann-9.1.0
cd -- "$glm_study_root"
if [[ -e rotation-v1.bin || -e glm_rotation_bridge_v1.so ]]; then
  echo "Refusing to overwrite native artifacts; build a new version in a fresh directory." >&2
  exit 1
fi
c++ ../native-resident-20261005/compile-kernel.cpp \
  -I"$glm_cann_root/include" -L"$glm_cann_root/lib64" \
  -lacl_rtc -lascendcl -o compile-kernel
./compile-kernel rotation.cpp rotation-v1.bin
ninja -f build.ninja
# Before loading, generate a manifest with hashes of these exact outputs.
# Do not overwrite a version already loaded by resident workers.
