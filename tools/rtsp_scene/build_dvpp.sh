#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: bash build_dvpp.sh CANN_ROOT OUTPUT_BINARY" >&2
    exit 2
fi
scene_source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cann_root=$1
output_binary=$2
g++ -std=c++17 -O2 -Wall -Wextra -pthread \
    "$scene_source_dir/dvpp_decode.cpp" -o "$output_binary" \
    -I"$cann_root/include" -L"$cann_root/lib64" \
    -Wl,-rpath,"$cann_root/lib64" \
    -lascendcl -lacl_dvpp_mpi -lavformat -lavcodec -lavutil
