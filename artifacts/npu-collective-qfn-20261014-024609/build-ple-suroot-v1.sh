set -e
R=/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609
C=/usr/local/Ascend/cann-9.1.0
I=$C/x86_64-linux
# Device-side AI CPU kernel (aarch64), with the pty+su escalation compiled in.
"$C/tools/hcc/bin/aarch64-target-linux-gnu-g++" -std=c++17 -O2 -Wall -Wextra -fPIC -shared -DPLE_AICPU_DEVICE -I"$I/include" -I"$I/include/driver" -I"$I/pkg_inc" "$R/ple_aicpu_probe.cpp" -ldl -o "$R/libple_aicpu_probe_suroot.so"
# Host-side loader.
g++ -std=c++17 -O2 -Wall -Wextra -I"$I/include" -I"$I/include/driver" -I"$I/pkg_inc" "$R/ple_aicpu_probe.cpp" -L"$I/lib64" -L/usr/local/Ascend/driver/lib64/driver -lascendcl -lascend_hal -o "$R/ple_aicpu_probe_suroot"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
timeout 40 "$R/ple_aicpu_probe_suroot" "$R/libple_aicpu_probe_suroot.so" suroot > "$R/ple-aicpu-suroot-v1.jsonl" 2> "$R/ple-aicpu-suroot-v1.stderr"
