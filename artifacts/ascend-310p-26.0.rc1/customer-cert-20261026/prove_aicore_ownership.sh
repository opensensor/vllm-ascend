#!/usr/bin/env bash
# Prove that the AI Core binary executing on the card is OUR rebuilt binary.
#
# The 2026-10-26 session proved the BUILD works (a source edit produced a new
# .o and opc regenerated the descriptor sha256).  What it never proved is that
# the rebuilt binary is what actually RUNS on the card -- the serving harness
# died before any op was dispatched.
#
# This script closes that gap with a semantic, observable change:
#
#   1. baseline   : stock source -> build -> install -> run op -> record output
#   2. modified   : source with a scale factor of 2.0 on the output -> build
#                   -> install -> run op -> record output
#   3. assert     : modified output == 2 * baseline output
#
# Step 3 is what proves control: the only way the device output can change is
# if the card executed OUR recompiled AscendC binary.
#
# The source is restored to its original content at the end (trap).
set -euo pipefail

CANN=/usr/local/Ascend/cann-9.1.0
VENV="${AICPU_VENV:-/srv/ai/venvs/fork028}"
SRC=/srv/ai/src/glm-l1-wide-build-20261004/gmm/w2_swiglu_v310/op_kernel/w2_swiglu_v310.cpp
BUILD=/home/matteius/experiments/glm-decode-flags-20261006/build
PKG=/home/matteius/experiments/glm-decode-flags-20261006/opp/vendors/custom_transformer
KERNELDIR="$PKG/op_impl/ai_core/tbe/kernel/ascend310p/w2_swiglu_v310"
TARGET=w2_swiglu_v310_ascend310p_0
PROBE=/var/local/ascend-work/tools/prove_aicore_exec.py

BACKUP=/var/local/ascend-work/aicore-src-backup.cpp
cp "$SRC" "$BACKUP"
restore() {
  cp "$BACKUP" "$SRC"
  echo "--- source restored"
}
trap restore EXIT

echo "=== 0. baseline source sha256 ==="
sha256sum "$SRC"

echo
echo "=== 1. build BASELINE kernel ==="
sudo rm -rf /srv/ai/src/glm-l1-wide-build-20261004/build_cache/custom_operator
sudo env PATH="${VENV}/bin:${CANN}/x86_64-linux/bin:/usr/bin:/bin" \
  LD_LIBRARY_PATH="${CANN}/x86_64-linux/lib64:${CANN}/python/site-packages/tbe:${LD_LIBRARY_PATH:-}" \
  PYTHONPATH="${CANN}/python/site-packages:${PYTHONPATH:-}" \
  ASCEND_HOME_PATH="${CANN}" \
  bash -c "source ${CANN}/set_env.sh >/dev/null 2>&1; make -C '${BUILD}' '${TARGET}' 2>&1 | tail -6"

BASE_O=$(find "$BUILD/binary" -name 'W2SwigluV310_*.o' -newermt '-5 minutes' | head -1)
echo "baseline .o : $BASE_O"
sudo sha256sum "$BASE_O"
sudo cp "$BASE_O" "$KERNELDIR/"
sudo cp "${BASE_O%.o}.json" "$KERNELDIR/"
sudo chown matteius:matteius "$KERNELDIR"/* 2>/dev/null || true
grep -h '"sha256"' "$KERNELDIR"/*.json

echo
echo "=== 2. run BASELINE op, capture output ==="
export ASCEND_HOME_PATH="$CANN"
source "${CANN}/set_env.sh" >/dev/null 2>&1
export ASCEND_CUSTOM_OPP_PATH="$PKG"
cd /home/matteius/experiments/glm-decode-flags-20261006
"$VENV/bin/python" - <<'PY' > /var/local/ascend-work/aicore-baseline.txt
import torch, torch_npu
torch.ops.load_library("/home/matteius/experiments/glm-decode-flags-20261006/glm_decode_flags.so")
torch.npu.set_device(0)
torch.manual_seed(0)
x = torch.randn(8, 4096, dtype=torch.float16)
out = torch.ops._C_ascend.npu_w2_swiglu_310(x.npu()).float().cpu()
torch.save(x, "/var/local/ascend-work/aicore-x.pt")
torch.save(out, "/var/local/ascend-work/aicore-baseline.pt")
print("baseline checksum", float(out.double().sum()))
PY
cat /var/local/ascend-work/aicore-baseline.txt

echo
echo "=== 3. patch source: scale the output by 2.0 ==="
python3 - "$SRC" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
anchor = "      Cast(gateHalf, gate, RoundMode::CAST_NONE, count);"
assert anchor in s, "anchor not found -- source layout changed"
patched = s.replace(anchor,
    "      Muls(gate, gate, 2.0f, count);\n"
    "      PipeBarrier<PIPE_V>();\n" + anchor, 1)
open(p, "w").write(patched)
print("patched: added Muls(gate, gate, 2.0f, count) before the output cast")
PY
grep -n "2.0f" "$SRC"

echo
echo "=== 4. build MODIFIED kernel ==="
sudo rm -rf /srv/ai/src/glm-l1-wide-build-20261004/build_cache/custom_operator
sudo env PATH="${VENV}/bin:${CANN}/x86_64-linux/bin:/usr/bin:/bin" \
  LD_LIBRARY_PATH="${CANN}/x86_64-linux/lib64:${CANN}/python/site-packages/tbe:${LD_LIBRARY_PATH:-}" \
  PYTHONPATH="${CANN}/python/site-packages:${PYTHONPATH:-}" \
  ASCEND_HOME_PATH="${CANN}" \
  bash -c "source ${CANN}/set_env.sh >/dev/null 2>&1; make -C '${BUILD}' '${TARGET}' 2>&1 | tail -6"

MOD_O=$(find "$BUILD/binary" -name 'W2SwigluV310_*.o' -newermt '-5 minutes' | head -1)
echo "modified .o : $MOD_O"
sudo sha256sum "$MOD_O"
sudo cp "$MOD_O" "$KERNELDIR/"
sudo cp "${MOD_O%.o}.json" "$KERNELDIR/"
sudo chown matteius:matteius "$KERNELDIR"/* 2>/dev/null || true
grep -h '"sha256"' "$KERNELDIR"/*.json

echo
echo "=== 5. run MODIFIED op and compare against baseline ==="
export ASCEND_HOME_PATH="$CANN"
source "${CANN}/set_env.sh" >/dev/null 2>&1
export ASCEND_CUSTOM_OPP_PATH="$PKG"
cd /home/matteius/experiments/glm-decode-flags-20261006
"$VENV/bin/python" - <<'PY'
import torch, torch_npu
torch.ops.load_library("/home/matteius/experiments/glm-decode-flags-20261006/glm_decode_flags.so")
torch.npu.set_device(0)
x = torch.load("/var/local/ascend-work/aicore-x.pt")
base = torch.load("/var/local/ascend-work/aicore-baseline.pt")
mod = torch.ops._C_ascend.npu_w2_swiglu_310(x.npu()).float().cpu()
torch.save(mod, "/var/local/ascend-work/aicore-modified.pt")

scale = (mod / base.clamp(min=1e-6))
ratio_med = float(scale.median())
err = float((mod - 2.0 * base).abs().max())
rel = err / (2.0 * float(base.abs().max()) + 1e-9)
print(f"median(mod/base)  = {ratio_med:.6f}   (expect 2.0)")
print(f"max|mod - 2*base| = {err:.6f}  rel={rel:.8f}")
ok = abs(ratio_med - 2.0) < 0.01 and rel < 0.01
print("RESULT:", "OUR REBUILT KERNEL EXECUTED ON THE CARD" if ok else "NO PROOF -- output did not change as predicted")
raise SystemExit(0 if ok else 1)
PY
