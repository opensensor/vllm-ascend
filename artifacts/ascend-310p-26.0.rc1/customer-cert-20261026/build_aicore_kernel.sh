#!/usr/bin/env bash
# Rebuild an Ascend AI Core custom-op kernel on the Ascend 310P card host.
#
# Proven end to end on 2026-10-26: a real source change produced a new .o
# (sha256 40afe472… -> fd9302b8…) and opc regenerated the descriptor JSON
# with the matching "sha256" automatically.  No signing key is involved --
# the only integrity value is a plain sha256 that the vendor tool computes
# from our binary.
#
# Usage:
#   build_aicore_kernel.sh <operator-source-dir> <build-dir> <target>
#
# Example:
#   build_aicore_kernel.sh \
#     /srv/ai/src/glm-l1-wide-build-20261004/gmm/w2_swiglu_v310 \
#     /home/matteius/experiments/glm-decode-flags-20261006/build \
#     w2_swiglu_v310_ascend310p_0
set -euo pipefail

SRC_DIR="${1:?operator source dir}"
BUILD_DIR="${2:?build dir}"
TARGET="${3:?cmake target}"

CANN=/usr/local/Ascend/cann-9.1.0
VENV="${AICPU_VENV:-/srv/ai/venvs/fork028}"

# The build cache is keyed on the operator source; drop it or the change is
# silently skipped (a comment-only change also yields an identical .o).
CACHE_ROOT="$(dirname "$(dirname "$SRC_DIR")")/build_cache"
rm -rf "${CACHE_ROOT}/custom_operator" 2>/dev/null || true

sudo env \
  PATH="${VENV}/bin:${CANN}/x86_64-linux/bin:/usr/bin:/bin" \
  LD_LIBRARY_PATH="${CANN}/x86_64-linux/lib64:${CANN}/python/site-packages/tbe:${LD_LIBRARY_PATH:-}" \
  PYTHONPATH="${CANN}/python/site-packages:${PYTHONPATH:-}" \
  ASCEND_HOME_PATH="${CANN}" \
  bash -c "
    source ${CANN}/set_env.sh >/dev/null 2>&1
    make -C '${BUILD_DIR}' '${TARGET}' 2>&1 | tail -20
  "

echo
echo "=== built artefacts ==="
sudo find "${BUILD_DIR}/binary" -name '*.o' -newermt '-10 minutes' 2>/dev/null | while read -r f; do
  echo "  $f"
  sudo sha256sum "$f"
  if sudo test -f "${f%.o}.json"; then
    sudo grep -h '"sha256"' "${f%.o}.json"
  fi
done
