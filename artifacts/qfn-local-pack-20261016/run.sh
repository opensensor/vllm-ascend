#!/bin/bash
set -eo pipefail
source /srv/ai/src/qwen38-w4-hardware-20260927/qwen38-w4-hardware-env.sh >/dev/null 2>&1
export PYTHONPATH=/srv/ai/storage/artifacts/ple-host-gather-20261015/serving-source-v2:/srv/ai/storage/artifacts/npu-collective-qfn-20261014-024609/qfn-pinned-position-runtime:$PYTHONPATH
cd /srv/ai/storage/artifacts/ple-host-gather-20261015/serving-source-v2
exec /srv/ai/venvs/qwen38-w4-test-ce1862/bin/python "$@"
