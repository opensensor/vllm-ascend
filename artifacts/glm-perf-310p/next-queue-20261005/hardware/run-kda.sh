#!/usr/bin/env bash
set -u
root=/home/matteius/experiments/glm-next-queue-20261005
for label in baseline kda-head kda-skip; do
    package="$root/opp-$label"
    if [[ "$label" == baseline ]]; then package=/srv/ai/src/kda-persistent-scores-opp; fi
    bash "$root/isolated-env.sh" "$package" kda "$label" > "$root/kda-$label-fixed.log" 2>&1
    printf '%s %s\n' "$label" "$?" >> "$root/kda-fixed-exits.txt"
done
python3 /home/matteius/experiments/glm-kpool-live-score-20261005/launch.py --context 311040 --batch 640 --mixer native --resident public-local-control --scorer native --label queue10-baseline2
set +u
source /srv/ai/bin/ascend-env.sh
cd /srv/ai/src/glm-selective-w3-nz-test-20261004
/srv/ai/venvs/qwen38-w4-test-ce1862/bin/python "$root/run-python.py" > "$root/python-run-final.log" 2>&1
