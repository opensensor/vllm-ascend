#!/usr/bin/env bash
set -u
root=/home/matteius/experiments/glm-next-queue-20261005
for label in adaptive2 adaptive4; do
    bash "$root/isolated-env.sh" "$root/opp-$label" group "$label" > "$root/group-$label.log" 2>&1
    printf '%s %s\n' "$label" "$?" >> "$root/isolated-exits.txt"
done
bash "$root/isolated-env.sh" /srv/ai/src/kda-persistent-scores-opp kda baseline > "$root/kda-baseline.log" 2>&1
printf 'kda-baseline %s\n' "$?" >> "$root/isolated-exits.txt"
for label in kda-head kda-skip; do
    bash "$root/isolated-env.sh" "$root/opp-$label" kda "$label" > "$root/$label.log" 2>&1
    printf '%s %s\n' "$label" "$?" >> "$root/isolated-exits.txt"
done
python3 /home/matteius/experiments/glm-kpool-live-score-20261005/launch.py --context 311040 --batch 640 --mixer native --resident public-local-control --scorer native --label queue10-baseline2
set +u
source /srv/ai/bin/ascend-env.sh
cd /srv/ai/src/glm-selective-w3-nz-test-20261004
/srv/ai/venvs/qwen38-w4-test-ce1862/bin/python "$root/run-python.py" > "$root/python-run-final.log" 2>&1
