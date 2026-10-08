#!/usr/bin/env bash
set -euo pipefail

output=${1:?CSV output path required}
if [[ ! -s "$output" ]]; then
    printf 'utc_epoch,pid,read_bytes,rss_kb\n' > "$output"
fi
while true; do
    stamp=$(date -u +%s)
    ps -eo pid=,comm= | awk '$2 ~ /^VLLM::Worker/ {print $1}' | while read -r pid; do
        if [[ -r "/proc/$pid/io" && -r "/proc/$pid/status" ]]; then
            read_bytes=$(awk '$1 == "read_bytes:" {print $2}' "/proc/$pid/io")
            rss_kb=$(awk '$1 == "VmRSS:" {print $2}' "/proc/$pid/status")
            printf '%s,%s,%s,%s\n' "$stamp" "$pid" "$read_bytes" "$rss_kb" >> "$output"
        fi
    done
    sleep 3
done
