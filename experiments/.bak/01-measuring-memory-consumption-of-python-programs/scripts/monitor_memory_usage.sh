#!/bin/sh

MEMORY_USAGE_SAMPLING_WINDOW_IN_SECONDS="${MEMORY_USAGE_SAMPLING_WINDOW_IN_SECONDS:-0.2}"
MEMORY_USAGE_CURRENT_CGROUP_FILE_PATH="${MEMORY_USAGE_CURRENT_CGROUP_FILE_PATH:-/sys/fs/cgroup/memory.current}"
MEMORY_USAGE_SAMPLING_PRECISION="${MEMORY_USAGE_SAMPLING_PRECISION:-2}"

SLEEP_TIME=$(awk "BEGIN {print 1 * 10^(-$MEMORY_USAGE_SAMPLING_PRECISION)}")

echo "Timestamp,Memory Usage (kB)"

while true; do
  MEM_USAGE=$(( $(head -n 1 "${MEMORY_USAGE_CURRENT_CGROUP_FILE_PATH}" 2>/dev/null || 0) / 1024 ))
  TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S.%3N' 2>/dev/null)

  if [ -n "$MEM_USAGE" ] && [ -n "$TIMESTAMP" ]; then
    stdbuf -oL echo "${TIMESTAMP},${MEM_USAGE}"
  fi

  sleep "${SLEEP_TIME}"
done