#!/bin/sh

MEMORY_USAGE_SAMPLING_WINDOW_IN_SECONDS="${MEMORY_USAGE_SAMPLING_WINDOW_IN_SECONDS:-0.2}"
MEMORY_USAGE_CURRENT_CGROUP_FILE_PATH="${MEMORY_USAGE_CURRENT_CGROUP_FILE_PATH:-/sys/fs/cgroup/memory.current}"

echo "Timestamp,Memory Usage (kB)"

while true; do
  MEM_USAGE=$(( $(head -n 1 "${MEMORY_USAGE_CURRENT_CGROUP_FILE_PATH}") / 1024 ))

  if [ -n "${MEM_USAGE}" ]; then
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S.%3N')
    stdbuf -oL echo "${TIMESTAMP},${MEM_USAGE}"
  fi

  sleep "${MEMORY_USAGE_SAMPLING_WINDOW_IN_SECONDS}"
done