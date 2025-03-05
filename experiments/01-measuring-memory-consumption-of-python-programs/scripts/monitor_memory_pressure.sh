#!/bin/sh

MEMORY_PRESSURE_SAMPLING_PRECISION="${MEMORY_PRESSURE_SAMPLING_PRECISION:-2}"
MEMORY_PRESSURE_CGROUP_FILE_PATH="${MEMORY_PRESSURE_CGROUP_FILE_PATH:-/sys/fs/cgroup/memory.pressure}"
MEMORY_EVENTS_CGROUP_FILE_PATH="${MEMORY_EVENTS_CGROUP_FILE_PATH:-/sys/fs/cgroup/memory.events}"

SLEEP_TIME=$(awk "BEGIN {print 1 * 10^(-${MEMORY_PRESSURE_SAMPLING_PRECISION})}")

echo "Timestamp,Some (avg10),Some (avg60),Some (avg300),Full (avg10),Full (avg60),Full (avg300),OOM Kills,OOM Failures"

while true; do
  PSI_OUTPUT=$(cat "${MEMORY_PRESSURE_CGROUP_FILE_PATH}" 2>/dev/null || echo "")
  OOM_STATS=$(cat "${MEMORY_EVENTS_CGROUP_FILE_PATH}" 2>/dev/null || echo "")

  if [ -n "${PSI_OUTPUT}" ]; then
    SOME_AVG10=$(echo "${PSI_OUTPUT}" | grep "^some" | awk '{print $2}' | cut -d= -f2)
    SOME_AVG60=$(echo "${PSI_OUTPUT}" | grep "^some" | awk '{print $3}' | cut -d= -f2)
    SOME_AVG300=$(echo "${PSI_OUTPUT}" | grep "^some" | awk '{print $4}' | cut -d= -f2)

    FULL_AVG10=$(echo "${PSI_OUTPUT}" | grep "^full" | awk '{print $2}' | cut -d= -f2)
    FULL_AVG60=$(echo "${PSI_OUTPUT}" | grep "^full" | awk '{print $3}' | cut -d= -f2)
    FULL_AVG300=$(echo "${PSI_OUTPUT}" | grep "^full" | awk '{print $4}' | cut -d= -f2)

    CURR_OOM_KILL=$(echo "${OOM_STATS}" | awk '/^oom_kill/ {print $2}' | head -n 1)
    CURR_OOM_FAIL=$(echo "${OOM_STATS}" | awk '/^oom[^_]/ {print $2}' | head -n 1)

    CURR_OOM_KILL="${CURR_OOM_KILL:-0}"
    CURR_OOM_FAIL="${CURR_OOM_FAIL:-0}"

    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S.%3N' 2>/dev/null)

    echo "${TIMESTAMP},${SOME_AVG10},${SOME_AVG60},${SOME_AVG300},${FULL_AVG10},${FULL_AVG60},${FULL_AVG300},${CURR_OOM_KILL},${CURR_OOM_FAIL}"
  fi

  sleep "${SLEEP_TIME}"
done