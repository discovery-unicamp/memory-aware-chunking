#!/bin/sh

MEMORY_PRESSURE_SAMPLING_WINDOW_IN_SECONDS="${MEMORY_PRESSURE_SAMPLING_WINDOW_IN_SECONDS:-0.2}"
MEMORY_PRESSURE_SAMPLING_PRECISION="${MEMORY_PRESSURE_SAMPLING_PRECISION:-2}"
MEMORY_PRESSURE_CGROUP_FILE_PATH="${MEMORY_PRESSURE_CGROUP_FILE_PATH:-/proc/pressure/memory}"
MEMORY_EVENTS_CGROUP_FILE_PATH="${MEMORY_EVENTS_CGROUP_FILE_PATH:-/sys/fs/cgroup/memory.events}"

SLEEP_TIME=$(awk "BEGIN {print 1 * 10^(-$MEMORY_PRESSURE_SAMPLING_PRECISION)}")

echo "Timestamp,Some (avg10),Some (avg60),Some (avg300),Full (avg10),Full (avg60),Full (avg300),OOM Kills,OOM Failures"

while true; do
  PSI_OUTPUT=$(cat "$MEMORY_PRESSURE_CGROUP_FILE_PATH")
  CURR_OOM_KILL=$(grep "^oom_kill " "$MEMORY_EVENTS_CGROUP_FILE_PATH" | awk '{print $2}' || echo 0)
  CURR_OOM_FAIL=$(grep "^oom " "$MEMORY_EVENTS_CGROUP_FILE_PATH" | awk '{print $2}' || echo 0)

  if [ -n "$PSI_OUTPUT" ]; then
    SOME_AVG10=$(echo "$PSI_OUTPUT" | grep "^some" | awk '{print $2}' | cut -d= -f2)
    SOME_AVG60=$(echo "$PSI_OUTPUT" | grep "^some" | awk '{print $3}' | cut -d= -f2)
    SOME_AVG300=$(echo "$PSI_OUTPUT" | grep "^some" | awk '{print $4}' | cut -d= -f2)

    FULL_AVG10=$(echo "$PSI_OUTPUT" | grep "^full" | awk '{print $2}' | cut -d= -f2)
    FULL_AVG60=$(echo "$PSI_OUTPUT" | grep "^full" | awk '{print $3}' | cut -d= -f2)
    FULL_AVG300=$(echo "$PSI_OUTPUT" | grep "^full" | awk '{print $4}' | cut -d= -f2)

    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S.%3N')

    stdbuf -oL echo "${TIMESTAMP},${SOME_AVG10},${SOME_AVG60},${SOME_AVG300},${FULL_AVG10},${FULL_AVG60},${FULL_AVG300},${CURR_OOM_KILL},${CURR_OOM_FAIL}"
  fi

  sleep "${SLEEP_TIME}"
done