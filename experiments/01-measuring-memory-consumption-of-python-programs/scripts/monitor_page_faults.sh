#!/bin/sh

PAGE_FAULTS_SAMPLING_WINDOW_IN_SECONDS="${PAGE_FAULTS_SAMPLING_WINDOW_IN_SECONDS:-0.2}"
PAGE_FAULTS_MONITORED_PROCESS_NAME="${PAGE_FAULTS_MONITORED_PROCESS_NAME:-main.py}"
PAGE_FAULTS_PID_RETRY_MAX_ATTEMPTS="${PAGE_FAULTS_PID_RETRY_MAX_ATTEMPTS:-20}"
PAGE_FAULTS_PID_RETRY_SLEEP_TIME="${PAGE_FAULTS_PID_RETRY_SLEEP_TIME:-0.2}"

attempt=0
while [ $attempt -lt "${PAGE_FAULTS_PID_RETRY_MAX_ATTEMPTS}" ]; do
  PAGE_FAULTS_MONITORED_PROCESS_PID=$(pgrep -f "${PAGE_FAULTS_MONITORED_PROCESS_NAME}")

  if [ -n "${PAGE_FAULTS_MONITORED_PROCESS_PID}" ]; then
    break
  fi

  attempt=$((attempt + 1))
  sleep "${PAGE_FAULTS_PID_RETRY_SLEEP_TIME}"
done

if [ -z "$PAGE_FAULTS_MONITORED_PROCESS_PID" ]; then
  echo "Experiment process '${PAGE_FAULTS_MONITORED_PROCESS_NAME}' not found after ${PAGE_FAULTS_PID_RETRY_MAX_ATTEMPTS} attempts. Exiting..."
  exit 1
fi

echo "Timestamp,Minor Page Faults,Major Page Faults"

while true; do
  if [ ! -f "/proc/${PAGE_FAULTS_MONITORED_PROCESS_PID}/stat" ]; then
    exit 1
  fi

  STATS=$(cat "/proc/${PAGE_FAULTS_MONITORED_PROCESS_PID}/stat")

  TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S.%3N')
  MIN_FLT=$(echo "${STATS}" | awk '{print $10}')  # Minor Page Faults
  MAJ_FLT=$(echo "${STATS}" | awk '{print $12}')  # Major Page Faults

  echo "${TIMESTAMP},${MIN_FLT},${MAJ_FLT}"

  sleep "${PAGE_FAULTS_SAMPLING_WINDOW_IN_SECONDS}"
done