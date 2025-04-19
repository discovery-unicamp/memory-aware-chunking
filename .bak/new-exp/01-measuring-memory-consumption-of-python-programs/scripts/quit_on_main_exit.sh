#!/bin/bash

QUIT_ON_MAIN_LOG_FILE=${QUIT_ON_MAIN_LOG_FILE:-"/app/logs/quit-on-main.log"}

while true; do
  echo "Listening for events..." >> "$QUIT_ON_MAIN_LOG_FILE"
  echo "READY" >&1; sleep 0.01

  read -r header
  echo "$header" >> "$QUIT_ON_MAIN_LOG_FILE"

  len=$(echo "$header" | grep -oE 'len:[0-9]+' | cut -d: -f2)
  echo "Payload length: $len" >> "$QUIT_ON_MAIN_LOG_FILE"

  payload=""
  if [[ "$len" -gt 0 ]]; then
      IFS= read -r -n "$len" payload
  fi

  echo "Payload: $payload" >> "$QUIT_ON_MAIN_LOG_FILE"

  process_name=$(echo "$payload" | grep -oE 'processname:[^ ]+' | cut -d: -f2)
  echo "Detected process: $process_name" >> "$QUIT_ON_MAIN_LOG_FILE"

  if [[ "$process_name" == "quit_on_main_exit" ]]; then
      echo "Ignoring self-triggered exit event." >> "$QUIT_ON_MAIN_LOG_FILE"
  elif [[ "$process_name" == "main" ]]; then
      echo "Main process exited, shutting down..." >> "$QUIT_ON_MAIN_LOG_FILE"
      kill -SIGQUIT $PPID  # Shutdown Supervisor
  fi

  echo -e -n "RESULT 2\nOK" >&1; sleep 0.01
done