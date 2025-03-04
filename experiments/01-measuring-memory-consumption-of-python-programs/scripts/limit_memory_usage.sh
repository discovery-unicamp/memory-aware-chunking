#!/bin/bash

MEMORY_LIMIT_MB=${MEMORY_LIMIT_MB:"max"}
CGROUP_PATH=${CGROUP_PATH:-"/sys/fs/cgroup"}

if [[ "${MEMORY_LIMIT_MB}" == "max" ]]; then
    MEMORY_LIMIT_BYTES="max"
else
    MEMORY_LIMIT_BYTES=$((MEMORY_LIMIT_MB * 1024 * 1024))
fi

echo "$MEMORY_LIMIT_BYTES" > "${CGROUP_PATH}/memory.max"

echo "Cgroup setup complete with limit: ${MEMORY_LIMIT_BYTES} bytes"