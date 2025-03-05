FROM traceq AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

FROM base AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

FROM base AS final

RUN apt-get update && \
    apt-get install -y supervisor procps cgroup-tools && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG SAMPLING_PRECISION="2"

ARG MEMORY_USAGE_LOG_FILE="/app/logs/memory-usage.log"
ARG MEMORY_USAGE_CURRENT_CGROUP_FILE_PATH="/sys/fs/cgroup/memory.current"
ARG MEMORY_USAGE_SAMPLING_PRECISION=${MEMORY_USAGE_SAMPLING_PRECISION}

ARG PAGE_FAULTS_LOG_FILE="/app/logs/page-faults.log"
ARG PAGE_FAULTS_MONITORED_PROCESS_NAME="operate envelope"
ARG PAGE_FAULTS_PID_RETRY_MAX_ATTEMPTS="20"
ARG PAGE_FAULTS_PID_RETRY_SLEEP_TIME="0.2"
ARG PAGE_FAULTS_SAMPLING_PRECISION=${SAMPLING_PRECISION}

ARG MEMORY_PRESSURE_LOG_FILE="/app/logs/memory-pressure.log"
ARG MEMORY_PRESSURE_SAMPLING_PRECISION=${SAMPLING_PRECISION}
ARG MEMORY_PRESSURE_CGROUP_FILE_PATH="/sys/fs/cgroup/memory.pressure"
ARG MEMORY_FAILCNT_CGROUP_FILE_PATH="/sys/fs/cgroup/memory.events"

ARG MPLCONFIGDIR="/opt/app/mplconfig"

ARG MEMORY_LIMIT_MB="max"
ARG MEMORY_LIMIT_BYTES
ARG CGROUP_PATH="/sys/fs/cgroup"

ARG SUPERVISORD_LOG_FILE="/app/logs/supervisord.log"
ARG SUPERVISORD_PID_FILE="/opt/app/supervisord.pid"

ARG MAIN_LOG_FILE="/app/logs/main.log"

ARG LIMIT_MEMORY_USAGE_LOG_FILE="/app/logs/limit-memory-usage.log"

ARG QUIT_ON_MAIN_LOG_FILE="/app/logs/quit-on-main.log"
ENV MEMORY_USAGE_LOG_FILE=${MEMORY_USAGE_LOG_FILE} \
    MEMORY_USAGE_CURRENT_CGROUP_FILE_PATH=${MEMORY_USAGE_CURRENT_CGROUP_FILE_PATH} \
    MEMORY_USAGE_SAMPLING_PRECISION=${MEMORY_USAGE_SAMPLING_PRECISION} \
    PAGE_FAULTS_LOG_FILE=${PAGE_FAULTS_LOG_FILE} \
    PAGE_FAULTS_MONITORED_PROCESS_NAME=${PAGE_FAULTS_MONITORED_PROCESS_NAME} \
    PAGE_FAULTS_PID_RETRY_MAX_ATTEMPTS=${PAGE_FAULTS_PID_RETRY_MAX_ATTEMPTS} \
    PAGE_FAULTS_PID_RETRY_SLEEP_TIME=${PAGE_FAULTS_PID_RETRY_SLEEP_TIME} \
    PAGE_FAULTS_SAMPLING_PRECISION=${PAGE_FAULTS_SAMPLING_PRECISION} \
    MPLCONFIGDIR=${MPLCONFIGDIR} \
    MEMORY_LIMIT_MB=${MEMORY_LIMIT_MB} \
    MEMORY_LIMIT_BYTES=${MEMORY_LIMIT_BYTES} \
    CGROUP_PATH=${CGROUP_PATH} \
    SUPERVISORD_LOG_FILE=${SUPERVISORD_LOG_FILE} \
    SUPERVISORD_PID_FILE=${SUPERVISORD_PID_FILE} \
    MAIN_LOG_FILE=${MAIN_LOG_FILE} \
    LIMIT_MEMORY_USAGE_LOG_FILE=${LIMIT_MEMORY_USAGE_LOG_FILE} \
    QUIT_ON_MAIN_LOG_FILE=${QUIT_ON_MAIN_LOG_FILE} \
    MEMORY_PRESSURE_LOG_FILE=${MEMORY_PRESSURE_LOG_FILE} \
    MEMORY_PRESSURE_SAMPLING_PRECISION=${MEMORY_PRESSURE_SAMPLING_PRECISION} \
    MEMORY_PRESSURE_CGROUP_FILE_PATH=${MEMORY_PRESSURE_CGROUP_FILE_PATH} \
    MEMORY_FAILCNT_CGROUP_FILE_PATH=${MEMORY_FAILCNT_CGROUP_FILE_PATH}

RUN mkdir -p "$(dirname "${MEMORY_USAGE_LOG_FILE}")" && \
    mkdir -p ${MPLCONFIGDIR}

COPY experiment experiment
COPY scripts/monitor_memory_usage.sh scripts/monitor_memory_usage.sh
COPY scripts/monitor_page_faults.sh scripts/monitor_page_faults.sh
COPY scripts/monitor_memory_pressure.sh scripts/monitor_memory_pressure.sh
COPY scripts/limit_memory_usage.sh scripts/limit_memory_usage.sh
COPY scripts/quit_on_main_exit.sh scripts/quit_on_main_exit.sh
COPY supervisord.conf .

RUN chmod +x /app/scripts/monitor_memory_usage.sh && \
    chmod +x /app/scripts/monitor_page_faults.sh && \
    chmod +x /app/scripts/monitor_memory_pressure.sh && \
    chmod +x /app/scripts/limit_memory_usage.sh && \
    chmod +x /app/scripts/quit_on_main_exit.sh

COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

ENTRYPOINT ["sh", "-c", "RUN_COMMAND=\"$*\" supervisord -c /app/supervisord.conf; exit $(cat /tmp/main_exit_code 2>/dev/null || echo 0)", "--"]