FROM python:3.13.2-slim AS base

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
    apt-get install -y supervisor procps && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG SAMPLING_WINDOW_IN_SECONDS="0.2"

ARG MEMORY_USAGE_LOG_FILE="/app/logs/memory-usage.log"
ARG MEMORY_USAGE_SAMPLING_WINDOW_IN_SECONDS=${SAMPLING_WINDOW_IN_SECONDS}
ARG MEMORY_USAGE_CURRENT_CGROUP_FILE_PATH="/sys/fs/cgroup/memory.current"

ARG PAGE_FAULTS_SAMPLING_WINDOW_IN_SECONDS=${SAMPLING_WINDOW_IN_SECONDS}
ARG PAGE_FAULTS_LOG_FILE="/app/logs/page-faults.log"
ARG PAGE_FAULTS_MONITORED_PROCESS_NAME="operate envelope"
ARG PAGE_FAULTS_PID_RETRY_MAX_ATTEMPTS="20"
ARG PAGE_FAULTS_PID_RETRY_SLEEP_TIME="0.2"

ARG MPLCONFIGDIR="/opt/app/mplconfig"

ENV MEMORY_USAGE_LOG_FILE=${MEMORY_USAGE_LOG_FILE} \
    MEMORY_USAGE_SAMPLING_WINDOW_IN_SECONDS=${MEMORY_USAGE_SAMPLING_WINDOW_IN_SECONDS} \
    MEMORY_USAGE_CURRENT_CGROUP_FILE_PATH=${MEMORY_USAGE_CURRENT_CGROUP_FILE_PATH} \
    PAGE_FAULTS_SAMPLING_WINDOW_IN_SECONDS=${PAGE_FAULTS_SAMPLING_WINDOW_IN_SECONDS} \
    PAGE_FAULTS_LOG_FILE=${PAGE_FAULTS_LOG_FILE} \
    PAGE_FAULTS_MONITORED_PROCESS_NAME=${PAGE_FAULTS_MONITORED_PROCESS_NAME} \
    PAGE_FAULTS_PID_RETRY_MAX_ATTEMPTS=${PAGE_FAULTS_PID_RETRY_MAX_ATTEMPTS} \
    PAGE_FAULTS_PID_RETRY_SLEEP_TIME=${PAGE_FAULTS_PID_RETRY_SLEEP_TIME} \
    MPLCONFIGDIR=${MPLCONFIGDIR}

RUN mkdir -p "$(dirname "${MEMORY_USAGE_LOG_FILE}")" && \
    mkdir -p ${MPLCONFIGDIR}

COPY experiment experiment
COPY scripts/monitor_memory_usage.sh scripts/monitor_memory_usage.sh
COPY scripts/monitor_page_faults.sh scripts/monitor_page_faults.sh
COPY supervisord.conf .

RUN chmod +x /app/scripts/monitor_memory_usage.sh && \
    chmod +x /app/scripts/monitor_page_faults.sh

COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

RUN useradd -m appuser && \
    chown -R appuser /app && \
    chown -R appuser /opt/app

USER appuser

ENTRYPOINT ["sh", "-c", "RUN_COMMAND=\"$*\" supervisord -c /app/supervisord.conf", "--"]