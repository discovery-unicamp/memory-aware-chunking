FROM python:3.13.2-slim AS builder

ARG TRACEQ_HOME="/root/.local/src/traceq"

WORKDIR ${TRACEQ_HOME}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV PATH="/root/.cargo/bin:$PATH"

COPY . .

RUN pip install .

FROM python:3.13.2-slim AS final

COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin