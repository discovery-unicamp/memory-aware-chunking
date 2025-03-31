FROM python:3.13.2-slim AS builder

WORKDIR /builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl && \
    apt-get clean && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:$PATH"

COPY --from=traceq . ./traceq
RUN pip install ./traceq

COPY --from=common . ./common
RUN pip install ./common

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

FROM python:3.13.2-slim AS final

RUN apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ARG HOST_UID=1000
ARG HOST_GID=1000

RUN addgroup --gid ${HOST_GID} experiment && \
    adduser --uid ${HOST_UID} --gid ${HOST_GID} --disabled-password --gecos "" experiment

USER experiment

WORKDIR /experiment

COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY experiment .

ENTRYPOINT ["python"]
CMD ["collect_memory_profile.py"]