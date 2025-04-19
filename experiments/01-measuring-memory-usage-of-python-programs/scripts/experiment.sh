#!/usr/bin/env sh

################################################################################
# This script orchestrates:
#   1) Generating data.
#   2) Profiling memory usage of Envelope for each profiler
#   3) Collecting results into a single dataset.
#   4) Analyzing the dataset.
#
# The Docker container uses "libs/scripts/experiment.sh" as its entrypoint.
# We pass in the Python script to run via "--env EXPERIMENT_COMMAND=<script.py>".
################################################################################

set -e

################################################################################
# CONFIGURATION
################################################################################
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d%H%M%S)}"
CPUSET_CPUS="${CPUSET_CPUS:-0}"
ROOT_DIR=$(git rev-parse --show-toplevel)
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/01-measuring-memory-usage-of-python-programs/out/results/${TIMESTAMP}}"

# Experiment context
DIND_VOLUME_NAME="${DIND_VOLUME_NAME:-mac__exp-01__dind-storage}"
EXPERIMENT_IMAGE_TAG="${EXPERIMENT_IMAGE_TAG:-experiment:${TIMESTAMP}}"
EXPERIMENT_N_RUNS="${EXPERIMENT_N_RUNS:-5}"
EXPERIMENT_BUILD_CONTEXT="${EXPERIMENT_BUILD_CONTEXT:-${ROOT_DIR}/experiments/01-measuring-memory-usage-of-python-programs}"
EXPERIMENT_DOCKERFILE_PATH="${EXPERIMENT_DOCKERFILE_PATH:-${EXPERIMENT_BUILD_CONTEXT}/Dockerfile}"
EXPERIMENT_TRACEQ_BUILD_CONTEXT="${EXPERIMENT_TRACEQ_BUILD_CONTEXT:-${ROOT_DIR}/libs/traceq}"
EXPERIMENT_COMMON_BUILD_CONTEXT="${EXPERIMENT_COMMON_BUILD_CONTEXT:-${ROOT_DIR}/libs/common}"

# Data generation
DATASET_INLINES="${DATASET_INLINES:-600}"
DATASET_XLINES="${DATASET_XLINES:-600}"
DATASET_SAMPLES="${DATASET_SAMPLES:-600}"

# Filesystem-related variables
HOST_UID="${HOST_UID:-$(id -u)}"
HOST_GID="${HOST_GID:-$(id -g)}"

echo "Args:"
echo "  TIMESTAMP=${TIMESTAMP}"
echo "  CPUSET_CPUS=${CPUSET_CPUS}"
echo "  EXPERIMENT_IMAGE_TAG=${EXPERIMENT_IMAGE_TAG}"
echo "  EXPERIMENT_DOCKERFILE_PATH=${EXPERIMENT_DOCKERFILE_PATH}"
echo "  EXPERIMENT_BUILD_CONTEXT=${EXPERIMENT_BUILD_CONTEXT}"
echo "  EXPERIMENT_TRACEQ_BUILD_CONTEXT=${EXPERIMENT_TRACEQ_BUILD_CONTEXT}"
echo "  EXPERIMENT_COMMON_BUILD_CONTEXT=${EXPERIMENT_COMMON_BUILD_CONTEXT}"
echo "  EXPERIMENT_N_RUNS=${EXPERIMENT_N_RUNS}"
echo "  ROOT_DIR=${ROOT_DIR}"
echo "  DATASET_INLINES=${DATASET_INLINES}"
echo "  DATASET_XLINES=${DATASET_XLINES}"
echo "  DATASET_SAMPLES=${DATASET_SAMPLES}"
echo "  DIND_VOLUME_NAME=${DIND_VOLUME_NAME}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  HOST_UID=${HOST_UID}"
echo "  HOST_GID=${HOST_GID}"
echo

echo "Starting experiment ${TIMESTAMP}..."

################################################################################
# Create Docker Volume for DIND
################################################################################
echo "Creating Docker volume for DIND..."
if ! docker volume inspect "${VOLUME_NAME}" &>/dev/null; then
  docker volume create "${VOLUME_NAME}"
fi

echo "Creating output dir..."
mkdir -p "${OUTPUT_DIR}"

#################################################################################
## STEP 1: GENERATE DATA
#################################################################################
echo "Generating input data..."
docker run \
  --rm \
  --privileged \
  --entrypoint /bin/sh \
  --cpuset-cpus=0 \
  -v "${DIND_VOLUME_NAME}:/var/lib/docker:rw" \
  -v "${ROOT_DIR}/libs/common/scripts:/workspace:ro" \
  -v "${EXPERIMENT_BUILD_CONTEXT}:/mnt${EXPERIMENT_BUILD_CONTEXT}:ro" \
  -v "${EXPERIMENT_TRACEQ_BUILD_CONTEXT}:/mnt${EXPERIMENT_TRACEQ_BUILD_CONTEXT}:ro" \
  -v "${EXPERIMENT_COMMON_BUILD_CONTEXT}:/mnt${EXPERIMENT_COMMON_BUILD_CONTEXT}:ro" \
  -v "${OUTPUT_DIR}:/mnt${OUTPUT_DIR}:rw" \
  --env DOCKER_TLS_CERTDIR="" \
  --env HOST_UID="${HOST_UID}" \
  --env HOST_GID="${HOST_GID}" \
  --env EXPERIMENT_IMAGE_TAG="${EXPERIMENT_IMAGE_TAG}" \
  --env EXPERIMENT_DOCKERFILE_PATH="/mnt${EXPERIMENT_DOCKERFILE_PATH}" \
  --env EXPERIMENT_BUILD_CONTEXT="/mnt${EXPERIMENT_BUILD_CONTEXT}" \
  --env EXPERIMENT_EXTRA_CONTEXTS="--build-context traceq=/mnt${EXPERIMENT_TRACEQ_BUILD_CONTEXT} --build-context common=/mnt${EXPERIMENT_COMMON_BUILD_CONTEXT}" \
  --env EXPERIMENT_N_RUNS="1" \
  --env EXPERIMENT_CPUSET_CPUS="${CPUSET_CPUS}" \
  --env EXPERIMENT_COMMAND="generate_data.py" \
  --env EXPERIMENT_ENV=" \
    -e OUTPUT_DIR=/experiment/out/inputs \
    -e DATASET_INLINES=${DATASET_INLINES} \
    -e DATASET_XLINES=${DATASET_XLINES} \
    -e DATASET_SAMPLES=${DATASET_SAMPLES} \
  " \
  --env EXPERIMENT_VOLUMES="-v /mnt${OUTPUT_DIR}:/experiment/out:rw" \
  docker:28.0.1-dind \
  "/workspace/experiment.sh"

################################################################################
# STEP 2: Profiling the memory usage
################################################################################
echo "Profiling the memory usage for psutil..."
docker run \
  --rm \
  --privileged \
  --entrypoint /bin/sh \
  --cpuset-cpus=0 \
  -v "${DIND_VOLUME_NAME}:/var/lib/docker:rw" \
  -v "${ROOT_DIR}/libs/common/scripts:/workspace:ro" \
  -v "${EXPERIMENT_BUILD_CONTEXT}:/mnt${EXPERIMENT_BUILD_CONTEXT}:ro" \
  -v "${EXPERIMENT_TRACEQ_BUILD_CONTEXT}:/mnt${EXPERIMENT_TRACEQ_BUILD_CONTEXT}:ro" \
  -v "${EXPERIMENT_COMMON_BUILD_CONTEXT}:/mnt${EXPERIMENT_COMMON_BUILD_CONTEXT}:ro" \
  -v "${OUTPUT_DIR}:/mnt${OUTPUT_DIR}:rw" \
  --env DOCKER_TLS_CERTDIR="" \
  --env HOST_UID="${HOST_UID}" \
  --env HOST_GID="${HOST_GID}" \
  --env EXPERIMENT_IMAGE_TAG="${EXPERIMENT_IMAGE_TAG}" \
  --env EXPERIMENT_DOCKERFILE_PATH="/mnt${EXPERIMENT_DOCKERFILE_PATH}" \
  --env EXPERIMENT_BUILD_CONTEXT="/mnt${EXPERIMENT_BUILD_CONTEXT}" \
  --env EXPERIMENT_EXTRA_CONTEXTS="--build-context traceq=/mnt${EXPERIMENT_TRACEQ_BUILD_CONTEXT} --build-context common=/mnt${EXPERIMENT_COMMON_BUILD_CONTEXT}" \
  --env EXPERIMENT_N_RUNS="${EXPERIMENT_N_RUNS}" \
  --env EXPERIMENT_CPUSET_CPUS="${CPUSET_CPUS}" \
  --env EXPERIMENT_COMMAND="measure_with_psutil.py" \
  --env EXPERIMENT_ENV=" \
    -e SEGY_FILEPATH=/experiment/out/inputs/${DATASET_INLINES}-${DATASET_XLINES}-${DATASET_SAMPLES}.segy \
    -e OUTPUT_RESULT_PATH=/experiment/out/profiles/psutil.txt \
    -e APPEND_TIMESTAMP=True \
  " \
  --env EXPERIMENT_VOLUMES="-v /mnt${OUTPUT_DIR}:/experiment/out:rw" \
  docker:28.0.1-dind \
  "/workspace/experiment.sh"

echo "Profiling the memory usage for resource..."
docker run \
  --rm \
  --privileged \
  --entrypoint /bin/sh \
  --cpuset-cpus=0 \
  -v "${DIND_VOLUME_NAME}:/var/lib/docker:rw" \
  -v "${ROOT_DIR}/libs/common/scripts:/workspace:ro" \
  -v "${EXPERIMENT_BUILD_CONTEXT}:/mnt${EXPERIMENT_BUILD_CONTEXT}:ro" \
  -v "${EXPERIMENT_TRACEQ_BUILD_CONTEXT}:/mnt${EXPERIMENT_TRACEQ_BUILD_CONTEXT}:ro" \
  -v "${EXPERIMENT_COMMON_BUILD_CONTEXT}:/mnt${EXPERIMENT_COMMON_BUILD_CONTEXT}:ro" \
  -v "${OUTPUT_DIR}:/mnt${OUTPUT_DIR}:rw" \
  --env DOCKER_TLS_CERTDIR="" \
  --env HOST_UID="${HOST_UID}" \
  --env HOST_GID="${HOST_GID}" \
  --env EXPERIMENT_IMAGE_TAG="${EXPERIMENT_IMAGE_TAG}" \
  --env EXPERIMENT_DOCKERFILE_PATH="/mnt${EXPERIMENT_DOCKERFILE_PATH}" \
  --env EXPERIMENT_BUILD_CONTEXT="/mnt${EXPERIMENT_BUILD_CONTEXT}" \
  --env EXPERIMENT_EXTRA_CONTEXTS="--build-context traceq=/mnt${EXPERIMENT_TRACEQ_BUILD_CONTEXT} --build-context common=/mnt${EXPERIMENT_COMMON_BUILD_CONTEXT}" \
  --env EXPERIMENT_N_RUNS="${EXPERIMENT_N_RUNS}" \
  --env EXPERIMENT_CPUSET_CPUS="${CPUSET_CPUS}" \
  --env EXPERIMENT_COMMAND="measure_with_resource.py" \
  --env EXPERIMENT_ENV=" \
    -e SEGY_FILEPATH=/experiment/out/inputs/${DATASET_INLINES}-${DATASET_XLINES}-${DATASET_SAMPLES}.segy \
    -e OUTPUT_RESULT_PATH=/experiment/out/profiles/resource.txt \
    -e APPEND_TIMESTAMP=True \
  " \
  --env EXPERIMENT_VOLUMES="-v /mnt${OUTPUT_DIR}:/experiment/out:rw" \
  docker:28.0.1-dind \
  "/workspace/experiment.sh"

echo "Profiling the memory usage for tracemalloc..."
docker run \
  --rm \
  --privileged \
  --entrypoint /bin/sh \
  --cpuset-cpus=0 \
  -v "${DIND_VOLUME_NAME}:/var/lib/docker:rw" \
  -v "${ROOT_DIR}/libs/common/scripts:/workspace:ro" \
  -v "${EXPERIMENT_BUILD_CONTEXT}:/mnt${EXPERIMENT_BUILD_CONTEXT}:ro" \
  -v "${EXPERIMENT_TRACEQ_BUILD_CONTEXT}:/mnt${EXPERIMENT_TRACEQ_BUILD_CONTEXT}:ro" \
  -v "${EXPERIMENT_COMMON_BUILD_CONTEXT}:/mnt${EXPERIMENT_COMMON_BUILD_CONTEXT}:ro" \
  -v "${OUTPUT_DIR}:/mnt${OUTPUT_DIR}:rw" \
  --env DOCKER_TLS_CERTDIR="" \
  --env HOST_UID="${HOST_UID}" \
  --env HOST_GID="${HOST_GID}" \
  --env EXPERIMENT_IMAGE_TAG="${EXPERIMENT_IMAGE_TAG}" \
  --env EXPERIMENT_DOCKERFILE_PATH="/mnt${EXPERIMENT_DOCKERFILE_PATH}" \
  --env EXPERIMENT_BUILD_CONTEXT="/mnt${EXPERIMENT_BUILD_CONTEXT}" \
  --env EXPERIMENT_EXTRA_CONTEXTS="--build-context traceq=/mnt${EXPERIMENT_TRACEQ_BUILD_CONTEXT} --build-context common=/mnt${EXPERIMENT_COMMON_BUILD_CONTEXT}" \
  --env EXPERIMENT_N_RUNS="${EXPERIMENT_N_RUNS}" \
  --env EXPERIMENT_CPUSET_CPUS="${CPUSET_CPUS}" \
  --env EXPERIMENT_COMMAND="measure_with_tracemalloc.py" \
  --env EXPERIMENT_ENV=" \
    -e SEGY_FILEPATH=/experiment/out/inputs/${DATASET_INLINES}-${DATASET_XLINES}-${DATASET_SAMPLES}.segy \
    -e OUTPUT_RESULT_PATH=/experiment/out/profiles/tracemalloc.txt \
    -e APPEND_TIMESTAMP=True \
  " \
  --env EXPERIMENT_VOLUMES="-v /mnt${OUTPUT_DIR}:/experiment/out:rw" \
  docker:28.0.1-dind \
  "/workspace/experiment.sh"

echo "Profiling the memory usage for kernel..."
docker run \
  --rm \
  --privileged \
  --entrypoint /bin/sh \
  --cpuset-cpus=0 \
  -v "${DIND_VOLUME_NAME}:/var/lib/docker:rw" \
  -v "${ROOT_DIR}/libs/common/scripts:/workspace:ro" \
  -v "${EXPERIMENT_BUILD_CONTEXT}:/mnt${EXPERIMENT_BUILD_CONTEXT}:ro" \
  -v "${EXPERIMENT_TRACEQ_BUILD_CONTEXT}:/mnt${EXPERIMENT_TRACEQ_BUILD_CONTEXT}:ro" \
  -v "${EXPERIMENT_COMMON_BUILD_CONTEXT}:/mnt${EXPERIMENT_COMMON_BUILD_CONTEXT}:ro" \
  -v "${OUTPUT_DIR}:/mnt${OUTPUT_DIR}:rw" \
  --env DOCKER_TLS_CERTDIR="" \
  --env HOST_UID="${HOST_UID}" \
  --env HOST_GID="${HOST_GID}" \
  --env EXPERIMENT_IMAGE_TAG="${EXPERIMENT_IMAGE_TAG}" \
  --env EXPERIMENT_DOCKERFILE_PATH="/mnt${EXPERIMENT_DOCKERFILE_PATH}" \
  --env EXPERIMENT_BUILD_CONTEXT="/mnt${EXPERIMENT_BUILD_CONTEXT}" \
  --env EXPERIMENT_EXTRA_CONTEXTS="--build-context traceq=/mnt${EXPERIMENT_TRACEQ_BUILD_CONTEXT} --build-context common=/mnt${EXPERIMENT_COMMON_BUILD_CONTEXT}" \
  --env EXPERIMENT_N_RUNS="${EXPERIMENT_N_RUNS}" \
  --env EXPERIMENT_CPUSET_CPUS="${CPUSET_CPUS}" \
  --env EXPERIMENT_COMMAND="measure_with_kernel.py" \
  --env EXPERIMENT_ENV=" \
    -e SEGY_FILEPATH=/experiment/out/inputs/${DATASET_INLINES}-${DATASET_XLINES}-${DATASET_SAMPLES}.segy \
    -e OUTPUT_RESULT_PATH=/experiment/out/profiles/kernel.txt \
    -e APPEND_TIMESTAMP=True \
  " \
  --env EXPERIMENT_VOLUMES="-v /mnt${OUTPUT_DIR}:/experiment/out:rw" \
  docker:28.0.1-dind \
  "/workspace/experiment.sh"

echo "Profiling the memory usage for TraceQ with psutil backend..."
docker run \
  --rm \
  --privileged \
  --entrypoint /bin/sh \
  --cpuset-cpus=0 \
  -v "${DIND_VOLUME_NAME}:/var/lib/docker:rw" \
  -v "${ROOT_DIR}/libs/common/scripts:/workspace:ro" \
  -v "${EXPERIMENT_BUILD_CONTEXT}:/mnt${EXPERIMENT_BUILD_CONTEXT}:ro" \
  -v "${EXPERIMENT_TRACEQ_BUILD_CONTEXT}:/mnt${EXPERIMENT_TRACEQ_BUILD_CONTEXT}:ro" \
  -v "${EXPERIMENT_COMMON_BUILD_CONTEXT}:/mnt${EXPERIMENT_COMMON_BUILD_CONTEXT}:ro" \
  -v "${OUTPUT_DIR}:/mnt${OUTPUT_DIR}:rw" \
  --env DOCKER_TLS_CERTDIR="" \
  --env HOST_UID="${HOST_UID}" \
  --env HOST_GID="${HOST_GID}" \
  --env EXPERIMENT_IMAGE_TAG="${EXPERIMENT_IMAGE_TAG}" \
  --env EXPERIMENT_DOCKERFILE_PATH="/mnt${EXPERIMENT_DOCKERFILE_PATH}" \
  --env EXPERIMENT_BUILD_CONTEXT="/mnt${EXPERIMENT_BUILD_CONTEXT}" \
  --env EXPERIMENT_EXTRA_CONTEXTS="--build-context traceq=/mnt${EXPERIMENT_TRACEQ_BUILD_CONTEXT} --build-context common=/mnt${EXPERIMENT_COMMON_BUILD_CONTEXT}" \
  --env EXPERIMENT_N_RUNS="${EXPERIMENT_N_RUNS}" \
  --env EXPERIMENT_CPUSET_CPUS="${CPUSET_CPUS}" \
  --env EXPERIMENT_COMMAND="measure_with_traceq.py" \
  --env EXPERIMENT_ENV=" \
    -e SEGY_FILEPATH=/experiment/out/inputs/${DATASET_INLINES}-${DATASET_XLINES}-${DATASET_SAMPLES}.segy \
    -e OUTPUT_DIR=/experiment/out/profiles \
    -e SESSION_ID=traceq_psutil \
    -e TRACEQ_BACKEND=psutil \
  " \
  --env EXPERIMENT_VOLUMES="-v /mnt${OUTPUT_DIR}:/experiment/out:rw" \
  docker:28.0.1-dind \
  "/workspace/experiment.sh"

echo "Profiling the memory usage for TraceQ with resource backend..."
docker run \
  --rm \
  --privileged \
  --entrypoint /bin/sh \
  --cpuset-cpus=0 \
  -v "${DIND_VOLUME_NAME}:/var/lib/docker:rw" \
  -v "${ROOT_DIR}/libs/common/scripts:/workspace:ro" \
  -v "${EXPERIMENT_BUILD_CONTEXT}:/mnt${EXPERIMENT_BUILD_CONTEXT}:ro" \
  -v "${EXPERIMENT_TRACEQ_BUILD_CONTEXT}:/mnt${EXPERIMENT_TRACEQ_BUILD_CONTEXT}:ro" \
  -v "${EXPERIMENT_COMMON_BUILD_CONTEXT}:/mnt${EXPERIMENT_COMMON_BUILD_CONTEXT}:ro" \
  -v "${OUTPUT_DIR}:/mnt${OUTPUT_DIR}:rw" \
  --env DOCKER_TLS_CERTDIR="" \
  --env HOST_UID="${HOST_UID}" \
  --env HOST_GID="${HOST_GID}" \
  --env EXPERIMENT_IMAGE_TAG="${EXPERIMENT_IMAGE_TAG}" \
  --env EXPERIMENT_DOCKERFILE_PATH="/mnt${EXPERIMENT_DOCKERFILE_PATH}" \
  --env EXPERIMENT_BUILD_CONTEXT="/mnt${EXPERIMENT_BUILD_CONTEXT}" \
  --env EXPERIMENT_EXTRA_CONTEXTS="--build-context traceq=/mnt${EXPERIMENT_TRACEQ_BUILD_CONTEXT} --build-context common=/mnt${EXPERIMENT_COMMON_BUILD_CONTEXT}" \
  --env EXPERIMENT_N_RUNS="${EXPERIMENT_N_RUNS}" \
  --env EXPERIMENT_CPUSET_CPUS="${CPUSET_CPUS}" \
  --env EXPERIMENT_COMMAND="measure_with_traceq.py" \
  --env EXPERIMENT_ENV=" \
    -e SEGY_FILEPATH=/experiment/out/inputs/${DATASET_INLINES}-${DATASET_XLINES}-${DATASET_SAMPLES}.segy \
    -e OUTPUT_DIR=/experiment/out/profiles \
    -e SESSION_ID=traceq_resource \
    -e TRACEQ_BACKEND=resource \
  " \
  --env EXPERIMENT_VOLUMES="-v /mnt${OUTPUT_DIR}:/experiment/out:rw" \
  docker:28.0.1-dind \
  "/workspace/experiment.sh"

echo "Profiling the memory usage for TraceQ with tracemalloc backend..."
docker run \
  --rm \
  --privileged \
  --entrypoint /bin/sh \
  --cpuset-cpus=0 \
  -v "${DIND_VOLUME_NAME}:/var/lib/docker:rw" \
  -v "${ROOT_DIR}/libs/common/scripts:/workspace:ro" \
  -v "${EXPERIMENT_BUILD_CONTEXT}:/mnt${EXPERIMENT_BUILD_CONTEXT}:ro" \
  -v "${EXPERIMENT_TRACEQ_BUILD_CONTEXT}:/mnt${EXPERIMENT_TRACEQ_BUILD_CONTEXT}:ro" \
  -v "${EXPERIMENT_COMMON_BUILD_CONTEXT}:/mnt${EXPERIMENT_COMMON_BUILD_CONTEXT}:ro" \
  -v "${OUTPUT_DIR}:/mnt${OUTPUT_DIR}:rw" \
  --env DOCKER_TLS_CERTDIR="" \
  --env HOST_UID="${HOST_UID}" \
  --env HOST_GID="${HOST_GID}" \
  --env EXPERIMENT_IMAGE_TAG="${EXPERIMENT_IMAGE_TAG}" \
  --env EXPERIMENT_DOCKERFILE_PATH="/mnt${EXPERIMENT_DOCKERFILE_PATH}" \
  --env EXPERIMENT_BUILD_CONTEXT="/mnt${EXPERIMENT_BUILD_CONTEXT}" \
  --env EXPERIMENT_EXTRA_CONTEXTS="--build-context traceq=/mnt${EXPERIMENT_TRACEQ_BUILD_CONTEXT} --build-context common=/mnt${EXPERIMENT_COMMON_BUILD_CONTEXT}" \
  --env EXPERIMENT_N_RUNS="${EXPERIMENT_N_RUNS}" \
  --env EXPERIMENT_CPUSET_CPUS="${CPUSET_CPUS}" \
  --env EXPERIMENT_COMMAND="measure_with_traceq.py" \
  --env EXPERIMENT_ENV=" \
    -e SEGY_FILEPATH=/experiment/out/inputs/${DATASET_INLINES}-${DATASET_XLINES}-${DATASET_SAMPLES}.segy \
    -e OUTPUT_DIR=/experiment/out/profiles \
    -e SESSION_ID=traceq_tracemalloc \
    -e TRACEQ_BACKEND=tracemalloc \
  " \
  --env EXPERIMENT_VOLUMES="-v /mnt${OUTPUT_DIR}:/experiment/out:rw" \
  docker:28.0.1-dind \
  "/workspace/experiment.sh"

echo "Profiling the memory usage for TraceQ with kernel backend..."
docker run \
  --rm \
  --privileged \
  --entrypoint /bin/sh \
  --cpuset-cpus=0 \
  -v "${DIND_VOLUME_NAME}:/var/lib/docker:rw" \
  -v "${ROOT_DIR}/libs/common/scripts:/workspace:ro" \
  -v "${EXPERIMENT_BUILD_CONTEXT}:/mnt${EXPERIMENT_BUILD_CONTEXT}:ro" \
  -v "${EXPERIMENT_TRACEQ_BUILD_CONTEXT}:/mnt${EXPERIMENT_TRACEQ_BUILD_CONTEXT}:ro" \
  -v "${EXPERIMENT_COMMON_BUILD_CONTEXT}:/mnt${EXPERIMENT_COMMON_BUILD_CONTEXT}:ro" \
  -v "${OUTPUT_DIR}:/mnt${OUTPUT_DIR}:rw" \
  --env DOCKER_TLS_CERTDIR="" \
  --env HOST_UID="${HOST_UID}" \
  --env HOST_GID="${HOST_GID}" \
  --env EXPERIMENT_IMAGE_TAG="${EXPERIMENT_IMAGE_TAG}" \
  --env EXPERIMENT_DOCKERFILE_PATH="/mnt${EXPERIMENT_DOCKERFILE_PATH}" \
  --env EXPERIMENT_BUILD_CONTEXT="/mnt${EXPERIMENT_BUILD_CONTEXT}" \
  --env EXPERIMENT_EXTRA_CONTEXTS="--build-context traceq=/mnt${EXPERIMENT_TRACEQ_BUILD_CONTEXT} --build-context common=/mnt${EXPERIMENT_COMMON_BUILD_CONTEXT}" \
  --env EXPERIMENT_N_RUNS="${EXPERIMENT_N_RUNS}" \
  --env EXPERIMENT_CPUSET_CPUS="${CPUSET_CPUS}" \
  --env EXPERIMENT_COMMAND="measure_with_traceq.py" \
  --env EXPERIMENT_ENV=" \
    -e SEGY_FILEPATH=/experiment/out/inputs/${DATASET_INLINES}-${DATASET_XLINES}-${DATASET_SAMPLES}.segy \
    -e OUTPUT_DIR=/experiment/out/profiles \
    -e SESSION_ID=traceq_kernel \
    -e TRACEQ_BACKEND=kernel \
  " \
  --env EXPERIMENT_VOLUMES="-v /mnt${OUTPUT_DIR}:/experiment/out:rw" \
  docker:28.0.1-dind \
  "/workspace/experiment.sh"

#################################################################################
## STEP 3: Collect the results
#################################################################################
echo "Collecting results..."
docker run \
  --rm \
  --privileged \
  --entrypoint /bin/sh \
  --cpuset-cpus=0 \
  -v "${DIND_VOLUME_NAME}:/var/lib/docker:rw" \
  -v "${ROOT_DIR}/libs/common/scripts:/workspace:ro" \
  -v "${EXPERIMENT_BUILD_CONTEXT}:/mnt${EXPERIMENT_BUILD_CONTEXT}:ro" \
  -v "${EXPERIMENT_TRACEQ_BUILD_CONTEXT}:/mnt${EXPERIMENT_TRACEQ_BUILD_CONTEXT}:ro" \
  -v "${EXPERIMENT_COMMON_BUILD_CONTEXT}:/mnt${EXPERIMENT_COMMON_BUILD_CONTEXT}:ro" \
  -v "${OUTPUT_DIR}:/mnt${OUTPUT_DIR}:rw" \
  --env DOCKER_TLS_CERTDIR="" \
  --env HOST_UID="${HOST_UID}" \
  --env HOST_GID="${HOST_GID}" \
  --env EXPERIMENT_IMAGE_TAG="${EXPERIMENT_IMAGE_TAG}" \
  --env EXPERIMENT_DOCKERFILE_PATH="/mnt${EXPERIMENT_DOCKERFILE_PATH}" \
  --env EXPERIMENT_BUILD_CONTEXT="/mnt${EXPERIMENT_BUILD_CONTEXT}" \
  --env EXPERIMENT_EXTRA_CONTEXTS="--build-context traceq=/mnt${EXPERIMENT_TRACEQ_BUILD_CONTEXT} --build-context common=/mnt${EXPERIMENT_COMMON_BUILD_CONTEXT}" \
  --env EXPERIMENT_N_RUNS="1" \
  --env EXPERIMENT_CPUSET_CPUS="${CPUSET_CPUS}" \
  --env EXPERIMENT_COMMAND="collect_results.py" \
  --env EXPERIMENT_ENV=" \
    -e OUTPUT_DIR=/experiment/out \
  " \
  --env EXPERIMENT_VOLUMES="-v /mnt${OUTPUT_DIR}:/experiment/out:rw" \
  docker:28.0.1-dind \
  "/workspace/experiment.sh"

################################################################################
# STEP 4: Analyze the results
################################################################################
echo "Analyzing results..."
docker run \
  --rm \
  --privileged \
  --entrypoint /bin/sh \
  --cpuset-cpus=0 \
  -v "${DIND_VOLUME_NAME}:/var/lib/docker:rw" \
  -v "${ROOT_DIR}/libs/common/scripts:/workspace:ro" \
  -v "${EXPERIMENT_BUILD_CONTEXT}:/mnt${EXPERIMENT_BUILD_CONTEXT}:ro" \
  -v "${EXPERIMENT_TRACEQ_BUILD_CONTEXT}:/mnt${EXPERIMENT_TRACEQ_BUILD_CONTEXT}:ro" \
  -v "${EXPERIMENT_COMMON_BUILD_CONTEXT}:/mnt${EXPERIMENT_COMMON_BUILD_CONTEXT}:ro" \
  -v "${OUTPUT_DIR}:/mnt${OUTPUT_DIR}:rw" \
  --env DOCKER_TLS_CERTDIR="" \
  --env HOST_UID="${HOST_UID}" \
  --env HOST_GID="${HOST_GID}" \
  --env EXPERIMENT_IMAGE_TAG="${EXPERIMENT_IMAGE_TAG}" \
  --env EXPERIMENT_DOCKERFILE_PATH="/mnt${EXPERIMENT_DOCKERFILE_PATH}" \
  --env EXPERIMENT_BUILD_CONTEXT="/mnt${EXPERIMENT_BUILD_CONTEXT}" \
  --env EXPERIMENT_EXTRA_CONTEXTS="--build-context traceq=/mnt${EXPERIMENT_TRACEQ_BUILD_CONTEXT} --build-context common=/mnt${EXPERIMENT_COMMON_BUILD_CONTEXT}" \
  --env EXPERIMENT_N_RUNS="1" \
  --env EXPERIMENT_CPUSET_CPUS="${CPUSET_CPUS}" \
  --env EXPERIMENT_COMMAND="analyze_results.py" \
  --env EXPERIMENT_ENV=" \
    -e OUTPUT_DIR=/experiment/out \
  " \
  --env EXPERIMENT_VOLUMES="-v /mnt${OUTPUT_DIR}:/experiment/out:rw" \
  docker:28.0.1-dind \
  "/workspace/experiment.sh"

echo
echo "Measuring memory usage experiment complete!"
echo "Results directory: ${OUTPUT_DIR}"