#!/usr/bin/env bash

################################################################################
# This script orchestrates:
#   1) Generating synthetic data.
#   2) Running multiple scenarios (single worker, 2 workers, small n, large n)
#      combined with multiple chunking modes (auto, manual, memaware).
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
ROOT_DIR="$(git rev-parse --show-toplevel)"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/03-improving-data-parallelism-using-memory-aware-chunking/out/results/${TIMESTAMP}}"

# Experiment context
DIND_VOLUME_NAME="${DIND_VOLUME_NAME:-mac__exp-03-dind-storage}"
EXPERIMENT_IMAGE_TAG="${EXPERIMENT_IMAGE_TAG:-memory-aware-chunking:${TIMESTAMP}}"
EXPERIMENT_N_RUNS="${EXPERIMENT_N_RUNS:-3}"
EXPERIMENT_BUILD_CONTEXT="${EXPERIMENT_BUILD_CONTEXT:-${ROOT_DIR}/experiments/03-improving-data-parallelism-using-memory-aware-chunking}"
EXPERIMENT_DOCKERFILE_PATH="${EXPERIMENT_DOCKERFILE_PATH:-${EXPERIMENT_BUILD_CONTEXT}/Dockerfile}"
EXPERIMENT_COMMON_BUILD_CONTEXT="${EXPERIMENT_COMMON_BUILD_CONTEXT:-${ROOT_DIR}/libs/common}"

# Data generation
DATASET_INITIAL_SIZE="${DATASET_INITIAL_SIZE:-100}"
DATASET_FINAL_SIZE="${DATASET_FINAL_SIZE:-400}"
DATASET_STEP_SIZE="${DATASET_STEP_SIZE:-100}"

# Scenarios
WORKER_SCENARIOS="${WORKER_SCENARIOS:-single:1,two:2,smalln:4,bign:8}"
CHUNKING_MODES="${CHUNKING_MODES:-auto,evenly_split,memaware}"
GST3D_MODEL_FILE="${GST3D_MODEL_FILE:-${ROOT_DIR}/experiments/02-predicting-memory-consumption-from-input-shapes/out/results/20250331101842/best_models/gst3d.pkl}"

# Filesystem-related variables
HOST_UID="${HOST_UID:-$(id -u)}"
HOST_GID="${HOST_GID:-$(id -g)}"

# Host memory limit
MEMORY_LIMIT_GB="${MEMORY_LIMIT_GB:-16}"

echo "Args:"
echo "  TIMESTAMP=${TIMESTAMP}"
echo "  CPUSET_CPUS=${CPUSET_CPUS}"
echo "  EXPERIMENT_IMAGE_TAG=${EXPERIMENT_IMAGE_TAG}"
echo "  EXPERIMENT_BUILD_CONTEXT=${EXPERIMENT_BUILD_CONTEXT}"
echo "  EXPERIMENT_COMMON_BUILD_CONTEXT=${EXPERIMENT_COMMON_BUILD_CONTEXT}"
echo "  EXPERIMENT_DOCKERFILE_PATH=${EXPERIMENT_DOCKERFILE_PATH}"
echo "  EXPERIMENT_N_RUNS=${EXPERIMENT_N_RUNS}"
echo "  ROOT_DIR=${ROOT_DIR}"
echo "  DATASET_INITIAL_SIZE=${DATASET_INITIAL_SIZE}"
echo "  DATASET_FINAL_SIZE=${DATASET_FINAL_SIZE}"
echo "  DATASET_STEP_SIZE=${DATASET_STEP_SIZE}"
echo "  WORKER_SCENARIOS=${WORKER_SCENARIOS}"
echo "  CHUNKING_MODES=${CHUNKING_MODES}"
echo "  GST3D_MODEL_FILE=${GST3D_MODEL_FILE}"
echo "  DIND_VOLUME_NAME=${DIND_VOLUME_NAME}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  HOST_UID=${HOST_UID}"
echo "  HOST_GID=${HOST_GID}"
echo "  MEMORY_LIMIT_GB=${MEMORY_LIMIT_GB}"
echo

echo "Starting Memory-Aware Chunking experiment [${TIMESTAMP}]..."

################################################################################
# Create Docker Volume for DIND
################################################################################
echo "Creating Docker volume for DIND..."
if ! docker volume inspect "${DIND_VOLUME_NAME}" >/dev/null 2>&1; then
  docker volume create "${DIND_VOLUME_NAME}"
fi

echo "Creating output directory..."
mkdir -p "${OUTPUT_DIR}"

################################################################################
# STEP 1: GENERATE DATA
################################################################################
echo "Generating data for memory-aware chunking..."

docker run \
  --rm \
  --privileged \
  --entrypoint /bin/sh \
  --cpuset-cpus="${CPUSET_CPUS}" \
  -v "${DIND_VOLUME_NAME}:/var/lib/docker:rw" \
  -v "${ROOT_DIR}/libs/common/scripts:/workspace:ro" \
  -v "${EXPERIMENT_BUILD_CONTEXT}:/mnt${EXPERIMENT_BUILD_CONTEXT}:ro" \
  -v "${EXPERIMENT_COMMON_BUILD_CONTEXT}:/mnt${EXPERIMENT_COMMON_BUILD_CONTEXT}:ro" \
  -v "${OUTPUT_DIR}:/mnt${OUTPUT_DIR}:rw" \
  --env DOCKER_TLS_CERTDIR="" \
  --env HOST_UID="${HOST_UID}" \
  --env HOST_GID="${HOST_GID}" \
  --env EXPERIMENT_IMAGE_TAG="${EXPERIMENT_IMAGE_TAG}" \
  --env EXPERIMENT_DOCKERFILE_PATH="/mnt${EXPERIMENT_DOCKERFILE_PATH}" \
  --env EXPERIMENT_BUILD_CONTEXT="/mnt${EXPERIMENT_BUILD_CONTEXT}" \
  --env EXPERIMENT_EXTRA_CONTEXTS="--build-context common=/mnt${EXPERIMENT_COMMON_BUILD_CONTEXT}" \
  --env EXPERIMENT_N_RUNS="1" \
  --env EXPERIMENT_CPUSET_CPUS="${CPUSET_CPUS}" \
  --env EXPERIMENT_COMMAND="generate_data.py" \
  --env EXPERIMENT_ENV=" \
    -e OUTPUT_DIR=/experiment/out/inputs \
    -e INITIAL_SIZE=${DATASET_INITIAL_SIZE} \
    -e FINAL_SIZE=${DATASET_FINAL_SIZE} \
    -e STEP_SIZE=${DATASET_STEP_SIZE} \
  " \
  --env EXPERIMENT_VOLUMES="-v /mnt${OUTPUT_DIR}:/experiment/out:rw" \
  docker:28.0.1-dind \
  "/workspace/experiment.sh"

################################################################################
# STEP 2: RUN SCENARIOS ON EACH SEG-Y FILE
################################################################################
IFS=',' read -r -a scenario_array <<< "${WORKER_SCENARIOS}"
IFS=',' read -r -a chunking_array <<< "${CHUNKING_MODES}"

echo

echo "Copying GST3D memory predictor to the output directory..."
mkdir -p "${OUTPUT_DIR}/models"
cp "${GST3D_MODEL_FILE}" "${OUTPUT_DIR}/models/gst3d.pkl"

echo "Running multiple scenarios on generated SEG-Y files..."
for segy_file in "${OUTPUT_DIR}/inputs"/*.segy; do
  [ -f "$segy_file" ] || continue

  segy_filename="$(basename "${segy_file}")"
  segy_name_noext="${segy_filename%.*}"

  echo "  Processing data: ${segy_filename}"

  for scenario_item in "${scenario_array[@]}"; do
    scenario_name="$(echo "${scenario_item}" | cut -d':' -f1)"
    scenario_workers="$(echo "${scenario_item}" | cut -d':' -f2)"

    echo "    Scenario: ${scenario_name} -> ${scenario_workers} workers"
    for chunk_mode in "${chunking_array[@]}"; do
      echo "      Chunk mode: ${chunk_mode}"

      docker run \
        --rm \
        --privileged \
        --entrypoint /bin/sh \
        --cpuset-cpus="${CPUSET_CPUS}" \
        -v "${DIND_VOLUME_NAME}:/var/lib/docker:rw" \
        -v "${ROOT_DIR}/libs/common/scripts:/workspace:ro" \
        -v "${EXPERIMENT_BUILD_CONTEXT}:/mnt${EXPERIMENT_BUILD_CONTEXT}:ro" \
        -v "${EXPERIMENT_COMMON_BUILD_CONTEXT}:/mnt${EXPERIMENT_COMMON_BUILD_CONTEXT}:ro" \
        -v "${OUTPUT_DIR}:/mnt${OUTPUT_DIR}:rw" \
        --env DOCKER_TLS_CERTDIR="" \
        --env HOST_UID="${HOST_UID}" \
        --env HOST_GID="${HOST_GID}" \
        --env EXPERIMENT_IMAGE_TAG="${EXPERIMENT_IMAGE_TAG}" \
        --env EXPERIMENT_DOCKERFILE_PATH="/mnt${EXPERIMENT_DOCKERFILE_PATH}" \
        --env EXPERIMENT_BUILD_CONTEXT="/mnt${EXPERIMENT_BUILD_CONTEXT}" \
        --env EXPERIMENT_EXTRA_CONTEXTS="--build-context common=/mnt${EXPERIMENT_COMMON_BUILD_CONTEXT}" \
        --env EXPERIMENT_N_RUNS="${EXPERIMENT_N_RUNS}" \
        --env EXPERIMENT_CPUSET_CPUS="${CPUSET_CPUS}" \
        --env EXPERIMENT_COMMAND="collect_profile.py" \
        --env EXPERIMENT_ENV=" \
          -e OUTPUT_DIR=/experiment/out/profiles \
          -e INPUT_PATH=/experiment/out/inputs/${segy_filename} \
          -e WORKER_COUNT=${scenario_workers} \
          -e CHUNKING_MODE=${chunk_mode} \
          -e MEMORY_LIMIT_GB=${MEMORY_LIMIT_GB} \
          -e GST3D_MODEL_FILE=/experiment/out/models/gst3d.pkl \
        " \
        --env EXPERIMENT_VOLUMES="-v /mnt${OUTPUT_DIR}:/experiment/out:rw" \
        docker:28.0.1-dind \
        "/workspace/experiment.sh"
    done
  done
done

#################################################################################
## STEP 3: COLLECT RESULTS
#################################################################################
echo
echo "Collecting results from the profiles..."

docker run \
  --rm \
  --privileged \
  --entrypoint /bin/sh \
  --cpuset-cpus="${CPUSET_CPUS}" \
  -v "${DIND_VOLUME_NAME}:/var/lib/docker:rw" \
  -v "${ROOT_DIR}/libs/common/scripts:/workspace:ro" \
  -v "${EXPERIMENT_BUILD_CONTEXT}:/mnt${EXPERIMENT_BUILD_CONTEXT}:ro" \
  -v "${EXPERIMENT_COMMON_BUILD_CONTEXT}:/mnt${EXPERIMENT_COMMON_BUILD_CONTEXT}:ro" \
  -v "${OUTPUT_DIR}:/mnt${OUTPUT_DIR}:rw" \
  --env DOCKER_TLS_CERTDIR="" \
  --env HOST_UID="${HOST_UID}" \
  --env HOST_GID="${HOST_GID}" \
  --env EXPERIMENT_IMAGE_TAG="${EXPERIMENT_IMAGE_TAG}" \
  --env EXPERIMENT_DOCKERFILE_PATH="/mnt${EXPERIMENT_DOCKERFILE_PATH}" \
  --env EXPERIMENT_BUILD_CONTEXT="/mnt${EXPERIMENT_BUILD_CONTEXT}" \
  --env EXPERIMENT_EXTRA_CONTEXTS="--build-context common=/mnt${EXPERIMENT_COMMON_BUILD_CONTEXT}" \
  --env EXPERIMENT_N_RUNS="1" \
  --env EXPERIMENT_CPUSET_CPUS="${CPUSET_CPUS}" \
  --env EXPERIMENT_COMMAND="collect_results.py" \
  --env EXPERIMENT_ENV=" \
    -e OUTPUT_DIR=/experiment/out \
  " \
  --env EXPERIMENT_VOLUMES="-v /mnt${OUTPUT_DIR}:/experiment/out:rw" \
  docker:28.0.1-dind \
  "/workspace/experiment.sh"

#################################################################################
## STEP 4: ANALYZE RESULTS
#################################################################################
echo
echo "Analyzing the results..."

docker run \
  --rm \
  --privileged \
  --entrypoint /bin/sh \
  --cpuset-cpus="${CPUSET_CPUS}" \
  -v "${DIND_VOLUME_NAME}:/var/lib/docker:rw" \
  -v "${ROOT_DIR}/libs/common/scripts:/workspace:ro" \
  -v "${EXPERIMENT_BUILD_CONTEXT}:/mnt${EXPERIMENT_BUILD_CONTEXT}:ro" \
  -v "${EXPERIMENT_COMMON_BUILD_CONTEXT}:/mnt${EXPERIMENT_COMMON_BUILD_CONTEXT}:ro" \
  -v "${OUTPUT_DIR}:/mnt${OUTPUT_DIR}:rw" \
  --env DOCKER_TLS_CERTDIR="" \
  --env HOST_UID="${HOST_UID}" \
  --env HOST_GID="${HOST_GID}" \
  --env EXPERIMENT_IMAGE_TAG="${EXPERIMENT_IMAGE_TAG}" \
  --env EXPERIMENT_DOCKERFILE_PATH="/mnt${EXPERIMENT_DOCKERFILE_PATH}" \
  --env EXPERIMENT_BUILD_CONTEXT="/mnt${EXPERIMENT_BUILD_CONTEXT}" \
  --env EXPERIMENT_EXTRA_CONTEXTS="--build-context common=/mnt${EXPERIMENT_COMMON_BUILD_CONTEXT}" \
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
echo "Memory-Aware Chunking experiment complete!"
echo "Results directory: ${OUTPUT_DIR}"