#!/usr/bin/env sh

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d%H%M%S)}"
CPUSET_CPUS="${CPUSET_CPUS:-0}"
EXPERIMENT_IMAGE_TAG="${EXPERIMENT_IMAGE_TAG:-experiment:${TIMESTAMP}}"
EXPERIMENT_N_RUNS="${EXPERIMENT_N_RUNS:-30}"
ROOT_DIR=$(git rev-parse --show-toplevel)
EXPERIMENT_BUILD_CONTEXT="${EXPERIMENT_BUILD_CONTEXT:-${ROOT_DIR}/experiments/02-predicting-memory-consumption-from-input-shapes}"
EXPERIMENT_DOCKERFILE_PATH="${EXPERIMENT_DOCKERFILE_PATH:-${EXPERIMENT_BUILD_CONTEXT}/Dockerfile}"
EXPERIMENT_TRACEQ_BUILD_CONTEXT="${EXPERIMENT_TRACEQ_BUILD_CONTEXT:-${ROOT_DIR}/libs/traceq}"
EXPERIMENT_COMMON_BUILD_CONTEXT="${EXPERIMENT_COMMON_BUILD_CONTEXT:-${ROOT_DIR}/libs/common}"
DATASET_FINAL_SIZE="${DATASET_FINAL_SIZE:-800}"
DATASET_STEP_SIZE="${DATASET_STEP_SIZE:-100}"
DIND_VOLUME_NAME="${DIND_VOLUME_NAME:-mac__dind-storage}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/02-predicting-memory-consumption-from-input-shapes/out/results/${TIMESTAMP}}"

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
echo "  DATASET_FINAL_SIZE=${DATASET_FINAL_SIZE}"
echo "  DATASET_STEP_SIZE=${DATASET_STEP_SIZE}"
echo "  DIND_VOLUME_NAME=${DIND_VOLUME_NAME}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo

echo "Starting experiment ${TIMESTAMP}..."

echo "Creating Docker volume for DIND..."
if ! docker volume inspect "${VOLUME_NAME}" &>/dev/null; then
  docker volume create "${VOLUME_NAME}"
fi

echo "Creating output dir..."
mkdir -p "${OUTPUT_DIR}"

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
  --env HOST_UID="$(id -u)" \
  --env HOST_GID="$(id -g)" \
  --env EXPERIMENT_IMAGE_TAG="${EXPERIMENT_IMAGE_TAG}" \
  --env EXPERIMENT_DOCKERFILE_PATH="/mnt${EXPERIMENT_DOCKERFILE_PATH}" \
  --env EXPERIMENT_BUILD_CONTEXT="/mnt${EXPERIMENT_BUILD_CONTEXT}" \
  --env EXPERIMENT_EXTRA_CONTEXTS="--build-context traceq=/mnt${EXPERIMENT_TRACEQ_BUILD_CONTEXT} --build-context common=/mnt${EXPERIMENT_COMMON_BUILD_CONTEXT}" \
  --env EXPERIMENT_N_RUNS="1" \
  --env EXPERIMENT_CPUSET_CPUS="${CPUSET_CPUS}" \
  --env EXPERIMENT_COMMAND="generate_data.py" \
  --env EXPERIMENT_ENV=" \
    -e OUTPUT_DIR=/experiment/out \
    -e FINAL_SIZE=${DATASET_FINAL_SIZE} \
    -e STEP_SIZE=${DATASET_STEP_SIZE} \
  " \
  --env EXPERIMENT_VOLUMES="-v /mnt${OUTPUT_DIR}/inputs:/experiment/out:rw" \
  docker:28.0.1-dind \
  "/workspace/experiment.sh"

echo "Collecting memory profile for Envelope..."
for file in "${OUTPUT_DIR}/inputs"/*.segy; do
  filename=$(basename "$file" .segy)
  session_id="envelope-${filename}"
  echo "  Processing file: $file (Session ID: ${session_id})"

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
    --env HOST_UID="$(id -u)" \
    --env HOST_GID="$(id -g)" \
    --env EXPERIMENT_IMAGE_TAG="${EXPERIMENT_IMAGE_TAG}" \
    --env EXPERIMENT_DOCKERFILE_PATH="/mnt${EXPERIMENT_DOCKERFILE_PATH}" \
    --env EXPERIMENT_BUILD_CONTEXT="/mnt${EXPERIMENT_BUILD_CONTEXT}" \
    --env EXPERIMENT_EXTRA_CONTEXTS="--build-context traceq=/mnt${EXPERIMENT_TRACEQ_BUILD_CONTEXT} --build-context common=/mnt${EXPERIMENT_COMMON_BUILD_CONTEXT}" \
    --env EXPERIMENT_N_RUNS="${EXPERIMENT_N_RUNS}" \
    --env EXPERIMENT_CPUSET_CPUS="${CPUSET_CPUS}" \
    --env EXPERIMENT_COMMAND="collect_memory_profile.py" \
    --env EXPERIMENT_ENV=" \
      -e ALGORITHM=envelope \
      -e OUTPUT_DIR=/experiment/out/profiles \
      -e SESSION_ID=${session_id} \
      -e INPUT_PATH=/experiment/out/inputs/${filename}.segy \
    " \
    --env EXPERIMENT_VOLUMES="-v /mnt${OUTPUT_DIR}:/experiment/out:rw" \
    docker:28.0.1-dind \
    "/workspace/experiment.sh"
done;

echo "Collecting memory profile for GST3D..."
for file in "${OUTPUT_DIR}/inputs"/*.segy; do
  filename=$(basename "$file" .segy)
  session_id="gst3d-${filename}"
  echo "  Processing file: $file (Session ID: ${session_id})"

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
    --env HOST_UID="$(id -u)" \
    --env HOST_GID="$(id -g)" \
    --env EXPERIMENT_IMAGE_TAG="${EXPERIMENT_IMAGE_TAG}" \
    --env EXPERIMENT_DOCKERFILE_PATH="/mnt${EXPERIMENT_DOCKERFILE_PATH}" \
    --env EXPERIMENT_BUILD_CONTEXT="/mnt${EXPERIMENT_BUILD_CONTEXT}" \
    --env EXPERIMENT_EXTRA_CONTEXTS="--build-context traceq=/mnt${EXPERIMENT_TRACEQ_BUILD_CONTEXT} --build-context common=/mnt${EXPERIMENT_COMMON_BUILD_CONTEXT}" \
    --env EXPERIMENT_N_RUNS="${EXPERIMENT_N_RUNS}" \
    --env EXPERIMENT_CPUSET_CPUS="${CPUSET_CPUS}" \
    --env EXPERIMENT_COMMAND="collect_memory_profile.py" \
    --env EXPERIMENT_ENV=" \
      -e ALGORITHM=gst3d \
      -e OUTPUT_DIR=/experiment/out/profiles \
      -e SESSION_ID=${session_id} \
      -e INPUT_PATH=/experiment/out/inputs/${filename}.segy \
    " \
    --env EXPERIMENT_VOLUMES="-v /mnt${OUTPUT_DIR}:/experiment/out:rw" \
    docker:28.0.1-dind \
    "/workspace/experiment.sh"
done;

echo "Collecting memory profile for Gaussian Filter..."
for file in "${OUTPUT_DIR}/inputs"/*.segy; do
  filename=$(basename "$file" .segy)
  session_id="gaussian-filter-${filename}"
  echo "  Processing file: $file (Session ID: ${session_id})"

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
    --env HOST_UID="$(id -u)" \
    --env HOST_GID="$(id -g)" \
    --env EXPERIMENT_IMAGE_TAG="${EXPERIMENT_IMAGE_TAG}" \
    --env EXPERIMENT_DOCKERFILE_PATH="/mnt${EXPERIMENT_DOCKERFILE_PATH}" \
    --env EXPERIMENT_BUILD_CONTEXT="/mnt${EXPERIMENT_BUILD_CONTEXT}" \
    --env EXPERIMENT_EXTRA_CONTEXTS="--build-context traceq=/mnt${EXPERIMENT_TRACEQ_BUILD_CONTEXT} --build-context common=/mnt${EXPERIMENT_COMMON_BUILD_CONTEXT}" \
    --env EXPERIMENT_N_RUNS="${EXPERIMENT_N_RUNS}" \
    --env EXPERIMENT_CPUSET_CPUS="${CPUSET_CPUS}" \
    --env EXPERIMENT_COMMAND="collect_memory_profile.py" \
    --env EXPERIMENT_ENV=" \
      -e ALGORITHM=gaussian_filter \
      -e OUTPUT_DIR=/experiment/out/profiles \
      -e SESSION_ID=${session_id} \
      -e INPUT_PATH=/experiment/out/inputs/${filename}.segy \
    " \
    --env EXPERIMENT_VOLUMES="-v /mnt${OUTPUT_DIR}:/experiment/out:rw" \
    docker:28.0.1-dind \
    "/workspace/experiment.sh"
done;

echo "Collecting the results..."
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
  --env HOST_UID="$(id -u)" \
  --env HOST_GID="$(id -g)" \
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
  --env EXPERIMENT_VOLUMES="-v /mnt${OUTPUT_DIR}/inputs:/experiment/out:rw" \
  docker:28.0.1-dind \
  "/workspace/experiment.sh"

echo "Analysing the results..."
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
  --env HOST_UID="$(id -u)" \
  --env HOST_GID="$(id -g)" \
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
  --env EXPERIMENT_VOLUMES="-v /mnt${OUTPUT_DIR}/inputs:/experiment/out:rw" \
  docker:28.0.1-dind \
  "/workspace/experiment.sh"