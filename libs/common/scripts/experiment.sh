#!/bin/sh

dockerd --host=unix:///var/run/docker.sock --host=tcp://0.0.0.0:2375 --storage-driver=vfs &

until docker info >/dev/null 2>&1; do
  echo "Waiting for Docker to be ready..."
  sleep 1
done
echo "Docker inside dind is ready"

echo "Building Docker image..."
docker_build_command="\
  docker buildx build \
    ${EXPERIMENT_EXTRA_CONTEXTS} \
    -t ${EXPERIMENT_IMAGE_TAG} \
    -f ${EXPERIMENT_DOCKERFILE_PATH} \
    ${EXPERIMENT_BUILD_CONTEXT} \
"
if ! eval "$docker_build_command"; then
  echo "Failed to build the image."
  echo "Build command used: ${docker_build_command}"
  exit 1
fi

for i in $(seq 1 "${EXPERIMENT_N_RUNS}"); do
  echo "Running container (${i}/${EXPERIMENT_N_RUNS})..."
  docker_run_command="\
    docker run \
      --rm \
      --cpuset-cpus=${EXPERIMENT_CPUSET_CPUS} \
      ${EXPERIMENT_VOLUMES} \
      ${EXPERIMENT_ENV} \
      docker.io/library/${EXPERIMENT_IMAGE_TAG} \
      ${EXPERIMENT_COMMAND} \
  "
  if ! eval "$docker_run_command"; then
    echo "Failed to run the container."
    echo "Run command used: ${docker_run_command}"
    exit 1
  fi

  sync && echo 3 > /proc/sys/vm/drop_caches
done

echo "Execution completed. Exiting docker:dind container."