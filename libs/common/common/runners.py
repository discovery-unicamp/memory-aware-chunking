import os

import docker
from docker.client import DockerClient
from docker.errors import NotFound

from .transformers import transform_to_container_path, transform_to_context_name

__all__ = ["run_isolated_container"]


def run_isolated_container(
    client=docker.from_env(),
    experiment_command: str = "",
    experiment_dockerfile_name: str = "Dockerfile",
    experiment_build_context: str = ".",
    experiment_image_tag: str = "experiment",
    experiment_extra_contexts: list[str] = None,
    experiment_volumes: dict[str, str] = None,
    experiment_env: dict[str, str] = None,
    experiment_n_runs: int = 1,
    experiment_dind_volume_name: str = "mac__dind-storage",
    cpuset_cpus: str = "0",
    scripts_path: str = os.path.abspath(os.path.join(__file__, "..", "..", "scripts")),
    auto_remove: bool = True,
):
    experiment_extra_contexts = experiment_extra_contexts or []
    experiment_volumes = experiment_volumes or {}
    experiment_env = experiment_env or {}
    dind_volume_name = __create_dind_storage(client, experiment_dind_volume_name)

    dind_container_volumes = {
        scripts_path: {
            "bind": "/workspace",
            "mode": "ro",
        },
        os.path.abspath(experiment_build_context): {
            "bind": transform_to_container_path(experiment_build_context),
            "mode": "ro",
        },
        dind_volume_name: {
            "bind": "/var/lib/docker",
            "mode": "rw",
        },
        **{
            os.path.abspath(context): {
                "bind": transform_to_container_path(context),
                "mode": "ro",
            }
            for context in experiment_extra_contexts
        },
        **{
            os.path.abspath(host): {
                "bind": transform_to_container_path(host),
                "mode": "rw",
            }
            for host, _ in experiment_volumes.items()
        },
    }

    experiment_volume_args = " ".join(
        f"-v {transform_to_container_path(host)}:{container}"
        for host, container in experiment_volumes.items()
    )

    experiment_extra_context_args = " ".join(
        f"--build-context {transform_to_context_name(context)}={transform_to_container_path(context)}"
        for context in experiment_extra_contexts
    )

    experiment_env_args = " ".join(
        f"-e {key}={value}" for key, value in experiment_env.items()
    )

    experiment_container_build_context = transform_to_container_path(
        experiment_build_context
    )
    experiment_dockerfile_path = os.path.join(
        experiment_container_build_context,
        experiment_dockerfile_name,
    )

    print(f"Running isolated container...")
    container = client.containers.run(
        image="docker:28.0.1-dind",
        auto_remove=auto_remove,
        privileged=True,
        detach=True,
        entrypoint="/bin/sh",
        command=["/workspace/experiment.sh"],
        cpuset_cpus=cpuset_cpus,
        volumes=dind_container_volumes,
        environment={
            "DOCKER_TLS_CERTDIR": "",
            "EXPERIMENT_IMAGE_TAG": experiment_image_tag,
            "EXPERIMENT_DOCKERFILE_PATH": experiment_dockerfile_path,
            "EXPERIMENT_BUILD_CONTEXT": experiment_container_build_context,
            "EXPERIMENT_EXTRA_CONTEXTS": experiment_extra_context_args,
            "EXPERIMENT_N_RUNS": experiment_n_runs,
            "EXPERIMENT_CPUSET_CPUS": cpuset_cpus,
            "EXPERIMENT_VOLUMES": experiment_volume_args,
            "EXPERIMENT_ENV": experiment_env_args,
            "EXPERIMENT_COMMAND": experiment_command,
        },
    )

    result = container.wait()
    status_code = result["StatusCode"]
    print(f"Finished running isolated container. Exit status: {status_code}")

    return status_code


def __create_dind_storage(
    client: DockerClient,
    volume_name: str = "mac__dind-storage",
) -> str:
    try:
        client.volumes.get(volume_name)  # Check if volume exists
        print(f"Using existing Docker volume: {volume_name}")
    except NotFound:
        print(f"Creating new Docker volume: {volume_name}")
        client.volumes.create(name=volume_name)

    return volume_name
