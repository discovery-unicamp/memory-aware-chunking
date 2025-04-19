import os
import subprocess
import time
from typing import Tuple, Any

from docker.models.containers import Container
from docker.models.images import Image
from loguru import logger

import docker
from docker import DockerClient

__all__ = [
    "get_vfs_client",
    "build_image",
    "drop_caches_in_container",
    "run_isolated_container",
]


def build_image(image_name: str, client: DockerClient) -> Image:
    logger.info("Building TraceQ image...")
    traceq_image, _ = client.images.build(
        path="../../../../libs/traceq",
        tag="traceq",
        rm=True,
    )
    logger.info("TraceQ image built.")

    logger.info("Building experiment image...")

    experiment_image, _ = client.images.build(
        path="../",
        tag=image_name,
        rm=True,
    )

    logger.info("Experiment image built.")

    return experiment_image


def get_vfs_client(
    host_out_dir: str,
    didn_container_name: str = "vfs-host",
    sleep_time: int = 2,
    max_retries: int = 20,
) -> Tuple[DockerClient, Container]:
    client = docker.from_env()
    vfs_container = client.containers.run(
        image="docker:dind",
        name=didn_container_name,
        auto_remove=True,
        privileged=True,
        detach=True,
        ports={"2375/tcp": 2375},
        command=["--storage-driver=vfs"],
        environment={
            "DOCKER_TLS_CERTDIR": "",
        },
        volumes={
            os.path.abspath(host_out_dir): {
                "bind": "/mnt/out",
                "mode": "rw",
            },
        },
    )

    logger.info("Start dind dockerd...")

    is_dockerd_running = __wait_for_dockerd(sleep_time, max_retries)
    if not is_dockerd_running:
        raise Exception("Dockerd inside dind is not operational.")

    vfs_client = docker.DockerClient(base_url="tcp://localhost:2375")

    return (vfs_client, vfs_container)


def run_isolated_container(
    client: DockerClient,
    image_name: str,
    command: list[str],
    environment: dict[str, str] = None,
    volumes: dict[str, dict[str, str]] = None,
    detach: Any = True,
    auto_remove: bool = True,
    privileged: bool = True,
    nano_cpus: int = int(
        1 * 1e9
    ),  # Strictly allocate 1 CPU (1 * 1,000,000,000 nanoCPUs)
    cpu_shares: int = 1024,
    cpuset_cpus: str = "0",
    container_ancestor: str = "docker:dind",
) -> Container:
    drop_caches_in_container(container_ancestor)

    logger.info("Launching in isolated container...")
    container = client.containers.run(
        image=image_name,
        detach=detach,
        auto_remove=auto_remove,
        privileged=privileged,
        nano_cpus=nano_cpus,
        cpu_shares=cpu_shares,
        cpuset_cpus=cpuset_cpus,
        command=command,
        environment=environment,
        volumes=volumes,
    )
    result = container.wait()
    logger.info("Finished running in isolated container")

    return result


def drop_caches_in_container(container_ancestor: str = "docker:dind"):
    container_id = subprocess.run(
        ["docker", "ps", "-q", "--filter", f"ancestor={container_ancestor}"],
        capture_output=True,
        text=True,
    ).stdout.strip()

    if not container_id:
        logger.error(f"{container_ancestor} container not found")
        return

    drop_cmd = (
        f"docker exec {container_id} sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches'"
    )
    subprocess.run(drop_cmd, shell=True, check=True)
    logger.info(f"Dropped caches inside {container_ancestor} container")


def __wait_for_dockerd(sleep_time: int, max_retries: int) -> bool:
    for _ in range(max_retries):
        try:
            subprocess.run(["docker", "-H", "tcp://localhost:2375", "info"], check=True)
            logger.info("Dockerd inside dind is fully operational.")
            return True
        except subprocess.CalledProcessError:
            logger.info("Waiting for dockerd to start...")
            time.sleep(sleep_time)

    return False
