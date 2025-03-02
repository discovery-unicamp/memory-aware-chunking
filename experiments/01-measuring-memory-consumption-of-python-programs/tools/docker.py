import os
import subprocess
import time
from typing import Tuple

import docker
from docker import DockerClient
from docker.models.containers import Container
from docker.models.images import Image
from loguru import logger

__all__ = ["get_vfs_client", "build_image"]


def build_image(image_name: str, client: DockerClient) -> Image:
    logger.info("Building TraceQ image...")
    traceq_image, _ = client.images.build(
        path="../../../libs/traceq",
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
