import os

__all__ = [
    "transform_b_to_gb",
    "transform_kb_to_gb",
    "transform_b_to_mb",
    "transform_mb_to_gb",
    "transform_ns_to_s",
    "transform_to_container_path",
    "transform_to_context_name",
]


def transform_b_to_gb(b_value: int) -> float:
    return b_value / (1024**3)


def transform_kb_to_gb(kb_value: int) -> float:
    return kb_value / (1024**2)


def transform_b_to_mb(b_value: int) -> float:
    return b_value / (1024**2)


def transform_mb_to_gb(mb_value: int) -> float:
    return mb_value / 1024


def transform_ns_to_s(ns_value: int) -> float:
    return ns_value / 1e9


def transform_to_container_path(path: str) -> str:
    return f"/mnt/{os.path.abspath(path)}".replace("//", "/")


def transform_to_context_name(path: str) -> str:
    return path.split("/")[-1]
