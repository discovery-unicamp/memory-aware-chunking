import argparse
from typing import Literal

__all__ = ["ArgsNamespace"]


class ArgsNamespace(argparse.Namespace):
    command: Literal["generate-data", "operate"]

    # generate-data
    inlines: int
    xlines: int
    samples: int
    prefix: str
    output_dir: str

    # operate
    operator: Literal["envelope"]
    segy_path: str
    memory_profiler: Literal[None, "psutil", "resource", "tracemalloc", "kernel"]
    memory_profile_output_dir: str
    memory_profile_session_id: str
