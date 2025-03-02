import argparse
from typing import Literal

__all__ = ["ArgsNamespace"]


class ArgsNamespace(argparse.Namespace):
    command: Literal["generate-data", "operate"]

    # generate-data
    inlines: int
    xlines: int
    samples: int
    output_dir: str
    prefix: str

    # operate
    operator: Literal["envelope"]
    segy_path: str
