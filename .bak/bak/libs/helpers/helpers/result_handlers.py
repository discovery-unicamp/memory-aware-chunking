import gzip
import os
from typing import List

import msgpack
from traceq.profiler.loaders import load_profile

__all__ = ["load_profile_results", "get_peak", "get_unit"]

DEFAULT_MEM_USAGE_METRIC = "kernel_memory_usage"


def load_profile_results(output_path: str):
    session_paths = __get_session_paths(output_path)

    __normalize_metadata(session_paths)
    session_names = __get_session_names(session_paths)

    return zip(session_names, session_paths)


def get_peak(profile_path: str, metric: str = DEFAULT_MEM_USAGE_METRIC):
    profile = load_profile(profile_path)
    data = profile["experiment"]

    return max(item[metric] for item in data)


def get_unit(profile_path: str, metric: str = DEFAULT_MEM_USAGE_METRIC):
    profile = load_profile(profile_path)
    return profile["metadata"][f"{metric}_unit"]


def __get_session_paths(directory_path: str):
    dirs = __list_directories(directory_path)
    shapes = [
        shape
        for shape in dirs
        if "experiment" not in shape and "checkpoints" not in shape
    ]

    return [__find_profiles(shape)[0] for shape in shapes]


def __list_directories(path: str):
    entries = os.listdir(path)

    directories = [
        os.path.join(path, entry)
        for entry in entries
        if os.path.isdir(os.path.join(path, entry))
    ]
    return directories


def __find_profiles(directory: str):
    parquet_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".prof"):
                full_path = os.path.join(root, file)
                parquet_files.append(full_path)
    return parquet_files


def __get_session_names(sessions: List[str]):
    return [
        os.path.basename(session).split("/")[-1].split(".")[0] for session in sessions
    ]


def __normalize_metadata(sessions: List[str]):
    for session in sessions:
        profile = load_profile(session)

        metadata = profile["metadata"]
        metadata_dict = {k: v for k, v in metadata.items()}

        entrypoint_segy_filepath = metadata_dict.pop("entrypoint_segy_filepath", None)
        if entrypoint_segy_filepath:
            entrypoint_shape = os.path.basename(entrypoint_segy_filepath).split(".")[0]
            entrypoint_shape = f"({entrypoint_shape.replace('-', ',')})"
            metadata_dict["entrypoint_shape"] = entrypoint_shape

        new_metadata = {k: v for k, v in metadata_dict.items()}
        profile["metadata"] = new_metadata

        with gzip.open(session, "wb") as f:
            packed = msgpack.packb(profile)
            f.write(packed)
