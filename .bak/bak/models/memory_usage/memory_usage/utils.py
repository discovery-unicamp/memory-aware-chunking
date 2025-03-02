import os
import traceq

from datetime import datetime


__all__ = ["create_output_dir", "dataset_shape_from_path", "group_output_to_folder"]


def create_output_dir(attribute: str):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_name = f"../output/{attribute}-{timestamp}"
    os.makedirs(dir_name, exist_ok=True)

    return dir_name


def dataset_shape_from_path(dataset_path: str) -> str:
    return dataset_path.split("/")[-1].split(".")[0]


def group_output_to_folder(folder: str) -> str:
    already_is_using_folder = dowser.context.config.output_dir.as_posix().endswith(
        folder
    )

    return (
        traceq.context.config.output_dir
        if already_is_using_folder
        else f"{dowser.context.config.output_dir}/{folder}"
    )
