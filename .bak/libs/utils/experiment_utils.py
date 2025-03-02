import os
from datetime import datetime

__all__ = ["generate_output_dir_path"]


def generate_output_dir_path(
        experiment_name: str,
        output_base_dir: str = "./outputs",
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    timestamped_experiment_name = f"{experiment_name}-{timestamp}"
    path = os.path.join(output_base_dir, timestamped_experiment_name)
    os.makedirs(path, exist_ok=True)

    return path
