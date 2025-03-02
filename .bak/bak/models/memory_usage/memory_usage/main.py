import os
from typing import Dict

import dowser
from dowser import get_logger

from constants import DEFAULTS
from data_collection import collect_data
from utils import create_output_dir


def run(
    attribute: str,
    max_inlines: int,
    max_crosslines: int,
    max_samples: int,
    log_level: str,
    amount_of_datasets: int,
    precision: int,
    min_dimension: int,
) -> None:
    _bootstrap(attribute, log_level, precision)

    logger = get_logger()

    logger.info("Starting experiment collection")
    collected_data = collect_data(
        max_inlines,
        max_crosslines,
        max_samples,
        amount_of_datasets,
        attribute,
        min_dimension=min_dimension,
    )
    logger.debug(f"Collected experiment Session ID: {collected_data}")


def run_from_env(defaults: Dict[str, str] = DEFAULTS) -> None:
    arg_keys = [
        "attribute",
        "max_inlines",
        "max_crosslines",
        "max_samples",
        "log_level",
        "amount_of_datasets",
        "precision",
        "min_dimension",
    ]
    arg_transformers = [str, int, int, int, str, int, int, int]
    args = [
        transformer(os.environ.get(key.upper(), defaults.get(key.lower())))
        for key, transformer in zip(arg_keys, arg_transformers)
    ]

    _validate(*args)
    run(*args)


def _bootstrap(attribute: str, log_level: str, precision: int) -> None:
    output_dir = create_output_dir(attribute)

    dowser.load_config(
        {
            "output_dir": output_dir,
            "logger": {
                "level": log_level,
                "enabled_transports": ["CONSOLE", "FILE"],
            },
            "profiler": {
                "precision": precision,
            },
        }
    )


def _validate(attribute: str, *_) -> None:
    if not attribute or attribute == "None":
        raise ValueError("Attribute is required")


if __name__ == "__main__":
    run_from_env()
