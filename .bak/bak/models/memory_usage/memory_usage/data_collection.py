import importlib
import importlib
import json
from typing import List, Callable, Dict

import numpy as np
import traceq
from helpers.datasets import generate_seismic_data
from traceq import get_logger
from traceq.profiler import Profile

from utils import group_output_to_folder, dataset_shape_from_path

__all__ = ["collect_data"]


def collect_data(
    max_inlines: int,
    max_crosslines: int,
    max_samples: int,
    amount_of_datasets: int,
    attribute: str,
    min_dimension: int,
) -> Dict[str, float]:
    logger = get_logger()

    input_datasets = _generate_many_evenly_spaced(
        max_inlines,
        max_crosslines,
        max_samples,
        amount_of_datasets,
        min_dimension=min_dimension,
    )
    attribute_handler = _get_attribute_handler(attribute)
    profile_paths = _profile_datasets(attribute_handler, input_datasets)
    profiles = _read_profiles(profile_paths)

    data = _parse_profiles(profiles)
    _store_collected_data(data)

    logger.info("Data collection finished")

    return data


def _generate_many_evenly_spaced(
    max_inlines: int,
    max_crosslines: int,
    max_samples: int,
    amount_of_datasets: int,
    data_folder: str = "experiment",
    min_dimension: int = 100,
) -> List[str]:
    logger = traceq.get_logger()
    logger.info(
        f"Generating {amount_of_datasets} sets of synthetic experiment with distributed shapes within max bounds "
        f"({max_inlines}, {max_crosslines}, {max_samples})"
    )

    filepaths = []
    output_dir = f"{traceq.context.config.output_dir}/{data_folder}"
    shapes = _generate_even_shapes(
        max_inlines,
        max_crosslines,
        max_samples,
        amount_of_datasets,
        min=min_dimension,
    )
    amount_of_generated_shapes = len(shapes)
    logger.info(
        f"Calculated {amount_of_generated_shapes} shapes to evenly generate experiment for"
    )

    for i in range(amount_of_generated_shapes):
        num_inlines, num_crosslines, num_samples = shapes[i]

        logger.info(
            f"Generating synthetic experiment {i+1}/{amount_of_generated_shapes} with shape "
            f"({num_inlines}, {num_crosslines}, {num_samples})"
        )

        filepath = generate_seismic_data(
            num_inlines,
            num_crosslines,
            num_samples,
            output_dir=output_dir,
        )

        filepaths.append(filepath)
        logger.info(
            f"Finished generating synthetic experiment {i+1}/{amount_of_generated_shapes}. Data stored on: {filepath}"
        )

    logger.info("All synthetic experiment generated")

    return filepaths


def _generate_even_shapes(
    max_inlines,
    max_crosslines,
    max_samples,
    num_shapes,
    min=100,
):
    steps_per_dim = int(np.ceil(np.power(num_shapes, 1 / 3)))

    inlines_steps = np.linspace(min, max_inlines, steps_per_dim, endpoint=False)
    crosslines_steps = np.linspace(min, max_crosslines, steps_per_dim, endpoint=False)
    samples_steps = np.linspace(min, max_samples, steps_per_dim, endpoint=False)

    inlines_steps[-1] = max_inlines
    crosslines_steps[-1] = max_crosslines
    samples_steps[-1] = max_samples

    shapes = []
    for inlines in inlines_steps:
        for crosslines in crosslines_steps:
            for samples in samples_steps:
                shapes.append((int(inlines), int(crosslines), int(samples)))

    while len(shapes) < num_shapes:
        shapes += shapes[: (num_shapes - len(shapes))]

    return shapes[:num_shapes]


def _profile_datasets(handler: Callable, dataset_paths: List[str]) -> List[str]:
    logger = get_logger()

    logger.info(
        f'Profiling datasets with handler "{handler.__module__}.{handler.__name__}"'
    )

    profiles = []
    for i, dataset_path in enumerate(dataset_paths):
        logger.debug(
            f"Profiling dataset with path {dataset_path}. Dataset {i+1}/{len(dataset_paths)}"
        )
        profile = _profile_single(handler, dataset_path)
        profiles.append(profile)

    return profiles


def _profile_single(handler: Callable, dataset_path: str) -> str:
    logger = get_logger()

    session_id = dataset_shape_from_path(dataset_path)
    output_dir = group_output_to_folder("profiles")

    logger.debug(
        f'Profiling dataset with handler "{handler.__module__}.{handler.__name__}" using path "{dataset_path}"'
    )
    logger.debug(f"Session ID: {session_id}")

    traceq.load_config(
        {
            "output_dir": output_dir,
            "profiler": {
                "session_id": session_id,
            },
        }
    )
    traceq.profile(handler, dataset_path)

    return f"{output_dir}/{session_id}.prof"


def _get_attribute_handler(attribute: str) -> Callable:
    logger = get_logger()

    logger.info(f'Loading attribute handler for "{attribute}"')

    module = importlib.import_module(f"seismic.attributes.{attribute}")
    handler = module.run
    logger.debug(f"Attribute handler loaded: {handler}")

    return handler


def _read_profiles(profile_paths: str) -> List[Profile]:
    return [
        traceq.profiler.load_profile(profile_path) for profile_path in profile_paths
    ]


def _get_peak_from_profile(profile: Profile) -> float:
    return max(profile["experiment"], key=lambda x: x["kernel_memory_usage"])[
        "kernel_memory_usage"
    ]


def _parse_profiles(profiles: List[Profile]) -> Dict:
    logger = get_logger()
    data = {
        "memory_unit": None,
        "peaks": {},
    }

    logger.info("Parsing the peak experiment from profiles")
    for profile in profiles:
        if data["memory_unit"] is None:
            logger.debug("Setting memory unit")
            data["memory_unit"] = profile["metadata"]["kernel_memory_usage_unit"]

        shape = dataset_shape_from_path(profile["metadata"]["entrypoint_segy_filepath"])
        logger.debug(f"Getting peak from profile with shape {shape}")

        peak = _get_peak_from_profile(profile)
        data["peaks"][shape] = peak

    return data


def _store_collected_data(data: Dict) -> None:
    logger = get_logger()

    output_dir = traceq.context.config.output_dir.parent
    output_path = f"{output_dir}/collected_data.json"

    logger.debug(f"Saving collected experiment to {output_path}")
    json.dump(data, open(output_path, "w"), indent=4, sort_keys=True)
