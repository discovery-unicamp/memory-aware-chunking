import os
import random
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import segyio
from loguru import logger
from scipy.signal import convolve

__all__ = [
    "generate_seismic_data",
    "load_segy",
    "render_random_inline",
]


def generate_seismic_data(
    inlines: int = 100,
    xlines: int = 100,
    samples: int = 100,
    dt: float = 0.004,
    frequency: int = 25,
    length: int = 250,
    output_dir: Optional[str] = None,
    prefix: str = "",
) -> str:
    filepath = None
    if output_dir:
        filename = f"{inlines}-{xlines}-{samples}.segy"
        filename = f"{prefix}-{filename}" if prefix else filename
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            logger.info(
                f"Skipping generation of synthetic data for shape ({inlines}, {xlines}, {samples}) as it already exists"
            )
            return filepath

    logger.info(f"Generating synthetic data for shape ({inlines}, {xlines}, {samples})")

    reflectivity = np.random.rand(samples) * 2 - 1
    wavelet = __ricker_wavelet(frequency, length, dt)

    seismic_data = np.array(
        [
            __generate_synthetic_seismic(
                reflectivity,
                wavelet,
                xlines,
                samples,
            )
            for _ in range(inlines)
        ]
    ).astype(np.float32)

    logger.info("Synthetic data generated successfully")

    if output_dir:
        __save_seismic_to_segy(seismic_data, filepath, dt)

    logger.info(f"Synthetic data saved to {filepath}")

    return filepath if output_dir else seismic_data


def load_segy(segy_file_path: str) -> np.ndarray:
    with segyio.open(segy_file_path, "r", strict=False) as segyfile:
        return segyio.tools.cube(segyfile)


def render_random_inline(segy_data: np.ndarray) -> None:
    num_inlines = segy_data.shape[0]
    random_inline_idx = random.randint(0, num_inlines - 1)

    plt.figure(figsize=(10, 6))
    plt.imshow(segy_data[random_inline_idx, :, :], cmap="gray", aspect="auto")
    plt.title(f"Inline {random_inline_idx}")
    plt.xlabel("Crossline")
    plt.ylabel("Samples")
    plt.colorbar(label="Amplitude")
    plt.show()


def __ricker_wavelet(frequency: int, length: int, dt: float) -> np.ndarray:
    t = np.linspace(-length / 2, (length / 2) - dt, length) * dt
    y = (1 - 2 * (np.pi**2) * (frequency**2) * (t**2)) * np.exp(
        -(np.pi**2) * (frequency**2) * (t**2)
    )

    return y


def __generate_synthetic_seismic(
    reflectivity: float,
    wavelet: np.ndarray,
    num_traces: int,
    num_samples: int,
) -> np.ndarray:
    seismic_data = np.zeros((num_traces, num_samples))
    for i in range(num_traces):
        seismic_data[i, :] = convolve(reflectivity, wavelet, mode="same")

    return seismic_data


def __save_seismic_to_segy(
    data: np.ndarray,
    filename: str,
    sample_interval: int,
) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    num_inlines, num_samples, num_crosslines = data.shape
    sample_interval_microseconds = int(sample_interval * 1e6)

    spec = segyio.spec()
    spec.sorting = 2
    spec.format = 1
    spec.samples = np.arange(num_samples) * sample_interval
    spec.ilines = np.arange(1, num_inlines + 1)
    spec.xlines = np.arange(1, num_crosslines + 1)
    spec.tracecount = num_inlines * num_crosslines

    with segyio.create(filename, spec) as segyfile:
        trace_index = 0
        for iline in spec.ilines:
            for xline in spec.xlines:
                segyfile.header[trace_index] = {
                    segyio.su.iline: iline,
                    segyio.su.xline: xline,
                }
                segyfile.trace[trace_index] = data[iline - 1, :, xline - 1].flatten()
                trace_index += 1

        segyfile.bin[segyio.BinField.Interval] = sample_interval_microseconds
        segyfile.bin[segyio.BinField.Traces] = spec.tracecount
        segyfile.bin[segyio.BinField.Samples] = num_samples
