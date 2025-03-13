import numpy as np
from common.loaders import load_segy
from scipy import signal

__all__ = ["envelope_from_ndarray", "envelope_from_segy"]


def envelope_from_ndarray(data: np.ndarray):
    print(f"Calculating envelope with data shape: {data.shape}")

    analytical_trace = signal.hilbert(data)
    result = np.abs(analytical_trace)

    print(f"Envelope calculated with result shape: {result.shape}")
    print(f"Envelope calculated with result: {result}")

    return result


def envelope_from_segy(segy_path: str):
    data = load_segy(segy_path)
    return envelope_from_ndarray(data)
