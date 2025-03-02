import numpy as np
from loguru import logger
from scipy import signal

__all__ = ["envelope_from_ndarray"]


def envelope_from_ndarray(data: np.ndarray):
    logger.info(f"Calculating envelope with data shape: {data.shape}")

    analytical_trace = signal.hilbert(data)
    result = np.abs(analytical_trace)

    logger.info(f"Envelope calculated with result shape: {result.shape}")
    logger.debug(f"Envelope calculated with result: {result}")

    return result
