import numpy as np
from common.loaders import load_segy
from scipy import ndimage as ndi

__all__ = [
    "gaussian_filter_from_ndarray",
    "gaussian_filter_from_segy",
]


def gaussian_filter_from_ndarray(
    data: np.ndarray,
    sigma=(3, 3, 3),
):
    print(f"Data shape: {data.shape}")
    return ndi.gaussian_filter(data, sigma=sigma)


def gaussian_filter_from_segy(segy_path: str):
    print(f"Calculating Gaussian Filter for segy: {segy_path}")
    data = load_segy(segy_path)
    return gaussian_filter_from_ndarray(data)
