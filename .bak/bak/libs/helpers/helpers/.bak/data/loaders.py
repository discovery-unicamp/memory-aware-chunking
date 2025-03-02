import segyio
import numpy as np


def load_segy(segy_file_path: str) -> np.ndarray:
    with segyio.open(segy_file_path, "r", strict=False) as segyfile:
        return segyio.tools.cube(segyfile)
