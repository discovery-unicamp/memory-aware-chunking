import random

import matplotlib.pyplot as plt
import numpy as np
import segyio

__all__ = ["load_segy", "render_random_inline"]


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
