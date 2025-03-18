import numpy as np
from common.loaders import load_segy
from scipy import ndimage as ndi

__all__ = [
    "gradient_structure_tensor_from_ndarray",
    "gradient_structure_tensor_from_segy",
]


def first_derivative(X: np.ndarray, axis=-1):
    kernel = (3, 3, 3)
    axes = [ax for ax in range(X.ndim) if ax != axis]

    # First derivative along the main axis
    result0 = ndi.correlate1d(X, weights=np.array([-0.5, 0.0, 0.5]), axis=axis)

    # Apply smoothing filters along the other two axes
    result1 = ndi.correlate1d(
        result0, weights=np.array([0.178947, 0.642105, 0.178947]), axis=axes[0]
    )
    result2 = ndi.correlate1d(
        result1, weights=np.array([0.178947, 0.642105, 0.178947]), axis=axes[1]
    )

    return trim_numpy_array(result2, kernel)


def compute_gradient_structure_tensor(gi, gj, gk, kernel):
    hw = tuple(np.array(kernel) // 2)

    gi2 = ndi.uniform_filter(gi * gi, size=kernel, mode="reflect")
    gj2 = ndi.uniform_filter(gj * gj, size=kernel, mode="reflect")
    gk2 = ndi.uniform_filter(gk * gk, size=kernel, mode="reflect")
    gigj = ndi.uniform_filter(gi * gj, size=kernel, mode="reflect")
    gigk = ndi.uniform_filter(gi * gk, size=kernel, mode="reflect")
    gjgk = ndi.uniform_filter(gj * gk, size=kernel, mode="reflect")

    return gi2, gj2, gk2, gigj, gigk, gjgk


def compute_3d_dip(gi2, gj2, gk2, gigj, gigk, gjgk):
    shape = gi2.shape

    gst = np.array([[gi2, gigj, gigk], [gigj, gj2, gjgk], [gigk, gjgk, gk2]])

    # Move axes to align dimensions correctly
    gst = np.moveaxis(gst, [0, 1], [-2, -1])
    gst = gst.reshape((-1, 3, 3))

    # Compute eigenvalues and eigenvectors
    evals, evecs = np.linalg.eigh(gst)
    ndx = evals.argsort(axis=-1)
    evecs = evecs[np.arange(gst.shape[0]), :, ndx[:, -1]]

    # Normalize eigenvectors
    norm_factor = np.linalg.norm(evecs, axis=-1)
    evecs = evecs / norm_factor[:, None]

    # Compute dip angle
    dip = np.arccos(evecs[:, 2])
    dip = dip.reshape(shape)

    return np.rad2deg(dip)


def gradient_structure_tensor_from_ndarray(data: np.ndarray, kernel=(3, 3, 3)):
    gi = first_derivative(data, axis=0)
    gj = first_derivative(data, axis=1)
    gk = first_derivative(data, axis=2)

    gi2, gj2, gk2, gigj, gigk, gjgk = compute_gradient_structure_tensor(
        gi, gj, gk, kernel
    )

    return compute_3d_dip(gi2, gj2, gk2, gigj, gigk, gjgk)


def gradient_structure_tensor_from_segy(segy_path: str, kernel=(3, 3, 3)):
    data = load_segy(segy_path)
    return gradient_structure_tensor_from_ndarray(data, kernel)


def trim_numpy_array(arr: np.ndarray, kernel_size):
    pad = tuple(k // 2 for k in kernel_size)
    slices = tuple(slice(p, -p if p > 0 else None) for p in pad)
    return arr[slices]
