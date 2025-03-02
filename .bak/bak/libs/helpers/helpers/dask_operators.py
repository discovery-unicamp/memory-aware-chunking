import dask.array as da
import numpy as np
import segyio
from scipy import ndimage as ndi
from scipy import signal

__all__ = [
    "load_segy",
    "envelope_from_segy",
    "envelope_from_ndarray",
    "gradient_structure_tensor_from_segy",
    "gradient_structure_tensor_from_ndarray",
    "gradient_structure_tensor_from_dask_array"
]

#### Data Loader

def load_segy(segy_file_path: str) -> np.ndarray:
    with segyio.open(segy_file_path, "r", strict=False) as segyfile:
        return segyio.tools.cube(segyfile)

#### Dask Helpers

def trim_dask_array(in_data, kernel, hw=None, boundary='reflect'):
    if hw is None:
        hw = tuple(np.array(kernel) // 2)
    axes = {0: hw[0], 1: hw[1], 2: hw[2]}
    
    return da.overlap.trim_internal(in_data, axes=axes, boundary=boundary)

#### Envelope


def envelope_from_segy(segy_path: str):
    data = load_segy(segy_path)
    return envelope_from_ndarray(data)


def envelope_from_ndarray(data: np.ndarray):
    analytical_trace = da.map_blocks(signal.hilbert, data, dtype=data.dtype)
    absolute = da.absolute(analytical_trace)

    return absolute.compute()

#### Gradient Structure Tensor (GST) for Dip Calculation

def first_derivative(X, axis=-1):
    """Compute the first derivative along a given axis using finite differences."""
    kernel = (3, 3, 3)
    axes = [ax for ax in range(X.ndim) if ax != axis]

    result0 = X.map_blocks(ndi.correlate1d,
                           weights=np.array([-0.5, 0.0, 0.5]),
                           axis=axis, dtype=X.dtype,
                           meta=np.array((), dtype=X.dtype))

    result1 = result0.map_blocks(ndi.correlate1d,
                                 weights=np.array([0.178947, 0.642105, 0.178947]),
                                 axis=axes[0], dtype=X.dtype,
                                 meta=np.array((), dtype=X.dtype))

    result2 = result1.map_blocks(ndi.correlate1d,
                                 weights=np.array([0.178947, 0.642105, 0.178947]),
                                 axis=axes[1], dtype=X.dtype,
                                 meta=np.array((), dtype=X.dtype))

    return trim_dask_array(result2, kernel)


def compute_gradient_structure_tensor(gi, gj, gk, kernel):
    """Compute the inner product of gradients for gradient structure tensor."""
    hw = tuple(np.array(kernel) // 2)

    gi2 = (gi * gi).map_overlap(ndi.uniform_filter, depth=hw, boundary='reflect', dtype=gi.dtype, size=kernel)
    gj2 = (gj * gj).map_overlap(ndi.uniform_filter, depth=hw, boundary='reflect', dtype=gj.dtype, size=kernel)
    gk2 = (gk * gk).map_overlap(ndi.uniform_filter, depth=hw, boundary='reflect', dtype=gk.dtype, size=kernel)
    gigj = (gi * gj).map_overlap(ndi.uniform_filter, depth=hw, boundary='reflect', dtype=gj.dtype, size=kernel)
    gigk = (gi * gk).map_overlap(ndi.uniform_filter, depth=hw, boundary='reflect', dtype=gk.dtype, size=kernel)
    gjgk = (gj * gk).map_overlap(ndi.uniform_filter, depth=hw, boundary='reflect', dtype=gj.dtype, size=kernel)

    return gi2, gj2, gk2, gigj, gigk, gjgk


def compute_3d_dip(gi2, gj2, gk2, gigj, gigk, gjgk):
    """Function to compute 3D dip from Gradient Structure Tensor."""
    shape = gi2.shape

    gst = np.array([[gi2, gigj, gigk],
                    [gigj, gj2, gjgk],
                    [gigk, gjgk, gk2]])

    gst = np.moveaxis(gst, [0, 1], [-2, -1])
    gst = gst.reshape((-1, 3, 3))

    evals, evecs = np.linalg.eigh(gst)
    ndx = evals.argsort(axis=-1)
    evecs = evecs[np.arange(gst.shape[0]), :, ndx[:, -1]]

    norm_factor = np.linalg.norm(evecs, axis=-1)
    evecs = evecs / norm_factor[:, None]

    dip = np.arccos(evecs[:, 2])
    dip = dip.reshape(shape)

    return np.rad2deg(dip)

def gradient_structure_tensor_from_dask_array(dask_array: da.Array, kernel=(3, 3, 3)):
    # Compute first derivatives along each axis using Dask array
    gi = first_derivative(dask_array, axis=0)
    gj = first_derivative(dask_array, axis=1)
    gk = first_derivative(dask_array, axis=2)

    # Compute gradient structure tensor components using Dask array
    gi2, gj2, gk2, gigj, gigk, gjgk = compute_gradient_structure_tensor(gi, gj, gk, kernel)
    
    # Map the 3D dip computation across the Dask chunks
    result = da.map_blocks(compute_3d_dip, gi2, gj2, gk2, gigj, gigk, gjgk, dtype=dask_array.dtype)
    
    return result.compute()

def gradient_structure_tensor_from_ndarray(data: np.ndarray, kernel=(3, 3, 3)):
    gi = first_derivative(da.from_array(data, chunks='auto'), axis=0)
    gj = first_derivative(da.from_array(data, chunks='auto'), axis=1)
    gk = first_derivative(da.from_array(data, chunks='auto'), axis=2)

    gi2, gj2, gk2, gigj, gigk, gjgk = compute_gradient_structure_tensor(gi, gj, gk, kernel)
    result = da.map_blocks(compute_3d_dip, gi2, gj2, gk2, gigj, gigk, gjgk, dtype=data.dtype)

    return result.compute()


def gradient_structure_tensor_from_segy(segy_path: str, kernel=(3, 3, 3)):
    data = load_segy(segy_path)
    return gradient_structure_tensor_from_ndarray(data, kernel)