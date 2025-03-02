import dask.array as da
import numpy as np
from datasets import load_segy
from loggers import get_named_logger
from scipy import ndimage as ndi

__all__ = [
    "gradient_structure_tensor_from_segy",
]


def gradient_structure_tensor_from_segy(segy_path, chunks="auto", kernel=(3, 3, 3), logger=get_named_logger('gst3d')):
    data = load_segy(segy_path)
    dask_array = da.from_array(data, chunks=chunks)

    logger.info(f"Calculating GST3D for {segy_path}")
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Data chunks: {dask_array.chunks}")

    # Compute first derivatives along each axis using Dask array
    gi = __first_derivative(dask_array, axis=0)
    gj = __first_derivative(dask_array, axis=1)
    gk = __first_derivative(dask_array, axis=2)

    # Compute gradient structure tensor components using Dask array
    gi2, gj2, gk2, gigj, gigk, gjgk = __compute_gradient_structure_tensor(
        gi, gj, gk, kernel
    )

    # Map the 3D dip computation across the Dask chunks
    return da.map_blocks(
        __compute_3d_dip, gi2, gj2, gk2, gigj, gigk, gjgk, dtype=dask_array.dtype
    )


def __first_derivative(X, axis=-1):
    """Compute the first derivative along a given axis using finite differences."""
    kernel = (3, 3, 3)
    axes = [ax for ax in range(X.ndim) if ax != axis]

    result0 = X.map_blocks(
        ndi.correlate1d,
        weights=np.array([-0.5, 0.0, 0.5]),
        axis=axis,
        dtype=X.dtype,
        meta=np.array((), dtype=X.dtype),
    )

    result1 = result0.map_blocks(
        ndi.correlate1d,
        weights=np.array([0.178947, 0.642105, 0.178947]),
        axis=axes[0],
        dtype=X.dtype,
        meta=np.array((), dtype=X.dtype),
    )

    result2 = result1.map_blocks(
        ndi.correlate1d,
        weights=np.array([0.178947, 0.642105, 0.178947]),
        axis=axes[1],
        dtype=X.dtype,
        meta=np.array((), dtype=X.dtype),
    )

    return __trim_dask_array(result2, kernel)


def __compute_gradient_structure_tensor(gi, gj, gk, kernel):
    """Compute the inner product of gradients for gradient structure tensor."""
    hw = tuple(np.array(kernel) // 2)

    gi2 = (gi * gi).map_overlap(
        ndi.uniform_filter, depth=hw, boundary="reflect", dtype=gi.dtype, size=kernel
    )
    gj2 = (gj * gj).map_overlap(
        ndi.uniform_filter, depth=hw, boundary="reflect", dtype=gj.dtype, size=kernel
    )
    gk2 = (gk * gk).map_overlap(
        ndi.uniform_filter, depth=hw, boundary="reflect", dtype=gk.dtype, size=kernel
    )
    gigj = (gi * gj).map_overlap(
        ndi.uniform_filter, depth=hw, boundary="reflect", dtype=gj.dtype, size=kernel
    )
    gigk = (gi * gk).map_overlap(
        ndi.uniform_filter, depth=hw, boundary="reflect", dtype=gk.dtype, size=kernel
    )
    gjgk = (gj * gk).map_overlap(
        ndi.uniform_filter, depth=hw, boundary="reflect", dtype=gj.dtype, size=kernel
    )

    return gi2, gj2, gk2, gigj, gigk, gjgk


def __compute_3d_dip(gi2, gj2, gk2, gigj, gigk, gjgk):
    """Function to compute 3D dip from Gradient Structure Tensor."""
    shape = gi2.shape

    gst = np.array([[gi2, gigj, gigk], [gigj, gj2, gjgk], [gigk, gjgk, gk2]])

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


def __trim_dask_array(in_data, kernel, hw=None, boundary="reflect"):
    if hw is None:
        hw = tuple(np.array(kernel) // 2)
    axes = {0: hw[0], 1: hw[1], 2: hw[2]}

    return da.overlap.trim_internal(in_data, axes=axes, boundary=boundary)
