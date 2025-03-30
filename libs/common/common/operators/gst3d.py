import dask.array as da
import numpy as np
from common.loaders import load_segy
from common.transformers import transform_to_dask_array
from scipy import ndimage as ndi

__all__ = [
    "gradient_structure_tensor_from_ndarray",
    "gradient_structure_tensor_from_dask_array",
    "gradient_structure_tensor_from_segy",
]

# ------------------------------------------------------------------------------
# Constants & Helpers
# ------------------------------------------------------------------------------

_DERIV_WEIGHTS = np.array([-0.5, 0.0, 0.5])
_SMOOTH_WEIGHTS = np.array([0.178947, 0.642105, 0.178947])


def _correlate1d_local(chunk, weights, axis, mode="reflect"):
    return ndi.correlate1d(chunk, weights=weights, axis=axis, mode=mode)


def _uniform_filter_local(chunk, size, mode="reflect"):
    return ndi.uniform_filter(chunk, size=size, mode=mode)


def trim_numpy_array(arr, kernel_size):
    pad = tuple(k // 2 for k in kernel_size)
    slices = tuple(slice(p, -p if p > 0 else None) for p in pad)
    return arr[slices]


# ------------------------------------------------------------------------------
# NumPy Pipeline
# ------------------------------------------------------------------------------


def first_derivative_ndarray(X: np.ndarray, axis=-1, kernel=(3, 3, 3)):
    axes = [ax for ax in range(X.ndim) if ax != axis]

    result0 = ndi.correlate1d(X, weights=_DERIV_WEIGHTS, axis=axis, mode="reflect")

    result1 = ndi.correlate1d(
        result0, weights=_SMOOTH_WEIGHTS, axis=axes[0], mode="reflect"
    )
    result2 = ndi.correlate1d(
        result1, weights=_SMOOTH_WEIGHTS, axis=axes[1], mode="reflect"
    )

    # Trim the boundaries
    return trim_numpy_array(result2, kernel)


def compute_gradient_structure_tensor_ndarray(gi, gj, gk, kernel):
    gi2 = ndi.uniform_filter(gi * gi, size=kernel, mode="reflect")
    gj2 = ndi.uniform_filter(gj * gj, size=kernel, mode="reflect")
    gk2 = ndi.uniform_filter(gk * gk, size=kernel, mode="reflect")
    gigj = ndi.uniform_filter(gi * gj, size=kernel, mode="reflect")
    gigk = ndi.uniform_filter(gi * gk, size=kernel, mode="reflect")
    gjgk = ndi.uniform_filter(gj * gk, size=kernel, mode="reflect")

    return gi2, gj2, gk2, gigj, gigk, gjgk


# ------------------------------------------------------------------------------
# Dask Pipeline
# ------------------------------------------------------------------------------


def first_derivative_dask(X: da.Array, axis=-1, kernel=(3, 3, 3)):
    axes = [ax for ax in range(X.ndim) if ax != axis]

    depth = 1
    depth_dict = {d: depth for d in range(X.ndim)}

    result0 = X.map_overlap(
        _correlate1d_local,
        weights=_DERIV_WEIGHTS,
        axis=axis,
        mode="reflect",
        depth=depth_dict,
        boundary="reflect",
    )

    result1 = result0.map_overlap(
        _correlate1d_local,
        weights=_SMOOTH_WEIGHTS,
        axis=axes[0],
        mode="reflect",
        depth=depth_dict,
        boundary="reflect",
    )
    result2 = result1.map_overlap(
        _correlate1d_local,
        weights=_SMOOTH_WEIGHTS,
        axis=axes[1],
        mode="reflect",
        depth=depth_dict,
        boundary="reflect",
    )

    return trim_numpy_array(result2, kernel)


def compute_gradient_structure_tensor_dask(gi, gj, gk, kernel):
    depth_dict = {d: (k // 2) for d, k in enumerate(kernel)}

    def uf(arr):
        return arr.map_overlap(
            _uniform_filter_local,
            size=kernel,
            mode="reflect",
            depth=depth_dict,
            boundary="reflect",
        )

    gi2 = uf(gi * gi)
    gj2 = uf(gj * gj)
    gk2 = uf(gk * gk)
    gigj = uf(gi * gj)
    gigk = uf(gi * gk)
    gjgk = uf(gj * gk)

    return gi2, gj2, gk2, gigj, gigk, gjgk


# ------------------------------------------------------------------------------
# Dask-based 3D Dip Computation (Chunked Eigen-Decomposition)
# ------------------------------------------------------------------------------


def compute_3d_dip_dask(gi2, gj2, gk2, gigj, gigk, gjgk):
    stacked = da.stack([gi2, gigj, gigk, gigj, gj2, gjgk, gigk, gjgk, gk2], axis=-1)

    def _eig_per_voxel(block):
        shp = block.shape
        n_voxels = np.prod(shp[:-1])
        block_2d = block.reshape(n_voxels, 9)

        mat = np.zeros((n_voxels, 3, 3), dtype=block.dtype)
        mat[:, 0, 0] = block_2d[:, 0]
        mat[:, 0, 1] = block_2d[:, 1]
        mat[:, 0, 2] = block_2d[:, 2]

        mat[:, 1, 0] = block_2d[:, 3]
        mat[:, 1, 1] = block_2d[:, 4]
        mat[:, 1, 2] = block_2d[:, 5]

        mat[:, 2, 0] = block_2d[:, 6]
        mat[:, 2, 1] = block_2d[:, 7]
        mat[:, 2, 2] = block_2d[:, 8]

        evals, evecs = np.linalg.eigh(mat)
        ndx = evals.argsort(axis=-1)
        largest_evec = evecs[np.arange(n_voxels), :, ndx[:, -1]]

        norms = np.linalg.norm(largest_evec, axis=-1)
        largest_evec = largest_evec / norms[:, None]

        dip = np.degrees(np.arccos(largest_evec[:, 2]))

        dip_3d = dip.reshape(shp[:-1])
        return dip_3d

    dip_map = da.map_blocks(
        _eig_per_voxel,
        stacked,
        dtype=stacked.dtype,
        drop_axis=[-1],
    )

    return dip_map


# ------------------------------------------------------------------------------
# Shared Dip Computation (NumPy version)
# ------------------------------------------------------------------------------


def compute_3d_dip(gi2, gj2, gk2, gigj, gigk, gjgk):
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


# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------


def gradient_structure_tensor_from_ndarray(data: np.ndarray, kernel=(3, 3, 3)):
    gi = first_derivative_ndarray(data, axis=0, kernel=kernel)
    gj = first_derivative_ndarray(data, axis=1, kernel=kernel)
    gk = first_derivative_ndarray(data, axis=2, kernel=kernel)

    gi2, gj2, gk2, gigj, gigk, gjgk = compute_gradient_structure_tensor_ndarray(
        gi, gj, gk, kernel
    )

    return compute_3d_dip(gi2, gj2, gk2, gigj, gigk, gjgk)


def gradient_structure_tensor_from_dask_array(data: da.Array, kernel=(3, 3, 3)):
    gi = first_derivative_dask(data, axis=0, kernel=kernel)
    gj = first_derivative_dask(data, axis=1, kernel=kernel)
    gk = first_derivative_dask(data, axis=2, kernel=kernel)

    gi2, gj2, gk2, gigj, gigk, gjgk = compute_gradient_structure_tensor_dask(
        gi, gj, gk, kernel
    )

    return compute_3d_dip_dask(gi2, gj2, gk2, gigj, gigk, gjgk)


def gradient_structure_tensor_from_segy(
    segy_path: str,
    kernel=(3, 3, 3),
    use_dask=False,
    dask_chunks="auto",
):
    if use_dask:
        data = load_segy(segy_path)
        da_data = transform_to_dask_array(data, chunks=dask_chunks)
        print("Loaded data shape:", da_data.shape)
        print("Loaded data chunk sizes:", da_data.chunks)
        return gradient_structure_tensor_from_dask_array(da_data, kernel)
    else:
        data = load_segy(segy_path)
        return gradient_structure_tensor_from_ndarray(data, kernel)
