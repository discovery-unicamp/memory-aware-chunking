import dask.array as da
from datasets import load_segy
from loggers import get_named_logger
from scipy import ndimage as ndi

__all__ = [
    "gaussian_filter_from_segy",
]


def gaussian_filter_from_segy(segy_path, chunks="auto", sigma=(3, 3, 3), logger=get_named_logger('gaussian_filter')):
    data = load_segy(segy_path)
    dask_array = da.from_array(data, chunks=chunks)

    logger.info(f"Calculating Gaussian Filter for {segy_path}")
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Data chunks: {dask_array.chunks}")

    return dask_array.map_blocks(
        ndi.gaussian_filter, sigma=sigma, dtype=dask_array.dtype
    )