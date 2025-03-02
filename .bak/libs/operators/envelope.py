import dask.array as da
from datasets.seismic import load_segy
from loggers import get_named_logger
from scipy import signal

__all__ = ["envelope_from_segy"]


def envelope_from_segy(segy_path, chunks="auto", logger=get_named_logger('envelope')):
    data = load_segy(segy_path)
    dask_array = da.from_array(data, chunks=chunks)

    logger.info(f"Calculating envelope for {segy_path}")
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Data chunks: {dask_array.chunks}")

    analytical_trace = da.map_blocks(signal.hilbert, dask_array, dtype=data.dtype)
    return da.absolute(analytical_trace)
