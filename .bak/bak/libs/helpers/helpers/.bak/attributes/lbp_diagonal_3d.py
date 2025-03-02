def run(segy_filepath: str, n_workers: int = 1, single_threaded: bool = True):
    from dasf_seismic.attributes.texture import LocalBinaryPattern3D
    from seismic.cluster import run_attribute

    quality = LocalBinaryPattern3D()

    return run_attribute(quality, segy_filepath, n_workers, single_threaded)
