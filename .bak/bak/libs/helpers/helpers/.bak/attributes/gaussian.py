def run(segy_filepath: str, n_workers: int = 1, single_threaded: bool = True):
    from dasf_seismic.attributes.noise_reduction import Gaussian
    from seismic.cluster import run_attribute

    quality = Gaussian()

    return run_attribute(quality, segy_filepath, n_workers, single_threaded)
