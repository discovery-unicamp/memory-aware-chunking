from typing import Any
from dasf_seismic.transforms.transforms import Transform
from traceq import get_logger
from .data.loaders import load_segy


def run_attribute(
    attribute: Transform,
    input: Any,
    n_workers: int = 1,
    single_threaded: bool = True,
):
    logger = get_logger()
    logger.debug(f"Running attribute {attribute.__class__.__name__}")
    input = load_segy(input)

    if single_threaded:
        return __run_single_threaded(attribute, input)
    else:
        return __run_in_local_cluster(attribute, input, n_workers)


def __run_single_threaded(attribute: Transform, input: Any):
    return attribute._transform_cpu(X=input)


def __run_in_local_cluster(attribute: Transform, input: Any, n_workers: int):
    pipeline = __build_pipeline(n_workers=n_workers)
    pipeline.add(attribute, X=lambda: input)
    pipeline.run()

    return pipeline.get_result_from(attribute).compute()


def __build_pipeline(n_workers: int = 1):
    from dasf.pipeline import Pipeline
    from dasf.pipeline.executors import DaskPipelineExecutor

    dask = DaskPipelineExecutor(local=True, cluster_kwargs={"n_workers": n_workers})
    pipeline = Pipeline("Pipeline", executor=dask)

    return pipeline
