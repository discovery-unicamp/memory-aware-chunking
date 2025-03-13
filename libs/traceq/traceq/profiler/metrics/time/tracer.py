import time

from traceq.common.transformers import convert_to_unit
from traceq.profiler.types import CapturedTrace

from .types import TimeUnit

__all__ = ["on_sample"]


def get_unix_timestamp(unit: TimeUnit = "ns", offset: float = time.time_ns() - time.perf_counter_ns()) -> float:
    unix_timestamp = time.perf_counter_ns() + offset
    return convert_to_unit(unit, "ns", unix_timestamp) if unit != "ns" else unix_timestamp


def capture_trace() -> CapturedTrace:
    return "unix_timestamp", get_unix_timestamp()


on_sample = capture_trace
