from typing import Callable

import traceq
from loguru import logger

from interfaces import ArgsNamespace

__all__ = ["profile_memory_usage"]


def profile_memory_usage(
    function: Callable,
    args: ArgsNamespace,
    *function_args,
    **function_kwargs,
):
    logger.info("Profiling memory usage...")
    logger.debug(f"Arguments: {args}")

    match args.memory_profiler:
        case "psutil":
            __profile_with_psutil(
                function,
                args,
                *function_args,
                **function_kwargs,
            )
        case _:
            raise ValueError(f"Memory profiler {args.memory_profiler} not supported.")


def __profile_with_psutil(
    function: Callable,
    args: ArgsNamespace,
    *function_args,
    **function_kwargs,
):
    traceq.load_config(
        {
            "output_dir": args.memory_profile_output_dir,
            "profiler": {
                "session_id": args.memory_profile_session_id,
                "memory_usage": {
                    "enabled_backends": ["psutil"],
                },
            },
        }
    )

    traceq.profile(function, *function_args, **function_kwargs)
