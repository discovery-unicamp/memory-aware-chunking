import os

import traceq
from common.loaders import load_segy
from common.operators.envelope import envelope_from_ndarray

traceq_backend = os.getenv("TRACEQ_BACKEND", "kernel")
segy_filepath = os.getenv("SEGY_FILEPATH", "data.sgy")
session_id = os.getenv("SESSION_ID", "memory_profile")
output_dir = os.getenv("OUTPUT_DIR", "./out")


def handler(filepath):
    data = load_segy(filepath)
    return envelope_from_ndarray(data)


def main():
    traceq.load_config(
        {
            "output_dir": output_dir,
            "profiler": {
                "session_id": session_id,
                "memory_usage": {
                    "enabled_backends": [traceq_backend],
                },
            },
        }
    )

    traceq.profile(handler, segy_filepath)


if __name__ == "__main__":
    main()
