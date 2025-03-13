import os
import random

import traceq
from common.operators import envelope

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./out/profiles")
SESSION_ID = os.getenv("SESSION_ID", str(random.randint))
PROFILER = os.getenv("PROFILER", "kernel")
INPUT_PATH = os.getenv("INPUT_PATH", "./inputs/input.segy")


def main():
    traceq.load_config(
        {
            "output_dir": OUTPUT_DIR,
            "profiler": {
                "session_id": SESSION_ID,
                "memory_usage": {
                    "enabled_backends": [PROFILER],
                },
            },
        }
    )

    function_args = (INPUT_PATH,)

    traceq.profile(
        envelope.envelope_from_segy,
        *function_args,
    )


if __name__ == "__main__":
    main()
