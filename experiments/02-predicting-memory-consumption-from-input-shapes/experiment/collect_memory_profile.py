import os
import random

import traceq
from common.operators import envelope, gst3d, gaussian_filter

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./out/profiles")
SESSION_ID = os.getenv("SESSION_ID", str(random.randint))
PROFILER = os.getenv("PROFILER", "kernel")
INPUT_PATH = os.getenv("INPUT_PATH", "./inputs/input.segy")
ALGORITHM = os.getenv("ALGORITHM", "envelope")


def main():
    print("Collecting memory profile...")
    print("Using args:")
    print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"  SESSION_ID: {SESSION_ID}")
    print(f"  PROFILER: {PROFILER}")
    print(f"  INPUT_PATH: {INPUT_PATH}")
    print(f"  ALGORITHM: {ALGORITHM}")
    print()

    algorithms = {
        "envelope": envelope.envelope_from_segy,
        "gst3d": gst3d.gradient_structure_tensor_from_segy,
        "gaussian_filter": gaussian_filter.gaussian_filter_from_segy,
    }
    enabled_algorithm = algorithms[ALGORITHM]
    print(f"Enabled algorithm: {enabled_algorithm}")

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
        enabled_algorithm,
        *function_args,
    )


if __name__ == "__main__":
    main()
