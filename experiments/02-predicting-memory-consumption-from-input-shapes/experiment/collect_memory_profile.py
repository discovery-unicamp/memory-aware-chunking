"""
Script for collecting a memory profile of a seismic operator (e.g., Envelope, GST3D,
Gaussian Filter) using TraceQ. The selected algorithm and other configs are controlled
by environment variables.

Environment Variables:
- OUTPUT_DIR (default: './out/profiles')
- SESSION_ID (default: random int if not provided)
- PROFILER (default: 'kernel')
- INPUT_PATH (default: './inputs/input.segy')
- ALGORITHM (default: 'envelope')

Usage Example:
    OUTPUT_DIR='./out/profiles' ALGORITHM='gst3d' python collect_memory_profile.py
"""

import os
import random

import traceq
from common.operators import envelope, gst3d, gaussian_filter

# ------------------------------------------------------------------------------
# Global Defaults / Constants
# ------------------------------------------------------------------------------
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./out/profiles")
SESSION_ID = os.getenv("SESSION_ID", str(random.randint(0, 999999)))
PROFILER = os.getenv("PROFILER", "kernel")
INPUT_PATH = os.getenv("INPUT_PATH", "./inputs/input.segy")
ALGORITHM = os.getenv("ALGORITHM", "envelope")


def main():
    """
    Main entry point that orchestrates the memory profiling.

    1. Reads environment variables for output/config.
    2. Chooses the correct seismic algorithm to run.
    3. Configures traceq with the selected profiler and session ID.
    4. Executes the chosen function with memory usage profiling.
    """
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

    # Ensure the chosen ALGORITHM is valid; default to 'envelope' otherwise
    if ALGORITHM not in algorithms:
        print(f"Warning: Unknown ALGORITHM '{ALGORITHM}'. Falling back to 'envelope'.")
        enabled_algorithm = envelope.envelope_from_segy
    else:
        enabled_algorithm = algorithms[ALGORITHM]

    print(f"Selected algorithm: {enabled_algorithm.__name__}")

    # Configure traceq for memory usage
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

    # Execute the chosen function under traceq.profile
    traceq.profile(enabled_algorithm, INPUT_PATH)


if __name__ == "__main__":
    main()
