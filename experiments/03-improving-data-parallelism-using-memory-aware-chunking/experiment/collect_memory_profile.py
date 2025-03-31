"""
Script for collecting a memory profile of a seismic operator (e.g., Envelope, GST3D,
Gaussian Filter) using Dask. The selected algorithm and other configs are controlled
by environment variables.

Environment Variables:
- OUTPUT_DIR (default: './out/profiles')
- SESSION_ID (default: random int if not provided)
- INPUT_PATH (default: './inputs/input.segy')
- ALGORITHM (default: 'gst3d')
- MONITORING_INTERVAL (default: 0.2 seconds)
- DASK_CHUNKS_INLINES (default: 0)
- DASK_CHUNKS_XLINES (default: 0)
- DASK_CHUNKS_SAMPLES (default: 0)

Usage Example:
    OUTPUT_DIR='./out/profiles' ALGORITHM='gst3d' python collect_memory_profile.py
"""

import json
import os
import random
import threading
import time

from common.operators import gst3d
from dask.distributed import LocalCluster, Client

# ------------------------------------------------------------------------------
# Global Defaults / Constants
# ------------------------------------------------------------------------------
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./out/profiles")
SESSION_ID = os.getenv("SESSION_ID", str(random.randint(0, 999999)))
INPUT_PATH = os.getenv("INPUT_PATH", "./inputs/input.segy")
ALGORITHM = os.getenv("ALGORITHM", "gst3d")
MONITORING_INTERVAL = float(os.getenv("MONITORING_INTERVAL", 0.2))
DASK_CHUNKS_INLINES = int(os.getenv("DASK_CHUNKS_INLINES", "0"))
DASK_CHUNKS_XLINES = int(os.getenv("DASK_CHUNKS_XLINES", "0"))
DASK_CHUNKS_SAMPLES = int(os.getenv("DASK_CHUNKS_SAMPLES", "0"))

TIMESTAMP = time.time()
MONITORING = False


def main():
    """
    Main entry point that orchestrates the memory profiling.

    1. Reads environment variables for output/config.
    2. Chooses the correct seismic algorithm to run.
    3. Configures the Dask local cluster.
    4. Executes the chosen function with memory usage profiling.
    """
    print("Collecting memory profile...")
    print("Using args:")
    print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"  SESSION_ID: {SESSION_ID}")
    print(f"  INPUT_PATH: {INPUT_PATH}")
    print(f"  ALGORITHM: {ALGORITHM}")
    print(f"  MONITORING_INTERVAL: {MONITORING_INTERVAL}")
    print(f"  DASK_CHUNKS_INLINES: {DASK_CHUNKS_INLINES}")
    print(f"  DASK_CHUNKS_XLINES: {DASK_CHUNKS_XLINES}")
    print(f"  DASK_CHUNKS_SAMPLES: {DASK_CHUNKS_SAMPLES}")
    print()

    algorithms = {
        "gst3d": gst3d.gradient_structure_tensor_from_segy,
    }

    if ALGORITHM not in algorithms:
        print(f"Warning: Unknown ALGORITHM '{ALGORITHM}'. Falling back to 'gst3d'.")
        enabled_algorithm = gst3d.gradient_structure_tensor_from_segy
    else:
        enabled_algorithm = algorithms[ALGORITHM]

    print(f"Selected algorithm: {enabled_algorithm.__name__}")

    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=1,
    )
    print("Created local Cluster")
    client = Client(cluster)
    print(
        f"Launched Dask client with {len(client.scheduler_info()['workers'])} workers."
    )

    profile = {
        "metadata": {
            "session_id": SESSION_ID,
            "timestamp": TIMESTAMP,
            "algorithm": ALGORITHM,
            "input_path": INPUT_PATH,
            "dask_memory_usage_unit": "bytes",
            "unix_timestamp_unit": "seconds",
        },
        "data": [],
    }
    monitoring_thread = __start_monitoring(client, profile)
    print(f"Profiler started, monitoring every {MONITORING_INTERVAL} seconds.")

    print("Starting to execute the algorithm...")
    dask_chunks = (
        (DASK_CHUNKS_INLINES, DASK_CHUNKS_XLINES, DASK_CHUNKS_SAMPLES)
        if (
            DASK_CHUNKS_INLINES > 0
            and DASK_CHUNKS_XLINES > 0
            and DASK_CHUNKS_SAMPLES > 0
        )
        else "auto"
    )
    graph = enabled_algorithm(INPUT_PATH, use_dask=True, dask_chunks=dask_chunks)
    graph.compute()
    print("Algorithm execution completed.")

    __stop_monitoring(monitoring_thread)
    print(f"Profiler stopped. Collected {len(profile['data'])} data points.")

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{ALGORITHM}-{SESSION_ID}-{TIMESTAMP}.json",
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=4)

    print(f"Memory profile saved to {output_file}.")


def __monitor_memory(client, profile, interval=MONITORING_INTERVAL):
    while MONITORING:
        info = client.scheduler_info()
        timestamp = time.time()

        for addr, worker_info in info["workers"].items():
            profile["data"].append(
                {
                    "unix_timestamp": timestamp,
                    "worker_addr": addr,
                    "dask_memory_usage": worker_info.get("metrics", {}).get(
                        "memory", 0
                    ),
                }
            )
        time.sleep(interval)


def __start_monitoring(client, memory_usage_history):
    global MONITORING
    MONITORING = True
    thread = threading.Thread(
        target=__monitor_memory,
        daemon=True,
        args=(
            client,
            memory_usage_history,
        ),
    )
    thread.start()

    return thread


def __stop_monitoring(thread):
    global MONITORING
    MONITORING = False
    thread.join()


if __name__ == "__main__":
    main()
