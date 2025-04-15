"""
Collects a memory profile while processing a .segy file using Dask and
a memory-aware (or alternative) chunking strategy.

Environment Variables (with defaults in parentheses):
  SESSION_ID (random int)
  OUTPUT_DIR (./out/profiles)
  INPUT_PATH (./out/inputs/input.segy)
  WORKER_COUNT (1)
  CHUNKING_MODE (auto) -> [auto, evenly_split, memaware]
  GST3D_MODEL_FILE (./out/models/gst3d.pkl)
  MEMORY_LIMIT_GB (32)
  SAFETY_FACTOR (0.8)
  MONITORING_INTERVAL (0.2)
"""

import json
import os
import pickle
import random
import sys
import threading
import traceback
from itertools import permutations
from time import time, sleep
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from common.operators.gst3d import gradient_structure_tensor_from_segy
from dask.distributed import LocalCluster, Client

# Global variables for controlling memory monitoring thread
MONITORING = False


def main():
    # Parse environment variables
    session_id = os.getenv("SESSION_ID", str(random.randint(0, 999999)))
    output_dir = os.getenv("OUTPUT_DIR", "./out/profiles")
    input_path = os.getenv("INPUT_PATH", "./out/inputs/input.segy")
    worker_count = int(os.getenv("WORKER_COUNT", "1"))
    chunking_mode = os.getenv("CHUNKING_MODE", "auto")
    gst3d_model_file = os.getenv("GST3D_MODEL_FILE", "./out/models/gst3d.pkl")
    memory_limit_gb = int(os.getenv("MEMORY_LIMIT_GB", "32"))
    safety_factor = float(os.getenv("SAFETY_FACTOR", "0.8"))
    monitoring_interval = float(os.getenv("MONITORING_INTERVAL", "0.2"))

    # Infer the shape from the input file name (e.g. "100-200-300.segy")
    shape = _parse_shape_from_filename(input_path)

    # Memory limit per worker (GB)
    memory_limit_per_worker_gb = memory_limit_gb / worker_count
    unix_timestamp = int(time())

    print("Collecting Memory Profile for Memory-Aware Chunking...")
    print(f"  OUTPUT_DIR={output_dir}")
    print(f"  SESSION_ID={session_id}")
    print(f"  INPUT_PATH={input_path}")
    print(f"  WORKER_COUNT={worker_count}")
    print(f"  CHUNKING_MODE={chunking_mode}")
    print(f"  GST3D_MODEL_FILE={gst3d_model_file}")
    print(f"  MEMORY_LIMIT_GB={memory_limit_gb}")
    print(f"  SAFETY_FACTOR={safety_factor}")
    print(f"  SHAPE={shape}")
    print(f"  UNIX_TIMESTAMP={unix_timestamp}")

    os.makedirs(output_dir, exist_ok=True)

    # Set up a Dask local cluster
    cluster = LocalCluster(
        n_workers=worker_count,
        threads_per_worker=1,
        memory_limit=f"{memory_limit_per_worker_gb}GB",
    )
    client = Client(cluster)
    memory_usage_history = {addr: [] for addr in client.scheduler_info()["workers"]}

    # Pick chunk size based on chunking_mode
    if chunking_mode == "auto":
        chunk_size = "auto"
    elif chunking_mode == "evenly_split":
        chunk_size = _find_evenly_split_chunk_size(shape, worker_count)
    elif chunking_mode == "memaware":
        chunk_size = _find_memaware_chunk_size(
            shape,
            gst3d_model_file,
            memory_limit_per_worker_gb,
            safety_factor,
        )
    else:
        chunk_size = "auto"

    print(f"  Resolved chunk_size = {chunk_size}")

    # Main try/finally block to ensure cluster is closed even on error
    try:
        monitoring_thread = _start_monitoring(
            client, memory_usage_history, monitoring_interval
        )
        start_time = time()

        # Run the actual data processing (Gradient Structure Tensor)
        dip_map = gradient_structure_tensor_from_segy(
            input_path,
            use_dask=True,
            dask_chunks=chunk_size,
        )
        dip_result = dip_map.compute()

        elapsed_time = time() - start_time
        _stop_monitoring(monitoring_thread)

        print(f"Dip result shape: {dip_result.shape}")
        print(f"Dip min: {dip_result.min()}, Dip max: {dip_result.max()}")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

        # Build and write JSON profile
        prof_filename = f"{_shape_to_str(shape)}-{chunking_mode}-{worker_count}-{unix_timestamp}-{session_id}.json"
        prof_path = os.path.join(output_dir, prof_filename)
        _write_profile_json(
            prof_path,
            memory_usage_history,
            session_id,
            worker_count,
            chunking_mode,
            unix_timestamp,
            shape,
            chunk_size,
            elapsed_time,
        )

        print(f"Profile saved to {prof_path}")

    except Exception as ex:
        print("Error occurred during memory profiling:")
        traceback.print_exc()
    finally:
        client.close()
        cluster.close()

    print("Memory profiling step complete.")
    return 0


def _parse_shape_from_filename(filepath: str) -> Tuple[int, int, int]:
    """
    Extracts a 3D shape from a filename if it contains digits separated by '-'
    before the .segy extension. E.g. '100-200-300.segy' -> (100, 200, 300).
    """
    basename = os.path.splitext(os.path.basename(filepath))[0]
    parts = [int(x) for x in basename.split("-") if x.isdigit()]
    return tuple(parts) if len(parts) == 3 else (0, 0, 0)


def _find_evenly_split_chunk_size(
    shape: Tuple[int, int, int], n_workers: int
) -> Optional[Tuple[int, int, int]]:
    """
    Attempts to factor the shape into n_workers sub-volumes that are as
    balanced as possible, returning a chunk shape (cx, cy, cz).
    """

    def factor_triplets(n: int):
        """Generate triplets (a,b,c) where a*b*c = n."""
        triplets = []
        for a in range(1, n + 1):
            if n % a != 0:
                continue
            for b in range(1, (n // a) + 1):
                if (n // a) % b != 0:
                    continue
                c = (n // a) // b
                triplets.append((a, b, c))
        return triplets

    best_chunk = None
    best_balance = float("inf")

    for fx, fy, fz in factor_triplets(n_workers):
        for perm_fx, perm_fy, perm_fz in permutations((fx, fy, fz)):
            if all(
                shape[i] % f == 0 for i, f in enumerate((perm_fx, perm_fy, perm_fz))
            ):
                candidate = tuple(
                    shape[i] // f for i, f in enumerate((perm_fx, perm_fy, perm_fz))
                )
                mean_dim = sum(candidate) / 3.0
                balance = sum((dim - mean_dim) ** 2 for dim in candidate)
                if balance < best_balance:
                    best_chunk = candidate
                    best_balance = balance

    return best_chunk


def _find_memaware_chunk_size(
    data_shape: Tuple[int, int, int],
    model_path: str,
    mem_limit_gb: float,
    safety_factor: float,
) -> Optional[Tuple[int, int, int]]:
    """
    Uses a pretrained model (gst3d.pkl) to estimate memory usage per voxel,
    then chooses a cubic chunk size that fits into mem_limit_gb * safety_factor.
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    total_voxels = data_shape[0] * data_shape[1] * data_shape[2]
    df = pd.DataFrame({"volume": [total_voxels]})
    predicted_mem_gb = model.predict(df[["volume"]])[0]

    print(f"Estimated memory usage for entire volume: {predicted_mem_gb:.2f} GB")
    voxel_cost_gb = predicted_mem_gb / total_voxels if total_voxels > 0 else 0.0

    max_memory = mem_limit_gb * safety_factor
    max_voxels = max_memory / voxel_cost_gb if voxel_cost_gb > 0 else 1
    side = int(np.cbrt(max_voxels))

    # Try to find a chunk (s, s, s) that evenly divides each dimension
    for s in range(side, 0, -1):
        if all(d % s == 0 for d in data_shape):
            if (s**3) * voxel_cost_gb <= max_memory:
                return (s, s, s)
    return None


def _start_monitoring(
    client: Client, memory_usage_history: dict, interval: float
) -> threading.Thread:
    """
    Starts a background thread that periodically queries the scheduler for
    each worker's memory usage, storing the values in memory_usage_history.
    """
    global MONITORING
    MONITORING = True
    t = threading.Thread(
        target=_monitor_memory,
        args=(client, memory_usage_history, interval),
        daemon=True,
    )
    t.start()
    return t


def _stop_monitoring(t: threading.Thread):
    """Stops the monitoring thread."""
    global MONITORING
    MONITORING = False
    t.join()


def _monitor_memory(client: Client, memory_usage_history: dict, interval: float):
    """Continuously polls worker memory usage until MONITORING is set to False."""
    global MONITORING
    while MONITORING:
        info = client.scheduler_info()
        for addr, worker_info in info["workers"].items():
            memory_usage_history[addr].append(
                worker_info.get("metrics", {}).get("memory", 0)
            )
        sleep(interval)


def _write_profile_json(
    path: str,
    memory_usage_history: dict,
    session_id: str,
    worker_count: int,
    chunking_mode: str,
    unix_timestamp: int,
    shape: Tuple[int, int, int],
    chunk_size,
    exec_time: float,
):
    """
    Writes a JSON file with metadata (execution time, chunking info) and
    memory usage details (peak, avg, history) for each worker.
    """
    # Prepare memory usage data
    usage_dict = {}
    for addr, usage_list in memory_usage_history.items():
        if usage_list:
            peak = max(usage_list)
            avg = float(sum(usage_list)) / len(usage_list)
        else:
            peak, avg = None, None
        usage_dict[addr] = {
            "peak_memory_usage": peak,
            "avg_memory_usage": avg,
            "memory_usage_history": usage_list,
        }

    metadata = {
        "session_id": session_id,
        "worker_count": worker_count,
        "chunking_mode": chunking_mode,
        "unix_timestamp": unix_timestamp,
        "shape": _shape_to_str(shape),
        "chunk_size": _shape_to_str(chunk_size),
        "execution_time_unit": "seconds",
        "memory_usage_unit": "bytes",
    }
    data = {
        "execution_time": exec_time,
        "memory_usage": usage_dict,
    }

    with open(path, "w") as f:
        json.dump({"metadata": metadata, "data": data}, f, indent=2)


def _shape_to_str(shape):
    """Helper for turning tuples into e.g. '100-200-300'."""
    if isinstance(shape, (list, tuple)):
        return "-".join(map(str, shape))
    return str(shape)


if __name__ == "__main__":
    sys.exit(main())
