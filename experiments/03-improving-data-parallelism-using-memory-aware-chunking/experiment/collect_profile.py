"""
collect_memory_profile_memory_aware.py

Runs a memory-aware chunking operation on a given .segy file under different
worker counts and chunking modes, then records memory usage.

Environment Variables:
  - SESSION_ID (default: random int if not provided)
  - OUTPUT_DIR (default: './out/profiles')
  - INPUT_PATH (default: './out/inputs/input.segy')
  - WORKER_COUNT (default: '1')
  - CHUNKING_MODE (default: 'auto') -> can be 'auto', 'evenly_split', 'memaware'
  - GST3D_MODEL_FILE (default: './out/models/gst3d.pkl')
  - SAFETY_FACTOR (default: '0.8') -> for memory limit calculations
  - MEMORY_LIMIT_GB (default: '32') -> max memory per worker in GB
  - MONITORING_INTERVAL (default: '0.2') -> interval for memory monitoring in seconds
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

SESSION_ID = os.getenv("SESSION_ID", str(random.randint(0, 999999)))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./out/profiles")
INPUT_PATH = os.getenv("INPUT_PATH", "./out/inputs/input.segy")
WORKER_COUNT = int(os.getenv("WORKER_COUNT", "1"))
CHUNKING_MODE = os.getenv("CHUNKING_MODE", "auto")
GST3D_MODEL_FILE = os.getenv("GST3D_MODEL_FILE", "./out/models/gst3d.pkl")
MEMORY_LIMIT_GB = int(os.getenv("MEMORY_LIMIT_GB", "32"))
SAFETY_FACTOR = float(os.getenv("SAFETY_FACTOR", "0.8"))
MONITORING_INTERVAL = float(os.getenv("MONITORING_INTERVAL", "0.2"))

SHAPE = tuple(
    int(s) for s in INPUT_PATH.split(".")[0].split("/")[-1].split("-") if s.isdigit()
)

UNIX_TIMESTAMP = int(time())
MONITORING = False
MEMORY_LIMIT_PER_WORKER_GB = MEMORY_LIMIT_GB / WORKER_COUNT


def main():
    print("Collecting Memory Profile for Memory-Aware Chunking...")
    print(f"  OUTPUT_DIR={OUTPUT_DIR}")
    print(f"  SESSION_ID={SESSION_ID}")
    print(f"  INPUT_PATH={INPUT_PATH}")
    print(f"  WORKER_COUNT={WORKER_COUNT}")
    print(f"  CHUNKING_MODE={CHUNKING_MODE}")
    print(f"  GST3D_MODEL_FILE={GST3D_MODEL_FILE}")
    print(f"  MEMORY_LIMIT_GB={MEMORY_LIMIT_GB}")
    print(f"  SAFETY_FACTOR={SAFETY_FACTOR}")
    print(f"  SHAPE={SHAPE}")
    print(f"  UNIX_TIMESTAMP={UNIX_TIMESTAMP}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cluster = LocalCluster(
        n_workers=WORKER_COUNT,
        threads_per_worker=1,
        memory_limit=f"{MEMORY_LIMIT_PER_WORKER_GB}GB",
    )
    client = Client(cluster)
    memory_usage_history = {addr: [] for addr in client.scheduler_info()["workers"]}

    if CHUNKING_MODE == "auto":
        chunk_size = "auto"
    elif CHUNKING_MODE == "evenly_split":
        chunk_size = _find_evenly_split_chunk_size()
    elif CHUNKING_MODE == "memaware":
        chunk_size = _find_memaware_chunk_size()
    else:
        chunk_size = "auto"

    print(f"  Resolved chunk_size = {chunk_size}")

    try:
        monitoring_thread = _start_monitoring(client, memory_usage_history)
        start_time = time()
        dip_map = gradient_structure_tensor_from_segy(
            INPUT_PATH,
            use_dask=True,
            dask_chunks=chunk_size,
        )
        dip_result = dip_map.compute()
        end_time = time()
        _stop_monitoring(monitoring_thread)
        elapsed_time = end_time - start_time

        print(f"Dip result shape: {dip_result.shape}")
        print(f"Dip min: {dip_result.min()}, Dip max: {dip_result.max()}")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

        shape_name = _shape_to_name(SHAPE)
        prof_filename = f"{shape_name}-{CHUNKING_MODE}-{WORKER_COUNT}-{UNIX_TIMESTAMP}-{SESSION_ID}.json"
        prof_path = os.path.join(OUTPUT_DIR, prof_filename)

        peak_memory_usages = {k: max(v) for k, v in memory_usage_history.items()}
        for addr, mem_bytes in peak_memory_usages.items():
            print(f"Worker {addr} peak memory: {mem_bytes / (1024 ** 3):.2f} GB")

        metadata = {
            "session_id": SESSION_ID,
            "worker_count": WORKER_COUNT,
            "chunking_mode": CHUNKING_MODE,
            "unix_timestamp": UNIX_TIMESTAMP,
            "shape": shape_name,
            "chunk_size": _shape_to_name(chunk_size),
            "execution_time_unit": "seconds",
            "memory_usage_unit": "bytes",
        }
        data = {
            "execution_time": elapsed_time,
            "memory_usage": {
                addr: {
                    "peak_memory_usage": max(mem_bytes),
                    "avg_memory_usage": sum(mem_bytes) / len(mem_bytes),
                    "memory_usage_history": mem_bytes,
                }
                for addr, mem_bytes in memory_usage_history.items()
            },
        }

        with open(prof_path, "w") as f:
            json.dump({"metadata": metadata, "data": data}, f, indent=2)

        print(f"Profile saved to {prof_path}")

    except Exception as e:
        print("Error occurred during memory profiling:")
        traceback.print_exc()

    finally:
        client.close()
        cluster.close()

    print("Memory profiling step complete.")


def _find_evenly_split_chunk_size(
    shape: Tuple[int, int, int] = SHAPE,
    n_workers: int = WORKER_COUNT,
) -> Optional[Tuple[int, int, int]]:
    def factor_triplets(n: int):
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
                chunk = tuple(
                    shape[i] // f for i, f in enumerate((perm_fx, perm_fy, perm_fz))
                )
                mean = sum(chunk) / 3
                balance = sum((c - mean) ** 2 for c in chunk)
                if balance < best_balance:
                    best_chunk = chunk
                    best_balance = balance

    return best_chunk


def _find_memaware_chunk_size(
    data_shape=SHAPE,
    model_path=GST3D_MODEL_FILE,
    mem_limit=MEMORY_LIMIT_PER_WORKER_GB,
    safety_factor=SAFETY_FACTOR,
):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    df = pd.DataFrame({"volume": [data_shape[0] * data_shape[1] * data_shape[2]]})
    predicted_mem_gb = model.predict(df[["volume"]])[0]
    print(f"Estimated memory usage: {predicted_mem_gb} GB")

    voxel_cost = predicted_mem_gb / np.prod(data_shape)
    max_mem = mem_limit * safety_factor
    max_voxels = max_mem / voxel_cost
    side = int(np.cbrt(max_voxels))

    for s in range(side, 0, -1):
        if all(d % s == 0 for d in data_shape):
            if (s**3) * voxel_cost <= max_mem:
                return (s, s, s)


def _start_monitoring(client, memory_usage_history):
    global MONITORING
    MONITORING = True
    thread = threading.Thread(
        target=_monitor_memory,
        daemon=True,
        args=(client, memory_usage_history),
    )
    thread.start()
    return thread


def _stop_monitoring(thread):
    global MONITORING
    MONITORING = False
    thread.join()


def _monitor_memory(client, memory_usage_history):
    while MONITORING:
        info = client.scheduler_info()
        for addr, worker_info in info["workers"].items():
            memory_usage_history[addr].append(
                worker_info.get("metrics", {}).get("memory", 0)
            )
        sleep(MONITORING_INTERVAL)


def _shape_to_name(shape):
    if isinstance(shape, (list, tuple)):
        return "-".join(map(str, shape))
    else:
        return str(shape)


if __name__ == "__main__":
    sys.exit(main())
