import os
import threading
import time
import tracemalloc

from common.loaders import load_segy
from common.operators.envelope import envelope_from_ndarray

segy_filepath = os.getenv("SEGY_FILEPATH", "data.sgy")
polling_interval = float(os.getenv("POLL_INTERVAL", "0.05"))
timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
output_result_path = os.getenv("OUTPUT_RESULT_PATH", "resource_result.txt")
append_timestamp = bool(os.getenv("APPEND_TIMESTAMP", "False"))

output_filename, output_ext = os.path.splitext(output_result_path)
output_result_path = (
    f"{output_filename}-{timestamp}{output_ext}"
    if append_timestamp
    else output_result_path
)
output_dir = os.path.dirname(output_result_path)
os.makedirs(output_dir, exist_ok=True)

data = load_segy(segy_filepath)

tracemalloc.start()

memory_snapshots = []
stop_polling = False


def poll_tracemalloc():
    while not stop_polling:
        current, peak = tracemalloc.get_traced_memory()
        current_mb = current / (1024.0 * 1024.0)
        peak_mb = peak / (1024.0 * 1024.0)
        memory_snapshots.append((current_mb, peak_mb))
        time.sleep(polling_interval)


t = threading.Thread(target=poll_tracemalloc)
t.start()

_ = envelope_from_ndarray(data)

stop_polling = True
t.join()
tracemalloc.stop()

with open(output_result_path, "w") as f:
    for curr, _ in memory_snapshots:
        f.write(f"{curr}\n")
