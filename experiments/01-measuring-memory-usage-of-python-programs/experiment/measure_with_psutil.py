import os
import threading
import time

import psutil
from common.loaders import load_segy
from common.operators.envelope import envelope_from_ndarray

segy_filepath = os.getenv("SEGY_FILEPATH", "data.sgy")
polling_interval = float(os.getenv("POLL_INTERVAL", "0.05"))
timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
output_result_path = os.getenv("OUTPUT_RESULT_PATH", "psutil_result.txt")
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

memory_readings = []
stop_polling = False
process = psutil.Process(os.getpid())


def poll_memory():
    while not stop_polling:
        rss_in_bytes = process.memory_info().rss
        rss_in_mb = rss_in_bytes / (1024.0 * 1024.0)
        memory_readings.append(rss_in_mb)
        time.sleep(polling_interval)


t = threading.Thread(target=poll_memory)
t.start()

# Run the operator
_ = envelope_from_ndarray(data)

stop_polling = True
t.join()

# Write all memory readings to a text file, one per line
with open(output_result_path, "w") as f:
    for reading in memory_readings:
        f.write(f"{reading}\n")
