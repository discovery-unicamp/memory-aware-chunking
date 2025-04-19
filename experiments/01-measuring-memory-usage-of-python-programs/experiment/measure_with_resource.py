import os
import resource
import threading
import time

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

stop_polling = False
memory_readings = []


def poll_resource_peak():
    while not stop_polling:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        peak_so_far_mb = usage.ru_maxrss / 1024.0
        memory_readings.append(peak_so_far_mb)
        time.sleep(polling_interval)


# Start polling in a background thread
t = threading.Thread(target=poll_resource_peak)
t.start()

# Run the seismic operator
_ = envelope_from_ndarray(data)

stop_polling = True
t.join()

# Write the entire time series (one measurement per line)
with open(output_result_path, "w") as f:
    for val in memory_readings:
        f.write(f"{val}\n")
