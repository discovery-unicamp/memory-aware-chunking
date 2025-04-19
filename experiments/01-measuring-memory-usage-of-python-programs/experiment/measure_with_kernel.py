import os
import threading
import time

from common.loaders import load_segy
from common.operators.envelope import envelope_from_ndarray

segy_filepath = os.getenv("SEGY_FILEPATH", "data.sgy")
output_result_path = os.getenv("OUTPUT_RESULT_PATH", "proc_result.txt")
polling_interval = float(os.getenv("POLL_INTERVAL", "0.05"))

data = load_segy(segy_filepath)

memory_readings = []
stop_polling = False
page_size = os.sysconf("SC_PAGE_SIZE")


def poll_proc():
    while not stop_polling:
        with open("/proc/self/statm", "r") as f:
            fields = f.read().split()
            # fields[1] is the resident set size in pages
            if len(fields) >= 2:
                resident_pages = int(fields[1])
                rss_in_mb = (resident_pages * page_size) / (1024.0 * 1024.0)
                memory_readings.append(rss_in_mb)
        time.sleep(polling_interval)


t = threading.Thread(target=poll_proc)
t.start()

_ = envelope_from_ndarray(data)

stop_polling = True
t.join()

# Write time series to file
with open(output_result_path, "w") as f:
    for reading in memory_readings:
        f.write(f"{reading}\n")
