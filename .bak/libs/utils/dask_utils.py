import resource
import sys
import time

from distributed.diagnostics.plugin import SchedulerPlugin

__all__ = ["get_worker_peak_memory", "OverheadPlugin"]


def get_worker_peak_memory():
    ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform.startswith('linux'):
        # On Linux, ru_maxrss is in kilobytes
        memory_in_bytes = ru_maxrss * 1024
    else:
        # On macOS and other systems, ru_maxrss is in bytes
        memory_in_bytes = ru_maxrss

    return memory_in_bytes


class OverheadPlugin(SchedulerPlugin):
    def __init__(self):
        self.task_times = {}
        self.overheads = []

    def update_graph(self, scheduler, dsk=None, keys=None, restrictions=None, **kwargs):
        submission_time = time.perf_counter()
        if keys:
            for key in keys:
                self.task_times[key] = {'submission_time': submission_time}

    def transition(self, key, start, finish, *args, **kwargs):
        current_time = time.perf_counter()
        if key not in self.task_times:
            self.task_times[key] = {}

        if finish == 'waiting':
            self.task_times[key]['waiting_time'] = current_time

        elif finish == 'processing':
            self.task_times[key]['processing_time'] = current_time

            submission_time = self.task_times[key].get('submission_time', current_time)
            waiting_time = self.task_times[key].get('waiting_time', submission_time)
            overhead = self.task_times[key]['processing_time'] - waiting_time

            self.overheads.append(overhead)

            # Clean up to save memory
            del self.task_times[key]

    def get_overhead_stats(self):
        if self.overheads:
            total_overhead = sum(self.overheads)
            max_overhead = max(self.overheads)
            avg_overhead = total_overhead / len(self.overheads)
            return total_overhead, max_overhead, avg_overhead
        else:
            return 0.0, 0.0, 0.0
