"""
Collects memory usage results from multiple runs, focusing on:
 - average peak usage per tool (mean of each run's peak),
 - an "averaged timeline" of memory usage for each tool.

Output:
1) profiles_detail.csv -> columns: [tool, step, memory_mb]
   - memory_mb is the average across runs at that step index
2) profiles_summary.csv -> columns: [tool, peak_memory_avg]
   - peak_memory_avg is the mean of run-level peaks for that tool
"""

import csv
import glob
import os
import statistics

import traceq

# Conversion factors to MB
UNIT_FACTORS = {
    "b": 1 / (1024 * 1024),
    "kb": 1 / 1024,
    "mb": 1.0,
    "gb": 1024.0,
}


def to_mb(value, unit):
    return value * UNIT_FACTORS.get(unit.lower(), 1.0)


def parse_txt(filepath):
    """
    Reads a .txt file of floats (each line = memory usage in MB).
    Returns a list of float values (one per step).
    """
    readings = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                val = float(line)
                readings.append(val)
            except ValueError:
                pass
    return readings


def parse_prof(filepath):
    """
    Reads a .prof file using traceq.load_profile(...).
    Extracts the memory usage array from the relevant key,
    converting to MB if needed.
    """
    prof = traceq.load_profile(filepath)
    # E.g. 'traceq_psutil-12345.prof' => tool_full='traceq_psutil'
    fname = os.path.basename(filepath).split(".prof")[0]
    if "-" in fname:
        tool_full, _run_id = fname.split("-", 1)
    else:
        tool_full = fname

    # The memory usage key is something like "psutil_memory_usage"
    # if the tool is "traceq_psutil"
    if tool_full.startswith("traceq_"):
        suffix = tool_full.replace("traceq_", "")
    else:
        suffix = tool_full

    unit_key = f"{suffix}_memory_usage_unit"
    unit = prof["metadata"].get(unit_key, "mb")

    values = []
    for entry in prof["data"]:
        raw = entry.get(f"{suffix}_memory_usage")
        if raw is not None:
            values.append(to_mb(raw, unit))

    return values


def main():
    out_dir = os.getenv("OUTPUT_DIR", "./out")
    profiles_dir = os.path.join(out_dir, "profiles")
    results_dir = os.path.join(out_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Dictionary for storing all runs per tool: { tool_name: [ [vals_run1], [vals_run2], ... ] }
    tool_runs = {}

    # 1) Parse any .txt files
    for txt_file in glob.glob(os.path.join(profiles_dir, "*.txt")):
        fname = os.path.basename(txt_file)
        base, _ = os.path.splitext(fname)  # e.g. 'psutil-1'
        if "-" in base:
            tool, _run_id = base.rsplit("-", 1)
        else:
            tool = base
        # Read memory data:
        vals = parse_txt(txt_file)

        tool_runs.setdefault(tool, []).append(vals)

    # 2) Parse any .prof files
    for prof_file in glob.glob(os.path.join(profiles_dir, "*.prof")):
        fname = os.path.basename(prof_file)
        base, _ = os.path.splitext(fname)  # e.g. 'traceq_psutil-12345'
        if "-" in base:
            tool, _run_id = base.split("-", 1)
        else:
            tool = base
        # Example tool might be 'traceq_psutil'
        vals = []
        try:
            vals = parse_prof(prof_file)
        except Exception as e:
            print(f"Skipping {prof_file} due to parse error: {e}")
            continue

        tool_runs.setdefault(tool, []).append(vals)

    # Prepare data for final CSVs
    # We'll produce two tables:
    #  1) detail_rows = [tool, step, memory_mb] -> averaged timeline
    #  2) summary_rows = [tool, peak_memory_avg] -> mean of run-level peaks

    detail_rows = []
    summary_rows = []

    for tool, runs in tool_runs.items():
        if not runs:
            continue
        # Compute run-level peaks:
        run_peaks = []
        for run_vals in runs:
            if run_vals:
                run_peaks.append(max(run_vals))

        # Average peak usage for this tool:
        if run_peaks:
            avg_peak = statistics.mean(run_peaks)
        else:
            avg_peak = 0.0

        summary_rows.append({"tool": tool, "peak_memory_avg": avg_peak})

        # Build an "averaged timeline" across runs:
        #  - find the max length
        max_len = max(len(r) for r in runs if r)
        #  - for step i in [0..max_len-1], gather the i-th reading from each run that has it
        for i in range(max_len):
            vals_at_step = []
            for run_vals in runs:
                if i < len(run_vals):
                    vals_at_step.append(run_vals[i])
            if vals_at_step:
                avg_mem = statistics.mean(vals_at_step)
                detail_rows.append({"tool": tool, "step": i, "memory_mb": avg_mem})

    # Write detail CSV
    detail_csv_path = os.path.join(results_dir, "profiles_detail.csv")
    with open(detail_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tool", "step", "memory_mb"])
        writer.writeheader()
        writer.writerows(detail_rows)

    # Write summary CSV
    summary_csv_path = os.path.join(results_dir, "profiles_summary.csv")
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tool", "peak_memory_avg"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print("Collection complete.")
    print(f"- Wrote {detail_csv_path}")
    print(f"- Wrote {summary_csv_path}")


if __name__ == "__main__":
    main()
