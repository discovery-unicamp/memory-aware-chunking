#!/usr/bin/env python3
import json
import os
import re
import sys
from collections import defaultdict

import pandas as pd


def parse_filename(filename: str):
    """
    Parse filenames of the form:
      <inlines>-<xlines>-<samples>-<chunking_mode>-<n_workers>-<timestamp>-<random_id>.json

    Returns a tuple:
      (inlines, xlines, samples, chunk_mode, worker_count, timestamp, random_id)

    or None if it doesn't match the pattern.
    """
    pattern = r"^(\d+)-(\d+)-(\d+)-([^-]+)-(\d+)-(\d+)-(\d+)\.json$"
    match = re.match(pattern, filename)
    if match:
        inlines = int(match.group(1))
        xlines = int(match.group(2))
        samples = int(match.group(3))
        chunk_mode = match.group(4)
        worker_count = int(match.group(5))
        timestamp = int(match.group(6))
        random_id = match.group(7)
        return (
            inlines,
            xlines,
            samples,
            chunk_mode,
            worker_count,
            timestamp,
            random_id,
        )
    return None


def main():
    """
    Collects all JSON profile results from OUTPUT_DIR/profiles, parses them,
    creates a "summary" and a "detail" CSV.

    Additionally, if in a given shape the largest set of worker counts is {1,2,4,8}
    but we only have data for chunk_mode='auto' at 8 workers (not for 'memaware' at 8),
    we automatically insert an OOM row in the summary for that missing combination.
    """
    output_dir = os.getenv("OUTPUT_DIR", "./out/results")
    profiles_dir = os.path.join(output_dir, "profiles")

    if not os.path.isdir(profiles_dir):
        print(f"[collect_results] Profiles directory not found: {profiles_dir}")
        sys.exit(1)

    summary_data = []
    detail_data = []

    # We'll keep track of all (shape, chunk_mode) combos and all worker_counts that appear
    data_catalog = defaultdict(
        lambda: {"all_workers": set(), "chunk_modes": defaultdict(set)}
    )

    # We'll store actual JSON-run-based summary in a dictionary keyed by the shape+params
    actual_summary_entries = {}
    detail_entries_list = []

    # -----------------------------
    # 1) GATHER ACTUAL FILE RESULTS
    # -----------------------------
    all_files = [f for f in os.listdir(profiles_dir) if f.endswith(".json")]
    for filename in sorted(all_files):
        parsed = parse_filename(filename)
        if not parsed:
            print(f"[collect_results] Skipping '{filename}' (doesn't match pattern).")
            continue

        (inlines, xlines, samples, chunk_mode, worker_count, ts, rnd_id) = parsed
        filepath = os.path.join(profiles_dir, filename)

        # Update the "catalog" with the shape/chunking/worker_count
        shape_key = (inlines, xlines, samples)
        data_catalog[shape_key]["all_workers"].add(worker_count)
        data_catalog[shape_key]["chunk_modes"][chunk_mode].add(worker_count)

        try:
            with open(filepath, "r") as f:
                profile_data = json.load(f)
        except Exception as e:
            print(f"[collect_results] Error reading {filepath}: {e}")
            continue

        metadata = profile_data.get("metadata", {})
        data_section = profile_data.get("data", {})

        session_id = metadata.get("session_id", None)
        execution_time = data_section.get("execution_time")  # float or None
        memory_usage = data_section.get("memory_usage", {})

        # If memory_usage is empty, we treat it as OOM or partial failure
        if not memory_usage:
            summary_data.append(
                {
                    "inlines": inlines,
                    "xlines": xlines,
                    "samples": samples,
                    "chunking_mode": chunk_mode,
                    "worker_count": worker_count,
                    "timestamp": ts,
                    "session_id": session_id,
                    "random_id": rnd_id,
                    "execution_time_sec": execution_time,
                    "peak_memory_usage_bytes": None,
                    "avg_memory_usage_bytes": None,
                    "oom_or_failed": True,
                }
            )
            actual_summary_entries[
                (inlines, xlines, samples, chunk_mode, worker_count, ts, rnd_id)
            ] = summary_data[-1]
            continue

        # Collect memory usage
        worker_peaks = []
        worker_avgs = []

        for worker_addr, usage_info in memory_usage.items():
            peak_mem = usage_info.get("peak_memory_usage")
            avg_mem = usage_info.get("avg_memory_usage")
            history = usage_info.get("memory_usage_history", [])

            detail_data.append(
                {
                    "inlines": inlines,
                    "xlines": xlines,
                    "samples": samples,
                    "chunking_mode": chunk_mode,
                    "worker_count": worker_count,
                    "timestamp": ts,
                    "session_id": session_id,
                    "random_id": rnd_id,
                    "worker_addr": worker_addr,
                    "peak_memory_usage_bytes": peak_mem,
                    "avg_memory_usage_bytes": avg_mem,
                    "memory_usage_history": history,
                }
            )

            if peak_mem is not None:
                worker_peaks.append(peak_mem)
            if avg_mem is not None:
                worker_avgs.append(avg_mem)

        overall_peak = max(worker_peaks) if worker_peaks else None
        overall_avg = (sum(worker_avgs) / len(worker_avgs)) if worker_avgs else None

        summary_item = {
            "inlines": inlines,
            "xlines": xlines,
            "samples": samples,
            "chunking_mode": chunk_mode,
            "worker_count": worker_count,
            "timestamp": ts,
            "session_id": session_id,
            "random_id": rnd_id,
            "execution_time_sec": execution_time,
            "peak_memory_usage_bytes": overall_peak,
            "avg_memory_usage_bytes": overall_avg,
            "oom_or_failed": False,
        }

        summary_data.append(summary_item)
        actual_summary_entries[
            (inlines, xlines, samples, chunk_mode, worker_count, ts, rnd_id)
        ] = summary_item

    # -----------------------------------------------------------------------
    # 2) FILL IN MISSING SCENARIOS (OOM) IF A SHAPE USES A GIVEN WORKER COUNT
    #    FOR ANY CHUNK_MODE, THEN EVERY CHUNK_MODE FOR THAT SHAPE GETS IT TOO.
    # -----------------------------------------------------------------------
    for shape_key, shape_info in data_catalog.items():
        (inlines, xlines, samples) = shape_key
        chunk_modes_dict = shape_info["chunk_modes"]
        union_of_all_workers = set()
        for cm in chunk_modes_dict:
            union_of_all_workers |= chunk_modes_dict[cm]

        all_chunk_modes_for_shape = set(chunk_modes_dict.keys())

        # Check for missing combos -> fill as OOM
        for cm in all_chunk_modes_for_shape:
            for wcount in union_of_all_workers:
                found_any = False
                for key_tuple in actual_summary_entries.keys():
                    (
                        sh_inlines,
                        sh_xlines,
                        sh_samples,
                        sh_chunk_mode,
                        sh_wcount,
                        sh_ts,
                        sh_rid,
                    ) = key_tuple
                    if (
                        sh_inlines == inlines
                        and sh_xlines == xlines
                        and sh_samples == samples
                        and sh_chunk_mode == cm
                        and sh_wcount == wcount
                    ):
                        found_any = True
                        break

                if not found_any:
                    summary_data.append(
                        {
                            "inlines": inlines,
                            "xlines": xlines,
                            "samples": samples,
                            "chunking_mode": cm,
                            "worker_count": wcount,
                            "timestamp": None,
                            "session_id": None,
                            "random_id": None,
                            "execution_time_sec": None,
                            "peak_memory_usage_bytes": None,
                            "avg_memory_usage_bytes": None,
                            "oom_or_failed": True,
                        }
                    )

    # --------------------------------------------
    # 3) CREATE DATAFRAMES & SAVE summary + detail
    # --------------------------------------------
    summary_df = pd.DataFrame(summary_data)
    detail_df = pd.DataFrame(detail_data)

    if not summary_df.empty:
        summary_df.sort_values(
            by=[
                "inlines",
                "xlines",
                "samples",
                "chunking_mode",
                "worker_count",
                "timestamp",
            ],
            inplace=True,
            na_position="last",
        )
    if not detail_df.empty:
        detail_df.sort_values(
            by=[
                "inlines",
                "xlines",
                "samples",
                "chunking_mode",
                "worker_count",
                "timestamp",
                "worker_addr",
            ],
            inplace=True,
        )

    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    summary_csv_path = os.path.join(results_dir, "profiles_summary.csv")
    detail_csv_path = os.path.join(results_dir, "profiles_detail.csv")

    summary_df.to_csv(summary_csv_path, index=False)
    detail_df.to_csv(detail_csv_path, index=False)

    print(f"[collect_results] Summary CSV saved to {summary_csv_path}")
    print(f"[collect_results] Detail CSV saved to {detail_csv_path}")
    print("[collect_results] Done.")


if __name__ == "__main__":
    main()
