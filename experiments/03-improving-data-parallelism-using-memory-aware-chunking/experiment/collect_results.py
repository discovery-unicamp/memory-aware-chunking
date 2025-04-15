"""
Parses all JSON memory profile results found in OUTPUT_DIR/profiles, then
consolidates them into two CSV files: a summary (one row per run) and a
detail (one row per worker per run).

Missing (OOM) combinations are inferred for shapes that have data in one
chunk mode but not in others, ensuring consistent coverage.
"""

import json
import os
import re
import sys
from collections import defaultdict

import pandas as pd


def main():
    output_dir = os.getenv("OUTPUT_DIR", "./out/results")
    profiles_dir = os.path.join(output_dir, "profiles")

    if not os.path.isdir(profiles_dir):
        print(f"[collect_results] Profiles directory not found: {profiles_dir}")
        sys.exit(1)

    summary_data = []
    detail_data = []

    # Catalog to track shapes, chunk_modes, and worker counts in runs
    data_catalog = defaultdict(
        lambda: {"all_workers": set(), "chunk_modes": defaultdict(set)}
    )
    actual_summary_entries = {}

    all_files = sorted(f for f in os.listdir(profiles_dir) if f.endswith(".json"))
    for filename in all_files:
        parsed = _parse_filename(filename)
        if not parsed:
            print(f"[collect_results] Skipping '{filename}' (doesn't match pattern).")
            continue

        (inlines, xlines, samples, chunk_mode, worker_count, ts, rnd_id) = parsed
        filepath = os.path.join(profiles_dir, filename)

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
        session_id = metadata.get("session_id")
        execution_time = data_section.get("execution_time")
        memory_usage = data_section.get("memory_usage", {})

        # If memory usage is empty, mark as OOM or failed
        if not memory_usage:
            summary_data.append(
                _make_summary_dict(
                    inlines,
                    xlines,
                    samples,
                    chunk_mode,
                    worker_count,
                    ts,
                    session_id,
                    rnd_id,
                    execution_time,
                    peak_mem=None,
                    avg_mem=None,
                    oom=True,
                )
            )
            actual_summary_entries[
                (inlines, xlines, samples, chunk_mode, worker_count, ts, rnd_id)
            ] = summary_data[-1]
            continue

        # Otherwise collect stats and detail rows
        worker_peaks = []
        worker_avgs = []
        for worker_addr, usage_info in memory_usage.items():
            peak = usage_info.get("peak_memory_usage")
            avg = usage_info.get("avg_memory_usage")
            hist = usage_info.get("memory_usage_history", [])

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
                    "peak_memory_usage_bytes": peak,
                    "avg_memory_usage_bytes": avg,
                    "memory_usage_history": hist,
                }
            )

            if peak is not None:
                worker_peaks.append(peak)
            if avg is not None:
                worker_avgs.append(avg)

        overall_peak = max(worker_peaks) if worker_peaks else None
        overall_avg = sum(worker_avgs) / len(worker_avgs) if worker_avgs else None

        summary_entry = _make_summary_dict(
            inlines,
            xlines,
            samples,
            chunk_mode,
            worker_count,
            ts,
            session_id,
            rnd_id,
            execution_time,
            peak_mem=overall_peak,
            avg_mem=overall_avg,
            oom=False,
        )
        summary_data.append(summary_entry)
        actual_summary_entries[
            (inlines, xlines, samples, chunk_mode, worker_count, ts, rnd_id)
        ] = summary_entry

    # Fill in missing (OOM) rows if shape+worker_count is used in one chunk mode but not another
    _fill_missing_oom_rows(summary_data, data_catalog, actual_summary_entries)

    # Convert to DataFrame, sort, and save
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


def _parse_filename(filename: str):
    """
    Expected format:
      <inlines>-<xlines>-<samples>-<chunk_mode>-<worker_count>-<timestamp>-<random_id>.json
    Returns a tuple of (inlines, xlines, samples, chunk_mode, worker_count, timestamp, random_id) or None.
    """
    pattern = r"^(\d+)-(\d+)-(\d+)-([^-]+)-(\d+)-(\d+)-(\d+)\.json$"
    match = re.match(pattern, filename)
    if not match:
        return None

    return (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3)),
        match.group(4),
        int(match.group(5)),
        int(match.group(6)),
        match.group(7),
    )


def _make_summary_dict(
    inlines,
    xlines,
    samples,
    chunk_mode,
    worker_count,
    ts,
    session_id,
    rnd_id,
    exec_time,
    peak_mem,
    avg_mem,
    oom,
):
    return {
        "inlines": inlines,
        "xlines": xlines,
        "samples": samples,
        "chunking_mode": chunk_mode,
        "worker_count": worker_count,
        "timestamp": ts,
        "session_id": session_id,
        "random_id": rnd_id,
        "execution_time_sec": exec_time,
        "peak_memory_usage_bytes": peak_mem,
        "avg_memory_usage_bytes": avg_mem,
        "oom_or_failed": oom,
    }


def _fill_missing_oom_rows(summary_data, data_catalog, actual_summary_entries):
    """
    For each shape, if we have multiple chunking modes, ensure that worker
    counts used by any mode are also present in others. If missing, insert OOM row.
    """
    for shape_key, shape_info in data_catalog.items():
        (inlines, xlines, samples) = shape_key
        chunk_modes_dict = shape_info["chunk_modes"]
        union_of_all_workers = set()
        for cm in chunk_modes_dict:
            union_of_all_workers |= chunk_modes_dict[cm]

        all_modes_for_shape = set(chunk_modes_dict.keys())
        for cm in all_modes_for_shape:
            for wc in union_of_all_workers:
                found_any = any(
                    k[0] == inlines
                    and k[1] == xlines
                    and k[2] == samples
                    and k[3] == cm
                    and k[4] == wc
                    for k in actual_summary_entries
                )
                if not found_any:
                    summary_data.append(
                        _make_summary_dict(
                            inlines,
                            xlines,
                            samples,
                            cm,
                            wc,
                            ts=None,
                            session_id=None,
                            rnd_id=None,
                            exec_time=None,
                            peak_mem=None,
                            avg_mem=None,
                            oom=True,
                        )
                    )


if __name__ == "__main__":
    main()
