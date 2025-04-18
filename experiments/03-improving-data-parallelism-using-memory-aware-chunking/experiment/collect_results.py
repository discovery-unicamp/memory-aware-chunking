"""
Collect and consolidate JSON memory-profile results into summary and detail CSV files.
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import pandas as pd

FILENAME_PATTERN = re.compile(
    r"^(?P<inlines>\d+)-(?P<xlines>\d+)-(?P<samples>\d+)-"
    r"(?P<mode>[^-]+)-(?P<workers>\d+)-(?P<ts>\d+)-(?P<rnd>\d+)\.json$"
)
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./out")
PROFILES_DIR = os.getenv("PROFILES_DIR", f"{OUTPUT_DIR}/profiles")
RESULTS_DIR = os.getenv("RESULTS_DIR", f"{OUTPUT_DIR}/results")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Consolidate JSON profiles into CSV summaries and details."
    )
    parser.add_argument(
        "--profiles-dir", default=PROFILES_DIR, help="Directory with .json profiles"
    )
    parser.add_argument(
        "--results-dir", default=RESULTS_DIR, help="Output directory for CSV files"
    )
    return parser.parse_args()


def parse_filename(name):
    match = FILENAME_PATTERN.match(name)
    if not match:
        return None
    parts = match.groupdict()
    return (
        int(parts["inlines"]),
        int(parts["xlines"]),
        int(parts["samples"]),
        parts["mode"],
        int(parts["workers"]),
        parts["ts"],
        parts["rnd"],
    )


def make_summary(
    inlines,
    xlines,
    samples,
    mode,
    workers,
    ts,
    session_id,
    rnd,
    exec_time,
    peak,
    avg,
    oom,
):
    return {
        "inlines": inlines,
        "xlines": xlines,
        "samples": samples,
        "chunking_mode": mode,
        "worker_count": workers,
        "timestamp": ts,
        "session_id": session_id,
        "random_id": rnd,
        "execution_time_sec": exec_time,
        "peak_memory_usage_bytes": peak,
        "avg_memory_usage_bytes": avg,
        "oom_or_failed": oom,
    }


def fill_missing(summary, catalog, seen):
    for shape, info in catalog.items():
        inl, xl, smp = shape
        workers_union = set().union(*info["modes"].values())
        for mode, wset in info["modes"].items():
            for w in workers_union:
                key = (inl, xl, smp, mode, w)
                if not any(
                    (inl, xl, smp, mode, w)
                    == (
                        s["inlines"],
                        s["xlines"],
                        s["samples"],
                        s["chunking_mode"],
                        s["worker_count"],
                    )
                    for s in summary
                ):
                    summary.append(
                        make_summary(
                            inl,
                            xl,
                            smp,
                            mode,
                            w,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            True,
                        )
                    )


def main():
    args = parse_args()
    profiles = args.profiles_dir
    results_dir = args.results_dir
    if not os.path.isdir(profiles):
        print(f"Profiles dir not found: {profiles}")
        sys.exit(1)

    summary = []
    detail = []
    catalog = defaultdict(lambda: {"workers": set(), "modes": defaultdict(set)})

    for fname in sorted(os.listdir(profiles)):
        if not fname.endswith(".json"):
            continue
        parsed = parse_filename(fname)
        if not parsed:
            print(f"Skipping {fname}")
            continue
        inl, xl, smp, mode, workers, ts, rnd = parsed
        path = os.path.join(profiles, fname)
        catalog[(inl, xl, smp)]["workers"].add(workers)
        catalog[(inl, xl, smp)]["modes"][mode].add(workers)

        try:
            data = json.load(open(path))
        except Exception as e:
            print(f"Error reading {path}: {e}")
            continue

        meta = data.get("metadata", {})
        body = data.get("data", {})
        sid = meta.get("session_id")
        exec_time = body.get("execution_time")
        mem = body.get("memory_usage", {})

        if not mem:
            summary.append(
                make_summary(
                    inl,
                    xl,
                    smp,
                    mode,
                    workers,
                    ts,
                    sid,
                    rnd,
                    exec_time,
                    None,
                    None,
                    True,
                )
            )
            continue

        peaks, avgs = [], []
        for addr, info in mem.items():
            p = info.get("peak_memory_usage")
            a = info.get("avg_memory_usage")
            hist = info.get("memory_usage_history", [])
            detail.append(
                {
                    "inlines": inl,
                    "xlines": xl,
                    "samples": smp,
                    "chunking_mode": mode,
                    "worker_count": workers,
                    "timestamp": ts,
                    "session_id": sid,
                    "random_id": rnd,
                    "worker_addr": addr,
                    "peak_memory_usage_bytes": p,
                    "avg_memory_usage_bytes": a,
                    "memory_usage_history": hist,
                }
            )
            if p is not None:
                peaks.append(p)
            if a is not None:
                avgs.append(a)

        peak_all = max(peaks) if peaks else None
        avg_all = sum(avgs) / len(avgs) if avgs else None
        summary.append(
            make_summary(
                inl,
                xl,
                smp,
                mode,
                workers,
                ts,
                sid,
                rnd,
                exec_time,
                peak_all,
                avg_all,
                False,
            )
        )

    fill_missing(summary, catalog, None)

    os.makedirs(results_dir, exist_ok=True)
    pd.DataFrame(summary).sort_values(
        ["inlines", "xlines", "samples", "chunking_mode", "worker_count", "timestamp"]
    ).to_csv(os.path.join(results_dir, "profiles_summary.csv"), index=False)
    pd.DataFrame(detail).sort_values(
        [
            "inlines",
            "xlines",
            "samples",
            "chunking_mode",
            "worker_count",
            "timestamp",
            "worker_addr",
        ]
    ).to_csv(os.path.join(results_dir, "profiles_detail.csv"), index=False)

    print(f"Saved summary and detail CSVs to {results_dir}")


if __name__ == "__main__":
    main()
