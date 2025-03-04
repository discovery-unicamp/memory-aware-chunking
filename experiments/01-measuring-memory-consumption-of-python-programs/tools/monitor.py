import pandas as pd
from loguru import logger

__all__ = [
    "memory_usage_log_to_df",
    "page_faults_log_to_df",
    "memory_pressure_log_to_df",
]


def memory_usage_log_to_df(log_file: str) -> pd.DataFrame:
    logger.info(f"Reading memory usage log) file: {log_file}")

    with open(log_file, "r") as f:
        lines = f.readlines()

    data = []

    for line in lines[1:]:
        timestamp, memory_usage = line.strip().split(",")
        data.append(
            {
                "timestamp": timestamp,
                "memory_usage_kb": int(memory_usage),
            }
        )

    return pd.DataFrame(data)


def page_faults_log_to_df(log_file: str) -> pd.DataFrame:
    logger.info(f"Reading page faults log file: {log_file}")

    with open(log_file, "r") as f:
        lines = f.readlines()

    data = []

    for line in lines[1:]:
        timestamp, minor_page_faults, major_page_faults = line.strip().split(",")
        data.append(
            {
                "timestamp": timestamp,
                "minor_page_faults": int(minor_page_faults),
                "major_page_faults": int(major_page_faults),
            }
        )

    return pd.DataFrame(data)


def memory_pressure_log_to_df(log_file: str) -> pd.DataFrame:
    logger.info(f"Reading memory pressure log file: {log_file}")

    with open(log_file, "r") as f:
        lines = f.readlines()

    data = []

    for line in lines[1:]:
        (
            timestamp,
            some10,
            some60,
            some300,
            full10,
            full60,
            full300,
            oom_kills,
            oom_failures,
        ) = line.strip().split(",")
        data.append(
            {
                "timestamp": timestamp,
                "some10": float(some10),
                "some60": float(some60),
                "some300": float(some300),
                "full10": float(full10),
                "full60": float(full60),
                "full300": float(full300),
                "oom_kills": int(oom_kills),
                "oom_failures": int(oom_failures),
            }
        )

    return pd.DataFrame(data)
