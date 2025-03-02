import pandas as pd
from loguru import logger

__all__ = ["memory_usage_log_to_df", "page_faults_log_to_df"]


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
