from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")

LOGS_DIR = Path("../../../logs_processed/second-pass")
OUTPUT_FILE = LOGS_DIR /"aggregated.csv"

FILE_NAME_PATTERN = re.compile(
    r"^job_(?P<job_id>\d+)_iter_(?P<iteration>\d+)_batch_(?P<batch>\d+)_sgd_(?P<sgd>\d+)_entropy_(?P<entropy>[\d_]+)\.csv$"
)


def parse_file_name(file_name: str) -> dict[str, int | float]:
    match = FILE_NAME_PATTERN.match(file_name)
    if match is None:
        raise ValueError(f"Unexpected file name format: {file_name}")

    entropy_token = match.group("entropy").replace("_", ".", 1)

    return {
        "job_id"            : int(match.group("job_id")),
        "iteration"         : int(match.group("iteration")),
        "train_batch_size"  : int(match.group("batch")),
        "sgd_minibatch_size": int(match.group("sgd")),
        "entropy"           : float(entropy_token),
    }


def build_aggregate_row(csv_file: Path) -> dict[str, object]:
    parsed_values = parse_file_name(csv_file.name)
    dataframe = pd.read_csv(csv_file)

    max_reward = float(dataframe["reward"].max())
    is_target_reached = (max_reward >= 6.0)
    is_training_stable = len(dataframe) == parsed_values["iteration"]

    return {
        **parsed_values,
        "max_reward"       : max_reward,
        "is_target_reached": is_target_reached,
        "is_training_stable": is_training_stable,
    }


def main() -> None:
    aggregate_rows = []

    for csv_file in sorted(LOGS_DIR.glob("*.csv")):
        if csv_file.name == OUTPUT_FILE.name:
            logging.info("skipping aggregated.csv, it is most likely from a previous run")
            continue
        if FILE_NAME_PATTERN.match(csv_file.name) is None:
            logging.error("Unexpected file name format: %s", csv_file.name)
            continue

        aggregate_rows.append(build_aggregate_row(csv_file))

    df= pd.DataFrame(aggregate_rows)
    df.sort_values("job_id", inplace=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved {len(df)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
