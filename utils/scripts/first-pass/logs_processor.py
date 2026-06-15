from pathlib import Path
import math
import re

import pandas as pd


LOGS_DIR = Path("../../../logs/first-pass")
OUTPUT_DIR = Path("../../../logs_processed")
CRASH_LINE_PREFIX = "Traceback"

LOG_LINE_PATTERN = re.compile(
    r"^iter\s+(?P<iteration_number>\d+)\s+\|\s+"
    r"reward\s+(?P<reward_min>\S+)\/\s*(?P<reward_mean>\S+)\/\s*(?P<reward_max>\S+)\s+\|\s+"
    r"target>=6\s+(?P<target>yes|no)\s+\|"
)


def parse_log_file(log_file: Path) -> pd.DataFrame:
    rows = []
    has_started = False

    with log_file.open("r", encoding="utf-8") as file:
        for line in file:
            if not has_started:
                if line.startswith("iter"):
                    has_started = True
                else:
                    continue

            match = LOG_LINE_PATTERN.match(line)
            if not match:
                if line.startswith(CRASH_LINE_PREFIX):
                    break
                continue

            reward = float(match.group("reward_max"))
            if math.isnan(reward):
                reward = 0.0

            rows.append(
                {
                    "iteration_number": int(match.group("iteration_number")),
                    "reward": reward,
                    "is_target_reached": match.group("target") == "yes",
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log_files = sorted(LOGS_DIR.glob("*.log"))
    for log_file in log_files:
        dataframe = parse_log_file(log_file)
        output_file = OUTPUT_DIR / f"{log_file.stem}.csv"

        if dataframe.empty:
            if output_file.exists():
                output_file.unlink()
            print(f"Skipped {log_file} because it has no parsed rows")
            continue

        dataframe.to_csv(output_file, index=False)
        print(f"Saved {len(dataframe)} rows to {output_file}")


if __name__ == "__main__":
    main()
