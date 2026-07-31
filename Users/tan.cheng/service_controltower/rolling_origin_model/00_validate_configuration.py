"""Validate local paths, target selection, and target-horizon coverage."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
from target_events import load_target_events


def main() -> None:
    required = {
        "fault": Path(config.FAULT_FILE),
        "fluid": Path(config.FLUID_FILE),
        "maintenance": Path(config.MAINTENANCE_FILE),
        "operation": Path(config.OPERATION_RAW_FILE),
        "target": Path(config.TARGET_FILE),
    }
    missing = {name: path for name, path in required.items() if not path.exists()}
    if missing:
        formatted = "\n".join(f"  {name}: {path}" for name, path in missing.items())
        raise FileNotFoundError(f"Missing required files:\n{formatted}")

    events = load_target_events()
    data_through_proxy = pd.Timestamp(events.by_machine_day["event_date"].max())
    latest_training_snapshot = pd.Timestamp(config.TRAIN_SNAPSHOT_END)
    required_training_through = latest_training_snapshot + pd.Timedelta(
        days=config.HORIZON_DAYS - 1
    )

    print("Configuration is valid")
    print(f"  project directory: {config.PROJECT_DIR}")
    print(f"  enriched data:     {config.DATA_DIR}")
    print(f"  target source:     {config.TARGET_SOURCE}")
    print(f"  target file:       {config.TARGET_FILE}")
    print(f"  target machine-days: {len(events.by_machine_day):,}")
    print(f"  target max event date: {data_through_proxy.date()}")
    print(f"  output directory:  {config.OUTPUT_DIR}")
    print(f"  model directory:   {config.MODEL_DIR}")

    if data_through_proxy < required_training_through:
        print(
            "WARNING: the maximum observed target-event date precedes the end "
            "of the latest configured 90-day training outcome window. This may "
            "mean the last training/anchor dates are right-censored, or simply "
            "that no events occurred near the extract end. Set dates based on "
            "the known source refresh date, not only the maximum event date."
        )
        print(f"  latest training snapshot: {latest_training_snapshot.date()}")
        print(f"  outcome data needed through: {required_training_through.date()}")


if __name__ == "__main__":
    main()
