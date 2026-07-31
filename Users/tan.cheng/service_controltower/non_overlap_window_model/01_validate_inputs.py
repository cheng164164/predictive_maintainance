"""Step 01: validate configured source files and the selected target source."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import config

CONDITION_REQUIRED: dict[str, tuple[Path, set[str]]] = {
    "fault_codes": (
        config.FAULT_CODES_PATH,
        {"full_model", "serial_number", "event_date", "fault_code"},
    ),
    "fluid_samples": (
        config.FLUID_SAMPLES_PATH,
        {"FULL_MODEL", "SERIAL", "sample_drawn_date", "LABS_SAMPLE_NUMBER"},
    ),
    "maintenance": (
        config.MAINTENANCE_PATH,
        {"full_model", "SERIAL", "event_date", "EVENT_NAME_ID"},
    ),
    "operation": (
        config.OPERATION_PATH,
        {"full_model", "SERIAL", "LOCAL_DATE", "smr_hours"},
    ),
}

TARGET_REQUIRED = {
    "warranty": (
        config.WARRANTY_PATH,
        {"full_model", "serial", "local_date", "claim_type_description", "failure_smr", "critical_fail_part_number"},
    ),
    "physical_failure": (
        config.PHYSICAL_FAILURE_PATH,
        {"machine", "event_date"},
    ),
}


def _check(name: str, path: Path, required: set[str]) -> dict[str, object]:
    row: dict[str, object] = {"source": name, "path": str(path), "exists": path.exists()}
    if not path.exists():
        row.update(status="missing_file", missing_columns=sorted(required))
        return row
    columns = set(pd.read_csv(path, nrows=0).columns)
    missing = sorted(required - columns)
    row.update(
        status="ok" if not missing else "missing_columns",
        available_column_count=len(columns),
        required_column_count=len(required),
        missing_columns=missing,
    )
    return row


def main() -> None:
    if config.TARGET_SOURCE not in TARGET_REQUIRED:
        raise ValueError("TARGET_SOURCE must be 'warranty' or 'physical_failure'.")
    config.STEP_01_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [_check(name, path, columns) for name, (path, columns) in CONDITION_REQUIRED.items()]
    target_path, target_columns = TARGET_REQUIRED[config.TARGET_SOURCE]
    rows.append(_check(f"target_{config.TARGET_SOURCE}", target_path, target_columns))
    inventory = pd.DataFrame(rows)
    inventory.to_csv(config.STEP_01_OUTPUT_DIR / "source_file_inventory.csv", index=False)
    failed = inventory["status"].ne("ok").any()
    summary = {
        "status": "failed" if failed else "passed",
        "target_source": config.TARGET_SOURCE,
        "lookback_days": config.LOOKBACK_DAYS,
        "horizon_days": config.HORIZON_DAYS,
        "data_dir": str(config.DATA_DIR),
        "sources": rows,
    }
    (config.STEP_01_OUTPUT_DIR / "input_validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(inventory.to_string(index=False))
    print(f"\nSelected target source: {config.TARGET_SOURCE}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
