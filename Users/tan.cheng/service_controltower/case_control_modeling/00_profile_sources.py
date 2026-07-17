"""Step 00: profile source files used for case-control modeling."""
from __future__ import annotations

import pandas as pd

import config
from cc_utils import ensure_dir, load_sources, write_json


def _profile_one(name: str, df: pd.DataFrame, date_col: str | None = None) -> dict:
    row = {
        "source": name,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "unique_machines": int(df["machine_key"].nunique(dropna=True)) if "machine_key" in df.columns else None,
    }
    if date_col and date_col in df.columns and len(df):
        row["date_min"] = pd.to_datetime(df[date_col]).min()
        row["date_max"] = pd.to_datetime(df[date_col]).max()
    return row


def run() -> None:
    step_dir = config.OUTPUT_DIR / "00_source_profile"
    ensure_dir(step_dir)
    sources = load_sources(config, include_operation=True)

    profile_rows = [
        _profile_one("warranty", sources["warranty"], "claim_date"),
        _profile_one("fault", sources["fault"], "event_date"),
        _profile_one("fluid", sources["fluid"], "sample_drawn_date"),
        _profile_one("maintenance", sources["maintenance"], "event_date"),
        _profile_one("operation", sources["operation"], "LOCAL_DATE"),
    ]
    profile = pd.DataFrame(profile_rows)
    profile.to_csv(step_dir / "source_profile_summary.csv", index=False)

    warranty_machines = set(sources["warranty"]["machine_key"].dropna().unique())
    overlap_rows = []
    for name in ["fault", "fluid", "maintenance", "operation"]:
        machines = set(sources[name]["machine_key"].dropna().unique())
        overlap_rows.append({
            "source": name,
            "source_unique_machines": len(machines),
            "warranty_unique_machines": len(warranty_machines),
            "source_machine_overlap_with_warranty": len(machines & warranty_machines),
            "warranty_machine_coverage_rate": len(machines & warranty_machines) / len(warranty_machines) if warranty_machines else None,
        })
    pd.DataFrame(overlap_rows).to_csv(step_dir / "source_warranty_overlap_summary.csv", index=False)

    write_json(
        {
            "step": "00_profile_sources",
            "source_dir": str(config.SOURCE_DIR),
            "output_dir": str(step_dir),
            "max_valid_event_date": config.MAX_VALID_EVENT_DATE,
            "min_valid_event_date": config.MIN_VALID_EVENT_DATE,
        },
        step_dir / "run_summary.json",
    )
    print(f"00_profile_sources completed. Outputs: {step_dir}")


if __name__ == "__main__":
    run()
