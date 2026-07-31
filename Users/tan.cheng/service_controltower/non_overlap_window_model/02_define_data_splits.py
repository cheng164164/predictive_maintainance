"""Step 02: define exact 90-day chronological train/validation/test periods."""
from __future__ import annotations

import pandas as pd

import config
from modeling_90d import define_native_split
from ninety_day_feature_builder import build_machine_roster, build_segment_grid
from pipeline_90d_io import load_selected_target_events, write_json


def main() -> None:
    config.STEP_02_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    roster = build_machine_roster()
    grid = build_segment_grid(roster)
    events, summary = load_selected_target_events()
    file_observation_end = pd.Timestamp(events["event_date"].max()).normalize()
    label_observation_end = (
        pd.Timestamp(config.LABEL_OBSERVATION_END_OVERRIDE).normalize()
        if config.LABEL_OBSERVATION_END_OVERRIDE
        else file_observation_end
    )
    if label_observation_end > file_observation_end:
        raise ValueError(
            f"LABEL_OBSERVATION_END_OVERRIDE={label_observation_end.date()} exceeds "
            f"the selected target file end date {file_observation_end.date()}."
        )

    split_dates = define_native_split(grid, label_observation_end)
    grid["target_window_start"] = grid["period_end"] + pd.Timedelta(days=1)
    grid["target_window_end"] = grid["period_end"] + pd.Timedelta(days=config.HORIZON_DAYS)
    grid["label_complete_90d"] = grid["target_window_end"].le(label_observation_end).astype("int8")
    grid["split"] = "purged_gap"
    grid.loc[grid["label_complete_90d"].eq(0), "split"] = "label_incomplete"
    for name, starts in split_dates.items():
        grid.loc[grid["period_start"].isin(starts), "split"] = name

    grid.to_csv(config.SPLIT_PLAN_PATH, index=False, compression="gzip")
    period_summary = (
        grid.groupby(["period_start", "period_end", "split", "label_complete_90d"], as_index=False)
        .agg(rows=("machine_key", "size"), machines=("machine_key", "nunique"))
        .sort_values("period_start")
    )
    period_summary.to_csv(config.STEP_02_OUTPUT_DIR / "split_period_assignments.csv", index=False)
    split_summary = (
        grid.groupby("split", as_index=False)
        .agg(
            rows=("machine_key", "size"),
            machines=("machine_key", "nunique"),
            periods=("period_start", "nunique"),
            first_period_start=("period_start", "min"),
            last_period_start=("period_start", "max"),
        )
    )
    split_summary.to_csv(config.STEP_02_OUTPUT_DIR / "split_summary.csv", index=False)
    summary.to_csv(config.STEP_02_OUTPUT_DIR / "target_event_summary.csv", index=False)
    write_json(
        {
            "target_source": config.TARGET_SOURCE,
            "target_file_observation_end": str(file_observation_end.date()),
            "label_observation_end": str(label_observation_end.date()),
            "lookback_days": config.LOOKBACK_DAYS,
            "segment_stride_days": config.SEGMENT_STRIDE_DAYS,
            "horizon_days": config.HORIZON_DAYS,
            "splits": {
                name: [str(pd.Timestamp(value).date()) for value in values]
                for name, values in split_dates.items()
            },
        },
        config.SPLIT_DEFINITION_PATH,
    )
    print(split_summary.to_string(index=False))
    print(f"\nSplit definition: {config.SPLIT_DEFINITION_PATH}")


if __name__ == "__main__":
    main()
