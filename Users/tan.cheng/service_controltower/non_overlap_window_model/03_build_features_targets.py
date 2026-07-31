"""Step 03: build exact 90-day condition features and selected-target labels."""
from __future__ import annotations

import json

import pandas as pd

import config
from modeling_90d import annotate_target_outcomes
from ninety_day_feature_builder import build_condition_features
from pipeline_90d_io import load_selected_target_events


def main() -> None:
    config.STEP_03_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not config.SPLIT_PLAN_PATH.exists():
        raise FileNotFoundError(f"Run Step 02 first: {config.SPLIT_PLAN_PATH}")

    built = build_condition_features(force_rebuild=config.FORCE_REBUILD_CONDITION_FEATURES)
    events, target_summary = load_selected_target_events()
    segment = annotate_target_outcomes(built.segment_features, events)
    anchor = annotate_target_outcomes(built.anchor_features, events)
    segment["target_source"] = config.TARGET_SOURCE
    anchor["target_source"] = config.TARGET_SOURCE

    split_plan = pd.read_csv(
        config.SPLIT_PLAN_PATH,
        usecols=["machine_key", "period_start", "split", "label_complete_90d"],
        parse_dates=["period_start"],
        low_memory=False,
    )
    segment = segment.drop(columns=["label_complete_90d"], errors="ignore").merge(
        split_plan,
        on=["machine_key", "period_start"],
        how="left",
        validate="one_to_one",
    )
    if segment["split"].isna().any():
        raise ValueError("Some segment rows did not match the frozen Step 02 split plan.")

    segment.to_pickle(config.FULL_DATASET_PATH)
    if config.WRITE_FULL_ANCHOR_FEATURES:
        anchor.to_pickle(config.ANCHOR_DATASET_PATH)

    for split_name in ("training", "validation", "locked_test", "purged_gap", "label_incomplete"):
        subset = segment[segment["split"].eq(split_name)].copy()
        subset.to_csv(config.split_dataset_path(split_name), index=False, compression="gzip")

    outcome_summary = (
        segment[segment["split"].isin(["training", "validation", "locked_test"])]
        .groupby("split", as_index=False)
        .agg(
            rows=("machine_key", "size"),
            machines=("machine_key", "nunique"),
            periods=("period_start", "nunique"),
            positives=("target_90d", "sum"),
            prevalence=("target_90d", "mean"),
        )
    )
    outcome_summary.to_csv(config.STEP_03_OUTPUT_DIR / "split_outcome_summary.csv", index=False)
    target_summary.to_csv(config.STEP_03_OUTPUT_DIR / "target_event_summary.csv", index=False)
    built.source_audit.to_csv(config.STEP_03_OUTPUT_DIR / "condition_source_feature_audit.csv", index=False)
    metadata = {
        "target_source": config.TARGET_SOURCE,
        "lookback_days": config.LOOKBACK_DAYS,
        "horizon_days": config.HORIZON_DAYS,
        "machines": int(built.roster["machine_key"].nunique()),
        "segment_rows": int(len(segment)),
        "anchor_rows": int(len(anchor)),
        "target_events": int(len(events)),
        "target_event_machines": int(events["machine_key"].nunique()),
    }
    (config.STEP_03_OUTPUT_DIR / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(outcome_summary.to_string(index=False))
    print(f"\nSegment dataset: {config.FULL_DATASET_PATH}")
    print(f"Anchor dataset:  {config.ANCHOR_DATASET_PATH}")


if __name__ == "__main__":
    main()
