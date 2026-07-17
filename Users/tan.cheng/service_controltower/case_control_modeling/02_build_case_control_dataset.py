"""Step 02: build window-based case-control datasets and train/validation/test splits."""
from __future__ import annotations

import pandas as pd

import config
from cc_utils import (
    build_case_control_base_rows,
    build_machine_master,
    build_window_features,
    ensure_dir,
    load_sources,
    split_case_control_train_validation_test,
    validate_dataset_features,
    window_dataset_id,
    write_json,
)


def _load_episodes() -> pd.DataFrame:
    path = config.OUTPUT_DIR / "01_claim_episodes" / "claim_episodes.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Claim episode file not found: {path}. Run 01_build_claim_episodes.py first."
        )
    episodes = pd.read_csv(path, parse_dates=["claim_date", "episode_end_date"])
    return episodes


def _split_counts(df: pd.DataFrame) -> dict:
    rows = {}
    if "split" not in df.columns:
        return rows
    for split_name, sub in df.groupby("split", dropna=False):
        prefix = str(split_name)
        rows[f"{prefix}_rows"] = int(len(sub))
        rows[f"{prefix}_positive_rows"] = int(sub["target"].sum()) if "target" in sub.columns else None
        rows[f"{prefix}_control_rows"] = int((sub["target"] == 0).sum()) if "target" in sub.columns else None
        rows[f"{prefix}_groups"] = int(sub["case_control_group_id"].nunique(dropna=True)) if "case_control_group_id" in sub.columns else None
    return rows


def run() -> None:
    step_dir = config.OUTPUT_DIR / "02_case_control_datasets"
    ensure_dir(step_dir)

    print("Loading sources for case-control dataset build...")
    sources = load_sources(config, include_operation=True)
    episodes = _load_episodes()
    machine_master = build_machine_master(sources)
    machine_master.to_csv(step_dir / "machine_master_source_coverage.csv", index=False)

    dataset_index_rows = []
    for window_config in config.WINDOW_CONFIGS:
        dataset_id = window_dataset_id(window_config, config)
        dataset_dir = step_dir / dataset_id
        ensure_dir(dataset_dir)
        print(f"Building dataset: {dataset_id}")

        base_rows, control_audit = build_case_control_base_rows(
            episodes=episodes,
            machine_master=machine_master,
            sources=sources,
            window_config=window_config,
            config=config,
        )
        base_rows.to_csv(dataset_dir / "case_control_base_rows.csv", index=False)
        control_audit.to_csv(dataset_dir / "control_sampling_audit.csv", index=False)

        if base_rows.empty:
            print(f"  [WARN] No rows generated for {dataset_id}")
            continue

        full_df = build_window_features(base_rows, sources=sources, episodes=episodes)
        validate_dataset_features(full_df, config)
        full_df, split_summary = split_case_control_train_validation_test(full_df, config)

        full_path = dataset_dir / "case_control_dataset_with_split.csv"
        train_path = dataset_dir / "case_control_training_dataset.csv"
        validation_path = dataset_dir / "case_control_validation_dataset.csv"
        test_path = dataset_dir / "case_control_test_dataset.csv"

        full_df.to_csv(full_path, index=False)
        full_df[full_df["split"].eq("train")].to_csv(train_path, index=False)
        full_df[full_df["split"].eq("validation")].to_csv(validation_path, index=False)
        full_df[full_df["split"].eq("test")].to_csv(test_path, index=False)
        split_summary.to_csv(dataset_dir / "split_summary.csv", index=False)

        group_split = (
            full_df[["case_control_group_id", "split", config.SPLIT_DATE_COL]]
            .drop_duplicates()
            .sort_values([config.SPLIT_DATE_COL, "case_control_group_id"], kind="mergesort")
        )
        group_split.to_csv(dataset_dir / "case_control_group_split_assignments.csv", index=False)

        positive_rows = int((full_df["target"] == 1).sum())
        control_rows = int((full_df["target"] == 0).sum())
        train_df = full_df[full_df["split"].eq("train")]
        validation_df = full_df[full_df["split"].eq("validation")]
        test_df = full_df[full_df["split"].eq("test")]

        summary = {
            "dataset_id": dataset_id,
            "window_config": window_config,
            "lead_max_days": int(window_config["lead_max_days"]),
            "lead_min_days": int(window_config["lead_min_days"]),
            "controls_per_positive_case_requested": int(config.CONTROLS_PER_POSITIVE_CASE),
            "rows": int(len(full_df)),
            "positive_rows": positive_rows,
            "control_rows": control_rows,
            "positive_rate": positive_rows / len(full_df) if len(full_df) else None,
            "unique_case_control_groups": int(full_df["case_control_group_id"].nunique(dropna=True)),
            "unique_machines": int(full_df["machine_key"].nunique(dropna=True)),
            "numeric_features": config.NUMERIC_FEATURES,
            "categorical_features": config.CATEGORICAL_FEATURES,
            "base_numeric_feature_count": len(getattr(config, "BASE_NUMERIC_FEATURES", [])),
            "component_features_enabled": bool(getattr(config, "ENABLE_COMPONENT_FEATURES", False)),
            "component_feature_groups": getattr(config, "COMPONENT_FEATURE_GROUPS", {}),
            "component_numeric_feature_count": len(getattr(config, "COMPONENT_NUMERIC_FEATURES", [])),
            "full_dataset_path": str(full_path),
            "training_dataset_path": str(train_path),
            "validation_dataset_path": str(validation_path),
            "test_dataset_path": str(test_path),
            "split_date_col": str(config.SPLIT_DATE_COL),
            "train_ratio": float(config.TRAIN_RATIO),
            "validation_ratio": float(config.VALIDATION_RATIO),
            "test_ratio": float(config.TEST_RATIO),
            "control_no_claim_days_after_window_end": int(config.CONTROL_NO_CLAIM_DAYS_AFTER_WINDOW_END),
            "control_exclude_prior_claim_days_before_window_start": int(config.CONTROL_EXCLUDE_PRIOR_CLAIM_DAYS_BEFORE_WINDOW_START),
            "control_match_on_full_model": bool(config.CONTROL_MATCH_ON_FULL_MODEL),
            "control_require_source_coverage_overlap_window": bool(config.CONTROL_REQUIRE_SOURCE_COVERAGE_OVERLAP_WINDOW),
            "require_positive_source_coverage_overlap_window": bool(config.REQUIRE_POSITIVE_SOURCE_COVERAGE_OVERLAP_WINDOW),
            "max_positive_cases_per_window": config.MAX_POSITIVE_CASES_PER_WINDOW,
        }
        summary.update(_split_counts(full_df))
        write_json(summary, dataset_dir / "dataset_summary.json")

        # Simple feature-level target-rate sanity table for fast inspection.
        availability_cols = [
            "has_any_source_window",
            "has_fault_window",
            "has_fluid_window",
            "has_maintenance_window",
            "has_operation_window",
        ]
        avail_rows = []
        for col in availability_cols:
            if col in full_df.columns:
                for value, sub in full_df.groupby(col, dropna=False):
                    avail_rows.append({
                        "feature": col,
                        "value": value,
                        "rows": int(len(sub)),
                        "positive_rate": float(sub["target"].mean()) if len(sub) else None,
                    })
        pd.DataFrame(avail_rows).to_csv(dataset_dir / "source_availability_target_rates.csv", index=False)

        dataset_index_row = {
            "dataset_id": dataset_id,
            "window_name": window_config["name"],
            "lead_max_days": int(window_config["lead_max_days"]),
            "lead_min_days": int(window_config["lead_min_days"]),
            "rows": int(len(full_df)),
            "positive_rows": positive_rows,
            "control_rows": control_rows,
            "full_dataset_path": str(full_path),
            "training_dataset_path": str(train_path),
            "validation_dataset_path": str(validation_path),
            "test_dataset_path": str(test_path),
            "dataset_dir": str(dataset_dir),
        }
        dataset_index_row.update(_split_counts(full_df))
        dataset_index_rows.append(dataset_index_row)

        print(
            f"  saved {full_path} rows={len(full_df):,} "
            f"positives={positive_rows:,} controls={control_rows:,}; "
            f"train={len(train_df):,} validation={len(validation_df):,} test={len(test_df):,}"
        )

    dataset_index = pd.DataFrame(dataset_index_rows)
    dataset_index.to_csv(step_dir / "dataset_index.csv", index=False)
    write_json(
        {
            "step": "02_build_case_control_dataset",
            "output_dir": str(step_dir),
            "dataset_count": int(len(dataset_index)),
            "window_configs": config.WINDOW_CONFIGS,
            "controls_per_positive_case": int(config.CONTROLS_PER_POSITIVE_CASE),
            "component_features_enabled": bool(getattr(config, "ENABLE_COMPONENT_FEATURES", False)),
            "component_feature_groups": getattr(config, "COMPONENT_FEATURE_GROUPS", {}),
            "split": {
                "split_date_col": str(config.SPLIT_DATE_COL),
                "train_ratio": float(config.TRAIN_RATIO),
                "validation_ratio": float(config.VALIDATION_RATIO),
                "test_ratio": float(config.TEST_RATIO),
                "group_column": "case_control_group_id",
            },
        },
        step_dir / "run_summary.json",
    )
    print(f"02_build_case_control_dataset completed. Outputs: {step_dir}")


if __name__ == "__main__":
    run()
