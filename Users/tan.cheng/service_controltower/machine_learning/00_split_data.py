"""Step 00: create chronological train / validation / test split reports."""
from __future__ import annotations

import pandas as pd

import config
from ml_utils import (
    apply_sentinel_cleaning,
    chronological_split,
    ensure_dir,
    make_split_summary,
    read_snapshot,
    resolve_input_path,
    source_features_for_prepared_features,
    validate_source_features,
    write_json,
)


def _ordered_unique(values):
    out = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def _load_and_prepare_snapshot():
    """Load snapshot data, apply configured early drops, and clean 9999 sentinels."""

    df = read_snapshot(config.INPUT_DATA_PATH, config.DATE_COL)

    source_cols_to_drop = [
        c for c in getattr(config, "SOURCE_COLUMNS_TO_DROP_BEFORE_MODELING", []) if c in df.columns
    ]
    if source_cols_to_drop:
        df = df.drop(columns=source_cols_to_drop)

    df, sentinel_report = apply_sentinel_cleaning(
        df=df,
        enabled=getattr(config, "SENTINEL_CLEANING_ENABLED", False),
        sentinel_value=getattr(config, "SENTINEL_VALUE", 9999),
        columns_to_clean=getattr(config, "SENTINEL_COLUMNS_TO_CLEAN", {}),
        replace_with=getattr(config, "SENTINEL_REPLACE_WITH", None),
    )
    return df, source_cols_to_drop, sentinel_report


def _export_reduced_snapshot_dataframe(df, step_dir):
    """Optionally save a source-level reduced dataframe for one model variant.

    The export keeps identifier/date/target columns for traceability, then keeps
    only the source-level feature columns required by the configured model
    variant. It does not include the train/validation/test split column.
    """

    if not getattr(config, "SAVE_REDUCED_SNAPSHOT_DATAFRAME", False):
        return None

    variant = str(getattr(config, "REDUCED_SNAPSHOT_MODEL_VARIANT", "C"))
    if variant not in config.FEATURE_SETS:
        raise ValueError(
            f"REDUCED_SNAPSHOT_MODEL_VARIANT='{variant}' is not defined in config.FEATURE_SETS."
        )

    selected_prepared = config.FEATURE_SETS[variant]
    required_source_features = source_features_for_prepared_features(
        selected_prepared,
        config.PREPARED_TO_SOURCE_FEATURE,
    )
    present_source_features, missing_source_features = validate_source_features(
        df=df,
        source_features=required_source_features,
        error_on_missing=config.ERROR_ON_MISSING_SOURCE_FEATURES,
    )

    required_context_cols = _ordered_unique(
        list(getattr(config, "ID_COLS", []) or [])
        + [getattr(config, "DATE_COL", None), getattr(config, "TARGET_COL", None)]
    )
    missing_context_cols = [c for c in required_context_cols if c not in df.columns]
    if missing_context_cols:
        raise ValueError(
            "Reduced snapshot export requires these context columns, but they are missing: "
            f"{missing_context_cols}"
        )

    # Reduced source-level export: keep machine ID/date/target, then model feature columns.
    # The split column is intentionally excluded.
    export_cols = _ordered_unique(required_context_cols + present_source_features)
    reduced_df = df[export_cols].copy()

    output_filename = getattr(
        config,
        "REDUCED_SNAPSHOT_OUTPUT_FILENAME",
        f"06_snapshot_dataframe_model_{variant}_reduced_snapshot.csv",
    )
    output_path = step_dir / output_filename
    reduced_df.to_csv(output_path, index=False)

    metadata = {
        "model_variant": variant,
        "output_path": str(output_path),
        "rows": int(len(reduced_df)),
        "columns": int(len(reduced_df.columns)),
        "context_columns": required_context_cols,
        "feature_columns": present_source_features,
        "selected_prepared_feature_count": int(len(selected_prepared)),
        "required_source_feature_count": int(len(required_source_features)),
        "present_source_feature_count": int(len(present_source_features)),
        "missing_source_feature_count": int(len(missing_source_features)),
        "missing_source_features": missing_source_features,
        "excluded_columns": {
            "split_col": "split",
        },
        "categorical_features_kept_in_source_format": True,
    }
    write_json(metadata, step_dir / "06_snapshot_dataframe_model_reduced_metadata.json")

    print(
        f"Reduced snapshot for Model {variant} saved: {output_path} "
        f"({len(reduced_df)} rows, {len(required_context_cols)} context columns, "
        f"{len(present_source_features)} feature columns)"
    )
    return metadata


def run() -> None:
    step_dir = config.OUTPUT_DIR / "00_data_split"
    ensure_dir(step_dir)

    resolved_input = resolve_input_path(config.INPUT_DATA_PATH)
    df, source_cols_to_drop, sentinel_report = _load_and_prepare_snapshot()

    if not sentinel_report.empty:
        sentinel_report.to_csv(step_dir / "01b_sentinel_cleaning_report.csv", index=False)
        cleaned_count = int((sentinel_report["status"] == "cleaned").sum())
        print(f"Sentinel cleaning completed for {cleaned_count} configured columns.")

    if config.TARGET_COL not in df.columns:
        raise ValueError(f"TARGET_COL='{config.TARGET_COL}' was not found in input data.")

    split, effective_ratios = chronological_split(
        df=df,
        date_col=config.DATE_COL,
        train_ratio=config.TRAIN_RATIO,
        validation_ratio=config.VALIDATION_RATIO,
        test_ratio=config.TEST_RATIO,
        secondary_sort_cols=config.SECONDARY_SORT_COLS,
    )

    split_summary = make_split_summary(split, config.TARGET_COL, config.DATE_COL, config.ID_COLS)
    split_summary.to_csv(step_dir / "01_split_summary.csv", index=False)
    split.split_assignments.to_csv(step_dir / "02_split_assignments.csv", index=False)

    feature_set_summary = []
    feature_set_detail_rows = []
    for variant, features in config.FEATURE_SETS.items():
        sources = []
        for feature_idx, feature in enumerate(features, start=1):
            source = config.PREPARED_TO_SOURCE_FEATURE[feature]
            if source not in sources:
                sources.append(source)

            feature_set_detail_rows.append(
                {
                    "model_variant": variant,
                    "feature_index": feature_idx,
                    "prepared_feature": feature,
                    "source_feature": source,
                    "is_machine_context_feature": bool(
                        feature in set(getattr(config, "PROTECTED_MACHINE_CONTEXT_PREPARED_FEATURES", []))
                    ),
                    "is_sentinel_indicator_feature": bool(
                        source in set(getattr(config, "SENTINEL_COLUMNS_TO_CLEAN", {}).values())
                    ),
                    "included_in_model_A": feature in set(config.FEATURE_SETS.get("A", [])),
                    "included_in_model_B": feature in set(config.FEATURE_SETS.get("B", [])),
                    "included_in_model_C": feature in set(config.FEATURE_SETS.get("C", [])),
                    "included_in_model_D": feature in set(config.FEATURE_SETS.get("D", [])),
                }
            )

        missing_sources = [c for c in sources if c not in df.columns]
        feature_set_summary.append(
            {
                "model_variant": variant,
                "selected_prepared_feature_count": len(features),
                "required_source_feature_count": len(sources),
                "missing_required_source_feature_count": len(missing_sources),
                "missing_required_source_features": ";".join(missing_sources),
            }
        )

    pd.DataFrame(feature_set_summary).to_csv(step_dir / "03_feature_set_summary.csv", index=False)

    feature_set_detail_df = pd.DataFrame(feature_set_detail_rows)
    feature_set_detail_df.to_csv(step_dir / "04_feature_sets_prepared_features.csv", index=False)

    for variant in ["C", "D"]:
        if variant in config.FEATURE_SETS:
            model_features_df = feature_set_detail_df[
                feature_set_detail_df["model_variant"] == variant
            ].copy()
            if variant == "D":
                model_features_df.insert(
                    5,
                    "model_d_feature_source_reason",
                    model_features_df.apply(
                        lambda row: "retained_from_model_C"
                        if row["prepared_feature"] in set(config.FEATURE_SETS.get("C", []))
                        else "added_back_machine_context",
                        axis=1,
                    ),
                )
            model_features_df.to_csv(step_dir / f"05_model_{variant}_prepared_features.csv", index=False)
            print(
                f"Model {variant} prepared feature list saved: "
                f"{step_dir / f'05_model_{variant}_prepared_features.csv'} "
                f"({len(model_features_df)} features)"
            )

    reduced_snapshot_metadata = _export_reduced_snapshot_dataframe(df, step_dir)

    write_json(
        {
            "step": "00_split_data",
            "input_data_path_configured": str(config.INPUT_DATA_PATH),
            "input_data_path_resolved": str(resolved_input),
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "target_col": config.TARGET_COL,
            "date_col": config.DATE_COL,
            "configured_split_ratios": {
                "train": config.TRAIN_RATIO,
                "validation": config.VALIDATION_RATIO,
                "test": config.TEST_RATIO,
            },
            "effective_normalized_split_ratios": effective_ratios,
            "source_columns_dropped_before_modeling": source_cols_to_drop,
            "sentinel_cleaning_enabled": bool(getattr(config, "SENTINEL_CLEANING_ENABLED", False)),
            "sentinel_cleaning_report_file": "01b_sentinel_cleaning_report.csv" if not sentinel_report.empty else None,
            "reduced_snapshot_export_enabled": bool(getattr(config, "SAVE_REDUCED_SNAPSHOT_DATAFRAME", False)),
            "reduced_snapshot_export": reduced_snapshot_metadata,
            "output_dir": str(step_dir),
        },
        step_dir / "00_run_summary.json",
    )
    print(f"00_split_data completed. Outputs: {step_dir}")


if __name__ == "__main__":
    run()
